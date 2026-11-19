"""Solvers for Objective 1: maximise the final score.

random, greedy, expectimax (d2-d8, plus a transposition-table version used for
d6-d8), minimax with alpha-beta (d2-d8, plus d5 with 4-tile spawns) and flat
Monte Carlo rollout search (60, 300, 1000 rollouts per move). MCTS-3000 runs
on a TPU and lives in mcts_tpu.py.

Search depth counts plies: player move, tile spawn, player move, ...
so depth d plans ceil(d/2) player moves.
"""
import math
import os
import random
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

from engine import apply_move, apply_move_grid, spawn_options, empty_cells, \
    get_empty_cells, DIRECTIONS, DIR_CODE, max_tile, Game


@njit(cache=True)
def heuristic_eval(grid):
    """Board score: empty cells, monotonic rows/columns, smoothness, max tile in a
    corner, and the max tile itself.
    """
    h, w = grid.shape
    empties = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                empties += 1

    mono = 0.0
    for r in range(h):
        inc = 0.0
        dec = 0.0
        for c in range(w - 1):
            diff = float(grid[r, c + 1]) - float(grid[r, c])
            if diff > 0:
                inc += diff
            else:
                dec += -diff
        mono -= min(inc, dec)
    for c in range(w):
        inc = 0.0
        dec = 0.0
        for r in range(h - 1):
            diff = float(grid[r + 1, c]) - float(grid[r, c])
            if diff > 0:
                inc += diff
            else:
                dec += -diff
        mono -= min(inc, dec)

    smooth = 0.0
    for r in range(h):
        for c in range(w - 1):
            if grid[r, c] != 0 and grid[r, c + 1] != 0:
                smooth -= abs(float(grid[r, c]) - float(grid[r, c + 1]))
    for r in range(h - 1):
        for c in range(w):
            if grid[r, c] != 0 and grid[r + 1, c] != 0:
                smooth -= abs(float(grid[r, c]) - float(grid[r + 1, c]))

    m = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] > m:
                m = grid[r, c]

    corner_bonus = 0.0
    if grid[0, 0] == m or grid[0, w - 1] == m or grid[h - 1, 0] == m or grid[h - 1, w - 1] == m:
        corner_bonus = float(m)

    return empties * 270.0 + mono * 1.0 + smooth * 0.1 + corner_bonus * 3.0 + float(m)


_TT_EMPTY_TAG = np.int64(-9223372036854775808)  # int64 min, used as "slot never written" sentinel


@njit(cache=True, inline="always")
def _tt_array_lookup(h1, h2, tag_table, val_table):
    idx = h1 & (tag_table.shape[0] - 1)
    if tag_table[idx] == h2:
        return True, val_table[idx]
    return False, 0.0


@njit(cache=True, inline="always")
def _tt_array_store(h1, h2, value, tag_table, val_table):
    idx = h1 & (tag_table.shape[0] - 1)
    tag_table[idx] = h2
    val_table[idx] = value


def make_tt_tables(size_bits=20):
    """Allocate an empty transposition table (tags and values) with 2**size_bits slots."""
    size = 1 << size_bits
    tag_table = np.full(size, _TT_EMPTY_TAG, dtype=np.int64)
    val_table = np.zeros(size, dtype=np.float64)
    return tag_table, val_table


# ---------- Opening-book warm start (precomputed, read-only, per-process) ----------

def save_tt_tables(tag_table, val_table, path):
    np.savez(path, tag=tag_table, val=val_table)


def load_tt_tables(path):
    data = np.load(path)
    return data["tag"].copy(), data["val"].copy()


def default_book_path(width, height, depth, book_dir="."):
    """File name of the opening book for a board size and depth."""
    return os.path.join(book_dir, f"book_{width}x{height}_d{depth}.npz")


def build_opening_book(width, height, depth, seeds, size_bits=20):
    """Pre-fill a transposition table by searching the first move of several games."""
    tag_table, val_table = make_tt_tables(size_bits=size_bits)
    for seed in seeds:
        g = Game(width=width, height=height, seed=seed)
        moves = g.legal_moves_with_results()
        for d, new_grid, gained in moves:
            expectimax_value_tt(new_grid, depth - 1, False, tag_table, val_table)
    return tag_table, val_table


@njit(cache=True, inline="always")
def _tt_hash(grid, depth, is_player):
    """Two 64-bit hashes of (grid, depth, is_player) used as the table key."""
    h1 = np.int64(-3750763034362895579)  # fnv offset basis variants
    h2 = np.int64(1469598103934665603)
    prime1 = np.int64(1099511628211)
    prime2 = np.int64(2654435761)
    flat = grid.ravel()
    for i in range(flat.shape[0]):
        v = np.int64(flat[i]) + np.int64(1)
        h1 = (h1 ^ v) * prime1
        h2 = (h2 ^ (v * prime2)) * prime1
    turn = np.int64(1) if is_player else np.int64(0)
    h1 = (h1 ^ np.int64(depth) ^ (turn << 1)) * prime1
    h2 = (h2 ^ (np.int64(depth) * prime2) ^ turn) * prime1
    return h1, h2


@njit(cache=True)
def expectimax_value_tt(grid, depth, is_player, tag_table, val_table):
    """Expectimax with a transposition table; same values as expectimax_value, computed
    with an explicit stack.
    """
    h, w = grid.shape
    max_frames = depth + 2
    max_cells = h * w

    s_grid = np.zeros((max_frames, h, w), dtype=grid.dtype)
    s_depth = np.zeros(max_frames, dtype=np.int64)
    s_isplayer = np.zeros(max_frames, dtype=np.int8)
    s_key1 = np.zeros(max_frames, dtype=np.int64)
    s_key2 = np.zeros(max_frames, dtype=np.int64)
    s_visited = np.zeros(max_frames, dtype=np.int8)
    s_dir = np.zeros(max_frames, dtype=np.int64)
    s_moved = np.zeros(max_frames, dtype=np.int8)
    s_best = np.zeros(max_frames, dtype=np.float64)
    s_gained = np.zeros(max_frames, dtype=np.float64)
    # chance-node fields
    s_idx = np.zeros(max_frames, dtype=np.int64)
    s_subphase = np.zeros(max_frames, dtype=np.int64)  # 0=about to try val=1, 1=val=1 launched, 2=val=2 launched
    s_total = np.zeros(max_frames, dtype=np.float64)
    s_ncells = np.zeros(max_frames, dtype=np.int64)
    s_pcell = np.zeros(max_frames, dtype=np.float64)
    s_cells = np.zeros((max_frames, max_cells, 2), dtype=np.int64)

    sp = 0
    s_grid[0] = grid
    s_depth[0] = depth
    s_isplayer[0] = 1 if is_player else 0

    pending_return = 0.0

    while sp >= 0:
        top = sp
        d_here = s_depth[top]

        if d_here == 0:
            pending_return = heuristic_eval(s_grid[top])
            sp -= 1
            if sp < 0:
                break
            top2 = sp
            if s_isplayer[top2] == 1:
                v = s_gained[top2] + pending_return
                if v > s_best[top2]:
                    s_best[top2] = v
            else:
                if s_subphase[top2] == 1:
                    s_total[top2] += s_pcell[top2] * 0.9 * pending_return
                else:
                    s_total[top2] += s_pcell[top2] * 0.1 * pending_return
            continue

        if s_visited[top] == 0:
            s_visited[top] = 1
            k1, k2 = _tt_hash(s_grid[top], d_here, s_isplayer[top] == 1)
            s_key1[top] = k1
            s_key2[top] = k2
            hit, cached_val = _tt_array_lookup(k1, k2, tag_table, val_table)
            if hit:
                pending_return = cached_val
                sp -= 1
                if sp < 0:
                    break
                top2 = sp
                if s_isplayer[top2] == 1:
                    v = s_gained[top2] + pending_return
                    if v > s_best[top2]:
                        s_best[top2] = v
                else:
                    if s_subphase[top2] == 1:
                        s_total[top2] += s_pcell[top2] * 0.9 * pending_return
                    else:
                        s_total[top2] += s_pcell[top2] * 0.1 * pending_return
                continue
            if s_isplayer[top] == 0:
                cells = get_empty_cells(s_grid[top])
                n = cells.shape[0]
                s_ncells[top] = n
                if n > 0:
                    s_pcell[top] = 1.0 / n
                    for i in range(n):
                        s_cells[top, i, 0] = cells[i, 0]
                        s_cells[top, i, 1] = cells[i, 1]

        if s_isplayer[top] == 1:
            pushed_child = False
            while s_dir[top] < 4:
                dd = s_dir[top]
                new_grid, gained, changed = apply_move_grid(s_grid[top], dd)
                s_dir[top] += 1
                if not changed:
                    continue
                s_moved[top] = 1
                s_gained[top] = gained
                sp += 1
                s_grid[sp] = new_grid
                s_depth[sp] = d_here - 1
                s_isplayer[sp] = 0
                s_visited[sp] = 0
                s_dir[sp] = 0
                s_moved[sp] = 0
                s_best[sp] = -1.0e18
                s_idx[sp] = 0
                s_subphase[sp] = 0
                s_total[sp] = 0.0
                pushed_child = True
                break
            if pushed_child:
                continue
            result = heuristic_eval(s_grid[top]) if s_moved[top] == 0 else s_best[top]
            _tt_array_store(s_key1[top], s_key2[top], result, tag_table, val_table)
            pending_return = result
            sp -= 1
            if sp < 0:
                break
            top2 = sp
            if s_isplayer[top2] == 1:
                v = s_gained[top2] + pending_return
                if v > s_best[top2]:
                    s_best[top2] = v
            else:
                if s_subphase[top2] == 1:
                    s_total[top2] += s_pcell[top2] * 0.9 * pending_return
                else:
                    s_total[top2] += s_pcell[top2] * 0.1 * pending_return
            continue

        else:
            n = s_ncells[top]
            if n == 0:
                result = heuristic_eval(s_grid[top])
                _tt_array_store(s_key1[top], s_key2[top], result, tag_table, val_table)
                pending_return = result
                sp -= 1
                if sp < 0:
                    break
                top2 = sp
                if s_isplayer[top2] == 1:
                    v = s_gained[top2] + pending_return
                    if v > s_best[top2]:
                        s_best[top2] = v
                else:
                    if s_subphase[top2] == 1:
                        s_total[top2] += s_pcell[top2] * 0.9 * pending_return
                    else:
                        s_total[top2] += s_pcell[top2] * 0.1 * pending_return
                continue

            pushed_child = False
            while s_idx[top] < n:
                i = s_idx[top]
                r = s_cells[top, i, 0]
                c = s_cells[top, i, 1]
                if s_subphase[top] == 0:
                    s_grid[top, r, c] = 1
                    s_subphase[top] = 1
                    sp += 1
                    s_grid[sp] = s_grid[top]  # copy current (just-mutated) parent grid into child slot
                    s_depth[sp] = d_here - 1
                    s_isplayer[sp] = 1
                    s_visited[sp] = 0
                    s_dir[sp] = 0
                    s_moved[sp] = 0
                    s_best[sp] = -1.0e18
                    pushed_child = True
                    break
                elif s_subphase[top] == 1:
                    s_grid[top, r, c] = 2
                    s_subphase[top] = 2
                    sp += 1
                    s_grid[sp] = s_grid[top]
                    s_depth[sp] = d_here - 1
                    s_isplayer[sp] = 1
                    s_visited[sp] = 0
                    s_dir[sp] = 0
                    s_moved[sp] = 0
                    s_best[sp] = -1.0e18
                    pushed_child = True
                    break
                else:  # subphase == 2, both branches for this cell done
                    s_grid[top, r, c] = 0  # backtrack — restore before next cell
                    s_idx[top] += 1
                    s_subphase[top] = 0
            if pushed_child:
                continue
            result = s_total[top]
            _tt_array_store(s_key1[top], s_key2[top], result, tag_table, val_table)
            pending_return = result
            sp -= 1
            if sp < 0:
                break
            top2 = sp
            if s_isplayer[top2] == 1:
                v = s_gained[top2] + pending_return
                if v > s_best[top2]:
                    s_best[top2] = v
            else:
                if s_subphase[top2] == 1:
                    s_total[top2] += s_pcell[top2] * 0.9 * pending_return
                else:
                    s_total[top2] += s_pcell[top2] * 0.1 * pending_return
            continue

    return pending_return


def make_expectimax_tt_policy(depth=3, table_size_bits=20, opening_book_path=None,
                               auto_book=True, book_dir="."):
    """Expectimax policy with a transposition table; plays the same moves as
    make_expectimax_policy.
    """
    state = {"tag_table": None, "val_table": None}

    def _ensure_table(width, height):
        if state["tag_table"] is not None:
            return
        path = opening_book_path
        if path is None and auto_book:
            candidate = default_book_path(width, height, depth, book_dir=book_dir)
            if os.path.exists(candidate):
                path = candidate
        if path is not None and os.path.exists(path):
            state["tag_table"], state["val_table"] = load_tt_tables(path)
        else:
            state["tag_table"], state["val_table"] = make_tt_tables(size_bits=table_size_bits)

    def policy(game, heartbeat=None):
        _ensure_table(game.w, game.h)
        moves = game.legal_moves_with_results()
        if not moves:
            return None
        best, best_val = None, -math.inf
        for d, new_grid, gained in moves:
            v = gained + expectimax_value_tt(new_grid, depth - 1, False,
                                              state["tag_table"], state["val_table"])
            if v > best_val:
                best_val, best = v, d
        return best
    return policy


# ---------- Minimax (adversarial spawn model, alpha-beta pruned) ----------

@njit(cache=True)
def _order_player_moves(grid):
    """Legal moves sorted best-first by points gained + heuristic, to help alpha-beta
    prune.
    """
    h, w = grid.shape
    order_grids = np.zeros((4, h, w), dtype=grid.dtype)
    order_gained = np.zeros(4, dtype=np.int64)
    order_scores = np.zeros(4, dtype=np.float64)
    count = 0
    for d in range(4):
        new_grid, gained, changed = apply_move_grid(grid, d)
        if not changed:
            continue
        score = gained + heuristic_eval(new_grid)
        pos = count
        while pos > 0 and order_scores[pos - 1] < score:
            order_scores[pos] = order_scores[pos - 1]
            order_gained[pos] = order_gained[pos - 1]
            order_grids[pos] = order_grids[pos - 1]
            pos -= 1
        order_scores[pos] = score
        order_gained[pos] = gained
        order_grids[pos] = new_grid
        count += 1
    return order_grids, order_gained, count


@njit
def minimax_value(grid, depth, is_player, alpha, beta, consider_four):
    """Alpha-beta minimax where the spawn is an adversary that picks the worst cell."""
    if depth == 0:
        return heuristic_eval(grid)

    if is_player:
        order_grids, order_gained, count = _order_player_moves(grid)
        if count == 0:
            return heuristic_eval(grid)
        best = -1.0e18
        for i in range(count):
            v = order_gained[i] + minimax_value(
                order_grids[i], depth - 1, False, alpha, beta, consider_four)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
    else:
        cells = get_empty_cells(grid)
        n = cells.shape[0]
        if n == 0:
            return heuristic_eval(grid)
        worst = 1.0e18
        for i in range(n):
            r = cells[i, 0]
            c = cells[i, 1]
            grid[r, c] = 1
            v = minimax_value(grid, depth - 1, True, alpha, beta, consider_four)
            if v < worst:
                worst = v
            if consider_four:
                grid[r, c] = 2
                v4 = minimax_value(grid, depth - 1, True, alpha, beta, consider_four)
                if v4 < worst:
                    worst = v4
            grid[r, c] = 0
            if worst < beta:
                beta = worst
            if alpha >= beta:
                break
        return worst


def make_minimax_policy(depth=3, consider_four_spawn=False):
    """Minimax policy. By default the adversary only places 2s;
    consider_four_spawn=True also tries 4s.
    """
    def policy(game, heartbeat=None):
        moves = game.legal_moves_with_results()
        if not moves:
            return None
        best, best_val = None, -math.inf
        alpha, beta = -math.inf, math.inf
        for d, new_grid, gained in moves:
            v = gained + minimax_value(new_grid, depth - 1, False, alpha, beta, consider_four_spawn)
            if v > best_val:
                best_val, best = v, d
            if best_val > alpha:
                alpha = best_val  # tighten alpha across root siblings too, same as any MAX node
        return best
    return policy


# ---------- Baselines + non-TT expectimax + CPU MCTS ----------

def random_policy(game, heartbeat=None):
    """Pick a legal move uniformly at random."""
    moves = game.legal_moves()
    return random.choice(moves) if moves else None


def greedy_policy(game, heartbeat=None):
    """Pick the move whose resulting board has the best heuristic value (one move ahead)."""
    moves = game.legal_moves_with_results()
    if not moves:
        return None
    best, best_val = None, -math.inf
    for d, new_grid, _ in moves:
        val = heuristic_eval(new_grid)
        if val > best_val:
            best_val, best = val, d
    return best


# ---------- Expectimax (fully jitted recursion) ----------

@njit
def expectimax_value(grid, depth, is_player):
    """Expected heuristic value: max over player moves, average over tile spawns."""
    if depth == 0:
        return heuristic_eval(grid)

    if is_player:
        best = -1.0e18
        moved = False
        for d in range(4):
            new_grid, gained, changed = apply_move_grid(grid, d)
            if not changed:
                continue
            moved = True
            v = gained + expectimax_value(new_grid, depth - 1, False)
            if v > best:
                best = v
        if not moved:
            return heuristic_eval(grid)
        return best
    else:
        cells = get_empty_cells(grid)
        n = cells.shape[0]
        if n == 0:
            return heuristic_eval(grid)
        p_cell = 1.0 / n
        total = 0.0
        for i in range(n):
            r = cells[i, 0]
            c = cells[i, 1]
            grid[r, c] = 1
            total += p_cell * 0.9 * expectimax_value(grid, depth - 1, True)
            grid[r, c] = 2
            total += p_cell * 0.1 * expectimax_value(grid, depth - 1, True)
            grid[r, c] = 0  # backtrack — must restore before the next cell/return
        return total


def make_expectimax_policy(depth=3):
    """Expectimax policy searching to the given depth (in plies)."""
    def policy(game, heartbeat=None):
        moves = game.legal_moves_with_results()
        if not moves:
            return None
        best, best_val = None, -math.inf
        for d, new_grid, gained in moves:
            v = gained + expectimax_value(new_grid, depth - 1, False)
            if v > best_val:
                best_val, best = v, d
        return best
    return policy


@njit(cache=True)
def random_rollout(grid, max_steps=40):
    """Play up to max_steps greedy moves (with random spawns) and return the heuristic of the final board."""
    g = grid.copy()
    steps = 0
    while steps < max_steps:
        best_d = -1
        best_v = -1.0e18
        best_grid = g
        moved_any = False
        for d in range(4):
            new_grid, gained, changed = apply_move_grid(g, d)
            if not changed:
                continue
            moved_any = True
            v = heuristic_eval(new_grid)
            if v > best_v:
                best_v = v
                best_d = d
                best_grid = new_grid
        if not moved_any:
            break
        g = best_grid
        cells = get_empty_cells(g)
        n = cells.shape[0]
        if n > 0:
            idx = np.random.randint(0, n)
            r = cells[idx, 0]
            c = cells[idx, 1]
            if np.random.random() < 0.9:
                g[r, c] = 1
            else:
                g[r, c] = 2
        steps += 1
    return heuristic_eval(g)


@njit(cache=True)
def _mcts_rollouts_total(new_grid, gained, per_move_iters):
    """Sum of rollout values for one candidate move, compiled with Numba."""
    total = 0.0
    for it in range(per_move_iters):
        cells = get_empty_cells(new_grid)
        n = cells.shape[0]
        if n > 0:
            idx = np.random.randint(0, n)
            r = cells[idx, 0]
            c = cells[idx, 1]
            g = new_grid.copy()
            if np.random.random() < 0.9:
                g[r, c] = 1
            else:
                g[r, c] = 2
        else:
            g = new_grid
        total += gained + random_rollout(g)
    return total


@njit(cache=True, parallel=True)
def _mcts_rollouts_total_parallel(new_grid, gained, per_move_iters):
    """Same as _mcts_rollouts_total but spreads rollouts across threads with prange."""
    total = 0.0
    for it in prange(per_move_iters):
        cells = get_empty_cells(new_grid)
        n = cells.shape[0]
        if n > 0:
            idx = np.random.randint(0, n)
            r = cells[idx, 0]
            c = cells[idx, 1]
            g = new_grid.copy()
            if np.random.random() < 0.9:
                g[r, c] = 1
            else:
                g[r, c] = 2
        else:
            g = new_grid
        total += gained + random_rollout(g)
    return total


def set_mcts_thread_count(n):
    """Limit the number of threads used by the parallel rollout version."""
    n = max(1, int(n))
    max_available = get_num_threads()
    set_num_threads(min(n, max_available))


def make_mcts_policy(iterations=60, seed=None, parallel=False):
    """Flat Monte Carlo: split the rollout budget across legal moves and pick the best average."""
    def policy(game, heartbeat=None):
        moves = game.legal_moves_with_results()
        if not moves:
            return None
        per_move_iters = max(1, iterations // len(moves))
        scores = {}
        for d, new_grid, gained in moves:
            if heartbeat is None:
                if parallel:
                    total = _mcts_rollouts_total_parallel(new_grid, gained, per_move_iters)
                else:
                    total = _mcts_rollouts_total(new_grid, gained, per_move_iters)
            else:
                total = 0.0
                for it in range(per_move_iters):
                    cells = empty_cells(new_grid)
                    if cells:
                        r, c = random.choice(cells)
                        g = new_grid.copy()
                        g[r, c] = 1 if random.random() < 0.9 else 2
                    else:
                        g = new_grid
                    total += gained + random_rollout(g)
                    heartbeat(f"direction={d}, rollout {it + 1}/{per_move_iters}")
            scores[d] = total / per_move_iters
        return max(scores, key=scores.get)
    return policy

SOLVERS = {
    # --- CPU MCTS: pulled to the front of the run order ---
    "mcts_60": lambda: make_mcts_policy(iterations=60),
    "mcts_1000": lambda: make_mcts_policy(iterations=1000),
    "mcts_300": lambda: make_mcts_policy(iterations=300),

    # --- Expectimax, transposition-table accelerated (memoization only, NO
    "expectimax_d6_tt": lambda: make_expectimax_tt_policy(depth=6),
    "expectimax_d7_tt": lambda: make_expectimax_tt_policy(depth=7),
    "expectimax_d8_tt": lambda: make_expectimax_tt_policy(depth=8),

    # --- Minimax tiers: alpha-beta pruning visits far fewer nodes than
    "minimax_d2": lambda: make_minimax_policy(depth=2),
    "minimax_d3": lambda: make_minimax_policy(depth=3),
    "minimax_d4": lambda: make_minimax_policy(depth=4),
    "minimax_d5": lambda: make_minimax_policy(depth=5),
    "minimax_d6": lambda: make_minimax_policy(depth=6),
    "minimax_d7": lambda: make_minimax_policy(depth=7),
    "minimax_d8": lambda: make_minimax_policy(depth=8),
    "minimax_d5_full": lambda: make_minimax_policy(depth=5, consider_four_spawn=True),

    # --- TPU-batched MCTS — implementation lives in mcts_tpu.py, lazily
    "mcts_3000_tpu": lambda: __import__("mcts_tpu").make_mcts_policy_tpu(iterations=3000),

    # --- Baselines, added on from the older solver set ---
    "random": lambda: random_policy,
    "greedy": lambda: greedy_policy,

    # --- Non-TT expectimax, added on from the older solver set. d6/d7/d8
    "expectimax_d2": lambda: make_expectimax_policy(depth=2),
    "expectimax_d3": lambda: make_expectimax_policy(depth=3),
    "expectimax_d4": lambda: make_expectimax_policy(depth=4),
    "expectimax_d5": lambda: make_expectimax_policy(depth=5),
    "expectimax_d6": lambda: make_expectimax_policy(depth=6),
    "expectimax_d7": lambda: make_expectimax_policy(depth=7),
    "expectimax_d8": lambda: make_expectimax_policy(depth=8),
}
