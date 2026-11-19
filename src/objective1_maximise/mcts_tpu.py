"""Monte Carlo rollout search on a TPU with JAX (used for mcts_3000_tpu).

Same algorithm as make_mcts_policy in solvers.py: for each legal move, run
greedy rollouts of up to 40 moves and pick the move with the best average.
All rollouts for a move are run as one vectorised batch. The rollout RNG is
not seeded, so results match the CPU version statistically, not bit for bit.
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax


def _merge_line_left_scan(line):
    """Merge one line of tile exponents to the left; returns (line, points)."""
    w = line.shape[0]

    def step(carry, v):
        out, idx, prev, prev_idx, score = carry
        is_zero = v == 0
        is_merge = (~is_zero) & (prev == v)
        is_new = (~is_zero) & (~is_merge)

        out = out.at[prev_idx].set(jnp.where(is_merge, v + 1, out[prev_idx]))
        out = out.at[idx].set(jnp.where(is_new, v, out[idx]))

        new_idx = idx + is_new.astype(jnp.int32)
        new_prev = jnp.where(is_merge, -1, jnp.where(is_new, v, prev))
        new_prev_idx = jnp.where(is_merge, -1, jnp.where(is_new, idx, prev_idx))
        new_score = score + jnp.where(is_merge, (1 << (v + 1)).astype(jnp.int64), jnp.int64(0))
        return (out, new_idx, new_prev, new_prev_idx, new_score), None

    init = (
        jnp.zeros(w, dtype=jnp.int32),
        jnp.int32(0),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.int64(0),
    )
    (out, _, _, _, score), _ = lax.scan(step, init, line)
    return out, score


def _apply_move_single(grid, direction):
    """Apply one move to a single board (JAX version of engine.apply_move_grid)."""
    h, w = grid.shape
    if direction in (0, 1):
        lines = grid  # (h, w), merge along axis=1
        if direction == 1:
            lines = lines[:, ::-1]
        merged, scores = jax.vmap(_merge_line_left_scan)(lines)
        if direction == 1:
            merged = merged[:, ::-1]
        new_grid = merged
        gained = jnp.sum(scores)
    else:
        lines = grid.T  # (w, h), merge along axis=1 after transpose
        if direction == 3:
            lines = lines[:, ::-1]
        merged, scores = jax.vmap(_merge_line_left_scan)(lines)
        if direction == 3:
            merged = merged[:, ::-1]
        new_grid = merged.T
        gained = jnp.sum(scores)
    changed = jnp.any(new_grid != grid)
    return new_grid, gained, changed


def _heuristic_eval_single(grid):
    """JAX version of solvers.heuristic_eval (same weights)."""
    h, w = grid.shape
    gf = grid.astype(jnp.float64)
    empties = jnp.sum(grid == 0)

    row_diff = gf[:, 1:] - gf[:, :-1]
    row_inc = jnp.sum(jnp.where(row_diff > 0, row_diff, 0.0), axis=1)
    row_dec = jnp.sum(jnp.where(row_diff <= 0, -row_diff, 0.0), axis=1)
    mono = -jnp.sum(jnp.minimum(row_inc, row_dec))

    col_diff = gf[1:, :] - gf[:-1, :]
    col_inc = jnp.sum(jnp.where(col_diff > 0, col_diff, 0.0), axis=0)
    col_dec = jnp.sum(jnp.where(col_diff <= 0, -col_diff, 0.0), axis=0)
    mono = mono - jnp.sum(jnp.minimum(col_inc, col_dec))

    row_pairs_mask = (grid[:, 1:] != 0) & (grid[:, :-1] != 0)
    smooth = -jnp.sum(jnp.where(row_pairs_mask, jnp.abs(gf[:, 1:] - gf[:, :-1]), 0.0))
    col_pairs_mask = (grid[1:, :] != 0) & (grid[:-1, :] != 0)
    smooth = smooth - jnp.sum(jnp.where(col_pairs_mask, jnp.abs(gf[1:, :] - gf[:-1, :]), 0.0))

    m = jnp.max(grid)
    is_corner_max = (grid[0, 0] == m) | (grid[0, w - 1] == m) | (grid[h - 1, 0] == m) | (grid[h - 1, w - 1] == m)
    corner_bonus = jnp.where(is_corner_max, m.astype(jnp.float64), 0.0)

    return (empties.astype(jnp.float64) * 270.0 + mono * 1.0 + smooth * 0.1
            + corner_bonus * 3.0 + m.astype(jnp.float64))


def _greedy_rollout_step(grid, key):
    """One rollout step: take the move with the best heuristic, then spawn a random
    tile.
    """
    h, w = grid.shape
    results = [_apply_move_single(grid, d) for d in range(4)]
    grids4 = jnp.stack([r[0] for r in results])  # (4, h, w)
    changed4 = jnp.stack([r[2] for r in results])  # (4,)
    vals4 = jax.vmap(_heuristic_eval_single)(grids4)  # (4,)
    vals4 = jnp.where(changed4, vals4, -1e18)
    moved_any = jnp.any(changed4)
    best_d = jnp.argmax(vals4)
    candidate_grid = grids4[best_d]
    new_grid = jnp.where(moved_any, candidate_grid, grid)

    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    empty_mask = (new_grid == 0).reshape(-1)
    noise = jax.random.uniform(subkey1, shape=empty_mask.shape)
    noise = jnp.where(empty_mask, noise, -1.0)
    has_empty = jnp.any(empty_mask)
    flat_idx = jnp.argmax(noise)
    r_idx, c_idx = flat_idx // w, flat_idx % w
    spawn_val = jnp.where(jax.random.uniform(subkey2) < 0.9, 1, 2).astype(jnp.int32)

    do_spawn = moved_any & has_empty
    new_grid = jnp.where(
        do_spawn,
        new_grid.at[r_idx, c_idx].set(spawn_val),
        new_grid,
    )
    return new_grid, key


def _rollout_one(grid0, key, max_steps=40):
    def body(i, carry):
        g, k = carry
        g2, k2 = _greedy_rollout_step(g, k)
        return (g2, k2)
    g_final, _ = lax.fori_loop(0, max_steps, body, (grid0, key))
    return _heuristic_eval_single(g_final)


def _batched_rollout_scores(new_grid, gained, keys, max_steps=40):
    """Average rollout value of one candidate move over a batch of random keys."""
    def one(key):
        subkey, spawn_key1, spawn_key2 = jax.random.split(key, 3)
        h, w = new_grid.shape
        empty_mask = (new_grid == 0).reshape(-1)
        noise = jax.random.uniform(spawn_key1, shape=empty_mask.shape)
        noise = jnp.where(empty_mask, noise, -1.0)
        has_empty = jnp.any(empty_mask)
        flat_idx = jnp.argmax(noise)
        r_idx, c_idx = flat_idx // w, flat_idx % w
        spawn_val = jnp.where(jax.random.uniform(spawn_key2) < 0.9, 1, 2).astype(jnp.int32)
        g0 = jnp.where(has_empty, new_grid.at[r_idx, c_idx].set(spawn_val), new_grid)
        roll = _rollout_one(g0, subkey, max_steps=max_steps)
        return gained + roll

    vals = jax.vmap(one)(keys)
    return jnp.mean(vals)


_batched_rollout_scores_jit = jax.jit(_batched_rollout_scores, static_argnames=())


def _rollout_sum_for_pmap(new_grid, gained, keys_shard, max_steps):
    """Sum of rollout values over this device's share of the keys (summed across
    devices afterwards).
    """
    def one(key):
        subkey, spawn_key1, spawn_key2 = jax.random.split(key, 3)
        h, w = new_grid.shape
        empty_mask = (new_grid == 0).reshape(-1)
        noise = jax.random.uniform(spawn_key1, shape=empty_mask.shape)
        noise = jnp.where(empty_mask, noise, -1.0)
        has_empty = jnp.any(empty_mask)
        flat_idx = jnp.argmax(noise)
        r_idx, c_idx = flat_idx // w, flat_idx % w
        spawn_val = jnp.where(jax.random.uniform(spawn_key2) < 0.9, 1, 2).astype(jnp.int32)
        g0 = jnp.where(has_empty, new_grid.at[r_idx, c_idx].set(spawn_val), new_grid)
        roll = _rollout_one(g0, subkey, max_steps=max_steps)
        return gained + roll

    vals = jax.vmap(one)(keys_shard)
    return jnp.sum(vals)


_pmap_rollout_sum = jax.pmap(
    _rollout_sum_for_pmap,
    in_axes=(None, None, 0, None),
    static_broadcasted_argnums=(3,),
)


def _batched_rollout_scores_multidevice(new_grid, gained, keys, max_steps=40):
    n_dev = jax.local_device_count()
    if n_dev <= 1:
        return float(_batched_rollout_scores_jit(new_grid, gained, keys))

    n = keys.shape[0]
    per_dev = max(1, n // n_dev)
    used = per_dev * n_dev
    keys_sharded = keys[:used].reshape((n_dev, per_dev) + keys.shape[1:])
    per_device_sums = _pmap_rollout_sum(new_grid, gained, keys_sharded, max_steps)
    return float(jnp.sum(per_device_sums)) / used


def make_mcts_policy_tpu(iterations=1000, seed=None, max_steps=40):
    key_state = {"key": jax.random.PRNGKey(seed if seed is not None else np.random.randint(0, 2**31 - 1))}
    n_dev = jax.local_device_count()

    def policy(game, heartbeat=None):
        moves = game.legal_moves_with_results()
        if not moves:
            return None
        raw_iters = max(1, iterations // len(moves))
        # round UP to a clean multiple of n_dev so every device gets an equal
        # shard with no truncated/dropped rollouts and no padding needed
        per_move_iters = ((raw_iters + n_dev - 1) // n_dev) * n_dev
        scores = {}
        for d, new_grid, gained in moves:
            key_state["key"], subkey = jax.random.split(key_state["key"])
            rollout_keys = jax.random.split(subkey, per_move_iters)
            grid_jax = jnp.asarray(new_grid, dtype=jnp.int32)
            mean_score = _batched_rollout_scores_multidevice(grid_jax, jnp.int32(gained), rollout_keys, max_steps)
            scores[d] = float(mean_score)
            if heartbeat is not None:
                heartbeat(f"direction={d}, {per_move_iters} rollouts sharded across {n_dev} device(s)")
        return max(scores, key=scores.get)
    return policy


def _rollout_val_for_pmap_grouped(grids_shard, gained_shard, keys_shard, max_steps):
    """Rollout values for this device's share of a batch of (game, move) pairs."""
    def one(grid, gained, key):
        subkey, spawn_key1, spawn_key2 = jax.random.split(key, 3)
        h, w = grid.shape
        empty_mask = (grid == 0).reshape(-1)
        noise = jax.random.uniform(spawn_key1, shape=empty_mask.shape)
        noise = jnp.where(empty_mask, noise, -1.0)
        has_empty = jnp.any(empty_mask)
        flat_idx = jnp.argmax(noise)
        r_idx, c_idx = flat_idx // w, flat_idx % w
        spawn_val = jnp.where(jax.random.uniform(spawn_key2) < 0.9, 1, 2).astype(jnp.int32)
        g0 = jnp.where(has_empty, grid.at[r_idx, c_idx].set(spawn_val), grid)
        roll = _rollout_one(g0, subkey, max_steps=max_steps)
        return gained + roll

    return jax.vmap(one)(grids_shard, gained_shard, keys_shard)


_pmap_rollout_val_grouped = jax.pmap(
    _rollout_val_for_pmap_grouped,
    in_axes=(0, 0, 0, None),  # grids/gained/keys all sharded now — different per rollout
    static_broadcasted_argnums=(3,),
)


def batched_group_scores(grids, gaineds, pair_id, rollouts_per_pair, key, max_steps=40):
    """Average rollout value for every (game, candidate move) pair in one batch."""
    P = grids.shape[0]
    R = rollouts_per_pair
    n_dev = jax.local_device_count()
    total = P * R
    keys_flat = jax.random.split(key, total)  # (P*R, 2)

    pids = jnp.repeat(jnp.arange(P), R)  # (P*R,) which pair each rollout belongs to
    grids_flat = grids[pids]  # (P*R, h, w)
    gained_flat = gaineds[pids]  # (P*R,)

    if n_dev <= 1:
        vals = jax.vmap(lambda g, ga, k: _rollout_val_for_pmap_grouped(
            g[None], ga[None], k[None], max_steps)[0]
        )(grids_flat, gained_flat, keys_flat)
    else:
        per_dev = total // n_dev
        used = per_dev * n_dev
        if used < total:
            grids_flat, gained_flat, keys_flat, pids = (
                grids_flat[:used], gained_flat[:used], keys_flat[:used], pids[:used]
            )
        g_sh = grids_flat.reshape((n_dev, per_dev) + grids_flat.shape[1:])
        ga_sh = gained_flat.reshape((n_dev, per_dev))
        k_sh = keys_flat.reshape((n_dev, per_dev) + keys_flat.shape[1:])
        vals_sh = _pmap_rollout_val_grouped(g_sh, ga_sh, k_sh, max_steps)
        vals = vals_sh.reshape(used)
        pids = pids[:used]

    sums = jnp.zeros(P, dtype=jnp.float64).at[pids].add(vals)
    counts = jnp.zeros(P, dtype=jnp.float64).at[pids].add(1.0)
    return sums / jnp.maximum(counts, 1.0)


def batched_group_scores_masked(grids, gaineds, iters, rollouts_per_pair, key, max_steps=40):
    """Like batched_group_scores, but pair p only counts its first iters[p] rollouts."""
    P = grids.shape[0]
    R = rollouts_per_pair
    n_dev = jax.local_device_count()
    total = P * R
    keys_flat = jax.random.split(key, total)

    pids = jnp.repeat(jnp.arange(P), R)  # which pair each lane belongs to
    lane = jnp.tile(jnp.arange(R), P)  # lane index within its pair
    w = (lane < iters[pids]).astype(jnp.float64)  # 1 = counted, 0 = masked
    grids_flat = grids[pids]
    gained_flat = gaineds[pids]

    if n_dev <= 1:
        vals = jax.vmap(lambda g, ga, k: _rollout_val_for_pmap_grouped(
            g[None], ga[None], k[None], max_steps)[0]
        )(grids_flat, gained_flat, keys_flat)
    else:
        per_dev = total // n_dev
        used = per_dev * n_dev
        if used < total:
            grids_flat, gained_flat, keys_flat, pids, w = (
                grids_flat[:used], gained_flat[:used], keys_flat[:used], pids[:used], w[:used]
            )
        g_sh = grids_flat.reshape((n_dev, per_dev) + grids_flat.shape[1:])
        ga_sh = gained_flat.reshape((n_dev, per_dev))
        k_sh = keys_flat.reshape((n_dev, per_dev) + keys_flat.shape[1:])
        vals = _pmap_rollout_val_grouped(g_sh, ga_sh, k_sh, max_steps).reshape(used)

    sums = jnp.zeros(P, dtype=jnp.float64).at[pids].add(vals * w)
    counts = jnp.zeros(P, dtype=jnp.float64).at[pids].add(w)
    return sums / jnp.maximum(counts, 1.0)


def make_mcts_batched_group(iterations=1000, max_steps=40, seed=None):
    key_state = {"key": jax.random.PRNGKey(seed if seed is not None else np.random.randint(0, 2**31 - 1))}
    n_dev = jax.local_device_count()

    def decide_batch(games):
        per_game_moves = [g.legal_moves_with_results() for g in games]
        pair_grids, pair_gained, owner = [], [], []
        for gi, moves in enumerate(per_game_moves):
            for d, new_grid, gained in moves:
                pair_grids.append(np.asarray(new_grid, dtype=np.int32))
                pair_gained.append(gained)
                owner.append((gi, d))
        if not pair_grids:
            return [None] * len(games)

        P = len(pair_grids)
        raw_iters = max(1, iterations // max(1, max(len(m) for m in per_game_moves)))
        per_pair_iters = ((raw_iters + n_dev - 1) // n_dev) * n_dev
        while (P * per_pair_iters) % max(n_dev, 1) != 0:
            per_pair_iters += n_dev

        grids_arr = jnp.asarray(np.stack(pair_grids), dtype=jnp.int32)
        gained_arr = jnp.asarray(np.array(pair_gained), dtype=jnp.int32)
        key_state["key"], subkey = jax.random.split(key_state["key"])
        means = batched_group_scores(grids_arr, gained_arr, None, per_pair_iters, subkey, max_steps)
        means = np.asarray(means)

        scores_per_game = [dict() for _ in games]
        for (gi, d), val in zip(owner, means):
            scores_per_game[gi][d] = float(val)

        out = []
        for gi, moves in enumerate(per_game_moves):
            if not moves:
                out.append(None)
            else:
                sc = scores_per_game[gi]
                out.append(max(sc, key=sc.get))
        return out

    return decide_batch


def make_mcts_batched_group_fixed(width, height, n_games, iterations=1000, max_steps=40,
                                   seed=None, max_moves_per_game=4):
    """Decide moves for a fixed-size group of games at once, keeping array shapes
    constant between calls.
    """
    key_state = {"key": jax.random.PRNGKey(seed if seed is not None else np.random.randint(0, 2**31 - 1))}
    n_dev = jax.local_device_count()
    P = n_games * max_moves_per_game

    def _round_up(x):
        return ((x + n_dev - 1) // n_dev) * n_dev

    def per_move_iters(n_legal):
        # exactly make_mcts_policy_tpu's per-move budget, so batched and
        # sequential runs of the same solver use the same search strength.
        return _round_up(max(1, iterations // n_legal))

    R = per_move_iters(2)

    def decide_batch(games):
        if len(games) != n_games:
            raise ValueError(f"decide_batch expects exactly {n_games} slots "
                              f"(pad with None for empty ones), got {len(games)}")

        grids_buf = np.zeros((P, height, width), dtype=np.int32)
        gained_buf = np.zeros(P, dtype=np.int32)
        iters_buf = np.zeros(P, dtype=np.int32)  # 0 = padding, never counted
        owner = [None] * P  # (game_index, direction) for real slots, None for padding
        forced = [None] * n_games  # games with exactly one legal move

        for gi, g in enumerate(games):
            base = gi * max_moves_per_game
            if g is None:
                continue
            moves = g.legal_moves_with_results()
            if len(moves) == 1:
                forced[gi] = moves[0][0]  # sequential policy would also return this
                continue
            if not moves:
                continue
            n_it = min(per_move_iters(len(moves)), R)
            for slot_offset, (d, new_grid, gained) in enumerate(moves[:max_moves_per_game]):
                idx = base + slot_offset
                grids_buf[idx] = new_grid
                gained_buf[idx] = gained
                iters_buf[idx] = n_it
                owner[idx] = (gi, d)

        scores_per_game = [dict() for _ in range(n_games)]
        if any(o is not None for o in owner):
            key_state["key"], subkey = jax.random.split(key_state["key"])
            means = batched_group_scores_masked(
                jnp.asarray(grids_buf), jnp.asarray(gained_buf), jnp.asarray(iters_buf),
                R, subkey, max_steps)
            means = np.asarray(means)
            for idx in range(P):
                if owner[idx] is not None:
                    gi, d = owner[idx]
                    scores_per_game[gi][d] = float(means[idx])

        out = []
        for gi, g in enumerate(games):
            if forced[gi] is not None:
                out.append(forced[gi])
                continue
            sc = scores_per_game[gi]
            out.append(max(sc, key=sc.get) if sc else None)
        return out

    return decide_batch
