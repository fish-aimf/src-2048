"""2048 game engine.

The board is a NumPy int array of tile exponents (0 = empty, n = tile 2**n),
so it works for any rectangular size. Move logic is compiled with Numba.
"""
import random
import numpy as np
from numba import njit

DIRECTIONS = ("left", "right", "up", "down")
DIR_CODE = {"left": 0, "right": 1, "up": 2, "down": 3}


@njit(cache=True)
def _merge_line_left(line):
    n = line.shape[0]
    out = np.zeros(n, dtype=np.int8)
    score = 0
    idx = 0
    prev = -1
    prev_idx = -1
    for i in range(n):
        v = line[i]
        if v == 0:
            continue
        if prev == v:
            out[prev_idx] = v + 1
            score += 1 << (v + 1)
            prev = -1
            prev_idx = -1
        else:
            out[idx] = v
            prev = v
            prev_idx = idx
            idx += 1
    return out, score


@njit(cache=True)
def apply_move_grid(grid, direction):
    """Slide and merge the board. direction: 0=left, 1=right, 2=up, 3=down."""
    h, w = grid.shape
    new_grid = grid.copy()
    gained = 0
    changed = False
    if direction == 0 or direction == 1:
        for r in range(h):
            line = grid[r].copy()
            if direction == 1:
                line = line[::-1].copy()
            merged, sc = _merge_line_left(line)
            if direction == 1:
                merged = merged[::-1].copy()
            gained += sc
            for c in range(w):
                if new_grid[r, c] != merged[c]:
                    changed = True
                new_grid[r, c] = merged[c]
    else:
        for c in range(w):
            line = grid[:, c].copy()
            if direction == 3:
                line = line[::-1].copy()
            merged, sc = _merge_line_left(line)
            if direction == 3:
                merged = merged[::-1].copy()
            gained += sc
            for r in range(h):
                if new_grid[r, c] != merged[r]:
                    changed = True
                new_grid[r, c] = merged[r]
    return new_grid, gained, changed


@njit(cache=True)
def has_moves(grid):
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                return True
    for r in range(h):
        for c in range(w - 1):
            if grid[r, c] == grid[r, c + 1]:
                return True
    for r in range(h - 1):
        for c in range(w):
            if grid[r, c] == grid[r + 1, c]:
                return True
    return False


@njit(cache=True)
def max_tile(grid):
    m = 0
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] > m:
                m = grid[r, c]
    return m


@njit(cache=True)
def board_sum(grid):
    """Sum of tile values on the board (a 4 and an 8 count as 12)."""
    total = 0
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] > 0:
                total += 1 << int(grid[r, c])
    return total


@njit(cache=True)
def legal_moves_full(grid):
    """Apply all four moves once; return the legal mask, the resulting grids and the
    points gained.
    """
    h, w = grid.shape
    mask = np.zeros(4, dtype=np.bool_)
    gained = np.zeros(4, dtype=np.int64)
    grids = np.zeros((4, h, w), dtype=grid.dtype)
    for d in range(4):
        new_grid, g, changed = apply_move_grid(grid, d)
        mask[d] = changed
        gained[d] = g
        grids[d] = new_grid
    return mask, grids, gained


@njit(cache=True)
def legal_moves_mask(grid):
    mask, _, _ = legal_moves_full(grid)
    return mask


@njit(cache=True)
def get_empty_cells(grid):
    h, w = grid.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                count += 1
    cells = np.zeros((count, 2), dtype=np.int64)
    idx = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                cells[idx, 0] = r
                cells[idx, 1] = c
                idx += 1
    return cells


def empty_cells(grid):
    """Empty cells as a list of (row, col) tuples."""
    cells = get_empty_cells(grid)
    return [(int(cells[i, 0]), int(cells[i, 1])) for i in range(cells.shape[0])]


def apply_move(grid, direction):
    """apply_move_grid with a direction name ('left', 'right', 'up', 'down')."""
    return apply_move_grid(grid, DIR_CODE[direction])


def spawn_options(grid):
    """Every possible spawn as (new_grid, probability): a 2 with p=0.9 or a 4 with
    p=0.1 in any empty cell.
    """
    cells = get_empty_cells(grid)
    n = cells.shape[0]
    if n == 0:
        return []
    options = []
    p_cell = 1.0 / n
    for i in range(n):
        r, c = int(cells[i, 0]), int(cells[i, 1])
        for val, p_val in ((1, 0.9), (2, 0.1)):
            new_grid = grid.copy()
            new_grid[r, c] = val
            options.append((new_grid, p_cell * p_val))
    return options


class Game:
    def __init__(self, width=4, height=4, seed=None):
        self.w, self.h = width, height
        self.rng = random.Random(seed)
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.score = 0
        self.moves_made = 0
        self._spawn()
        self._spawn()

    def _spawn(self):
        cells = empty_cells(self.grid)
        if not cells:
            return
        r, c = self.rng.choice(cells)
        self.grid[r, c] = 1 if self.rng.random() < 0.9 else 2

    def legal_moves(self):
        mask = legal_moves_mask(self.grid)
        return [DIRECTIONS[d] for d in range(4) if mask[d]]

    def legal_moves_with_results(self):
        """Legal moves with each move's resulting grid and points gained."""
        mask, grids, gained = legal_moves_full(self.grid)
        return [(DIRECTIONS[d], grids[d], int(gained[d])) for d in range(4) if mask[d]]

    def step(self, direction):
        new_grid, gained, changed = apply_move(self.grid, direction)
        if not changed:
            return False
        self.grid = new_grid
        self._spawn()
        self.score += int(gained)
        self.moves_made += 1
        return True

    def is_over(self):
        return not has_moves(self.grid)

    def max_tile_value(self):
        return 1 << int(max_tile(self.grid))

    def board_sum_value(self):
        return int(board_sum(self.grid))
