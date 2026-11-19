"""2048 environment for the Objective 3 prototype (DQN that minimises score).

Observation: one-hot tile planes padded to a fixed size so one network can
play several board sizes. Reward: minus the score gained, so maximising
return minimises the final score.
"""
import os
import sys
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from engine import Game, DIRECTIONS, apply_move, legal_moves_mask, has_moves

MAX_EXP = 23  # supports tile values up to 2**23 (way beyond any real game)
NUM_EXP_CHANNELS = MAX_EXP + 1  # one-hot channels for exponents 0..MAX_EXP
NUM_CHANNELS = NUM_EXP_CHANNELS + 1  # + in-bounds mask channel


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    legal_mask: np.ndarray
    score: int
    max_tile_exp: int


def encode_obs(grid, h, w, max_h, max_w):
    """Shared encoder: game grid (h x w array of exponents) -> padded CHW tensor."""
    obs = np.zeros((NUM_CHANNELS, max_h, max_w), dtype=np.float32)
    obs[NUM_EXP_CHANNELS, :h, :w] = 1.0
    for r in range(h):
        for c in range(w):
            e = min(int(grid[r, c]), MAX_EXP)
            obs[e, r, c] = 1.0
    return obs


class Env2048:
    """One 2048 game as an RL environment."""

    def __init__(self, width=4, height=4, max_h=8, max_w=8, max_steps=4000, seed=None):
        if height > max_h or width > max_w:
            raise ValueError(f"board {height}x{width} exceeds canvas {max_h}x{max_w}")
        self.w, self.h = width, height
        self.max_h, self.max_w = max_h, max_w
        self.max_steps = max_steps
        self.rng_seed = seed
        self.game = None
        self.steps_taken = 0

    def reset(self, seed=None):
        s = seed if seed is not None else self.rng_seed
        self.game = Game(self.w, self.h, seed=s)
        self.steps_taken = 0
        return self._obs(), self._legal_mask()

    def _obs(self):
        return encode_obs(self.game.grid, self.h, self.w, self.max_h, self.max_w)

    def _legal_mask(self):
        mask = legal_moves_mask(self.game.grid)
        return np.asarray(mask, dtype=bool)

    def step(self, action_idx: int) -> StepResult:
        assert self.game is not None, "call reset() first"
        direction = DIRECTIONS[action_idx]
        new_grid, gained, changed = apply_move(self.game.grid, direction)
        self.steps_taken += 1

        if not changed:
            terminated = not has_moves(self.game.grid)
            truncated = self.steps_taken >= self.max_steps
            return StepResult(self._obs(), 0.0, terminated, truncated,
                               self._legal_mask(), self.game.score, self._max_exp())

        self.game.grid = new_grid
        self.game.score += int(gained)
        self.game.moves_made += 1
        self.game._spawn()

        terminated = not has_moves(self.game.grid)
        truncated = (not terminated) and self.steps_taken >= self.max_steps
        reward = -float(gained)  # minimizing score == maximizing this reward
        return StepResult(self._obs(), reward, terminated, truncated,
                           self._legal_mask(), self.game.score, self._max_exp())

    def _max_exp(self):
        return int(self.game.grid.max())


class VecEnv2048:
    """Several Env2048 instances stepped together."""

    def __init__(self, n_envs, board_sizes, max_h=8, max_w=8, max_steps=4000, seed=None):
        """board_sizes: list of (w, h) tuples to sample from, one per env per reset."""
        self.n_envs = n_envs
        self.board_sizes = board_sizes
        self.max_h, self.max_w = max_h, max_w
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.envs = []
        for i in range(n_envs):
            w, h = self.board_sizes[self.rng.integers(len(self.board_sizes))]
            self.envs.append(Env2048(w, h, max_h, max_w, max_steps,
                                      seed=None if seed is None else seed + i))

    def reset_all(self):
        obs = np.zeros((self.n_envs, NUM_CHANNELS, self.max_h, self.max_w), dtype=np.float32)
        masks = np.zeros((self.n_envs, 4), dtype=bool)
        for i, e in enumerate(self.envs):
            o, m = e.reset()
            obs[i], masks[i] = o, m
        return obs, masks

    def step(self, actions):
        """actions: int array shape (n_envs,)."""
        n = self.n_envs
        obs = np.zeros((n, NUM_CHANNELS, self.max_h, self.max_w), dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=bool)
        masks = np.zeros((n, 4), dtype=bool)
        final_scores = np.full(n, -1, dtype=np.int64)  # -1 unless an episode ended this step
        for i, e in enumerate(self.envs):
            r = e.step(int(actions[i]))
            done = r.terminated or r.truncated
            if done:
                final_scores[i] = r.score
                w, h = self.board_sizes[self.rng.integers(len(self.board_sizes))]
                self.envs[i] = Env2048(w, h, self.max_h, self.max_w, self.max_steps)
                o, m = self.envs[i].reset()
                obs[i], masks[i] = o, m
                rewards[i] = r.reward
                dones[i] = True
            else:
                obs[i], masks[i] = r.obs, r.legal_mask
                rewards[i] = r.reward
                dones[i] = False
        return obs, rewards, dones, masks, final_scores
