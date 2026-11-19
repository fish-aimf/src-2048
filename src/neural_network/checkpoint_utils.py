"""Load saved DQN checkpoints and play games with them."""
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from env2048 import encode_obs, legal_moves_mask, NUM_CHANNELS
from engine import Game, apply_move, DIRECTIONS
from model import DuelingCNN


def find_checkpoints(ckpt_dir):
    """Sorted (step, label, path) for every saved checkpoint in ckpt_dir."""
    found = {}
    for path in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
        m = re.search(r"step_(\d+)\.pt$", path)
        if m:
            found[int(m.group(1))] = path
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest) and latest not in found.values():
        try:
            step = int(torch.load(latest, map_location="cpu")["step"])
            found[step] = latest
        except Exception:
            pass
    return [(step, f"step {step:,}", path) for step, path in sorted(found.items())]


def load_checkpoint(path, device="cpu"):
    """Loads a checkpoint and returns (net, board_sizes, canvas_h, canvas_w, step)."""
    ckpt = torch.load(path, map_location=device)
    board_sizes = ckpt.get("board_sizes", [(4, 4)])
    canvas_w = max(w for w, _ in board_sizes)
    canvas_h = max(h for _, h in board_sizes)
    net = DuelingCNN().to(device)
    with torch.no_grad():
        net(torch.zeros(2, NUM_CHANNELS, canvas_h, canvas_w, device=device))
    state = ckpt["online"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    return net, board_sizes, canvas_h, canvas_w, ckpt.get("step", 0)


@torch.no_grad()
def pick_action(net, grid, h, w, canvas_h, canvas_w, device="cpu"):
    """Returns (chosen_direction_name, {direction_name: q_value or None if illegal})."""
    obs = encode_obs(grid, h, w, canvas_h, canvas_w)
    legal = legal_moves_mask(grid)
    obs_t = torch.as_tensor(obs[None], device=device)
    q = net(obs_t)[0].cpu().numpy()
    q_by_dir = {DIRECTIONS[i]: (float(q[i]) if legal[i] else None) for i in range(4)}
    masked = np.where(legal, q, -np.inf)
    chosen = DIRECTIONS[int(np.argmax(masked))]
    return chosen, q_by_dir


def play_episode(net, w, h, canvas_h, canvas_w, max_steps, seed=None, device="cpu"):
    """Play one game with a checkpoint, without printing."""
    game = Game(w, h, seed=seed)
    steps = 0
    while steps < max_steps:
        if not game.legal_moves():
            break
        d, _ = pick_action(net, game.grid, h, w, canvas_h, canvas_w, device)
        game.step(d)
        steps += 1
    return game.score, steps
