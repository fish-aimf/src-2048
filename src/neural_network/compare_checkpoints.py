"""Evaluate every saved checkpoint and plot mean score against training steps."""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed — saves straight to a file
import matplotlib.pyplot as plt

from checkpoint_utils import find_checkpoints, load_checkpoint, play_episode
from baseline_random import run_random_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    p.add_argument("--board", type=str, default="4x4")
    p.add_argument("--episodes", type=int, default=10,
                    help="games per checkpoint — more = less noisy curve, slower")
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--include-random-baseline", action="store_true",
                    help="also plot the random-legal-move baseline as a flat "
                         "reference line, so 'better than random' is visible")
    p.add_argument("--out", type=str, default="progression.png")
    args = p.parse_args()

    w, h = (int(x) for x in args.board.lower().split("x"))
    checkpoints = find_checkpoints(args.ckpt_dir)
    if not checkpoints:
        print(f"No step_*.pt or latest.pt files found in {args.ckpt_dir}")
        return

    steps, means, mins, maxs = [], [], [], []
    for step, label, path in checkpoints:
        net, board_sizes, canvas_h, canvas_w, _ = load_checkpoint(path)
        if w > canvas_w or h > canvas_h:
            print(f"skipping {label}: board {w}x{h} exceeds this checkpoint's "
                  f"canvas {canvas_h}x{canvas_w}")
            continue
        scores = [play_episode(net, w, h, canvas_h, canvas_w, args.max_steps, seed=ep)[0]
                  for ep in range(args.episodes)]
        scores = np.array(scores)
        steps.append(step)
        means.append(scores.mean())
        mins.append(scores.min())
        maxs.append(scores.max())
        print(f"{label}: mean={scores.mean():.1f} min={scores.min()} max={scores.max()}")

    if not steps:
        print("Nothing plottable — every checkpoint's canvas was smaller than --board.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, means, marker="o", color="#2a6f97", label="mean score (lower = better)")
    ax.fill_between(steps, mins, maxs, alpha=0.15, color="#2a6f97", label="min–max range")

    if args.include_random_baseline:
        rng = np.random.default_rng(0)
        random_scores = [run_random_episode(w, h, args.max_steps, rng)
                          for _ in range(args.episodes)]
        ax.axhline(np.mean(random_scores), color="#c0392b", linestyle="--",
                    label=f"random baseline (mean={np.mean(random_scores):.0f})")

    ax.set_xlabel("training step")
    ax.set_ylabel(f"final score, {w}x{h} board ({args.episodes} games/checkpoint)")
    ax.set_title("Min-score agent: score vs. training progress")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
