"""Random-play baseline in the same environment, for comparison with the DQN."""
import argparse
import numpy as np

from env2048 import Env2048


def run_random_episode(w, h, max_steps, rng):
    env = Env2048(w, h, max_steps=max_steps)
    obs, mask = env.reset()
    while True:
        legal = np.where(mask)[0]
        a = rng.choice(legal)
        r = env.step(a)
        mask = r.legal_mask
        if r.terminated or r.truncated:
            return r.score


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--board-sizes", type=str, default="4x4,5x5,6x6")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    for tok in args.board_sizes.split(","):
        w, h = (int(x) for x in tok.lower().split("x"))
        scores = [run_random_episode(w, h, args.max_steps, rng) for _ in range(args.episodes)]
        scores = np.array(scores)
        print(f"{w}x{h} RANDOM: mean={scores.mean():.1f} min={scores.min()} "
              f"max={scores.max()} median={np.median(scores):.1f} (n={len(scores)})")
