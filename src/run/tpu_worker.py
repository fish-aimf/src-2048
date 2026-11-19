"""TPU worker for mcts_3000_tpu.

Plays several games at once and runs their rollouts as one batch on the TPU
(see objective1_maximise/mcts_tpu.py), then reports each finished game to
server.py.
"""
import argparse
import csv
import time

import requests

import paths  # noqa: F401  (sets up imports for the chosen objective)
import core
from mcts_tpu import make_mcts_batched_group_fixed
from engine import Game

_real_print = print


def print(*args, **kwargs):
    _real_print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)


# solver name -> iterations, matching solvers.py's SOLVERS registrations exactly
SOLVER_ITERATIONS = {
    "mcts_300_tpu": 300,
    "mcts_1000_tpu": 1000,
    "mcts_3000_tpu": 3000,
    "mcts_6000_tpu": 6000,
    "mcts_10000_tpu": 10000,
}


class GameSlot:
    """One game in the TPU batch, with what is needed to report its result row."""
    __slots__ = ("job", "game", "t0", "moves", "assignment_token")

    def __init__(self, job):
        self.job = job
        self.game = Game(width=job["width"], height=job["height"], seed=job["seed"])
        self.t0 = time.time()
        self.moves = 0
        self.assignment_token = job.get("assignment_token")

    def to_result(self):
        return {
            "job_id": self.job["job_id"],
            "assignment_token": self.assignment_token,
            "solver": self.job["solver"],
            "board": f"{self.job['width']}x{self.job['height']}",
            "seed": self.job["seed"],
            "score": self.game.score,
            "board_sum": self.game.board_sum_value(),
            "max_tile": self.game.max_tile_value(),
            "moves": self.moves,
            "seconds": round(time.time() - self.t0, 4),
            "timed_out": 0,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, required=True)
    ap.add_argument("--name", type=str, default="tpu-batch-worker")
    ap.add_argument("--n_games", type=int, default=16, help="games processed together per "
                     "round — this is the batch size that was tested and confirmed to work "
                     "well at 16 on real v5e-8 hardware. Larger may help more (round cost was "
                     "FLAT at 16, suggesting spare capacity) but is untested — try it and "
                     "compare against this default before trusting a bigger number blindly, "
                     "same as every other tuning number in this project.")
    ap.add_argument("--solver_filter", type=str, required=True,
                     help="comma-separated mcts_*_tpu names this worker should claim, e.g. "
                          "mcts_3000_tpu,mcts_6000_tpu,mcts_10000_tpu")
    ap.add_argument("--poll_interval", type=float, default=3.0)
    args = ap.parse_args()

    base = args.server.rstrip("/")
    session = requests.Session()
    solver_names = [s.strip() for s in args.solver_filter.split(",") if s.strip()]
    for s in solver_names:
        if s not in SOLVER_ITERATIONS:
            raise SystemExit(f"Unknown solver '{s}' — SOLVER_ITERATIONS in this file only "
                              f"knows about {list(SOLVER_ITERATIONS)}. Add it there first if "
                              f"you've registered a new mcts_*_tpu tier in solvers.py.")

    print(f"Batched TPU worker '{args.name}' — n_games={args.n_games}, "
          f"solvers={solver_names}, pulling from {base}")

    decide_cache = {}

    def get_decider(solver, width, height):
        key = (solver, width, height)
        if key not in decide_cache:
            print(f"  [compiling new batched decider for {key} — this is a one-time "
                  f"cost, ~15-30s expected on TPU hardware]")
            decide_cache[key] = make_mcts_batched_group_fixed(
                width=width, height=height, n_games=args.n_games,
                iterations=SOLVER_ITERATIONS[solver],
            )
        return decide_cache[key]

    def fetch_jobs(count):
        try:
            r = session.get(f"{base}/jobs", params={
                "name": args.name, "count": count, "solvers": args.solver_filter,
            }, timeout=20)
            r.raise_for_status()
            resp = r.json()
            return resp.get("jobs", []), resp.get("done", False)
        except requests.RequestException as e:
            print(f"  [connection issue: {e}] retrying in {args.poll_interval}s")
            return None, False

    def post_results(results):
        if not results:
            return
        try:
            r = session.post(f"{base}/results", json={"results": results, "worker": args.name}, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            job_ids = [res.get("job_id") for res in results]
            print(f"  [could not report {len(results)} result(s), job_ids={job_ids}: {e}] "
                  f"— server will requeue them after its timeout")

    server_done = False
    while not server_done:
        jobs, done_flag = fetch_jobs(args.n_games)
        if jobs is None:
            time.sleep(args.poll_interval)
            continue
        if not jobs:
            if done_flag:
                server_done = True
                print("Server reports all matching jobs complete. Exiting.")
            else:
                time.sleep(args.poll_interval)
            continue

        # Group this fetched batch by (solver, width, height) — should almost
        # always be one group given the queue's build order, but never assume it.
        groups = {}
        for job in jobs:
            key = (job["solver"], job["width"], job["height"])
            groups.setdefault(key, []).append(job)

        for (solver, width, height), group_jobs in groups.items():
            decide = get_decider(solver, width, height)
            slots = [GameSlot(j) for j in group_jobs]
            # pad to exactly n_games with None so decide_batch's fixed shape is honored
            padded = list(slots) + [None] * (args.n_games - len(slots))
            finished_results = []

            active_slots = list(slots)  # slots still being played
            round_input = list(padded)
            while any(s is not None for s in round_input):
                t0 = time.time()
                moves = decide([s.game if s is not None else None for s in round_input])
                round_t = time.time() - t0
                print(f"  [{solver} {width}x{height}] round: {len(active_slots)} active, "
                      f"{round_t:.2f}s")
                for i, m in enumerate(moves):
                    slot = round_input[i]
                    if slot is None:
                        continue
                    max_moves = slot.job.get("max_moves")
                    if m is None or (max_moves is not None and slot.moves >= max_moves):
                        finished_results.append(slot.to_result())
                        round_input[i] = None
                    else:
                        slot.game.step(m)
                        slot.moves += 1
                        flag = " [max_moves reached, will finish next round]" \
                            if max_moves is not None and slot.moves >= max_moves else ""
                        _ = flag  # handled on the next loop iteration's check above

            for res in finished_results:
                print(f"  {res['solver']} {res['board']} seed={res['seed']} "
                      f"done (score={res['score']}, {res['seconds']}s)")
            post_results(finished_results)

        if done_flag and not jobs:
            server_done = True


if __name__ == "__main__":
    main()
