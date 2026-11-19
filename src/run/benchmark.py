"""Run the benchmark on one machine.

Plays every (solver, board, seed) game in parallel with a process pool and
appends results to a CSV. Re-running with the same --out skips finished games.

    OBJECTIVE=2 python benchmark.py --games 5000 --workers 16 --out results.csv
"""
import argparse
import csv
import os
import sys
import threading
import time
from multiprocessing import Pool, Lock

import paths  # noqa: F401  (sets up imports for the chosen objective)
import core
from core import FIELDNAMES, apply_power_saving, parse_boards, build_all_jobs

# --- Timestamp every printed line ---
_real_print = print


def print(*args, **kwargs):
    _real_print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

_print_lock = None  # set in each worker via init_worker


def init_worker(lock, priority, efficiency_mode):
    global _print_lock
    _print_lock = lock
    apply_power_saving(priority, efficiency_mode)


def run_one_game(job):
    """Run one game through core.run_one_game, printing heartbeats for long games."""
    def hb(msg):
        with _print_lock:
            print(f"    ...working: {msg}")
    job = dict(job)
    job["on_heartbeat"] = hb
    return core.run_one_game(job)


def load_completed(out_path):
    completed = set()
    if os.path.exists(out_path):
        with open(out_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["solver"], row["board"], int(row["seed"])))
    return completed


class CpuMonitor:
    """Samples CPU usage in a background thread for the progress line."""
    def __init__(self):
        self.value = None
        self._stop = False
        if HAVE_PSUTIL:
            psutil.cpu_percent(interval=None)  # prime it
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop:
            self.value = psutil.cpu_percent(interval=2)

    def stop(self):
        self._stop = True

    def read(self):
        if not HAVE_PSUTIL:
            return "n/a (pip install psutil for this)"
        return f"{self.value}%" if self.value is not None else "measuring..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100, help="games per (solver, board) combo")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=str, default="results.csv")
    ap.add_argument("--boards", type=str, default="3x3,3x4,4x3,4x4")
    ap.add_argument("--solvers", type=str,
                     default="random,greedy,expectimax_d2,expectimax_d3,expectimax_d4,mcts_60,mcts_300,mcts_1000")
    ap.add_argument("--max_moves", type=int, default=None,
                     help="cap on moves per game. Default None = unlimited "
                          "(games run until they naturally reach game-over).")
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--time_budget", type=float, default=None,
                     help="max active-compute seconds per single game before it's cut off. "
                          "Default None = unlimited, games always run to completion "
                          "(a single expectimax_d5/d6 move can legitimately take a "
                          "long time — this will no longer cut it short).")
    ap.add_argument("--duty_cycle", type=float, default=1.0,
                     help="fraction of time each worker is actively computing vs. sleeping")
    ap.add_argument("--heartbeat_seconds", type=float, default=5.0,
                     help="print a progress line if a single game runs longer than this "
                          "(0 disables heartbeat)")
    ap.add_argument("--priority", type=str, default="idle", choices=["normal", "below_normal", "idle"],
                     help="OS scheduling priority for worker processes (Windows). "
                          "'idle' is most power-saving/heat-friendly, 'normal' is fastest.")
    ap.add_argument("--no_efficiency_mode", action="store_true",
                     help="disable requesting Windows Efficiency Mode (EcoQoS) for workers")
    ap.add_argument("--chunksize", type=int, default=8,
                     help="jobs batched per IPC round-trip to workers; higher reduces overhead "
                          "now that games are fast, but delays how often you see progress prints")
    ap.add_argument("--fsync_interval", type=float, default=1.0,
                     help="max seconds between forced disk syncs (fsync). Data is still written "
                          "and flushed to the OS after every game either way — this only affects "
                          "the small window of risk if the whole machine loses power mid-run, not "
                          "normal crashes/closing the terminal.")
    args = ap.parse_args()

    boards = parse_boards(args.boards)
    solver_names = args.solvers.split(",")

    all_jobs = build_all_jobs(solver_names, boards, args.games, args.seed_offset,
                               args.max_moves, args.time_budget, args.duty_cycle,
                               args.heartbeat_seconds)

    completed = load_completed(args.out)
    jobs = [j for j in all_jobs
            if (j["solver"], f"{j['width']}x{j['height']}", j["seed"]) not in completed]

    skipped = len(all_jobs) - len(jobs)
    if skipped:
        print(f"Resuming: {skipped} games already completed in {args.out}, skipping those.")
    budget_str = f"{args.time_budget}s/game" if args.time_budget is not None else "unlimited (no per-game cap)"
    print(f"Games remaining to run: {len(jobs)} across {args.workers} workers "
          f"(time budget: {budget_str})")
    if not HAVE_PSUTIL:
        print("(Tip: run 'pip install psutil' to see live CPU% and enable priority control.)")
    if os.name != "nt":
        print("(Note: efficiency-core preference / EcoQoS is Windows-only; on this OS only "
              "--priority and --duty_cycle have an effect.)")
    apply_power_saving(args.priority, not args.no_efficiency_mode)

    if not jobs:
        print("Nothing to do — all games already completed.")
        return

    file_exists = os.path.exists(args.out) and skipped > 0
    mode = "a" if file_exists else "w"
    t0 = time.time()
    done_count = skipped
    monitor = CpuMonitor()
    print_lock = Lock()

    with open(args.out, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        f.flush()

        with Pool(args.workers, initializer=init_worker,
                  initargs=(print_lock, args.priority, not args.no_efficiency_mode)) as pool:
            last_fsync = time.time()
            for res in pool.imap_unordered(run_one_game, jobs, chunksize=args.chunksize):
                writer.writerow(res)
                f.flush()
                now = time.time()
                if now - last_fsync >= args.fsync_interval:
                    os.fsync(f.fileno())
                    last_fsync = now
                done_count += 1
                if res["timed_out"]:
                    print(f"  [timed out] {res['solver']} on {res['board']} seed={res['seed']} "
                          f"after {res['seconds']}s ({res['moves']} moves)")
                if done_count % max(1, len(all_jobs) // 40) == 0:
                    elapsed = time.time() - t0
                    print(f"  {done_count}/{len(all_jobs)} total games done "
                          f"({elapsed:.1f}s this session, CPU: {monitor.read()})")
            os.fsync(f.fileno())  # final sync before exiting

    monitor.stop()
    print(f"Done. {done_count}/{len(all_jobs)} total games in {args.out} "
          f"({time.time()-t0:.1f}s this session).")


if __name__ == "__main__":
    main()
