"""Worker that pulls games from server.py and plays them on local cores.

    python worker.py --server http://<server-ip>:8765 --workers 8 --name laptop-2
"""
import argparse
import platform
import socket
import time
from multiprocessing import Pool

import requests

import paths  # noqa: F401  (sets up imports for the chosen objective)
import core
from core import apply_power_saving

_real_print = print


def print(*args, **kwargs):
    _real_print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)


def default_name():
    return platform.node() or socket.gethostname() or "unknown-worker"


def init_worker(priority, efficiency_mode):
    apply_power_saving(priority, efficiency_mode)


def run_job_and_report(args):
    """Runs inside a worker process."""
    job, duty_cycle_override = args
    if duty_cycle_override is not None:
        job = dict(job)
        job["duty_cycle"] = duty_cycle_override
    result = core.run_one_game(job)
    result["job_id"] = job["job_id"]
    result["assignment_token"] = job.get("assignment_token")  # echoed back for the server's staleness check
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, required=True,
                     help="coordinator base URL, e.g. http://100.x.y.z:8765")
    ap.add_argument("--name", type=str, default=default_name(),
                     help="label shown in the coordinator's live log (default: this machine's hostname)")
    ap.add_argument("--workers", type=int, default=8, help="local parallel game processes")
    ap.add_argument("--priority", type=str, default="idle", choices=["normal", "below_normal", "idle"])
    ap.add_argument("--no_efficiency_mode", action="store_true")
    ap.add_argument("--poll_interval", type=float, default=3.0,
                     help="seconds to wait before re-polling when the server has no job ready yet")
    ap.add_argument("--duty_cycle_override", type=float, default=None,
                     help="ignore whatever duty_cycle the coordinator assigned to each job and use "
                          "this instead — e.g. 1.0 to run this machine flat-out with no throttling "
                          "regardless of what home laptops are set to. Omit to just use the "
                          "coordinator's setting like normal.")
    ap.add_argument("--solver_filter", type=str, default=None,
                     help="comma-separated allowlist of solver names this worker instance should "
                          "EXCLUSIVELY pull jobs for, e.g. "
                          "'mcts_3000_tpu' — for running a small "
                          "TPU-dedicated pool alongside a separate CPU-only worker.py instance on "
                          "the same machine, both pulling from the same queue without ever handing "
                          "each other the wrong job type. Mutually exclusive with --exclude_solvers "
                          "(if both given, this one wins). Omit to accept any job type, as before.")
    ap.add_argument("--exclude_solvers", type=str, default=None,
                     help="comma-separated denylist — this worker will claim any job EXCEPT these "
                          "solver names. Handy for the CPU-pool side of a split: rather than listing "
                          "every CPU solver name, just exclude the *_tpu ones, e.g. "
                          "'mcts_3000_tpu'. Ignored if --solver_filter "
                          "is also given.")
    args = ap.parse_args()

    if not (args.server.startswith("http://") or args.server.startswith("https://")):
        print(f"  [WARNING] --server value '{args.server}' doesn't start with http:// or https:// "
              f"— this is very likely a typo and every request below will fail to connect.")

    base = args.server.rstrip("/")
    session = requests.Session()

    print(f"Worker '{args.name}' starting — {args.workers} local processes, "
          f"pulling from {base} (batched /jobs + /results)")

    def fetch_jobs(count):
        """Fetch up to `count` jobs in one request."""
        try:
            params = {"name": args.name, "count": count}
            if args.solver_filter:
                params["solvers"] = args.solver_filter
            elif args.exclude_solvers:
                params["exclude_solvers"] = args.exclude_solvers
            r = session.get(f"{base}/jobs", params=params, timeout=20)
            r.raise_for_status()
            resp = r.json()
            return resp.get("jobs", []), resp.get("done", False)
        except requests.RequestException as e:
            print(f"  [connection issue reaching server: {e}] retrying in {args.poll_interval}s")
            return None, False

    def post_results(results):
        """Post an entire batch of finished results in one request."""
        if not results:
            return
        try:
            r = session.post(f"{base}/results",
                              json={"results": results, "worker": args.name},
                              timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            job_ids = [res.get("job_id") for res in results]
            print(f"  [could not report {len(results)} result(s), job_ids={job_ids}: {e}] "
                  f"— server will requeue them after its timeout")

    with Pool(args.workers, initializer=init_worker,
              initargs=(args.priority, not args.no_efficiency_mode)) as pool:
        in_flight = {}  # async result -> job dict, so we know what to report
        server_done = False
        crash_streak = [0]  # consecutive crashed games; abort at 5 (see below)

        def top_up():
            nonlocal server_done
            open_slots = args.workers - len(in_flight)
            if open_slots <= 0:
                return
            jobs, done_flag = fetch_jobs(open_slots)
            if jobs is None:
                return  # connection issue already logged inside fetch_jobs
            for job in jobs:
                async_res = pool.apply_async(run_job_and_report, ((job, args.duty_cycle_override),))
                in_flight[async_res] = job
            if not jobs and done_flag:
                server_done = True

        while True:
            top_up()
            if server_done and not in_flight:
                print("Server reports all jobs complete. Exiting.")
                break
            if not in_flight:
                time.sleep(args.poll_interval)
                continue
            finished = [ar for ar in in_flight if ar.ready()]
            if not finished:
                time.sleep(0.5)
                continue

            batch = []
            for ar in finished:
                job = in_flight.pop(ar)
                try:
                    result = ar.get()
                except Exception as e:
                    crash_streak[0] += 1
                    print(f"  [game crashed, job_id={job['job_id']}: {type(e).__name__}: {e}] "
                          f"— server will requeue it after its timeout")
                    if crash_streak[0] >= 5:
                        # Every game is failing (broken install, missing solver...).
                        # Stop now instead of claiming the whole queue and locking it.
                        print(f"  [ABORT] {crash_streak[0]} games in a row crashed — this "
                              f"machine's setup is broken. Fix it, then restart server.py "
                              f"to release the claimed jobs.")
                        pool.terminate()
                        raise SystemExit(1)
                    continue
                crash_streak[0] = 0
                flag = " [TIMED OUT]" if result["timed_out"] else ""
                print(f"  {result['solver']} {result['board']} seed={result['seed']} "
                      f"done (score={result['score']}, {result['seconds']}s){flag}")
                batch.append(result)

            post_results(batch)  # single request for the whole batch


if __name__ == "__main__":
    main()
