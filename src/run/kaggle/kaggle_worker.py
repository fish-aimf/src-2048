"""Worker for short-lived machines such as Kaggle CPU notebooks.

Runs a quick self-test, sends a heartbeat every 30 minutes, only takes games
expected to finish before the session ends, and hands unfinished games back
near the end. Finished results are kept in an outbox until the server
confirms them, so a dropped connection loses nothing.

    python kaggle_worker.py --server https://<tunnel-address> --name kcpu-01
"""
import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import paths  # noqa: F401  (repository layout)
except ImportError:
    pass

import core

_real_print = print
OUTBOX_FILE = "outbox.json"
URL_OK = re.compile(r"^https://[A-Za-z0-9.-]+\.(trycloudflare\.com|ngrok-free\.dev|ngrok-free\.app|ts\.net)$")


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _real_print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)


def _cpu_times():
    """(busy, total) jiffies from /proc/stat (Linux)."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        return sum(v) - idle, sum(v)
    except Exception:
        return None


def _n_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def run_job(job):
    result = core.run_one_game(job)
    result["job_id"] = job["job_id"]
    result["assignment_token"] = job.get("assignment_token")
    return result


def latest_url_from_topic(topic):
    """Newest valid address published on ntfy.sh/<topic>, or None."""
    try:
        r = requests.get(f"https://ntfy.sh/{topic}/raw", params={"poll": "1", "since": "24h"}, timeout=20)
        r.raise_for_status()
        for line in reversed([l.strip() for l in r.text.splitlines() if l.strip()]):
            if URL_OK.match(line.rstrip("/")):
                return line.rstrip("/")
    except requests.RequestException:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--url_topic", default=None,
                     help="private ntfy.sh topic where the PC publishes its current tunnel address")
    ap.add_argument("--workers", type=int, default=_n_cpus())
    ap.add_argument("--session_hours", type=float, default=11.5)
    ap.add_argument("--stop_margin_min", type=float, default=10.0)
    ap.add_argument("--heartbeat_every", type=float, default=1800.0)
    ap.add_argument("--give_up_minutes", type=float, default=100.0,
                     help="exit after this long without contact — only once nothing is running "
                          "and every result has been delivered")
    ap.add_argument("--solver_filter", default=None)
    ap.add_argument("--exclude_solvers", default="mcts_3000_tpu")
    ap.add_argument("--poll_interval", type=float, default=120.0)
    ap.add_argument("--control_every", type=float, default=300.0,
                     help="seconds between quick checks for a remote stop / check-in request")
    ap.add_argument("--follow_after", type=float, default=300.0,
                     help="seconds without contact before checking for a new server address "
                          "(then re-checked at the same interval)")
    args = ap.parse_args()

    conn = {"base": args.server.rstrip("/"), "last_contact": time.time(), "last_lookup": 0.0}
    t_start = time.time()
    stop_at = t_start + args.session_hours * 3600 - args.stop_margin_min * 60

    def time_left():
        return max(0.0, stop_at - time.time())

    # ---- self-test before claiming anything
    try:
        t = time.time()
        core.run_one_game({"solver": "mcts_60", "width": 3, "height": 3, "seed": 0})
        print(f"[self-test OK] engine + solvers work ({time.time() - t:.1f}s incl. numba compile)")
    except Exception as e:
        print(f"[SELF-TEST FAILED] {type(e).__name__}: {e} — not connecting to the server.")
        raise SystemExit(1)

    session = requests.Session()
    session.headers["ngrok-skip-browser-warning"] = "1"

    def call(method, path, retries=2, quiet=False, **kw):
        for attempt in range(retries):
            try:
                r = session.request(method, conn["base"] + path, timeout=30, **kw)
                r.raise_for_status()
                conn["last_contact"] = time.time()
                return r.json()
            except requests.RequestException as e:
                if attempt == retries - 1:
                    if not quiet:
                        print(f"  [server unreachable: {method} {path.split('?')[0]}: {str(e)[:120]}] "
                              f"— will retry; games keep running")
                    return None
                time.sleep(5 * (attempt + 1))

    def maybe_follow_new_address():
        now = time.time()
        if (not args.url_topic or now - conn["last_contact"] < args.follow_after
                or now - conn["last_lookup"] < args.follow_after):
            return
        conn["last_lookup"] = now
        url = latest_url_from_topic(args.url_topic)
        if url and url != conn["base"]:
            print(f"  [new server address] {conn['base']} -> {url}")
            conn["base"] = url

    # ---- outbox: results not yet confirmed by the server, kept on disk too
    outbox = []
    try:
        with open(OUTBOX_FILE) as f:
            outbox = json.load(f)
        if outbox:
            print(f"  {len(outbox)} undelivered result(s) found in {OUTBOX_FILE}")
    except (OSError, ValueError):
        pass
    flush = {"next_try": 0.0, "backoff": 30.0}

    def save_outbox():
        try:
            with open(OUTBOX_FILE, "w") as f:
                json.dump(outbox, f)
        except OSError:
            pass

    def flush_outbox(force=False):
        if not outbox or (not force and time.time() < flush["next_try"]):
            return
        resp = call("POST", "/results", retries=1, quiet=True,
                    json={"results": outbox, "worker": args.name})
        if resp is not None:
            print(f"  delivered {len(outbox)} result(s)")
            outbox.clear()
            save_outbox()
            flush["backoff"] = 30.0
            flush["next_try"] = 0.0
        else:
            print(f"  [{len(outbox)} result(s) waiting — server unreachable, retry in "
                  f"{flush['backoff']:.0f}s]")
            flush["next_try"] = time.time() + flush["backoff"]
            flush["backoff"] = min(flush["backoff"] * 2, 300.0)

    def tokens(jobs):
        return [{"job_id": j["job_id"], "assignment_token": j.get("assignment_token")} for j in jobs]

    def release(jobs, reason):
        if jobs:
            call("POST", "/release", retries=2, quiet=True,
                 json={"worker": args.name, "jobs": tokens(jobs), "reason": reason})

    print(f"Worker '{args.name}': {args.workers} processes, session ends in "
          f"{args.session_hours:.1f}h, server {conn['base']}"
          + (f", address updates via ntfy.sh/{args.url_topic[:6]}…" if args.url_topic else ""))

    with Pool(args.workers) as pool:
        in_flight = {}  # AsyncResult -> job
        last_hb = time.time() - args.heartbeat_every + 300  # first heartbeat after 5 min
        cpu_prev = [_cpu_times()]

        def cpu_percent():
            now_t, prev = _cpu_times(), cpu_prev[0]
            cpu_prev[0] = now_t
            if not now_t or not prev or now_t[1] == prev[1]:
                return None
            return round(100.0 * _n_cpus() * (now_t[0] - prev[0]) / (now_t[1] - prev[1]), 1)

        next_fetch = 0.0
        next_control = time.time() + args.control_every
        seen_checkin = [None]
        stop_requested = False
        nothing_fits_count = 0
        crash_streak = 0
        server_done = False
        winding_down = False

        while True:
            now = time.time()
            maybe_follow_new_address()
            flush_outbox()

            # ---- remote control: stop request / manual check-in (cheap poll every 5 min)
            if now >= next_control:
                next_control = now + args.control_every
                ctl = call("GET", f"/control?name={args.name}", retries=1, quiet=True)
                if ctl is not None:
                    if ctl.get("stop"):
                        stop_requested = True
                    cid = ctl.get("checkin_id")
                    if seen_checkin[0] is None:
                        seen_checkin[0] = cid
                    elif cid != seen_checkin[0]:
                        seen_checkin[0] = cid
                        last_hb = 0.0  # heartbeat right now (below)

            # ---- end of session (or stop requested): deliver, hand back the rest, leave
            if now >= stop_at or stop_requested:
                why = "stop requested from the PC" if stop_requested else "session ending"
                print(f"{why[0].upper() + why[1:]} — handing back {len(in_flight)} unfinished game(s).")
                release(list(in_flight.values()), why)
                pool.terminate()
                end = time.time() + (args.stop_margin_min - 2) * 60
                while outbox and time.time() < end:
                    flush_outbox(force=True)
                    if outbox:
                        time.sleep(30)
                        maybe_follow_new_address()
                if outbox:
                    print(f"  {len(outbox)} result(s) could not be delivered (kept in {OUTBOX_FILE}).")
                call("POST", "/bye", retries=2, quiet=True, json={"worker": args.name, "reason": why})
                break

            # ---- server gone for a long time and nothing left to lose -> leave
            if (now - conn["last_contact"] > args.give_up_minutes * 60
                    and not in_flight and not outbox):
                print(f"No contact for {args.give_up_minutes:.0f} min and nothing pending — exiting "
                      f"so this notebook can be relaunched.")
                pool.terminate()
                break

            # ---- heartbeat
            if now - last_hb >= args.heartbeat_every:
                cpu = cpu_percent()
                resp = call("POST", "/heartbeat", json={
                    "worker": args.name, "jobs": tokens(in_flight.values()),
                    "time_left": time_left(), "cpu_percent": cpu})
                print(f"  [heartbeat] {len(in_flight)} game(s) running, CPU {cpu}% "
                      f"(max {100 * _n_cpus()}%), {len(outbox)} result(s) waiting, "
                      f"{time_left() / 3600:.1f}h left" + ("" if resp is not None else " — not delivered"))
                last_hb = now

            # ---- top up free slots
            free = args.workers - len(in_flight)
            if free > 0 and not server_done and not winding_down and now >= next_fetch:
                params = {"name": args.name, "count": free, "hb": "1", "time_left": int(time_left())}
                if args.solver_filter:
                    params["solvers"] = args.solver_filter
                elif args.exclude_solvers:
                    params["exclude_solvers"] = args.exclude_solvers
                resp = call("GET", "/jobs", retries=1, params=params)
                if resp is None:
                    next_fetch = now + 60
                else:
                    if resp.get("stop"):
                        stop_requested = True
                    for job in resp.get("jobs", []):
                        in_flight[pool.apply_async(run_job, (job,))] = job
                        print(f"  started {job['solver']} {job['width']}x{job['height']} seed={job['seed']}")
                    if not resp.get("jobs"):
                        next_fetch = now + args.poll_interval
                        if resp.get("done"):
                            server_done = True
                        elif resp.get("nothing_fits"):
                            nothing_fits_count += 1
                            if nothing_fits_count >= 2:
                                winding_down = True
                                print("  remaining games are too long for this session's time left — "
                                      "finishing current games, then exiting for a fresh session")
                        else:
                            nothing_fits_count = 0

            if (server_done or winding_down) and not in_flight and not outbox:
                print("No more work for this session. Exiting.")
                break

            # ---- collect finished games
            finished = [ar for ar in in_flight if ar.ready()]
            if not finished:
                time.sleep(2.0)
                continue
            for ar in finished:
                job = in_flight.pop(ar)
                try:
                    result = ar.get()
                except Exception as e:
                    crash_streak += 1
                    print(f"  [game crashed, job_id={job['job_id']}: {type(e).__name__}: {e}]")
                    release([job], "crashed")
                    if crash_streak >= 5:
                        print("[ABORT] 5 games in a row crashed — handing everything back and exiting.")
                        release(list(in_flight.values()), "worker aborting")
                        flush_outbox(force=True)
                        pool.terminate()
                        raise SystemExit(1)
                    continue
                crash_streak = 0
                print(f"  {result['solver']} {result['board']} seed={result['seed']} "
                      f"done (score={result['score']}, {result['seconds']}s)")
                outbox.append(result)
            save_outbox()
            flush_outbox()  # sends now unless we're in a retry back-off


if __name__ == "__main__":
    main()
