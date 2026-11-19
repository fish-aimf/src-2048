"""Job server for running the benchmark across several machines.

Holds the queue of (solver, board, seed) games and writes every returned
result to one CSV. Workers claim games in batches over HTTP. Workers that send
heartbeats get their games re-queued if they go silent, can hand games back
before their session ends, and are only given games expected to finish in the
time they have left.

    python server.py --out results.csv --port 8765
"""
import argparse
import csv
import os
import threading
import time
import uuid
from collections import defaultdict
from multiprocessing import Pool

from flask import Flask, jsonify, request, send_from_directory, abort

import paths  # noqa: F401  (sets up imports for the chosen objective)
import core
from core import FIELDNAMES, build_all_jobs, parse_boards, apply_power_saving

try:
    from waitress import serve as waitress_serve
    HAVE_WAITRESS = True
except ImportError:
    HAVE_WAITRESS = False

app = Flask(__name__)
state_lock = threading.Lock()
csv_lock = threading.Lock()

state = {
    "pending": [],
    "in_progress": {},
    "done_count": 0,
    "total_count": 0,
    "out_path": None,
    "job_timeout": 600.0,
    "heartbeat_timeout": 5400.0,
    "time_safety": 1.5,
    "fresh_session_s": 10.5 * 3600,
    "code_dir": None,
    "admin_token": None,  # required for /admin/* (set by the tray app)
    "stopping": False,  # remote stop: every worker hands back its games and exits
    "stop_workers": set(),  # remote stop for some workers only (session refresh)
    "checkin_id": int(time.time()) % 1000000,
                                                # heartbeat right after a restart (-> adoption)
    "stats": {},  # (solver, board) -> [count, total_seconds]  (all rows ever)
    "recent": None,
    "workers": {},  # name -> {"last_seen", "done", "time_left"}
    "remaining": defaultdict(int),  # (solver, board) -> games not yet written
    "completed": set(),  # (solver, board, seed) already in results.csv
    "plan_keys": set(),  # (solver, board, seed) of every game in this run
}

_real_print = print


def print(*args, **kwargs):
    _real_print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)


@app.errorhandler(Exception)
def handle_any_error(e):
    import traceback
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    print(f"  [UNHANDLED SERVER ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
    return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------- helpers

def _board(job):
    return f"{job['width']}x{job['height']}"


RECENT_N = 2000


def load_completed_and_stats(out_path):
    from collections import deque
    completed = set()
    stats = {}
    recent = deque(maxlen=RECENT_N)
    if os.path.exists(out_path):
        with open(out_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["solver"], row["board"])
                completed.add((row["solver"], row["board"], int(row["seed"])))
                try:
                    secs = float(row["seconds"])
                except (TypeError, ValueError):
                    continue
                s = stats.setdefault(key, [0, 0.0])
                s[0] += 1
                s[1] += secs
                recent.append((key, secs))
    return completed, stats, recent


def _speed_estimates():
    """Per-combo expected game length on the machines running now."""
    rec = defaultdict(lambda: [0, 0.0])
    for key, secs in state["recent"] or ():
        rec[key][0] += 1
        rec[key][1] += secs
    ratios = []
    older = {}
    for key, (n_all, t_all) in state["stats"].items():
        n_r, t_r = rec.get(key, (0, 0.0))
        n_o, t_o = n_all - n_r, t_all - t_r
        if n_o > 0:
            older[key] = t_o / n_o
        if n_r >= 10 and n_o >= 10 and t_o > 0:
            ratios.append((t_r / n_r) / (t_o / n_o))
    ratios.sort()
    factor = ratios[len(ratios) // 2] if ratios else 1.0
    est = {}
    for key in set(state["stats"]) | set(rec):
        n_r, t_r = rec.get(key, (0, 0.0))
        if n_r >= 5:
            est[key] = t_r / n_r
        elif key in older:
            est[key] = older[key] * factor
        elif n_r:
            est[key] = t_r / n_r
    return est, factor


def _estimate_seconds(job):
    """Expected length of this game on today's machines, or None if unknown."""
    now = time.time()
    cache = state.get("_est_cache")
    if not cache or now - cache[0] > 60:  # recompute at most once a minute
        cache = (now, *_speed_estimates())
        state["_est_cache"] = cache
    return cache[1].get((job["solver"], _board(job)))


def _fits(job, time_left):
    if time_left is None:
        return True
    est = _estimate_seconds(job)
    if est is None:
        return True  # never seen this combo: let it run rather than starve it
    if est * state["time_safety"] <= time_left:
        return True
    # A game longer than any session would never be handed out at all. Give
    # such games only to freshly started sessions (most time possible).
    return time_left >= state["fresh_session_s"]


def _touch_worker(name, time_left=None):
    w = state["workers"].setdefault(name, {"last_seen": 0.0, "done": 0, "time_left": None,
                                           "cpu_percent": None, "gone": False})
    w["last_seen"] = time.time()
    w["gone"] = False  # any contact = alive (a relaunched notebook reuses its name)
    if time_left is not None:
        w["time_left"] = time_left


def _requeue_timed_out():
    """Caller holds state_lock."""
    now = time.time()
    stale = []
    for jid, rec in state["in_progress"].items():
        if rec.get("hb"):
            if now - rec["last_seen"] > state["heartbeat_timeout"]:
                stale.append((jid, "no heartbeat"))
        elif now - rec["assigned_at"] > state["job_timeout"]:
            stale.append((jid, "job_timeout"))
    for jid, why in stale:
        rec = state["in_progress"].pop(jid)
        j = rec["job"]
        print(f"  [requeue: {why}] {rec['worker']} — {j['solver']} {_board(j)} seed={j['seed']}")
        state["pending"].insert(0, j)


def _record(job, worker, hb, now):
    token = uuid.uuid4().hex
    state["in_progress"][job["job_id"]] = {
        "job": job, "worker": worker, "assigned_at": now, "token": token,
        "hb": hb, "last_seen": now,
    }
    return token


def _filter_matches(job, allowed, excluded):
    s = job["solver"]
    if allowed is not None and s not in allowed:
        return False
    if excluded is not None and s in excluded:
        return False
    return True


def _claim_jobs_locked(worker, count, allowed=None, excluded=None, time_left=None, hb=False):
    _requeue_timed_out()
    claimed = []
    now = time.time()
    i = 0
    pending = state["pending"]
    while len(claimed) < count and i < len(pending):
        job = pending[i]
        if _filter_matches(job, allowed, excluded) and _fits(job, time_left):
            pending.pop(i)
            claimed.append((job, _record(job, worker, hb, now)))
        else:
            i += 1
    return claimed


def _claim_job_locked(worker):
    """Claim one job (used by the /job endpoint and the local process pool)."""
    _requeue_timed_out()
    for i, j in enumerate(state["pending"]):
        if not j["solver"].endswith("_tpu"):
            job = state["pending"].pop(i)
            return job, _record(job, worker, False, time.time())
    return None, None


def _write_result(row, worker, late=False):
    with csv_lock:
        with open(state["out_path"], "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
    key = (row["solver"], row["board"])
    with state_lock:
        state["done_count"] += 1
        state["remaining"][key] -= 1
        s = state["stats"].setdefault(key, [0, 0.0])
        try:
            secs = float(row["seconds"])
            s[0] += 1
            s[1] += secs
            if state["recent"] is not None:
                state["recent"].append((key, secs))
        except (TypeError, ValueError):
            pass
        _touch_worker(worker)
        state["workers"][worker]["done"] += 1
        done, total = state["done_count"], state["total_count"]
    flag = (" [TIMED OUT]" if row["timed_out"] else "") + (" [late result kept]" if late else "")
    print(f"  <- {worker}: {row['solver']} {row['board']} seed={row['seed']} "
          f"done (score={row['score']}, {row['seconds']}s){flag}  [{done}/{total}]")


def _accept_result(item, worker):
    """Validate one result payload and write it."""
    job_id = item.get("job_id")
    token = item.get("assignment_token")
    try:
        row = {k: item[k] for k in FIELDNAMES}
        key = (row["solver"], row["board"], int(row["seed"]))
    except (KeyError, TypeError, ValueError) as e:
        print(f"  [MALFORMED RESULT from {worker}: {e}] job_id={job_id} — ignored")
        return {"job_id": job_id, "ok": False, "error": f"malformed: {e}"}
    late = False
    with state_lock:
        if key in state["completed"]:
            rec = state["in_progress"].get(job_id)
            if rec is not None and rec.get("token") == token:
                del state["in_progress"][job_id]
            return {"job_id": job_id, "ok": True, "note": "already recorded, ignored"}
        rec = state["in_progress"].get(job_id)
        if rec is None or rec.get("token") != token:
            if key not in state["plan_keys"]:
                return {"job_id": job_id, "ok": True, "note": "not part of this run, ignored"}
            late = True
            # cancel any other copy of this game (queued or running elsewhere)
            for jid, r in list(state["in_progress"].items()):
                if (r["job"]["solver"], _board(r["job"]), r["job"]["seed"]) == key:
                    del state["in_progress"][jid]
            state["pending"] = [j for j in state["pending"]
                                if (j["solver"], _board(j), j["seed"]) != key]
        else:
            del state["in_progress"][job_id]
        state["completed"].add(key)  # claimed before the disk write: no double write
    _write_result(row, worker, late)
    return {"job_id": job_id, "ok": True}


def _parse_filters(args):
    sp = args.get("solvers", "").strip()
    ep = args.get("exclude_solvers", "").strip()
    allowed = set(s.strip() for s in sp.split(",") if s.strip()) if sp else None
    excluded = set(s.strip() for s in ep.split(",") if s.strip()) if ep else None
    return allowed, excluded


# ---------------------------------------------------------------- endpoints

@app.route("/job", methods=["GET"])
def get_job():
    worker = request.args.get("name", request.remote_addr)
    with state_lock:
        _touch_worker(worker)
        job, token = _claim_job_locked(worker)
        if job is None:
            return jsonify({"job": None, "done": len(state["in_progress"]) == 0}), 200
    job_out = dict(job)
    job_out["assignment_token"] = token
    print(f"  -> {worker}: {job['solver']} {_board(job)} seed={job['seed']}")
    return jsonify({"job": job_out, "done": False}), 200


@app.route("/result", methods=["POST"])
def post_result():
    payload = request.get_json(force=True)
    worker = payload.get("worker", request.remote_addr)
    out = _accept_result(payload, worker)
    return jsonify(out), (200 if out.get("ok") else 400)


@app.route("/jobs", methods=["GET"])
def get_jobs():
    """Hand out a batch of jobs. Query: name, count, solvers, exclude_solvers, hb,
    time_left.
    """
    worker = request.args.get("name", request.remote_addr)
    try:
        count = max(1, int(request.args.get("count", 1)))
    except (TypeError, ValueError):
        count = 1
    allowed, excluded = _parse_filters(request.args)
    hb = request.args.get("hb") == "1"
    try:
        time_left = float(request.args["time_left"]) if "time_left" in request.args else None
    except ValueError:
        time_left = None

    with state_lock:
        _touch_worker(worker, time_left)
        if state["stopping"] or worker in state["stop_workers"]:
            return jsonify({"jobs": [], "done": False, "nothing_fits": False, "stop": True}), 200
        claimed = _claim_jobs_locked(worker, count, allowed, excluded, time_left, hb)
        pending_match = any(_filter_matches(j, allowed, excluded) for j in state["pending"])
        inprog_match = any(_filter_matches(r["job"], allowed, excluded)
                           for r in state["in_progress"].values())
        done = (not pending_match) and (not inprog_match) and (not claimed)
        # "nothing fits my remaining time" is not "done": tell the worker so it can wind down
        nothing_fits = (not claimed) and pending_match
    jobs_out = []
    for job, token in claimed:
        j = dict(job)
        j["assignment_token"] = token
        jobs_out.append(j)
    if jobs_out:
        print(f"  -> {worker}: {len(jobs_out)} job(s)"
              + (f" (time_left {time_left/3600:.1f}h)" if time_left is not None else ""))
    return jsonify({"jobs": jobs_out, "done": done, "nothing_fits": nothing_fits}), 200


@app.route("/results", methods=["POST"])
def post_results():
    payload = request.get_json(force=True)
    worker = payload.get("worker", request.remote_addr)
    outcomes = [_accept_result(item, worker) for item in payload.get("results", [])]
    return jsonify({"ok": True, "results": outcomes}), 200


@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    """Refresh the jobs a worker still owns. Body: worker, jobs, time_left."""
    p = request.get_json(force=True)
    worker = p.get("worker", request.remote_addr)
    now = time.time()
    lost = []
    with state_lock:
        _touch_worker(worker, p.get("time_left"))
        if p.get("cpu_percent") is not None:
            state["workers"][worker]["cpu_percent"] = p.get("cpu_percent")
        adopted = 0
        for item in p.get("jobs", []):
            jid = item.get("job_id")
            rec = state["in_progress"].get(jid)
            if rec is not None and rec.get("token") == item.get("assignment_token"):
                rec["last_seen"] = now
                rec["hb"] = True
                continue
            if rec is None:
                for i, j in enumerate(state["pending"]):
                    if j["job_id"] == jid:
                        state["pending"].pop(i)
                        state["in_progress"][jid] = {
                            "job": j, "worker": worker, "assigned_at": now,
                            "token": item.get("assignment_token"), "hb": True, "last_seen": now}
                        adopted += 1
                        break
                else:
                    lost.append(jid)
            else:
                lost.append(jid)
        if adopted:
            print(f"  [adopted] {worker}: {adopted} running game(s) re-attached after restart")
    return jsonify({"ok": True, "lost": lost}), 200


@app.route("/release", methods=["POST"])
def release():
    """Put a worker's unfinished jobs back at the front of the queue. Body: worker,
    jobs, reason.
    """
    p = request.get_json(force=True)
    worker = p.get("worker", request.remote_addr)
    reason = p.get("reason", "released")
    n = 0
    with state_lock:
        _touch_worker(worker)
        for item in p.get("jobs", []):
            jid = item.get("job_id")
            rec = state["in_progress"].get(jid)
            if rec is not None and rec.get("token") == item.get("assignment_token"):
                del state["in_progress"][jid]
                state["pending"].insert(0, rec["job"])
                n += 1
    if n:
        print(f"  [release] {worker}: {n} job(s) back in the queue ({reason})")
    return jsonify({"ok": True, "released": n}), 200


@app.route("/code/<path:name>", methods=["GET"])
def code(name):
    d = state["code_dir"]
    if not d or not name.endswith(".py") or "/" in name or "\\" in name or ".." in name:
        abort(404)
    if not os.path.isfile(os.path.join(d, name)):
        abort(404)
    return send_from_directory(os.path.abspath(d), name, mimetype="text/plain")


@app.route("/control", methods=["GET"])
def control():
    """Polled by workers every few minutes: should this worker stop?"""
    worker = request.args.get("name", request.remote_addr)
    with state_lock:
        _touch_worker(worker)
        return jsonify({"stop": state["stopping"] or worker in state["stop_workers"],
                        "checkin_id": state["checkin_id"]}), 200


@app.route("/bye", methods=["POST"])
def bye():
    """A worker says it has stopped (after handing back its games)."""
    p = request.get_json(force=True)
    worker = p.get("worker", request.remote_addr)
    with state_lock:
        _touch_worker(worker)
        state["workers"][worker]["gone"] = True
        state["stop_workers"].discard(worker)  # its replacement (same name) may work normally
    print(f"  [bye] {worker} stopped ({p.get('reason', '')})")
    return jsonify({"ok": True}), 200


def _admin_ok():
    tok = state.get("admin_token")
    return bool(tok) and request.headers.get("X-Admin-Token") == tok


@app.route("/admin/stop", methods=["POST"])
def admin_stop():
    if not _admin_ok():
        abort(403)
    with state_lock:
        state["stopping"] = True
    print("  [STOP requested] workers will hand back their games and exit at their next check (<= 5 min)")
    return jsonify({"ok": True}), 200


@app.route("/admin/stop_worker", methods=["POST"])
def admin_stop_worker():
    """Stop one worker (session refresh): it hands back its games and exits at its next
    check.
    """
    if not _admin_ok():
        abort(403)
    name = (request.get_json(force=True) or {}).get("name")
    with state_lock:
        state["stop_workers"].add(name)
    print(f"  [refresh] asked {name} to hand back its games and end its session")
    return jsonify({"ok": True}), 200


@app.route("/admin/checkin", methods=["POST"])
def admin_checkin():
    if not _admin_ok():
        abort(403)
    with state_lock:
        state["checkin_id"] += 1
        n = state["checkin_id"]
    print(f"  [check-in requested #{n}] workers will report within ~5 min")
    return jsonify({"ok": True, "checkin_id": n}), 200


@app.route("/status", methods=["GET"])
def status():
    now = time.time()
    with state_lock:
        _requeue_timed_out()
        per_worker = defaultdict(int)
        running = defaultdict(list)
        for rec in state["in_progress"].values():
            per_worker[rec["worker"]] += 1
            running[rec["worker"]].append(
                {"game": f"{rec['job']['solver']} {_board(rec['job'])}",
                 "minutes": round((now - rec["assigned_at"]) / 60)})
        workers = {
            name: {
                "in_progress": per_worker.get(name, 0),
                "results": w["done"],
                "seconds_since_seen": round(now - w["last_seen"]),
                "time_left_h": None if w["time_left"] is None else round(w["time_left"] / 3600, 2),
                "cpu_percent": w.get("cpu_percent"),
                "gone": w.get("gone", False),
                "stop_requested": name in state["stop_workers"],
                "running": running.get(name, []),
            }
            for name, w in sorted(state["workers"].items())
        }
        remaining = {f"{s} {b}": n for (s, b), n in sorted(state["remaining"].items()) if n > 0}
        est, factor = _speed_estimates()
        est_min = {f"{s} {b}": round(est[(s, b)] / 60, 1) for (s, b), n in state["remaining"].items()
                   if n > 0 and (s, b) in est}
        return jsonify({
            "pending": len(state["pending"]),
            "in_progress": len(state["in_progress"]),
            "done": state["done_count"],
            "total": state["total_count"],
            "remaining_by_combo": remaining,
            "workers": workers,
            "est_minutes_per_game": est_min,
            "speed_factor": round(factor, 2),
            "stopping": state["stopping"],
            "checkin_id": state["checkin_id"],
        })


# ---------------------------------------------------------------- local pool (unchanged idea)

def local_worker_init(priority, efficiency_mode):
    apply_power_saving(priority, efficiency_mode)


def _run_local_job(job):
    return core.run_one_game(job)


def local_dispatch_loop(name, num_workers, priority, efficiency_mode):
    with Pool(num_workers, initializer=local_worker_init,
              initargs=(priority, efficiency_mode)) as pool:
        in_flight = {}

        def top_up():
            while len(in_flight) < num_workers:
                with state_lock:
                    job, token = _claim_job_locked(name)
                if job is None:
                    return
                in_flight[pool.apply_async(_run_local_job, (job,))] = (job, token)

        while True:
            top_up()
            if not in_flight:
                time.sleep(5.0)
                continue
            finished = [ar for ar in in_flight if ar.ready()]
            if not finished:
                time.sleep(0.2)
                continue
            for ar in finished:
                job, token = in_flight.pop(ar)
                try:
                    result = ar.get()
                except Exception as e:
                    print(f"  [local game crashed, job_id={job['job_id']}: {e}] — requeuing")
                    with state_lock:
                        rec = state["in_progress"].get(job["job_id"])
                        if rec is not None and rec.get("token") == token:
                            del state["in_progress"][job["job_id"]]
                            state["pending"].insert(0, job)
                    continue
                result["job_id"] = job["job_id"]
                result["assignment_token"] = token
                _accept_result(result, f"{name} (local)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", type=str, default="results.csv")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--boards", type=str, default="3x3,3x4,4x3,4x4")
    ap.add_argument("--solvers", type=str,
                     default="random,greedy,expectimax_d2,expectimax_d3,expectimax_d4,mcts_60,mcts_300,mcts_1000")
    ap.add_argument("--plan", type=str, default=None,
                     help="'solvers:boards:games;solvers:boards:games;...' — overrides "
                          "--solvers/--boards/--games. Queued in the order written.")
    ap.add_argument("--max_moves", type=int, default=None)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--time_budget", type=float, default=None)
    ap.add_argument("--duty_cycle", type=float, default=1.0)
    ap.add_argument("--heartbeat_seconds", type=float, default=5.0)
    ap.add_argument("--job_timeout", type=float, default=86400.0,
                     help="requeue after this long — only for OLD workers without heartbeats")
    ap.add_argument("--heartbeat_timeout", type=float, default=5400.0,
                     help="requeue a heartbeat worker's jobs after this many silent seconds")
    ap.add_argument("--time_safety", type=float, default=1.5,
                     help="only hand out a game if mean_length * this <= worker's time_left")
    ap.add_argument("--admin_token", type=str, default=os.environ.get("BENCH_ADMIN_TOKEN"),
                     help="secret for /admin/stop and /admin/checkin (the tray app sets it)")
    ap.add_argument("--fresh_session_hours", type=float, default=10.5,
                     help="a game expected to take longer than any session is only given to a "
                          "worker with at least this much session time left")
    ap.add_argument("--code_dir", type=str, default=None,
                     help="folder whose .py files are served at /code/<name> for notebooks")
    ap.add_argument("--local_workers", type=int, default=0)
    ap.add_argument("--local_name", type=str, default=None)
    ap.add_argument("--local_priority", type=str, default="idle",
                     choices=["normal", "below_normal", "idle"])
    ap.add_argument("--local_no_efficiency_mode", action="store_true")
    ap.add_argument("--priority", type=str, default="below_normal",
                     choices=["normal", "below_normal", "idle"])
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    if args.plan:
        groups = [g.strip() for g in args.plan.split(";") if g.strip()]
        specs = []
        for g in groups:
            g_solvers, g_boards, g_games = g.split(":")
            specs.append(([s.strip() for s in g_solvers.split(",") if s.strip()],
                          parse_boards(g_boards), int(g_games)))
    else:
        specs = [(args.solvers.split(","), parse_boards(args.boards), args.games)]

    all_jobs, seen = [], set()
    for solvers, boards, games in specs:
        for j in build_all_jobs(solvers, boards, games, args.seed_offset, args.max_moves,
                                args.time_budget, args.duty_cycle, args.heartbeat_seconds):
            k = (j["solver"], j["width"], j["height"], j["seed"])
            if k not in seen:
                seen.add(k)
                all_jobs.append(j)
    for i, j in enumerate(all_jobs):
        j["job_id"] = i

    completed, stats, recent = load_completed_and_stats(args.out)
    pending = [j for j in all_jobs if (j["solver"], _board(j), j["seed"]) not in completed]
    skipped = len(all_jobs) - len(pending)

    if not os.path.exists(args.out):
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    if args.code_dir and not os.path.isdir(args.code_dir):
        raise SystemExit(f"--code_dir '{args.code_dir}' does not exist")

    with state_lock:
        state["pending"] = pending
        state["out_path"] = args.out
        state["job_timeout"] = args.job_timeout
        state["heartbeat_timeout"] = args.heartbeat_timeout
        state["time_safety"] = args.time_safety
        state["code_dir"] = args.code_dir
        state["admin_token"] = args.admin_token
        state["stats"] = stats
        state["recent"] = recent
        state["fresh_session_s"] = args.fresh_session_hours * 3600
        state["completed"] = completed
        state["plan_keys"] = {(j["solver"], _board(j), j["seed"]) for j in all_jobs}
        state["done_count"] = skipped
        state["total_count"] = len(all_jobs)
        for j in pending:
            state["remaining"][(j["solver"], _board(j))] += 1

    if skipped:
        print(f"Resuming: {skipped} games already in {args.out}, skipping those.")
    print(f"Coordinator up on {args.host}:{args.port} — {len(pending)} games queued, writing to {args.out}")
    for (s, b), n in sorted(state["remaining"].items()):
        est = _estimate_seconds({"solver": s, "width": int(b.split('x')[0]), "height": int(b.split('x')[1])})
        print(f"    {s:18s} {b}: {n:5d} left"
              + (f"  (~{est/60:.0f} min/game)" if est else "  (no timing data yet)"))
    if args.code_dir:
        print(f"Serving worker code from {args.code_dir}: "
              + ", ".join(sorted(f for f in os.listdir(args.code_dir) if f.endswith('.py'))))
    print(f"Heartbeat timeout {args.heartbeat_timeout:.0f}s, old-worker job timeout {args.job_timeout:.0f}s. "
          f"GET /status for the live overview.")
    apply_power_saving(args.priority, True)

    if args.local_workers > 0:
        import platform
        local_name = args.local_name or platform.node() or "coordinator-local"
        print(f"Also computing locally: {args.local_workers} workers as '{local_name}'")
        threading.Thread(target=local_dispatch_loop,
                         args=(local_name, args.local_workers, args.local_priority,
                               not args.local_no_efficiency_mode), daemon=True).start()

    if HAVE_WAITRESS:
        waitress_serve(app, host=args.host, port=args.port, threads=args.threads)
    else:
        print("waitress not installed (pip install waitress) — using Flask's dev server.")
        app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
