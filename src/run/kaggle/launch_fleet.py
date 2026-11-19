"""Keep a set of Kaggle CPU notebooks running kaggle_worker.py.

Uses the Kaggle API. Every --interval seconds it restarts any notebook that
has finished its session and prints each worker's status from the server.
API keys are asked for at start-up and kept in memory only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "notebook_template.py")
KERNELS_DIR = os.path.join(HERE, "fleet_kernels")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # don't flash a console per kaggle call


def stamp(msg):
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"


def kaggle_cmd():
    exe = shutil.which("kaggle")
    return [exe] if exe else [sys.executable.replace("pythonw.exe", "python.exe"), "-m", "kaggle.cli"]


def account_env(username, key):
    env = dict(os.environ)
    for k in ("KAGGLE_CONFIG_DIR", "KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        env.pop(k, None)
    env["KAGGLE_USERNAME"] = username
    if key.startswith("KGAT_"):
        env["KAGGLE_API_TOKEN"] = key
    else:
        env["KAGGLE_KEY"] = key  # classic key from kaggle.json
    return env


def parse_key(user, key):
    """Accepts a bare key or a pasted kaggle.json."""
    key = key.strip()
    if key.startswith("{"):
        j = json.loads(key)
        return j.get("username", user), j["key"]
    return user, key


def run_kaggle(args, env):
    p = subprocess.run(kaggle_cmd() + args, capture_output=True, text=True, env=env,
                       creationflags=NO_WINDOW)
    return p.returncode, (p.stdout + p.stderr).strip()


def check_account(user, key):
    """Returns (ok, message, env)."""
    env = account_env(user, key)
    code, out = run_kaggle(["kernels", "list", "--mine", "--page-size", "1"], env)
    return code == 0, ("OK" if code == 0 else out[:200]), env


def write_kernel(username, slug, server, name, session_hours, topic=""):
    d = os.path.join(KERNELS_DIR, username, slug)
    os.makedirs(d, exist_ok=True)
    with open(TEMPLATE, encoding="utf-8") as f:
        code = (f.read().replace("__SERVER__", server)
                        .replace("__NAME__", name)
                        .replace("__SESSION_HOURS__", str(session_hours))
                        .replace("__TOPIC__", topic or ""))
    with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as f:
        f.write(code)
    meta = {
        "id": f"{username}/{slug}", "title": slug, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return d


def kernel_state(username, slug, env):
    code, out = run_kaggle(["kernels", "status", f"{username}/{slug}"], env)
    low = out.lower()
    if code != 0 or "404" in low or "not found" in low:
        return "missing", out
    for s in ("running", "queued", "complete", "error", "cancel"):
        if s in low:
            return s, out
    return "unknown", out


def server_status(url):
    try:
        req = urllib.request.Request(url.rstrip("/") + "/status",
                                     headers={"ngrok-skip-browser-warning": "1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def cpu_games_left(st):
    if st is None:
        return None
    return sum(n for combo, n in st.get("remaining_by_combo", {}).items()
               if not combo.split()[0].endswith("_tpu"))


def current_wifi():
    """Name (ssid) of the Wi-Fi network this PC is on, or None (wired / unknown / not
    Windows).
    """
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True,
                             text=True, creationflags=NO_WINDOW, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        k, _, v = line.partition(":")
        if k.strip().lower() == "ssid" and v.strip():
            return v.strip()
    return None


def run_fleet(accounts, get_server, status_url=None, per_account=5, prefix="bench2048-cpu",
              session_hours=11.5, interval=1800, stale_minutes=100,
              log=print, on_table=None, stop_event=None, topic="", wake_event=None,
              api_allowed=None, status_gap=(20, 60), push_gap=(45, 120)):
    """accounts: [{"user", "env"}]."""
    import random
    if api_allowed is None:
        api_allowed = lambda: (True, "")
    notebooks = []
    for a in accounts:
        tag = "".join(c for c in a["user"] if c.isalnum() or c in "-_")[:20]
        for i in range(1, per_account + 1):
            notebooks.append({"acc": a, "slug": f"{prefix}-{i:02d}", "name": f"{tag}-{i:02d}",
                              "last_push": 0.0, "state": "?", "state_at": 0.0})
    log(stamp(f"{len(accounts)} account(s), {len(notebooks)} notebooks (~{4 * len(notebooks)} cores)"))

    def stopped():
        return stop_event is not None and stop_event.is_set()

    def pause(lo_hi):
        end = time.time() + random.uniform(*lo_hi)
        while time.time() < end and not stopped():
            time.sleep(0.5)

    while not stopped():
        server = get_server()
        if not server:
            log(stamp("waiting for the tunnel address..."))
            pause((30, 30))
            continue
        st = server_status(status_url or server)
        left = cpu_games_left(st)
        if left == 0:
            log(stamp("No CPU games left — not launching any more notebooks."))
            if on_table:
                on_table([stamp("No CPU games left — fleet finished.")])
            break
        workers = (st or {}).get("workers", {})

        # 1) Kaggle status — only for notebooks the server hasn't heard from lately
        ok, why = api_allowed()
        api_line = "Kaggle API: allowed" if ok else f"Kaggle API: PAUSED ({why}) — nothing sent to Kaggle"
        queried = 0
        for nb in notebooks:
            w = workers.get(nb["name"])
            if w and not w.get("gone") and w["seconds_since_seen"] < 45 * 60:
                nb["state"], nb["state_at"] = "running", time.time()  # known from the server
                continue
            ok, why = api_allowed()
            if not ok or stopped():
                break
            if queried:
                pause(status_gap)
            nb["state"], _ = kernel_state(nb["acc"]["user"], nb["slug"], nb["acc"]["env"])
            nb["state_at"] = time.time()
            queried += 1

        # 2) table
        lines = [stamp(api_line)]
        if st is None:
            lines.append(stamp("server /status unreachable"))
        else:
            lines.append(stamp(f"server: {st['done']}/{st['total']} done | {st['pending']} pending | "
                               f"{st['in_progress']} running | CPU games left: {left}"))
        lines.append(f"{'notebook':36s} {'kaggle':9s} {'contact':>8s} {'CPU':>6s} {'games':>5s} {'results':>7s}")
        for nb in notebooks:
            w = workers.get(nb["name"])
            contact = f"{w['seconds_since_seen'] // 60}m" if w else "-"
            cpu = f"{w['cpu_percent']:.0f}%" if w and w.get("cpu_percent") is not None else "-"
            games = str(w["in_progress"]) if w else "-"
            results = str(w["results"]) if w else "-"
            note = ""
            if nb["state"] == "running" and w and w["seconds_since_seen"] > stale_minutes * 60:
                note = "  <-- silent, check its log on kaggle.com"
            label = f"{nb['acc']['user']}/{nb['slug']}"
            lines.append(f"{label:36s} {nb['state']:9s} {contact:>8s} {cpu:>6s} {games:>5s} {results:>7s}{note}")
        for line in lines:
            log(line)
        if on_table:
            on_table(lines)

        # 3) (re)launch notebooks that aren't running — staggered, only while allowed
        pushes = [nb for nb in notebooks
                  if nb["state"] in ("missing", "complete", "error", "cancel")
                  and time.time() - nb["last_push"] >= 15 * 60]
        launched = 0
        for nb in pushes:
            ok, why = api_allowed()
            if not ok or stopped():
                if not ok:
                    log(stamp(f"  launches paused ({why}) — {len(pushes) - launched} waiting"))
                break
            if launched:
                pause(push_gap)
            user, env = nb["acc"]["user"], nb["acc"]["env"]
            d = write_kernel(user, nb["slug"], server, nb["name"], session_hours, topic)
            code, out = run_kaggle(["kernels", "push", "-p", d], env)
            nb["last_push"] = time.time()
            nb["state"] = "queued" if code == 0 else nb["state"]
            launched += 1
            log(stamp(f"  -> {user}/{nb['slug']}: " + ("launched" if code == 0 else f"push FAILED: {out[:250]}")))

        base = interval if not launched else min(interval, 600)
        wait = int(base * random.uniform(0.8, 1.2))
        log(stamp(f"next check-in in ~{wait // 60} min"))
        end = time.time() + wait
        while time.time() < end and not stopped():
            if wake_event is not None and wake_event.is_set():  # "Check in now"
                wake_event.clear()
                break
            time.sleep(1)


def ask_accounts():
    print("Enter each Kaggle account. Leave the username blank when done.")
    accounts = []
    while True:
        user = input(f"  Account {len(accounts) + 1} — Kaggle username: ").strip()
        if not user:
            break
        try:
            user, key = parse_key(user, input(f"  API key for {user}: "))
        except (ValueError, KeyError):
            print("    couldn't read that as kaggle.json — paste just the key. Skipped.")
            continue
        if not key:
            print("    no key — skipped.")
            continue
        ok, msg, env = check_account(user, key)
        print(f"    {user}: {msg}" + ("" if ok else " — skipped"))
        if ok:
            accounts.append({"user": user, "env": env})
    return accounts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None, help="public tunnel URL of server.py")
    ap.add_argument("--server_file", default=None, help="text file holding the tunnel URL; re-read each check-in")
    ap.add_argument("--per_account", type=int, default=5)
    ap.add_argument("--prefix", default="bench2048-cpu")
    ap.add_argument("--session_hours", type=float, default=11.5)
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--stagger", type=int, default=10)
    ap.add_argument("--stale_minutes", type=int, default=100)
    args = ap.parse_args()

    def get_server():
        if args.server_file:
            try:
                with open(args.server_file, encoding="utf-8-sig") as f:
                    url = f.readline().strip()
                if url:
                    return url.rstrip("/")
            except OSError:
                pass
        return args.server.rstrip("/") if args.server else None

    code, out = run_kaggle(["--version"], dict(os.environ))
    if code != 0:
        raise SystemExit(f"`kaggle` command not working:\n{out}\nRun: pip install kaggle")
    accounts = ask_accounts()
    if not accounts:
        raise SystemExit("No accounts entered.")
    run_fleet(accounts, get_server, per_account=args.per_account, prefix=args.prefix,
              session_hours=args.session_hours, interval=args.interval,
              stale_minutes=args.stale_minutes)


if __name__ == "__main__":
    main()
