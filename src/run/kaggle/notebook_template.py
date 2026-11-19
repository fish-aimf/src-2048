# Kaggle notebook (script) for one 2048 benchmark CPU worker.
# launch_fleet.py fills in the values below and pushes one copy per notebook.
SERVER = "__SERVER__"
NAME = "__NAME__"
SESSION_HOURS = __SESSION_HOURS__
URL_TOPIC = "__TOPIC__"

import os
import re
import subprocess
import sys
import time
import urllib.request

t0 = time.time()
os.chdir("/kaggle/working")
URL_OK = re.compile(r"^https://[A-Za-z0-9.-]+\.(trycloudflare\.com|ngrok-free\.dev|ngrok-free\.app|ts\.net)$")


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "1",
                                               "User-Agent": "kaggle-2048-worker"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def newest_address():
    """Latest address the PC published (in case it changed since this notebook was
    pushed).
    """
    if not URL_TOPIC or URL_TOPIC.startswith("__"):
        return None
    try:
        text = get(f"https://ntfy.sh/{URL_TOPIC}/raw?poll=1&since=24h", timeout=20).decode()
        for line in reversed([l.strip().rstrip("/") for l in text.splitlines() if l.strip()]):
            if URL_OK.match(line):
                return line
    except Exception:
        pass
    return None


# 1) numba: use what the image has; install only if missing, pinned to versions
#    that work with numpy < 2.4.
try:
    import numba  # noqa: F401
    print("numba already installed:", numba.__version__, flush=True)
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "numpy>=2.0,<2.4", "numba>=0.61,<0.63"], check=False)

# 2) download the current worker code from the coordinator (no dataset needed)
FILES = ["core.py", "engine.py", "solvers.py", "kaggle_worker.py"]
for f in FILES:
    for attempt in range(10):
        try:
            data = get(f"{SERVER}/code/{f}")
            with open(f, "wb") as out:
                out.write(data)
            break
        except Exception as e:
            print(f"download {f} from {SERVER} failed ({e}), retrying...", flush=True)
            new = newest_address()
            if new and new != SERVER:
                print(f"switching to the newer address {new}", flush=True)
                SERVER = new
            time.sleep(30)
    else:
        raise SystemExit(f"could not download {f} — is the PC's server + tunnel running?")
print("code downloaded:", FILES, flush=True)

# 3) run the worker until the session is nearly over
hours_left = SESSION_HOURS - (time.time() - t0) / 3600
cmd = [sys.executable, "-u", "kaggle_worker.py", "--server", SERVER, "--name", NAME,
       "--session_hours", f"{hours_left:.3f}"]
if URL_TOPIC and not URL_TOPIC.startswith("__"):
    cmd += ["--url_topic", URL_TOPIC]
subprocess.run(cmd, check=False)
print("worker finished — session ending normally.", flush=True)
