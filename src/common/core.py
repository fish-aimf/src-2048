"""Plays one benchmark game and builds the job list.

Shared by every runner (benchmark.py, server.py, worker.py, kaggle_worker.py)
so a game is simulated the same way wherever it runs.
"""
import os
import time

from engine import Game
from solvers import SOLVERS

FIELDNAMES = ["solver", "board", "seed", "score", "board_sum", "max_tile", "moves", "seconds", "timed_out"]

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

PRIORITY_MAP = {}
if HAVE_PSUTIL and os.name == "nt":
    PRIORITY_MAP = {
        "normal": psutil.NORMAL_PRIORITY_CLASS,
        "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
        "idle": psutil.IDLE_PRIORITY_CLASS,
    }


def apply_power_saving(priority="idle", efficiency_mode=True):
    """On Windows, lower the process priority and enable efficiency mode so long runs
    don't hog the machine.
    """
    if HAVE_PSUTIL and priority in PRIORITY_MAP:
        try:
            psutil.Process().nice(PRIORITY_MAP[priority])
        except Exception:
            pass

    if efficiency_mode and os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
                _fields_ = [
                    ("Version", wintypes.ULONG),
                    ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG),
                ]

            PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
            ProcessPowerThrottling = 4

            state = PROCESS_POWER_THROTTLING_STATE()
            state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION
            state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            state.StateMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetProcessInformation(
                handle, ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state)
            )
        except Exception:
            pass


def run_one_game(job):
    """Play one game for a job dict (solver, width, height, seed, ...) and return one
    results row.
    """
    solver_name = job["solver"]
    width = job["width"]
    height = job["height"]
    max_moves = job.get("max_moves")
    seed = job["seed"]
    time_budget = job.get("time_budget")
    duty_cycle = job.get("duty_cycle", 1.0)
    heartbeat_seconds = job.get("heartbeat_seconds", 5.0)
    on_heartbeat = job.get("on_heartbeat")  # optional callable(str), not sent over network

    policy = SOLVERS[solver_name]()
    game = Game(width=width, height=height, seed=seed)
    t0 = time.time()
    last_heartbeat = t0
    moves = 0
    timed_out = 0
    active_elapsed = 0.0
    while not game.is_over() and (max_moves is None or moves < max_moves):
        now = time.time()
        if time_budget is not None and active_elapsed > time_budget:
            timed_out = 1
            break
        if on_heartbeat and heartbeat_seconds > 0 and (now - last_heartbeat) >= heartbeat_seconds:
            on_heartbeat(f"{solver_name} on {width}x{height} seed={seed} "
                         f"— move {moves}, {active_elapsed:.1f}s active work, score so far {game.score}")
            last_heartbeat = now
        move_t0 = time.time()
        d = policy(game)
        if d is None:
            break
        game.step(d)
        moves += 1
        work_time = time.time() - move_t0
        active_elapsed += work_time
        if duty_cycle < 1.0:
            time.sleep(work_time * (1.0 / duty_cycle - 1.0))
    elapsed = time.time() - t0
    return {
        "solver": solver_name,
        "board": f"{width}x{height}",
        "seed": seed,
        "score": game.score,
        "board_sum": game.board_sum_value(),
        "max_tile": game.max_tile_value(),
        "moves": moves,
        "seconds": round(elapsed, 4),
        "timed_out": timed_out,
    }


def parse_boards(s):
    boards = []
    for part in s.split(","):
        w, h = part.lower().split("x")
        boards.append((int(w), int(h)))
    return boards


def build_all_jobs(solver_names, boards, games, seed_offset, max_moves, time_budget, duty_cycle, heartbeat_seconds):
    """List every (solver, board, seed) game, skipping none; callers filter out
    finished ones.
    """
    all_jobs = []
    for solver_name in solver_names:
        for (w, h) in boards:
            for i in range(games):
                seed = seed_offset + i
                all_jobs.append({
                    "solver": solver_name, "width": w, "height": h,
                    "max_moves": max_moves, "seed": seed,
                    "time_budget": time_budget, "duty_cycle": duty_cycle,
                    "heartbeat_seconds": heartbeat_seconds,
                })
    return all_jobs
