"""Put the game engine and the chosen objective's solvers on the import path.

Set OBJECTIVE=1 (maximise score, default) or OBJECTIVE=2 (minimise score)
before running any script in this folder.
"""
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
OBJECTIVE = os.environ.get("OBJECTIVE", "1")
if OBJECTIVE not in ("1", "2"):
    sys.exit("OBJECTIVE must be 1 or 2")

SOLVER_DIR = SRC / ("objective1_maximise" if OBJECTIVE == "1" else "objective2_minimise")
sys.path[:0] = [str(SRC / "common"), str(SOLVER_DIR)]
