#!/usr/bin/env python3
"""Print how many games are done for each solver and board in a results CSV."""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict


def read_counts(csv_path):
    """Return {(solver, board): count} from the csv file."""
    counts = defaultdict(int)
    if not os.path.exists(csv_path):
        return counts

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return counts
        # normalize field names (strip whitespace) just in case
        fieldnames = [fn.strip() for fn in reader.fieldnames]
        for row in reader:
            row = {k.strip(): (v.strip() if v else v) for k, v in row.items()}
            solver = row.get("solver")
            board = row.get("board")
            if solver is None or board is None:
                continue
            counts[(solver, board)] += 1
    return counts


def print_progress(counts, total, solvers=None, boards=None):
    os.system("cls" if os.name == "nt" else "clear")

    keys = set(counts.keys())
    if solvers:
        keys = {k for k in keys if k[0] in solvers}
    if boards:
        keys = {k for k in keys if k[1] in boards}
    if solvers and boards:
        # also include combos with zero games so they show up as 0%
        for s in solvers:
            for b in boards:
                keys.add((s, b))

    rows = sorted(keys, key=lambda k: (k[1], k[0]))

    grand_done = 0
    grand_total = 0

    name_w = max([len(k[0]) for k in rows], default=6) + 2
    board_w = max([len(k[1]) for k in rows], default=5) + 2

    header = f"{'Solver'.ljust(name_w)}{'Board'.ljust(board_w)}Progress"
    print(header)
    print("-" * len(header))

    for solver, board in rows:
        done = counts.get((solver, board), 0)
        pct = (done / total * 100) if total else 0
        bar_len = 20
        filled = int(bar_len * min(done, total) / total) if total else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"{solver.ljust(name_w)}{board.ljust(board_w)}[{bar}] {done}/{total} ({pct:5.1f}%)")
        grand_done += done
        grand_total += total

    print("-" * len(header))
    overall_pct = (grand_done / grand_total * 100) if grand_total else 0
    print(f"TOTAL: {grand_done}/{grand_total} ({overall_pct:.1f}%)")
    print(f"\nLast updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description="Check progress of results.csv against expected total games.")
    parser.add_argument("--csv", default="results.csv", help="Path to results.csv")
    parser.add_argument("--total", type=int, default=5000, help="Expected number of games per (solver, board) combo")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit (no loop)")
    parser.add_argument(
        "--solvers",
        default="",
        help="Comma-separated list of solvers to include (default: all found in csv)",
    )
    parser.add_argument(
        "--boards",
        default="",
        help="Comma-separated list of board sizes to include (default: all found in csv)",
    )
    args = parser.parse_args()

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()] or None
    boards = [b.strip() for b in args.boards.split(",") if b.strip()] or None

    try:
        while True:
            counts = read_counts(args.csv)
            print_progress(counts, args.total, solvers=solvers, boards=boards)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
