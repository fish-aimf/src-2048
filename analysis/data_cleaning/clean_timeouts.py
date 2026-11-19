"""Remove timed-out games from a results CSV so they are re-run."""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="results.csv")
    ap.add_argument("--solvers", type=str, default=None,
                     help="comma-separated solver names to clean; omit to clean all solvers")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"{args.file} not found.")
        return

    target_solvers = set(args.solvers.split(",")) if args.solvers else None

    backup_path = args.file + ".bak"
    shutil.copy(args.file, backup_path)
    print(f"Backed up original to {backup_path}")

    with open(args.file, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept = []
    removed = 0
    for row in rows:
        is_timeout = row.get("timed_out") in ("1", 1, "True", "true")
        solver_matches = target_solvers is None or row["solver"] in target_solvers
        if is_timeout and solver_matches:
            removed += 1
        else:
            kept.append(row)

    with open(args.file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Removed {removed} timed_out rows"
          f"{' for solvers: ' + args.solvers if args.solvers else ' (all solvers)'}.")
    print(f"{len(kept)} rows remaining in {args.file}.")
    print("Rerun benchmark.py with the same command to regenerate the removed games.")


if __name__ == "__main__":
    main()
