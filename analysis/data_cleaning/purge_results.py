#!/usr/bin/env python3
"""Objective 2 clean-up: drop solver runs that are not part of the study
and sort the file by solver, board and seed. Writes <file>.bak first.
"""

import sys
import csv
import shutil

MCTS_ALGOS_TO_PURGE = {"mcts_3000_tpu", "mcts_6000_tpu", "mcts_10000_tpu"}
EXPECTIMAX_ALGO_TO_PURGE = "expectimax_d8"
EXPECTIMAX_BOARD_TO_PURGE = "5x5"


def should_purge(row):
    if not row:
        return False
    algo = row[0].strip()
    board = row[1].strip() if len(row) > 1 else ""

    if algo in MCTS_ALGOS_TO_PURGE:
        return True
    if algo == EXPECTIMAX_ALGO_TO_PURGE and board == EXPECTIMAX_BOARD_TO_PURGE:
        return True
    return False


def sort_key(row):
    solver = row[0].strip()
    board = row[1].strip() if len(row) > 1 else ""
    try:
        seed = int(row[2].strip())
    except (IndexError, ValueError):
        seed = 0
    return (solver, board, seed)


def main():
    if len(sys.argv) < 2:
        print("Usage: python purge_results.py input_file.tsv")
        sys.exit(1)

    input_path = sys.argv[1]
    backup_path = input_path + ".bak"

    with open(input_path, "r", newline="") as infile:
        sample = infile.read(4096)
        infile.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
            delimiter = dialect.delimiter
        except csv.Error:
            # Fall back: tab if any tabs present, else comma
            delimiter = "\t" if "\t" in sample else ","
        reader = csv.reader(infile, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        print("File is empty, nothing to do.")
        sys.exit(0)

    header = None
    data_rows = rows
    first_row = rows[0]
    is_header = False
    if first_row and first_row[0].strip().lower() == "solver":
        is_header = True
    elif len(first_row) > 2:
        try:
            int(first_row[2].strip())
        except ValueError:
            is_header = True

    if is_header:
        header = first_row
        data_rows = rows[1:]

    kept = []
    purged = 0
    for row in data_rows:
        if should_purge(row):
            purged += 1
            continue
        kept.append(row)

    kept.sort(key=sort_key)

    # Back up the original before overwriting.
    shutil.copyfile(input_path, backup_path)

    with open(input_path, "w", newline="") as outfile:
        writer = csv.writer(outfile, delimiter=delimiter)
        if header is not None:
            writer.writerow(header)
        writer.writerows(kept)

    print(f"Done. Kept {len(kept)} rows, purged {purged} rows.")
    print(f"File updated in place: {input_path}")
    print(f"Backup of original saved to: {backup_path}")


if __name__ == "__main__":
    main()
