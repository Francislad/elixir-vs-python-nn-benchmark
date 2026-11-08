#!/usr/bin/env python3
"""Repairs CSVs written under a comma-decimal locale. Usage: fix_csv_locale.py [--in-place] file.csv ..."""

import argparse
import csv
import sys

SUMMARY_HEADER = ["script", "run", "elapsed_seconds", "exit_code"]
STATS_HEADER = [
    "timestamp", "cpu_percent", "ram_used_mb", "ram_total_mb",
    "gpu_util_percent", "vram_used_mb", "vram_total_mb",
]


def fix_rows(rows, header, split_index):
    # Only rows with exactly one extra field are rejoined, so unrelated integer fields are not merged
    fixed = []
    n_fixed = 0
    n_unrecognized = 0
    expected_len = len(header)
    for row in rows:
        if len(row) == expected_len:
            fixed.append(row)
        elif len(row) == expected_len + 1:
            new_row = row[:split_index] + [row[split_index] + "." + row[split_index + 1]] + row[split_index + 2:]
            fixed.append(new_row)
            n_fixed += 1
        else:
            fixed.append(row)
            n_unrecognized += 1
    return fixed, n_fixed, n_unrecognized


def process_file(path, in_place):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        print(f"{path}: empty, skipping")
        return

    header, data = rows[0], rows[1:]

    if header == SUMMARY_HEADER:
        fixed_data, n_fixed, n_bad = fix_rows(data, header, split_index=2)
    elif header == STATS_HEADER:
        fixed_data, n_fixed, n_bad = fix_rows(data, header, split_index=1)
    else:
        print(f"{path}: header doesn't match a known shape, skipping ({header})", file=sys.stderr)
        return

    if n_bad:
        print(f"{path}: {n_bad} row(s) had an unexpected field count and were left untouched -- check manually", file=sys.stderr)

    out_path = path if in_place else path.rsplit(".csv", 1)[0] + ".fixed.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(fixed_data)

    print(f"{path}: fixed {n_fixed} row(s) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Repair comma-decimal-split CSV columns.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the original file instead of writing a .fixed.csv copy")
    parser.add_argument("files", nargs="+", help="CSV file(s) to repair")
    args = parser.parse_args()

    for path in args.files:
        process_file(path, args.in_place)


if __name__ == "__main__":
    main()
