#!/usr/bin/env python3
"""Compiles result CSVs into one xlsx. Usage: compile_results.py -o report.xlsx [LABEL=]DIR ..."""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

INVALID_SHEET_CHARS = r'[\\/*?:\[\]]'

COMPARISON_METRICS = ["cpu_percent", "ram_used_mb", "gpu_util_percent", "vram_used_mb"]


class RowWriter:
    # ws.max_row does not advance after a fully empty row, so spacer rows would shift row numbers

    def __init__(self, ws):
        self.ws = ws
        self.row = 0

    def append(self, values):
        self.row += 1
        self.ws.append(values)
        return self.row


def sanitize_sheet_name(name, used_names):
    name = re.sub(INVALID_SHEET_CHARS, "_", name).strip("'")
    if not name:
        name = "Sheet"
    name = name[:31]
    base = name
    i = 1
    while name in used_names:
        suffix = f"_{i}"
        name = base[: 31 - len(suffix)] + suffix
        i += 1
    used_names.add(name)
    return name


def add_stats_sheet(wb, used_names, base_label, csv_path, rel_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  (skipping empty file: {csv_path})")
        return None

    script_dir = os.path.basename(os.path.dirname(csv_path))
    run_match = re.search(r"run_(\d+)_stats", os.path.basename(csv_path))
    run_num = run_match.group(1) if run_match else "?"

    sheet_label = f"{base_label}_{script_dir}_r{run_num}"
    sheet_name = sanitize_sheet_name(sheet_label, used_names)
    ws = wb.create_sheet(sheet_name)
    rw = RowWriter(ws)

    rw.append([f"Source: {rel_path} (base: {base_label})"])
    rw.append([])

    columns = list(df.columns)
    header_row_idx = rw.append(columns)
    for cell in ws[header_row_idx]:
        cell.font = Font(bold=True)

    data_start = None
    data_end = None
    for row in df.itertuples(index=False):
        r = rw.append(list(row))
        if data_start is None:
            data_start = r
        data_end = r

    numeric_cols = set(df.select_dtypes(include="number").columns)

    rw.append([])
    avg_row = rw.append(["Average"])
    max_row = rw.append(["Max"])

    for idx, col in enumerate(columns, start=1):
        if col in numeric_cols:
            col_letter = get_column_letter(idx)
            ws.cell(row=avg_row, column=idx).value = (
                f"=AVERAGE({col_letter}{data_start}:{col_letter}{data_end})"
            )
            ws.cell(row=max_row, column=idx).value = (
                f"=MAX({col_letter}{data_start}:{col_letter}{data_end})"
            )

    for row_idx in (avg_row, max_row):
        for cell in ws[row_idx]:
            cell.font = Font(bold=True)

    for idx, col in enumerate(columns, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = max(14, len(str(col)) + 2)

    print(f"  + sheet '{sheet_name}' <- {rel_path}")

    return {
        "sheet_name": sheet_name,
        "base_label": base_label,
        "script_dir": script_dir,
        "run_num": run_num,
        "columns": columns,
        "data_start": data_start,
        "data_end": data_end,
        "avg_row": avg_row,
        "max_row": max_row,
    }


def col_letter_for(columns, colname):
    return get_column_letter(columns.index(colname) + 1)


def build_comparison_sheet(wb, used_names, script_dir, infos):
    labels = sorted({info["base_label"] for info in infos})
    if len(labels) < 2:
        return

    by_label = defaultdict(list)
    for info in infos:
        by_label[info["base_label"]].append(info)

    common_columns = infos[0]["columns"]
    metrics = [m for m in COMPARISON_METRICS if m in common_columns]

    sheet_name = sanitize_sheet_name(f"Cmp_{script_dir}", used_names)
    ws = wb.create_sheet(sheet_name)
    rw = RowWriter(ws)

    rw.append([f"Python vs Elixir comparison: {script_dir}"])
    ws.cell(row=1, column=1).font = Font(bold=True, size=14)
    rw.append(["(All values below are live formulas referencing the individual run sheets.)"])
    ws.column_dimensions["A"].width = 12
    for col in "BCDEFG":
        ws.column_dimensions[col].width = 16

    max_len = max(info["data_end"] - info["data_start"] + 1 for info in infos)

    for metric in metrics:
        rw.append([])
        title_row = rw.append([f"{metric} - average across runs, by platform"])
        ws.cell(row=title_row, column=1).font = Font(bold=True)

        headers = ["Sample"] + [f"{label}_avg" for label in labels]
        header_row = rw.append(headers)
        for cell in ws[header_row]:
            cell.font = Font(bold=True)

        data_start_row = None
        data_end_row = None
        for k in range(1, max_len + 1):
            row_values = [k]
            for label in labels:
                refs = []
                for info in by_label[label]:
                    row_in_sheet = info["data_start"] + k - 1
                    if row_in_sheet <= info["data_end"]:
                        col_letter = col_letter_for(info["columns"], metric)
                        refs.append(f"'{info['sheet_name']}'!{col_letter}{row_in_sheet}")
                if refs:
                    row_values.append(f'=IFERROR(AVERAGE({",".join(refs)}),"")')
                else:
                    row_values.append("")
            r = rw.append(row_values)
            if data_start_row is None:
                data_start_row = r
            data_end_row = r

        chart = LineChart()
        chart.title = f"{metric} - {script_dir}"
        chart.style = 2
        chart.y_axis.title = metric
        chart.x_axis.title = "Sample #"
        chart.width = 20
        chart.height = 9

        data_ref = Reference(ws, min_col=2, max_col=1 + len(labels), min_row=header_row, max_row=data_end_row)
        cats_ref = Reference(ws, min_col=1, min_row=data_start_row, max_row=data_end_row)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)

        anchor_col = get_column_letter(1 + len(labels) + 2)
        ws.add_chart(chart, f"{anchor_col}{header_row}")

    rw.append([])
    rw.append([])
    section_row = rw.append(["Speed & peak resource usage comparison"])
    ws.cell(row=section_row, column=1).font = Font(bold=True)

    metric_display = {
        "cpu_percent": "Peak CPU %",
        "ram_used_mb": "Peak RAM (MiB)",
        "gpu_util_percent": "Peak GPU %",
        "vram_used_mb": "Peak VRAM (MiB)",
    }
    headers2 = ["Platform", "Avg elapsed (s)"] + [metric_display[m] for m in metrics]
    header_row2 = rw.append(headers2)
    for cell in ws[header_row2]:
        cell.font = Font(bold=True)

    table_start_row = None
    table_end_row = None
    for label in labels:
        row = [label, f'=AVERAGEIFS(Summary!D:D,Summary!A:A,"{label}",Summary!B:B,"{script_dir}")']
        for metric in metrics:
            refs = [
                f"'{info['sheet_name']}'!{col_letter_for(info['columns'], metric)}{info['max_row']}"
                for info in by_label[label]
            ]
            row.append(f"=MAX({','.join(refs)})" if refs else "")
        r = rw.append(row)
        if table_start_row is None:
            table_start_row = r
        table_end_row = r

    bar = BarChart()
    bar.type = "col"
    bar.title = f"Avg elapsed training time (s) - {script_dir}"
    bar.y_axis.title = "seconds"
    bar.width = 14
    bar.height = 8
    data_ref = Reference(ws, min_col=2, max_col=2, min_row=header_row2, max_row=table_end_row)
    cats_ref = Reference(ws, min_col=1, min_row=table_start_row, max_row=table_end_row)
    bar.add_data(data_ref, titles_from_data=True)
    bar.set_categories(cats_ref)
    ws.add_chart(bar, f"A{table_end_row + 2}")

    print(f"  + sheet '{sheet_name}' (comparison: {', '.join(labels)})")


def main():
    parser = argparse.ArgumentParser(description="Compile benchmark CSVs into a multi-tab xlsx.")
    parser.add_argument("-o", "--output", default="benchmark_report.xlsx", help="Output .xlsx path")
    parser.add_argument("dirs", nargs="+", help="One or more base result directories to scan (recursively). Use LABEL=DIR to name the platform; otherwise the directory name is used")
    args = parser.parse_args()

    wb = Workbook()
    wb.remove(wb.active)
    used_names = set()
    summary_frames = []
    stats_infos = []

    for base in args.dirs:
        label = None
        if "=" in base:
            label, base = base.split("=", 1)
        if not os.path.isdir(base):
            print(f"Warning: '{base}' is not a directory, skipping.", file=sys.stderr)
            continue

        base_label = label or os.path.basename(os.path.normpath(base)) or base
        print(f"Scanning '{base}' (label: {base_label})...")

        stats_files = sorted(glob.glob(os.path.join(base, "**", "run_*_stats.csv"), recursive=True))
        for f in stats_files:
            rel = os.path.relpath(f, base)
            info = add_stats_sheet(wb, used_names, base_label, f, rel)
            if info:
                stats_infos.append(info)

        summary_files = sorted(glob.glob(os.path.join(base, "**", "summary.csv"), recursive=True))
        for sf in summary_files:
            df = pd.read_csv(sf)
            if df.empty:
                continue
            df.insert(0, "source_dir", base_label)
            summary_frames.append(df)

    if summary_frames:
        combined = pd.concat(summary_frames, ignore_index=True)
        ws = wb.create_sheet("Summary")
        rw = RowWriter(ws)
        header_row = rw.append(list(combined.columns))
        for cell in ws[header_row]:
            cell.font = Font(bold=True)
        for row in combined.itertuples(index=False):
            rw.append(list(row))
        for idx, col in enumerate(combined.columns, start=1):
            ws.column_dimensions[get_column_letter(idx)].width = max(14, len(str(col)) + 2)
        print(f"  + sheet 'Summary' <- {len(summary_frames)} summary.csv file(s)")

    by_script = defaultdict(list)
    for info in stats_infos:
        by_script[info["script_dir"]].append(info)

    for script_dir, infos in sorted(by_script.items()):
        build_comparison_sheet(wb, used_names, script_dir, infos)

    if not wb.sheetnames:
        print("No 'run_*_stats.csv' or 'summary.csv' files found under the given directories.", file=sys.stderr)
        sys.exit(1)

    def sort_key(ws):
        if ws.title == "Summary":
            return (0, ws.title)
        if ws.title.startswith("Cmp_"):
            return (1, ws.title)
        return (2, ws.title)

    wb._sheets = sorted(wb._sheets, key=sort_key)

    wb.save(args.output)
    print(f"\nSaved '{args.output}' with {len(wb.sheetnames)} sheet(s).")


if __name__ == "__main__":
    main()
