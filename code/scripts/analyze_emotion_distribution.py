#!/usr/bin/env python3
"""
Analyze emotion label distributions in an Excel file.

Given an .xlsx with columns like 'goemotion' and 'basic', this script computes
per-label percentages for each column:

- Row coverage %: fraction of rows where the label appears at least once
- Label share %: share of label occurrences among all label instances

Usage examples (from repo root):
  python code/scripts/analyze_emotion_distribution.py \
    --file resources/datasets/ch-simsv2s/前2500/meta刘彦秀(1).xlsx \
    --columns goemotion basic

Requirements:
  pip install pandas openpyxl matplotlib
"""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emotion percentage analysis for Excel columns")
    default_file = Path("resources/datasets/ch-simsv2s/前2500/meta刘彦秀(1).xlsx")
    p.add_argument("--file", type=Path, default=default_file, help="Path to Excel .xlsx file")
    p.add_argument(
        "--columns",
        nargs="+",
        default=["goemotion", "basic"],
        help="Column names to analyze (case-insensitive)",
    )
    p.add_argument(
        "--separators",
        default=r"[,\|;/、，]+",
        help="Regex of separators for multi-label cells (default: commas/pipes/semicolons/Chinese separators)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save outputs (default: same dir as --file)",
    )
    return p.parse_args()


def load_excel(path: Path):
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise SystemExit(
            "pandas is required. Please install with `pip install pandas openpyxl` and retry."
        )
    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise SystemExit(f"Failed to read Excel file: {path}\nDetails: {e}")
    return df


def find_columns(df, names: Iterable[str]) -> Dict[str, str]:
    # map requested name -> actual dataframe column name (case-insensitive)
    lower_to_actual = {c.lower(): c for c in df.columns}
    resolved = {}
    for n in names:
        key = n.lower()
        if key in lower_to_actual:
            resolved[n] = lower_to_actual[key]
        else:
            # try exact match first, then contains
            matches = [c for c in df.columns if c.lower() == key or key in c.lower()]
            if matches:
                resolved[n] = matches[0]
            else:
                raise SystemExit(
                    f"Column '{n}' not found. Available: {', '.join(map(str, df.columns))}"
                )
    return resolved


def split_labels(cell: str, sep_regex: str) -> List[str]:
    cell = str(cell).strip()
    if not cell or cell.lower() == "nan":
        return []
    # remove brackets like [xxx] if present
    cell = cell.strip("[](){}")
    # split by regex, filter empty
    parts = re.split(sep_regex, cell)
    return [p.strip() for p in parts if p.strip()]


def analyze_column(values: List[str], sep_regex: str) -> Tuple[Counter, Counter, int, int]:
    # instance_counts: counts of each label occurrence
    instance_counts: Counter = Counter()
    # row_counts: number of rows in which each label appears at least once
    row_counts: Counter = Counter()
    nonempty_rows = 0
    total_instances = 0
    for cell in values:
        labels = split_labels(cell, sep_regex)
        if not labels:
            continue
        nonempty_rows += 1
        # instance-level
        instance_counts.update(labels)
        total_instances += len(labels)
        # row-level (unique per row)
        for lab in set(labels):
            row_counts[lab] += 1
    return instance_counts, row_counts, nonempty_rows, total_instances


def print_report(name: str, instance_counts: Counter, row_counts: Counter, nonempty_rows: int, total_instances: int):
    if nonempty_rows == 0:
        print(f"[WARN] Column '{name}': no non-empty rows found.")
        return
    labels = sorted(instance_counts.keys(), key=lambda k: (instance_counts[k], k), reverse=True)
    print(f"\n===== Column: {name} =====")
    print(f"Non-empty rows: {nonempty_rows}")
    print(f"Total label instances: {total_instances}")
    print(f"{'Label':<24} {'RowCov%':>10} {'Share%':>10} {'Rows':>8} {'Inst':>8}")
    for lab in labels:
        rc = row_counts.get(lab, 0)
        ic = instance_counts.get(lab, 0)
        row_pct = (rc / nonempty_rows * 100.0) if nonempty_rows else 0.0
        share_pct = (ic / total_instances * 100.0) if total_instances else 0.0
        print(f"{lab:<24} {row_pct:10.2f} {share_pct:10.2f} {rc:8d} {ic:8d}")


def to_dataframe(instance_counts: Counter, row_counts: Counter, nonempty_rows: int, total_instances: int):
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    labels = sorted(instance_counts.keys(), key=lambda k: (instance_counts[k], k), reverse=True)
    rows = []
    for lab in labels:
        rc = int(row_counts.get(lab, 0))
        ic = int(instance_counts.get(lab, 0))
        row_pct = (rc / nonempty_rows * 100.0) if nonempty_rows else 0.0
        share_pct = (ic / total_instances * 100.0) if total_instances else 0.0
        rows.append({
            "label": lab,
            "row_count": rc,
            "inst_count": ic,
            "row_pct": row_pct,
            "share_pct": share_pct,
        })
    return pd.DataFrame(rows)


def plot_bars(df, title_prefix: str, out_png: Path):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print("[WARN] matplotlib not available; skip plotting. Install with `pip install matplotlib`.")
        return
    if df is None or df.empty:
        print("[WARN] Empty dataframe; skip plotting.")
        return
    # Sort by share_pct desc for consistent ordering
    df_plot = df.sort_values("share_pct", ascending=True)
    labels = df_plot["label"].tolist()
    share = df_plot["share_pct"].tolist()
    rowcov = df_plot["row_pct"].tolist()

    h = max(3.0, 0.35 * len(labels) + 2.0)
    fig, axes = plt.subplots(ncols=2, figsize=(12, h), sharey=True)
    fig.suptitle(f"{title_prefix} — Emotion Percentages")

    axes[0].barh(labels, rowcov, color="#4C78A8")
    axes[0].set_title("Row Coverage (%)")
    axes[0].set_xlabel("% of non-empty rows")

    axes[1].barh(labels, share, color="#F58518")
    axes[1].set_title("Label Share (%)")
    axes[1].set_xlabel("% of label instances")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    df = load_excel(args.file)
    resolved = find_columns(df, args.columns)
    out_dir = args.out_dir or args.file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for req_name, col_name in resolved.items():
        series = df[col_name].astype(str).fillna("").tolist()
        inst, rows, nrows, ninst = analyze_column(series, args.separators)
        print_report(req_name, inst, rows, nrows, ninst)
        # Save csv and figure
        dfout = to_dataframe(inst, rows, nrows, ninst)
        stem = args.file.stem
        if dfout is not None:
            csv_path = out_dir / f"{stem}_{req_name}_distribution.csv"
            try:
                dfout.to_csv(csv_path, index=False)
                print(f"[OK] Saved CSV: {csv_path}")
            except Exception as e:
                print(f"[WARN] Failed to save CSV: {e}")
        png_path = out_dir / f"{stem}_{req_name}_bars.png"
        plot_bars(dfout, f"{req_name}", png_path)
        if png_path.exists():
            print(f"[OK] Saved plot: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
