"""
Show train/val/test class incidence per dataset using split_by_subject.

Usage:
    python scripts/analyze_split.py --windows-csv path/to/windows.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_subject(
    windows: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    val_random_state: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mirror of fall_anticipation_cv.data.split_by_subject."""
    if "split" in windows.columns:
        train = windows[windows["split"] == "train"].reset_index(drop=True)
        val   = windows[windows["split"] == "val"].reset_index(drop=True)
        test  = windows[windows["split"] == "test"].reset_index(drop=True)
        if not train.empty and not val.empty and not test.empty:
            return train, val, test

    group_column = "split_group" if "split_group" in windows.columns else "subject"
    groups = windows[group_column]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss.split(windows, windows["y"], groups))

    train_val = windows.iloc[train_val_idx].reset_index(drop=True)
    test      = windows.iloc[test_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=val_random_state)
    train_idx, val_idx = next(
        gss_val.split(train_val, train_val["y"], train_val[group_column])
    )

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val   = train_val.iloc[val_idx].reset_index(drop=True)

    return train, val, test


def class_row(df: pd.DataFrame) -> dict:
    n     = len(df)
    n_pos = int((df["y"] == 1).sum())
    n_neg = int((df["y"] == 0).sum())
    return {
        "total":   n,
        "pos":     n_pos,
        "neg":     n_neg,
        "pos_pct": round(n_pos / n * 100, 1) if n > 0 else 0.0,
    }


def print_report(windows_csv: str) -> None:
    windows = pd.read_csv(windows_csv)

    print(f"\nWindows CSV : {windows_csv}")
    print(f"Total rows  : {len(windows):,}")

    train_df, val_df, test_df = split_by_subject(windows)
    splits = {"train": train_df, "val": val_df, "test": test_df}

    has_dataset = "dataset" in windows.columns
    datasets = sorted(windows["dataset"].dropna().unique()) if has_dataset else ["(all)"]

    sep = "=" * 72

    # ── per-split breakdown ──────────────────────────────────────────────────
    for split_name, df in splits.items():
        r = class_row(df)
        print(f"\n{sep}")
        print(
            f"  {split_name.upper():5s}  "
            f"total={r['total']:6,}  pos={r['pos']:4,} ({r['pos_pct']:5.1f}%)  neg={r['neg']:6,}"
        )
        print(sep)
        if has_dataset:
            for ds in datasets:
                sub = df[df["dataset"] == ds]
                if sub.empty:
                    print(f"    {ds:14s}  (no windows)")
                    continue
                sr = class_row(sub)
                print(
                    f"    {ds:14s}  "
                    f"total={sr['total']:6,}  "
                    f"pos={sr['pos']:4,} ({sr['pos_pct']:5.1f}%)  "
                    f"neg={sr['neg']:6,}"
                )

    # ── summary table: positive counts ───────────────────────────────────────
    print(f"\n{sep}")
    print("  POSITIVE WINDOW COUNTS  (pos / total)")
    print(sep)
    col_w = 20
    header = f"  {'dataset':14s}"
    for s in splits:
        header += f"  {s:>{col_w}}"
    header += f"  {'total':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ds in datasets:
        row = f"  {ds:14s}"
        row_total = 0
        for df in splits.values():
            sub = df[df["dataset"] == ds] if has_dataset else df
            n_pos   = int((sub["y"] == 1).sum())
            n_total = len(sub)
            cell    = f"{n_pos}/{n_total}"
            row    += f"  {cell:>{col_w}}"
            row_total += n_pos
        row += f"  {row_total:>10,}"
        print(row)

    print("  " + "-" * (len(header) - 2))
    totals_row = f"  {'TOTAL':14s}"
    grand_pos = 0
    for df in splits.values():
        n_pos   = int((df["y"] == 1).sum())
        n_total = len(df)
        cell    = f"{n_pos}/{n_total}"
        totals_row += f"  {cell:>{col_w}}"
        grand_pos  += n_pos
    totals_row += f"  {grand_pos:>10,}"
    print(totals_row)
    print(sep + "\n")

    # ── positive rate table ───────────────────────────────────────────────────
    print("  POSITIVE RATE (%) per split x dataset")
    print(sep)
    print(header.replace("pos / total", "pos %       "))
    print("  " + "-" * (len(header) - 2))
    for ds in datasets:
        row = f"  {ds:14s}"
        for df in splits.values():
            sub    = df[df["dataset"] == ds] if has_dataset else df
            n_pos  = int((sub["y"] == 1).sum())
            n      = len(sub)
            pct    = f"{n_pos/n*100:.1f}%" if n > 0 else "—"
            row   += f"  {pct:>{col_w}}"
        print(row)
    print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze train/val/test class balance.")
    parser.add_argument(
        "--windows-csv",
        required=True,
        help="Path to the window metadata CSV (e.g. windows_gmdcsa24.csv).",
    )
    args = parser.parse_args()

    if not Path(args.windows_csv).exists():
        print(f"ERROR: file not found: {args.windows_csv}", file=sys.stderr)
        sys.exit(1)

    print_report(args.windows_csv)


if __name__ == "__main__":
    main()
