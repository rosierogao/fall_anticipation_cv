"""Merge staged and OOPs feature metadata CSVs.

This intentionally only concatenates already-extracted feature metadata. It does
not run pose or V-JEPA extraction, so staged feature extraction is not repeated.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def merge_csvs(inputs: list[Path], output: Path) -> None:
    if len(inputs) < 2:
        raise ValueError("Provide at least two input CSV files.")

    import csv

    rows = []
    fieldnames = []
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input CSV: {input_path}")
        with input_path.open("r", newline="") as src_file:
            reader = csv.DictReader(src_file)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV: {input_path}")
            for field in reader.fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)
            rows.extend(reader)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as dst_file:
        writer = csv.DictWriter(dst_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_csvs([Path(path) for path in args.inputs], Path(args.output))


if __name__ == "__main__":
    main()
