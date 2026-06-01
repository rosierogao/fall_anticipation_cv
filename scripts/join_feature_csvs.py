"""Attach extracted feature paths to an authoritative window metadata CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["video_path", "window_start", "window_end", "y"]


def join_features(
    windows_csv: Path,
    feature_csvs: list[Path],
    output_csv: Path,
    required_feature_col: str,
) -> None:
    windows = pd.read_csv(windows_csv)
    feature_frames = [pd.read_csv(path) for path in feature_csvs]
    features = pd.concat(feature_frames, ignore_index=True, sort=False)

    missing_keys = [column for column in KEY_COLUMNS if column not in windows.columns]
    if missing_keys:
        raise ValueError(f"Window CSV is missing key columns: {missing_keys}")
    missing_keys = [column for column in KEY_COLUMNS if column not in features.columns]
    if missing_keys:
        raise ValueError(f"Feature CSVs are missing key columns: {missing_keys}")
    if required_feature_col not in features.columns:
        raise ValueError(f"Feature CSVs are missing {required_feature_col!r}.")

    feature_columns = [
        column
        for column in features.columns
        if column in KEY_COLUMNS or column not in windows.columns
    ]
    features = features[feature_columns].drop_duplicates(KEY_COLUMNS, keep="first")
    joined = windows.merge(features, on=KEY_COLUMNS, how="left", validate="one_to_one")

    missing_features = joined[required_feature_col].isna()
    if missing_features.any():
        print(
            f"Dropping {int(missing_features.sum())} rows without "
            f"{required_feature_col}.",
            flush=True,
        )
        joined = joined.loc[~missing_features].reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_csv, index=False)
    print(f"Wrote {len(joined)} rows to {output_csv}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument("--feature-csv", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--required-feature-col", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    join_features(
        windows_csv=Path(args.windows_csv),
        feature_csvs=[Path(path) for path in args.feature_csv],
        output_csv=Path(args.output_csv),
        required_feature_col=args.required_feature_col,
    )


if __name__ == "__main__":
    main()
