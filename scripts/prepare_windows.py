import argparse
from pathlib import Path

from fall_anticipation_cv.data import (
    assign_group_splits,
    build_window_dataframe,
    load_all_labels,
    sample_oops_negative_windows,
    validate_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fall-anticipation windows.")
    parser.add_argument("--data-root", required=True, help="Path to final_project_dataset.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--include-oops",
        action="store_true",
        help="Include the OOPs real fall dataset.",
    )
    parser.add_argument(
        "--oops-negative-ratio",
        type=float,
        default=3.0,
        help="Keep at most this many OOPs negative windows per OOPs positive window.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic OOPs negative sampling.",
    )
    parser.add_argument(
        "--assign-splits",
        action="store_true",
        help="Save a train/val/test split column assigned at subject/video-group level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = load_all_labels(args.data_root, include_oops=args.include_oops)
    windows = build_window_dataframe(labels)
    if args.include_oops:
        windows = sample_oops_negative_windows(
            windows,
            negative_to_positive_ratio=args.oops_negative_ratio,
            random_state=args.sample_seed,
        )
    if args.assign_splits:
        windows = assign_group_splits(windows, random_state=args.sample_seed)
    validate_windows(windows)
    windows.to_csv(output, index=False)

    print(f"Labels: {len(labels)}")
    print(f"Matched videos: {labels['video_exists'].sum()}/{len(labels)}")
    print(labels["dataset"].value_counts())
    print(f"Windows: {windows.shape}")
    print(windows["y"].value_counts())
    print(windows.groupby(["dataset", "y"]).size().unstack(fill_value=0))
    if "split" in windows.columns:
        print(windows.groupby(["dataset", "split", "y"]).size().unstack(fill_value=0))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
