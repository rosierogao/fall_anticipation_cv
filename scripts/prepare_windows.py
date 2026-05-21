import argparse
from pathlib import Path

from fall_anticipation_cv.data import (
    build_window_dataframe,
    load_gmd_labels,
    validate_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GMDCSA24 window metadata.")
    parser.add_argument("--data-root", required=True, help="Path to final_project_dataset.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = load_gmd_labels(args.data_root)
    windows = build_window_dataframe(labels)
    validate_windows(windows)
    windows.to_csv(output, index=False)

    print(f"Labels: {len(labels)}")
    print(f"Matched videos: {labels['video_exists'].sum()}/{len(labels)}")
    print(f"Windows: {windows.shape}")
    print(windows["y"].value_counts())
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
