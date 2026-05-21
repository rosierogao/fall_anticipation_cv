import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis.")
    parser.add_argument(
        "--data-root",
        default="/data",
        help="Dataset root or Modal mount root. Defaults to /data.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eda",
        help="Directory for EDA tables and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt

    from fall_anticipation_cv.data import load_gmd_labels

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_gmd_labels(args.data_root)

    label_counts = (
        df.groupby(["label", "label_name"])
        .size()
        .reset_index(name="num_videos")
        .sort_values("num_videos", ascending=False)
    )

    df["start_is_zero"] = df["start"] <= 1e-6
    start_stats = (
        df.groupby("label_name")
        .agg(total_clips=("label_name", "count"), start_zero=("start_is_zero", "sum"))
        .reset_index()
    )
    start_stats["pct_start_zero"] = (
        start_stats["start_zero"] / start_stats["total_clips"]
    )
    start_stats = start_stats.sort_values("pct_start_zero", ascending=False)

    fall_df = df[df["label_name"] == "fall"]

    label_counts_path = output_dir / "label_counts.csv"
    start_stats_path = output_dir / "start_zero_stats.csv"
    fall_hist_path = output_dir / "fall_start_time_distribution.png"

    label_counts.to_csv(label_counts_path, index=False)
    start_stats.to_csv(start_stats_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(fall_df["start"], bins=20)
    plt.xlabel("Fall start time (sec)")
    plt.ylabel("Count")
    plt.title("Distribution of Fall Start Times")
    plt.tight_layout()
    plt.savefig(fall_hist_path, dpi=200)
    plt.close()

    print("Label counts:")
    print(label_counts)
    print("\nStart-time zero statistics:")
    print(start_stats)
    print(f"\nSaved label counts: {label_counts_path}")
    print(f"Saved start-time stats: {start_stats_path}")
    print(f"Saved fall start-time histogram: {fall_hist_path}")


if __name__ == "__main__":
    main()
