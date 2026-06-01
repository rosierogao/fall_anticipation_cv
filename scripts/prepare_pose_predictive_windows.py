from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create observed/future pose feature files from RTMPose video poses."
    )
    parser.add_argument("--pose-windows-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--video-pose-dir",
        default=None,
        help="Directory containing per-video RTMPose arrays. Defaults to output_dir/../video_pose.",
    )
    parser.add_argument(
        "--feature-col",
        default="pose_predictive_feature_path",
        help="Output metadata column for observed/future pose npz files.",
    )
    return parser.parse_args()


def stable_video_id(video_path: str) -> str:
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:12]
    stem = Path(video_path).stem.replace(" ", "_")
    return f"{stem}_{digest}"


def resolve_video_pose_path(
    row: pd.Series,
    default_video_pose_dir: Path | None,
) -> Path:
    video_id = stable_video_id(str(row["video_path"]))
    if default_video_pose_dir is not None:
        candidate = default_video_pose_dir / f"{video_id}.npy"
        if candidate.exists():
            return candidate

    pose_feature_path = row.get("pose_feature_path")
    if isinstance(pose_feature_path, str) and pose_feature_path:
        feature_path = Path(pose_feature_path)
        inferred = feature_path.parent.parent / "video_pose" / f"{video_id}.npy"
        if inferred.exists():
            return inferred

    return (
        default_video_pose_dir / f"{video_id}.npy"
        if default_video_pose_dir is not None
        else Path(f"{video_id}.npy")
    )


def slice_pose(video_pose: np.ndarray, row: pd.Series, start: int, end: int) -> np.ndarray:
    indices = [
        int(round(sampled_idx * float(row["sample_interval"])))
        for sampled_idx in range(start, end)
    ]

    num_keypoints = video_pose.shape[1] if video_pose.ndim == 3 else 17
    frames = []
    for original_idx in indices:
        if 0 <= original_idx < len(video_pose):
            frames.append(video_pose[original_idx])
        else:
            frames.append(np.zeros((num_keypoints, 3), dtype=np.float32))

    if len(frames) == 0:
        return np.zeros((0, num_keypoints, 3), dtype=np.float32)
    return np.stack(frames, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_pose_dir = (
        Path(args.video_pose_dir)
        if args.video_pose_dir is not None
        else None
    )

    windows = pd.read_csv(args.pose_windows_csv)
    rows = []
    skipped_missing_video_pose = 0

    for row_idx, row in windows.iterrows():
        video_pose_path = resolve_video_pose_path(row, video_pose_dir)
        if not video_pose_path.exists():
            skipped_missing_video_pose += 1
            continue

        video_pose = np.load(video_pose_path)
        window_start = int(row["window_start"])
        window_end = int(row["window_end"])
        future_end = window_end + int(row["k_frames"])

        observed_pose = slice_pose(video_pose, row, window_start, window_end)
        future_pose = slice_pose(video_pose, row, window_end, future_end)

        feature_path = output_dir / f"pose_predictive_{row_idx:08d}.npz"
        np.savez_compressed(
            feature_path,
            observed_pose=observed_pose.astype(np.float32),
            future_pose=future_pose.astype(np.float32),
        )

        output_row = row.to_dict()
        output_row[args.feature_col] = str(feature_path)
        output_row["pose_future_steps"] = int(future_pose.shape[0])
        rows.append(output_row)

    output = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Saved pose predictive metadata: {output_path}")
    print(f"Saved {len(output)} observed/future pose files under: {output_dir}")
    if skipped_missing_video_pose:
        print(f"Skipped rows missing full-video pose: {skipped_missing_video_pose}")


if __name__ == "__main__":
    main()
