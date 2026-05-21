from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RTMPose features with MMPose for window metadata."
    )
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pose2d", default="human", help="MMPose pose2d alias/config.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-col", default="pose_feature_path")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video-start", type=int, default=0)
    parser.add_argument("--video-count", type=int, default=None)
    return parser.parse_args()


def stable_video_id(video_path: str) -> str:
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:12]
    stem = Path(video_path).stem.replace(" ", "_")
    return f"{stem}_{digest}"


def prediction_score(instance: dict) -> float:
    if "bbox_score" in instance:
        score = instance["bbox_score"]
        if isinstance(score, list):
            return float(score[0]) if score else 0.0
        return float(score)

    scores = instance.get("keypoint_scores", [])
    if len(scores) == 0:
        return 0.0
    return float(sum(scores) / len(scores))


def frame_to_pose(result: dict, num_keypoints: int | None) -> tuple[object, int]:
    import numpy as np

    predictions = result.get("predictions", [])
    if len(predictions) == 0:
        if num_keypoints is None:
            num_keypoints = 17
        return np.zeros((num_keypoints, 3), dtype=np.float32), num_keypoints

    # MMPose returns either [instances] for an image/frame or [[instances], ...]
    # for some batched inputs.
    instances = predictions[0] if isinstance(predictions[0], list) else predictions
    if len(instances) == 0:
        if num_keypoints is None:
            num_keypoints = 17
        return np.zeros((num_keypoints, 3), dtype=np.float32), num_keypoints

    instance = max(instances, key=prediction_score)
    keypoints = np.asarray(instance["keypoints"], dtype=np.float32)
    scores = np.asarray(instance.get("keypoint_scores", []), dtype=np.float32)

    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores.ndim == 2:
        scores = scores[0]

    num_keypoints = int(keypoints.shape[0])
    if scores.size == 0:
        scores = np.ones(num_keypoints, dtype=np.float32)

    return np.concatenate([keypoints, scores[:, None]], axis=1), num_keypoints


def extract_video_pose(inferencer, video_path: str) -> object:
    import cv2
    import numpy as np

    frames = []
    num_keypoints = None

    capture = cv2.VideoCapture(video_path)
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result_iter = inferencer(frame, show=False, return_vis=False)
        result = next(result_iter)
        pose, num_keypoints = frame_to_pose(result, num_keypoints)
        frames.append(pose)
    capture.release()

    if len(frames) == 0:
        num_keypoints = num_keypoints or 17
        return np.zeros((0, num_keypoints, 3), dtype=np.float32)

    return np.stack(frames, axis=0).astype(np.float32)


def build_window_feature(video_pose, row):
    import numpy as np

    indices = [
        int(round(sampled_idx * float(row["sample_interval"])))
        for sampled_idx in range(int(row["window_start"]), int(row["window_end"]))
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

    import numpy as np
    import pandas as pd
    from mmpose.apis import MMPoseInferencer

    output_dir = Path(args.output_dir)
    video_pose_dir = output_dir / "video_pose"
    window_pose_dir = output_dir / "window_pose"
    video_pose_dir.mkdir(parents=True, exist_ok=True)
    window_pose_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)
    inferencer = MMPoseInferencer(pose2d=args.pose2d, device=args.device)

    video_to_pose_path = {}
    unique_videos = windows["video_path"].drop_duplicates().tolist()
    if args.video_start:
        unique_videos = unique_videos[args.video_start :]
    if args.video_count is not None:
        unique_videos = unique_videos[: args.video_count]
    if args.max_videos is not None:
        unique_videos = unique_videos[: args.max_videos]

    for video_idx, video_path in enumerate(unique_videos, start=1):
        video_id = stable_video_id(video_path)
        pose_path = video_pose_dir / f"{video_id}.npy"
        if not pose_path.exists():
            print(
                f"[{video_idx}/{len(unique_videos)}] Extracting RTMPose: {video_path}",
                flush=True,
            )
            video_pose = extract_video_pose(inferencer, video_path)
            np.save(pose_path, video_pose)
        video_to_pose_path[video_path] = pose_path

    rows = []
    for row_idx, row in windows.iterrows():
        video_path = row["video_path"]
        if video_path not in video_to_pose_path:
            continue

        video_pose = np.load(video_to_pose_path[video_path])
        feature = build_window_feature(video_pose, row)
        feature_path = window_pose_dir / f"window_{row_idx:08d}.npy"
        np.save(feature_path, feature)

        output_row = row.to_dict()
        output_row[args.feature_col] = str(feature_path)
        rows.append(output_row)

    output = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved pose window metadata: {output_path}")
    print(f"Saved {len(output)} window feature files under: {window_pose_dir}")


if __name__ == "__main__":
    main()
