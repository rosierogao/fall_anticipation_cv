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
    parser.add_argument(
        "--decode-backend",
        choices=["ffmpeg", "opencv"],
        default="ffmpeg",
        help="Video frame decoder. ffmpeg is safer for fragile AVI containers.",
    )
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


def extract_pose_from_frames(inferencer, frames) -> object:
    import numpy as np

    poses = []
    num_keypoints = None
    for frame in frames:
        result_iter = inferencer(frame, show=False, return_vis=False)
        result = next(result_iter)
        pose, num_keypoints = frame_to_pose(result, num_keypoints)
        poses.append(pose)

    if len(poses) == 0:
        num_keypoints = num_keypoints or 17
        return np.zeros((0, num_keypoints, 3), dtype=np.float32)

    return np.stack(poses, axis=0).astype(np.float32)


def ffmpeg_frames(video_path: str):
    import subprocess
    import tempfile

    import cv2

    with tempfile.TemporaryDirectory(prefix="rtmpose_frames_") as tmpdir:
        frame_pattern = str(Path(tmpdir) / "frame_%08d.jpg")
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-an",
            "-q:v",
            "2",
            frame_pattern,
        ]
        subprocess.run(cmd, check=True)

        frame_paths = sorted(Path(tmpdir).glob("frame_*.jpg"))
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                yield frame


def opencv_frames(video_path: str):
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def extract_video_pose(inferencer, video_path: str, decode_backend: str) -> object:
    if decode_backend == "ffmpeg":
        video_pose = extract_pose_from_frames(inferencer, ffmpeg_frames(video_path))
    else:
        video_pose = extract_pose_from_frames(inferencer, opencv_frames(video_path))

    return video_pose


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
            video_pose = extract_video_pose(
                inferencer,
                video_path,
                args.decode_backend,
            )
            np.save(pose_path, video_pose)
        video_to_pose_path[video_path] = pose_path

    rows = []
    if output_path := Path(args.output_csv):
        if output_path.exists():
            previous = pd.read_csv(output_path)
            if args.feature_col in previous.columns:
                rows.extend(previous.to_dict("records"))

    existing_keys = {
        (row.get("video_path"), int(row.get("window_start", -1)), int(row.get("window_end", -1)))
        for row in rows
    }
    for row_idx, row in windows.iterrows():
        video_path = row["video_path"]
        if video_path not in video_to_pose_path:
            continue
        key = (video_path, int(row["window_start"]), int(row["window_end"]))
        if key in existing_keys:
            continue

        video_pose = np.load(video_to_pose_path[video_path])
        feature = build_window_feature(video_pose, row)
        feature_path = window_pose_dir / f"window_{row_idx:08d}.npy"
        np.save(feature_path, feature)

        output_row = row.to_dict()
        output_row[args.feature_col] = str(feature_path)
        rows.append(output_row)
        existing_keys.add(key)

    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved pose window metadata: {output_path}")
    print(f"Saved {len(output)} window feature files under: {window_pose_dir}")


if __name__ == "__main__":
    main()
