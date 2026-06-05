from pathlib import Path

import modal

APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-pose-classification")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "opencv-python-headless==4.10.0.84",
        "pandas<3.0",
        "scikit-learn",
        "torch==2.3.1",
        "tqdm",
    )
    .add_local_dir(
        LOCAL_PACKAGE_DIR,
        f"{PACKAGE_REMOTE_ROOT}/fall_anticipation_cv",
        copy=True,
    )
    .add_local_dir(LOCAL_SCRIPTS_DIR, f"{PACKAGE_REMOTE_ROOT}/scripts", copy=True)
    .env({"PYTHONPATH": PACKAGE_REMOTE_ROOT})
)

VIDEO_POSE_DIRS = [
    f"{DATASET_ROOT}/rtmpose_features/video_pose",
    f"{DATASET_ROOT}/rtmpose_features_oops/video_pose",
]


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    timeout=60 * 60 * 2,
)
def prepare_and_slice(
    windows_csv: str = f"{DATASET_ROOT}/windows_classification_oops.csv",
    pose_output_csv: str = f"{DATASET_ROOT}/pose_windows_classification_oops.csv",
    pose_output_dir: str = f"{DATASET_ROOT}/rtmpose_features_classification",
) -> dict:
    import hashlib
    import numpy as np
    import pandas as pd

    from fall_anticipation_cv.data import (
        assign_group_splits,
        build_classification_window_dataframe,
        load_all_labels,
        validate_windows,
    )

    labels = load_all_labels(DATA_ROOT, include_oops=True)
    windows = build_classification_window_dataframe(labels)
    windows = assign_group_splits(windows, random_state=42)
    validate_windows(windows)

    Path(windows_csv).parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(windows_csv, index=False)
    print(f"Classification windows: {len(windows)}")
    print(f"  Falls: {(windows['y'] == 1).sum()}")
    print(f"  Non-falls: {(windows['y'] == 0).sum()}")
    print(windows.groupby(["dataset", "y"]).size().unstack(fill_value=0))
    if "split" in windows.columns:
        print(windows.groupby(["split", "y"]).size().unstack(fill_value=0))

    output_dir = Path(pose_output_dir)
    window_pose_dir = output_dir / "window_pose"
    window_pose_dir.mkdir(parents=True, exist_ok=True)

    def stable_video_id(video_path: str) -> str:
        digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:12]
        stem = Path(video_path).stem.replace(" ", "_")
        return f"{stem}_{digest}"

    def find_video_pose(video_path: str) -> Path | None:
        video_id = stable_video_id(video_path)
        for d in VIDEO_POSE_DIRS:
            candidate = Path(d) / f"{video_id}.npy"
            if candidate.exists():
                return candidate
        return None

    rows = []
    missing = 0
    for row_idx, row in windows.iterrows():
        video_pose_path = find_video_pose(row["video_path"])
        if video_pose_path is None:
            missing += 1
            continue

        video_pose = np.load(video_pose_path)
        sample_interval = float(row["sample_interval"])
        window_start = int(row["window_start"])
        window_end = int(row["window_end"])

        num_keypoints = video_pose.shape[1] if video_pose.ndim == 3 else 17
        frames = []
        for sampled_idx in range(window_start, window_end):
            original_idx = int(round(sampled_idx * sample_interval))
            if 0 <= original_idx < len(video_pose):
                frames.append(video_pose[original_idx])
            else:
                frames.append(np.zeros((num_keypoints, 3), dtype=np.float32))

        if not frames:
            missing += 1
            continue

        feature = np.stack(frames, axis=0).astype(np.float32)
        feature_path = window_pose_dir / f"classification_{row_idx:06d}.npy"
        np.save(feature_path, feature)

        output_row = row.to_dict()
        output_row["pose_feature_path"] = str(feature_path)
        rows.append(output_row)

    pose_df = pd.DataFrame(rows)
    Path(pose_output_csv).parent.mkdir(parents=True, exist_ok=True)
    pose_df.to_csv(pose_output_csv, index=False)
    volume.commit()

    summary = {
        "windows_csv": windows_csv,
        "pose_csv": pose_output_csv,
        "total_windows": int(len(windows)),
        "pose_features_created": int(len(pose_df)),
        "missing_video_poses": int(missing),
    }
    print(f"\nPose features: {len(pose_df)} sliced, {missing} missing video poses")
    return summary


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="L4",
    timeout=60 * 60 * 2,
)
def train_classification(
    windows_csv: str = f"{DATASET_ROOT}/pose_windows_classification_oops.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/pose_classification.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/pose_classification_metrics.json",
    epochs: int = 30,
    batch_size: int = 16,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    max_seq_len: int = 512,
    patience: int = 6,
) -> str:
    import json
    import subprocess
    import sys

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/train_pose_classification.py",
        "--windows-csv", windows_csv,
        "--checkpoint", checkpoint,
        "--metrics", metrics,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--hidden-dim", str(hidden_dim),
        "--num-layers", str(num_layers),
        "--dropout", str(dropout),
        "--lr", str(lr),
        "--max-seq-len", str(max_seq_len),
        "--patience", str(patience),
        "--num-workers", "0",
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()

    metrics_content = json.loads(Path(metrics).read_text())
    return json.dumps(metrics_content, indent=2)


@app.local_entrypoint()
def main() -> None:
    result = prepare_and_slice.remote()
    print(f"\nPreparation complete:")
    print(f"  Windows: {result['total_windows']}")
    print(f"  Pose features: {result['pose_features_created']}")
    print(f"  Missing poses: {result['missing_video_poses']}")

    if result["pose_features_created"] == 0:
        print("No pose features created — cannot train. Check video pose cache.")
        return

    metrics_json = train_classification.remote(
        windows_csv=result["pose_csv"],
    )
    print(f"\nTraining complete. Metrics:")
    print(metrics_json)
