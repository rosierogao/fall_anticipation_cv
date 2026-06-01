from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-train-models")
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


def _run_script(cmd: list[str]) -> None:
    import subprocess

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 8,
)
def train_video_cnn_transformer(
    epochs: int = 5,
    batch_size: int = 16,
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/video_cnn_transformer_baseline.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/video_cnn_transformer_metrics.json",
    resume: bool = True,
    num_workers: int = 4,
) -> str:
    import sys

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/train_baseline.py",
        "--windows-csv",
        windows_csv,
        "--checkpoint",
        checkpoint,
        "--metrics",
        metrics,
        "--model",
        "transformer",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    ]
    if resume:
        cmd.append("--resume")

    _run_script(cmd)
    volume.commit()
    return metrics


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 8,
)
def train_pose_transformer(
    epochs: int = 5,
    batch_size: int = 32,
    windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/pose_transformer_normalized.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/pose_transformer_normalized_metrics.json",
    num_workers: int = 4,
) -> str:
    import sys
    from pathlib import Path

    if not Path(windows_csv).exists():
        raise FileNotFoundError(
            f"Missing pose feature metadata: {windows_csv}. "
            "Run RTMPose feature extraction before training PoseTransformerBaseline."
        )

    _run_script(
        [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/train_pose_baseline.py",
            "--windows-csv",
            windows_csv,
            "--feature-col",
            "pose_feature_path",
            "--checkpoint",
            checkpoint,
            "--metrics",
            metrics,
            "--model",
            "transformer",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
        ]
    )
    volume.commit()
    return metrics


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 8,
)
def train_pose_predictive(
    epochs: int = 5,
    batch_size: int = 32,
    pose_windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    windows_csv: str = f"{DATASET_ROOT}/pose_predictive_windows_rtmpose.csv",
    feature_dir: str = f"{DATASET_ROOT}/rtmpose_features/pose_predictive_windows",
    checkpoint: str = f"{DATASET_ROOT}/outputs/pose_seq2seq_predictive.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/pose_seq2seq_predictive_metrics.json",
    num_workers: int = 4,
) -> str:
    import sys
    from pathlib import Path

    if not Path(pose_windows_csv).exists():
        raise FileNotFoundError(
            f"Missing pose feature metadata: {pose_windows_csv}. "
            "Run RTMPose feature extraction before training PoseSeq2SeqPredictiveModel."
        )

    if not Path(windows_csv).exists():
        _run_script(
            [
                sys.executable,
                f"{PACKAGE_REMOTE_ROOT}/scripts/prepare_pose_predictive_windows.py",
                "--pose-windows-csv",
                pose_windows_csv,
                "--output-csv",
                windows_csv,
                "--output-dir",
                feature_dir,
            ]
        )
        volume.commit()

    _run_script(
        [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/train_pose_predictive.py",
            "--windows-csv",
            windows_csv,
            "--feature-col",
            "pose_predictive_feature_path",
            "--checkpoint",
            checkpoint,
            "--metrics",
            metrics,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
        ]
    )
    volume.commit()
    return metrics


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 2,
)
def evaluate_video_cnn_transformer(
    batch_size: int = 8,
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/video_cnn_transformer_baseline.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/video_cnn_transformer_h100_eval_metrics.json",
) -> str:
    import sys

    _run_script(
        [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/evaluate_video_checkpoint.py",
            "--windows-csv",
            windows_csv,
            "--checkpoint",
            checkpoint,
            "--metrics",
            metrics,
            "--model",
            "transformer",
            "--batch-size",
            str(batch_size),
            "--num-workers",
            "0",
        ]
    )
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    model: str = "video",
    action: str = "train",
    epochs: int = 5,
    batch_size: int | None = None,
    resume: bool = True,
    windows_csv: str | None = None,
    predictive_windows_csv: str | None = None,
    feature_dir: str | None = None,
    checkpoint: str | None = None,
    metrics: str | None = None,
    num_workers: int = 4,
) -> None:
    if action == "eval":
        if model != "video":
            raise ValueError("eval action currently supports model='video' only")
        evaluate_video_cnn_transformer.remote(batch_size=batch_size or 8)
        return

    if action != "train":
        raise ValueError("action must be 'train' or 'eval'")

    if model == "video":
        call = train_video_cnn_transformer.spawn(
            windows_csv=windows_csv or f"{DATASET_ROOT}/windows_gmdcsa24.csv",
            checkpoint=checkpoint or f"{DATASET_ROOT}/outputs/video_cnn_transformer_baseline.pt",
            metrics=metrics or f"{DATASET_ROOT}/outputs/video_cnn_transformer_metrics.json",
            epochs=epochs,
            batch_size=batch_size or 16,
            resume=resume,
            num_workers=num_workers,
        )
        print(f"Spawned video CNN/Transformer training: {call.object_id}", flush=True)
    elif model == "pose":
        call = train_pose_transformer.spawn(
            windows_csv=windows_csv or f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
            checkpoint=checkpoint or f"{DATASET_ROOT}/outputs/pose_transformer_normalized.pt",
            metrics=metrics or f"{DATASET_ROOT}/outputs/pose_transformer_normalized_metrics.json",
            epochs=epochs,
            batch_size=batch_size or 32,
            num_workers=num_workers,
        )
        print(f"Spawned pose Transformer training: {call.object_id}", flush=True)
    elif model == "pose_predictive":
        call = train_pose_predictive.spawn(
            pose_windows_csv=windows_csv or f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
            windows_csv=predictive_windows_csv
            or f"{DATASET_ROOT}/pose_predictive_windows_rtmpose.csv",
            feature_dir=feature_dir
            or f"{DATASET_ROOT}/rtmpose_features/pose_predictive_windows",
            epochs=epochs,
            batch_size=batch_size or 32,
            checkpoint=checkpoint or f"{DATASET_ROOT}/outputs/pose_seq2seq_predictive.pt",
            metrics=metrics or f"{DATASET_ROOT}/outputs/pose_seq2seq_predictive_metrics.json",
            num_workers=num_workers,
        )
        print(f"Spawned pose predictive training: {call.object_id}", flush=True)
    else:
        raise ValueError("model must be 'video', 'pose', or 'pose_predictive'")
