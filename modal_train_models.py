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
    gpu="L4",
    timeout=60 * 60 * 8,
)
def train_video_cnn_transformer(
    epochs: int = 5,
    batch_size: int = 16,
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/video_cnn_transformer_baseline.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/video_cnn_transformer_metrics.json",
) -> str:
    import sys

    _run_script(
        [
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
            "0",
        ]
    )
    volume.commit()
    return metrics


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="L4",
    timeout=60 * 60 * 8,
)
def train_pose_transformer(
    epochs: int = 5,
    batch_size: int = 32,
    windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/pose_transformer_baseline.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/pose_transformer_metrics.json",
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
            "0",
        ]
    )
    volume.commit()
    return metrics


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="L4",
    timeout=60 * 60 * 2,
)
def evaluate_video_cnn_transformer(
    batch_size: int = 8,
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/video_cnn_transformer_baseline.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/video_cnn_transformer_epoch1_eval_metrics.json",
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
) -> None:
    if action == "eval":
        if model != "video":
            raise ValueError("eval action currently supports model='video' only")
        evaluate_video_cnn_transformer.remote(batch_size=batch_size or 8)
        return

    if action != "train":
        raise ValueError("action must be 'train' or 'eval'")

    if model == "video":
        train_video_cnn_transformer.remote(
            epochs=epochs,
            batch_size=batch_size or 16,
        )
    elif model == "pose":
        train_pose_transformer.remote(
            epochs=epochs,
            batch_size=batch_size or 32,
        )
    else:
        raise ValueError("model must be 'video' or 'pose'")
