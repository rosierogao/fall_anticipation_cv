from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-fusion")
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
    timeout=60 * 60 * 4,
)
def train_pose_vjepa_fusion(
    windows_csv: str = f"{DATASET_ROOT}/vjepa_windows.csv",
    pose_windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/pose_vjepa_fusion.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/pose_vjepa_fusion_metrics.json",
    epochs: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
) -> str:
    import sys
    from pathlib import Path

    if not Path(windows_csv).exists():
        raise FileNotFoundError(f"Missing fusion window metadata: {windows_csv}")

    _run_script(
        [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/train_pose_vjepa_fusion.py",
            "--windows-csv",
            windows_csv,
            "--pose-windows-csv",
            pose_windows_csv,
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


@app.local_entrypoint()
def main(
    epochs: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    windows_csv: str = f"{DATASET_ROOT}/vjepa_windows.csv",
    pose_windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
) -> None:
    call = train_pose_vjepa_fusion.spawn(
        windows_csv=windows_csv,
        pose_windows_csv=pose_windows_csv,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"Spawned pose + V-JEPA fusion training: {call.object_id}", flush=True)
