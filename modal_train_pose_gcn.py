from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-train-pose-gcn")
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
def train_pose_gcn(
    windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/pose_gcn_transformer.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/pose_gcn_transformer_metrics.json",
    epochs: int = 20,
    batch_size: int = 32,
    gcn_hidden: int = 64,
    gcn_out: int = 128,
    d_model: int = 256,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.3,
    lr: float = 1e-3,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
    patience: int = 5,
) -> str:
    import sys
    from pathlib import Path

    if not Path(windows_csv).exists():
        raise FileNotFoundError(f"Missing pose windows CSV: {windows_csv}")

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/train_pose_gcn.py",
        "--windows-csv", windows_csv,
        "--checkpoint", checkpoint,
        "--metrics", metrics,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--gcn-hidden", str(gcn_hidden),
        "--gcn-out", str(gcn_out),
        "--d-model", str(d_model),
        "--num-layers", str(num_layers),
        "--num-heads", str(num_heads),
        "--dropout", str(dropout),
        "--lr", str(lr),
        "--label-smoothing", str(label_smoothing),
        "--grad-clip", str(grad_clip),
        "--patience", str(patience),
        "--num-workers", "0",
    ]

    _run_script(cmd)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    epochs: int = 20,
    d_model: int = 256,
    num_layers: int = 2,
    lr: float = 1e-3,
    label_smoothing: float = 0.0,
) -> None:
    metrics_path = train_pose_gcn.remote(
        epochs=epochs,
        d_model=d_model,
        num_layers=num_layers,
        lr=lr,
        label_smoothing=label_smoothing,
    )
    print(f"Done. Metrics saved to: {metrics_path}")
