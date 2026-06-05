from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-finetune-pose")
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
def finetune_pose_transformer(
    pretrained_checkpoint: str = f"{DATASET_ROOT}/outputs/pose_transformer_normalized.pt",
    windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/pose_transformer_finetuned.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/pose_transformer_finetuned_metrics.json",
    epochs: int = 10,
    batch_size: int = 32,
    num_layers: int | None = None,
    hidden_dim: int | None = None,
    lr: float = 1e-4,
    label_smoothing: float = 0.1,
    grad_clip: float = 1.0,
    patience: int = 4,
    freeze_projection: bool = True,
) -> str:
    import sys
    from pathlib import Path

    if not Path(pretrained_checkpoint).exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {pretrained_checkpoint}. "
            "Run train_pose_transformer first."
        )
    if not Path(windows_csv).exists():
        raise FileNotFoundError(
            f"Missing pose feature metadata: {windows_csv}. "
            "Run RTMPose feature extraction before fine-tuning."
        )

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/finetune_pose_transformer.py",
        "--pretrained-checkpoint", pretrained_checkpoint,
        "--windows-csv", windows_csv,
        "--checkpoint", checkpoint,
        "--metrics", metrics,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--label-smoothing", str(label_smoothing),
        "--grad-clip", str(grad_clip),
        "--patience", str(patience),
        "--num-workers", "0",
    ]
    if freeze_projection:
        cmd.append("--freeze-projection")
    if num_layers is not None:
        cmd.extend(["--num-layers", str(num_layers)])
    if hidden_dim is not None:
        cmd.extend(["--hidden-dim", str(hidden_dim)])

    _run_script(cmd)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    pretrained_checkpoint: str = f"{DATASET_ROOT}/outputs/pose_transformer_normalized.pt",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    freeze_projection: bool = True,
    num_layers: int | None = None,
) -> None:
    call = finetune_pose_transformer.spawn(
        pretrained_checkpoint=pretrained_checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        freeze_projection=freeze_projection,
        num_layers=num_layers,
    )
    print(f"Spawned pose transformer fine-tuning: {call.object_id}", flush=True)
