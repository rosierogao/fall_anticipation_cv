from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-vjepa-two-stage")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy<2.0",
        "opencv-python-headless==4.10.0.84",
        "pandas<3.0",
        "scikit-learn",
        "torch==2.3.1",
        "torchvision==0.18.1",
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
    timeout=60 * 60 * 2,
)
def train_two_stage(
    pretrain_csv: str = f"{DATASET_ROOT}/vjepa_windows_real_oops.csv",
    finetune_csv: str = f"{DATASET_ROOT}/vjepa_windows.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/vjepa_two_stage.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/vjepa_two_stage_metrics.json",
    model: str = "predictive",
    pretrain_epochs: int = 5,
    pretrain_lr: float = 1e-4,
    pretrain_batch_size: int = 32,
    pretrain_predictive_loss_weight: float = 0.2,
    finetune_epochs: int = 10,
    finetune_lr: float = 1e-5,
    finetune_batch_size: int = 32,
    finetune_predictive_loss_weight: float = 0.0,
    freeze_body: bool = False,
) -> str:
    import sys
    from pathlib import Path

    for path, label in [(pretrain_csv, "pretrain CSV"), (finetune_csv, "finetune CSV")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/train_vjepa_two_stage.py",
        "--pretrain-csv", pretrain_csv,
        "--finetune-csv", finetune_csv,
        "--checkpoint", checkpoint,
        "--metrics", metrics,
        "--model", model,
        "--pretrain-epochs", str(pretrain_epochs),
        "--pretrain-lr", str(pretrain_lr),
        "--pretrain-batch-size", str(pretrain_batch_size),
        "--pretrain-predictive-loss-weight", str(pretrain_predictive_loss_weight),
        "--finetune-epochs", str(finetune_epochs),
        "--finetune-lr", str(finetune_lr),
        "--finetune-batch-size", str(finetune_batch_size),
        "--finetune-predictive-loss-weight", str(finetune_predictive_loss_weight),
    ]
    if freeze_body:
        cmd.append("--freeze-body")

    _run_script(cmd)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    freeze_body: bool = False,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 10,
) -> None:
    result = train_two_stage.remote(
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        freeze_body=freeze_body,
    )
    print(f"Done. Metrics saved to: {result}")
