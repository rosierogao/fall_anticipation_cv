from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-vjepa")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "accelerate",
        "numpy<2.0",
        "opencv-python-headless==4.10.0.84",
        "pandas<3.0",
        "scikit-learn",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.57.3",
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
    timeout=60 * 60 * 12,
)
def extract_vjepa_latents(
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
    output_csv: str = f"{DATASET_ROOT}/vjepa_windows.csv",
    output_dir: str = f"{DATASET_ROOT}/vjepa_latents",
    model_name: str = "facebook/vjepa2-vitl-fpc64-256",
    batch_size: int = 2,
    max_windows: int | None = None,
) -> str:
    import sys

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/extract_vjepa_latents.py",
        "--windows-csv",
        windows_csv,
        "--output-csv",
        output_csv,
        "--output-dir",
        output_dir,
        "--model-name",
        model_name,
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda",
    ]
    if max_windows is not None:
        cmd.extend(["--max-windows", str(max_windows)])

    _run_script(cmd)
    volume.commit()
    return output_csv


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="A100",
    timeout=60 * 60 * 4,
)
def train_vjepa_predictive(
    windows_csv: str = f"{DATASET_ROOT}/vjepa_windows.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/vjepa_latent_predictive.pt",
    metrics: str = f"{DATASET_ROOT}/outputs/vjepa_latent_predictive_metrics.json",
    epochs: int = 5,
    batch_size: int = 32,
) -> str:
    import sys
    from pathlib import Path

    if not Path(windows_csv).exists():
        raise FileNotFoundError(
            f"Missing V-JEPA latent metadata: {windows_csv}. "
            "Run V-JEPA latent extraction before training."
        )

    _run_script(
        [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/train_vjepa_predictive.py",
            "--windows-csv",
            windows_csv,
            "--checkpoint",
            checkpoint,
            "--metrics",
            metrics,
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


@app.local_entrypoint()
def main(
    action: str = "extract",
    epochs: int = 5,
    batch_size: int = 2,
    max_windows: int | None = None,
) -> None:
    if action == "extract":
        call = extract_vjepa_latents.spawn(
            batch_size=batch_size,
            max_windows=max_windows,
        )
        print(f"Spawned V-JEPA latent extraction: {call.object_id}", flush=True)
    elif action == "train":
        call = train_vjepa_predictive.spawn(
            epochs=epochs,
            batch_size=batch_size,
        )
        print(f"Spawned V-JEPA predictive training: {call.object_id}", flush=True)
    elif action == "all":
        output_csv = extract_vjepa_latents.remote(
            batch_size=batch_size,
            max_windows=max_windows,
        )
        call = train_vjepa_predictive.spawn(
            windows_csv=output_csv,
            epochs=epochs,
            batch_size=32,
        )
        print(f"Spawned V-JEPA predictive training: {call.object_id}", flush=True)
    else:
        raise ValueError("action must be 'extract', 'train', or 'all'")
