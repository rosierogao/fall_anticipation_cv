from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-temporal-ablation")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "opencv-python-headless==4.10.0.84",
        "pandas<3.0",
        "scikit-learn",
        "torch==2.3.1",
    )
    .add_local_dir(
        LOCAL_PACKAGE_DIR,
        f"{PACKAGE_REMOTE_ROOT}/fall_anticipation_cv",
        copy=True,
    )
    .add_local_dir(LOCAL_SCRIPTS_DIR, f"{PACKAGE_REMOTE_ROOT}/scripts", copy=True)
    .env({"PYTHONPATH": PACKAGE_REMOTE_ROOT})
)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 2,
)
def run_temporal_ablation(
    output_dir: str = f"{DATASET_ROOT}/outputs/temporal_ablation_staged_caucafall",
    batch_size: int = 32,
    ablation_method: str = "delete",
    mask_value: str = "zero",
) -> str:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/evaluate_temporal_ablation.py",
        "--data-root",
        DATASET_ROOT,
        "--output-dir",
        output_dir,
        "--batch-size",
        str(batch_size),
        "--ablation-method",
        ablation_method,
        "--mask-value",
        mask_value,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()
    return output_dir


@app.local_entrypoint()
def main(
    output_dir: str = f"{DATASET_ROOT}/outputs/temporal_ablation_staged_caucafall",
    batch_size: int = 32,
    ablation_method: str = "delete",
    mask_value: str = "zero",
) -> None:
    saved_dir = run_temporal_ablation.remote(
        output_dir=output_dir,
        batch_size=batch_size,
        ablation_method=ablation_method,
        mask_value=mask_value,
    )
    print(f"Saved temporal ablation outputs: {saved_dir}", flush=True)
