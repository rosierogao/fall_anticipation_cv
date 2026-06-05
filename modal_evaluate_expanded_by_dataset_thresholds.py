from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-expanded-by-dataset-thresholds")
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


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 2,
)
def evaluate_expanded_by_dataset(
    output: str = f"{DATASET_ROOT}/outputs/expanded_by_dataset_threshold_metrics.json",
    batch_size: int = 32,
    target_recall: float = 0.75,
    datasets: str = "GMDCSA24,le2i,caucafall",
) -> str:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/evaluate_expanded_by_dataset_thresholds.py",
        "--data-root",
        DATASET_ROOT,
        "--output",
        output,
        "--batch-size",
        str(batch_size),
        "--datasets",
        *[dataset.strip() for dataset in datasets.split(",") if dataset.strip()],
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()
    return output


@app.local_entrypoint()
def main(
    output: str = f"{DATASET_ROOT}/outputs/expanded_by_dataset_threshold_metrics.json",
    batch_size: int = 32,
    target_recall: float = 0.75,
    datasets: str = "GMDCSA24,le2i,caucafall",
) -> None:
    output_path = evaluate_expanded_by_dataset.remote(
        output=output,
        batch_size=batch_size,
        target_recall=target_recall,
        datasets=datasets,
    )
    print(f"Saved expanded by-dataset metrics: {output_path}", flush=True)
