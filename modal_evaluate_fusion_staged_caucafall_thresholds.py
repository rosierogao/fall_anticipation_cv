from pathlib import Path

import modal

APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-fusion-staged-thresholds")
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
    .add_local_dir(LOCAL_PACKAGE_DIR, f"{PACKAGE_REMOTE_ROOT}/fall_anticipation_cv", copy=True)
    .add_local_dir(LOCAL_SCRIPTS_DIR, f"{PACKAGE_REMOTE_ROOT}/scripts", copy=True)
    .env({"PYTHONPATH": PACKAGE_REMOTE_ROOT})
)


@app.function(image=image, volumes={DATA_ROOT: volume}, gpu="H100", timeout=60 * 60)
def evaluate(
    output: str = f"{DATASET_ROOT}/outputs/pose_vjepa_fusion_staged_caucafall_threshold_metrics.json",
) -> str:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/evaluate_fusion_staged_caucafall_thresholds.py",
        "--data-root",
        DATASET_ROOT,
        "--output",
        output,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()
    return output


@app.local_entrypoint()
def main(output: str = f"{DATASET_ROOT}/outputs/pose_vjepa_fusion_staged_caucafall_threshold_metrics.json") -> None:
    result = evaluate.remote(output=output)
    print(f"Saved staged fusion threshold metrics: {result}", flush=True)
