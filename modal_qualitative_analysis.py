"""Modal wrapper for qualitative analysis (t-SNE, attention, saliency)."""

from pathlib import Path

import modal

APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-qualitative-analysis")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy<2.0",
        "matplotlib>=3.8",
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
    .env({"PYTHONPATH": f"{PACKAGE_REMOTE_ROOT}:{PACKAGE_REMOTE_ROOT}/scripts"})
)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 2,
)
def qualitative_analysis(
    output_dir: str = f"{DATASET_ROOT}/outputs/qualitative_analysis",
    dataset_preset: str = "staged_caucafall_oops",
    examples_per_bucket: int = 8,
    batch_size: int = 32,
) -> dict:
    from qualitative_analysis import run_analysis

    summary = run_analysis(
        data_root=DATASET_ROOT,
        output_dir=output_dir,
        dataset_preset=dataset_preset,
        examples_per_bucket=examples_per_bucket,
        batch_size=batch_size,
    )
    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    output_dir: str = f"{DATASET_ROOT}/outputs/qualitative_analysis",
    dataset_preset: str = "staged_caucafall_oops",
    examples_per_bucket: int = 8,
    batch_size: int = 32,
) -> None:
    summary = qualitative_analysis.remote(
        output_dir=output_dir,
        dataset_preset=dataset_preset,
        examples_per_bucket=examples_per_bucket,
        batch_size=batch_size,
    )
    print(summary)
