from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-vjepa-l1-experiment")
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


def _run(cmd: list[str]) -> None:
    import subprocess

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 12,
)
def train_and_tune_l1(
    epochs: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    output: str = f"{DATASET_ROOT}/outputs/vjepa_l1_two_policy_threshold_metrics.json",
) -> str:
    import sys

    runs = [
        {
            "windows_csv": f"{DATASET_ROOT}/vjepa_windows_staged_caucafall_joined.csv",
            "suffix": "staged_caucafall_fall_anticipation_l1",
        },
        {
            "windows_csv": f"{DATASET_ROOT}/vjepa_windows_staged_caucafall_oops.csv",
            "suffix": "staged_caucafall_oops_fall_anticipation_l1",
        },
    ]

    for run in runs:
        common_args = [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/train_vjepa_predictive.py",
            "--windows-csv",
            run["windows_csv"],
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--dropout",
            "0.35",
            "--weight-decay",
            "3e-4",
            "--lr-plateau-patience",
            "2",
            "--lr-plateau-factor",
            "0.5",
            "--num-workers",
            str(num_workers),
        ]

        _run(
            [
                *common_args,
                "--checkpoint",
                f"{DATASET_ROOT}/outputs/vjepa_baseline_{run['suffix']}.pt",
                "--metrics",
                f"{DATASET_ROOT}/outputs/vjepa_baseline_{run['suffix']}_metrics.json",
                "--model",
                "baseline",
            ]
        )
        volume.commit()

        _run(
            [
                *common_args,
                "--checkpoint",
                f"{DATASET_ROOT}/outputs/vjepa_latent_predictive_{run['suffix']}.pt",
                "--metrics",
                f"{DATASET_ROOT}/outputs/vjepa_latent_predictive_{run['suffix']}_metrics.json",
                "--model",
                "predictive",
                "--predictive-loss-weight",
                "0.2",
                "--predictive-loss",
                "l1",
            ]
        )
        volume.commit()

    _run(
        [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/evaluate_vjepa_l1_thresholds.py",
            "--data-root",
            DATASET_ROOT,
            "--output",
            output,
            "--batch-size",
            str(batch_size),
        ]
    )
    volume.commit()
    return output


@app.local_entrypoint()
def main(
    epochs: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    output: str = f"{DATASET_ROOT}/outputs/vjepa_l1_two_policy_threshold_metrics.json",
) -> None:
    result_path = train_and_tune_l1.remote(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        output=output,
    )
    print(f"Saved V-JEPA L1 threshold metrics: {result_path}", flush=True)
