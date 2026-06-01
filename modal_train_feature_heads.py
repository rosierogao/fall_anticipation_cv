from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-train-feature-heads")
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
    timeout=60 * 60 * 12,
)
def train_feature_heads(
    pose_windows_csv: str,
    vjepa_windows_csv: str,
    output_suffix: str,
    epochs: int = 5,
    batch_size: int = 32,
    model: str = "all",
    vjepa_num_workers: int = 4,
) -> dict:
    import sys
    from pathlib import Path

    for csv_path in (pose_windows_csv, vjepa_windows_csv):
        if not Path(csv_path).exists():
            raise FileNotFoundError(csv_path)

    outputs_dir = f"{DATASET_ROOT}/outputs"
    pose_metrics = f"{outputs_dir}/pose_transformer_{output_suffix}_metrics.json"
    vjepa_baseline_metrics = f"{outputs_dir}/vjepa_baseline_{output_suffix}_metrics.json"
    vjepa_predictive_metrics = (
        f"{outputs_dir}/vjepa_latent_predictive_{output_suffix}_metrics.json"
    )

    if model in {"all", "pose"}:
        _run_script(
            [
                sys.executable,
                f"{PACKAGE_REMOTE_ROOT}/scripts/train_pose_baseline.py",
                "--windows-csv",
                pose_windows_csv,
                "--feature-col",
                "pose_feature_path",
                "--checkpoint",
                f"{outputs_dir}/pose_transformer_{output_suffix}.pt",
                "--metrics",
                pose_metrics,
                "--model",
                "transformer",
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--num-workers",
                "4",
            ]
        )
        volume.commit()

    common_vjepa_args = [
        "--windows-csv",
        vjepa_windows_csv,
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
        str(vjepa_num_workers),
    ]

    if model in {"all", "vjepa", "vjepa-baseline"}:
        _run_script(
            [
                sys.executable,
                f"{PACKAGE_REMOTE_ROOT}/scripts/train_vjepa_predictive.py",
                *common_vjepa_args,
                "--checkpoint",
                f"{outputs_dir}/vjepa_baseline_{output_suffix}.pt",
                "--metrics",
                vjepa_baseline_metrics,
                "--model",
                "baseline",
            ]
        )
        volume.commit()

    if model in {"all", "vjepa", "vjepa-predictive"}:
        _run_script(
            [
                sys.executable,
                f"{PACKAGE_REMOTE_ROOT}/scripts/train_vjepa_predictive.py",
                *common_vjepa_args,
                "--checkpoint",
                f"{outputs_dir}/vjepa_latent_predictive_{output_suffix}.pt",
                "--metrics",
                vjepa_predictive_metrics,
                "--model",
                "predictive",
                "--predictive-loss-weight",
                "0.2",
            ]
        )
        volume.commit()

    return {
        "pose_metrics": pose_metrics,
        "vjepa_baseline_metrics": vjepa_baseline_metrics,
        "vjepa_predictive_metrics": vjepa_predictive_metrics,
    }


@app.local_entrypoint()
def main(
    pose_windows_csv: str,
    vjepa_windows_csv: str,
    output_suffix: str,
    epochs: int = 5,
    batch_size: int = 32,
    model: str = "all",
    vjepa_num_workers: int = 4,
) -> None:
    metrics = train_feature_heads.remote(
        pose_windows_csv=pose_windows_csv,
        vjepa_windows_csv=vjepa_windows_csv,
        output_suffix=output_suffix,
        epochs=epochs,
        batch_size=batch_size,
        model=model,
        vjepa_num_workers=vjepa_num_workers,
    )
    print(f"Completed feature-head training: {metrics}", flush=True)
