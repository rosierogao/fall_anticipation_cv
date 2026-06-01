from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-fallen-state")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

vjepa_image = (
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

pose_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "openmim",
        "numpy<2.0",
        "pandas<3.0",
        "scikit-learn",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "tqdm",
    )
    .run_commands(
        'mim install "mmengine>=0.7.0" "mmcv==2.1.0" "mmdet>=3.0.0,<3.3.0" "mmpose==1.3.2"',
        'python -m pip install --force-reinstall "numpy<2.0"',
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


def _metadata_is_complete(source_csv: str, feature_csv: str) -> bool:
    import pandas as pd
    from pathlib import Path

    feature_path = Path(feature_csv)
    if not feature_path.exists():
        return False

    source = pd.read_csv(source_csv)
    features = pd.read_csv(feature_csv)
    if len(features) == 0 or len(features) < len(source):
        return False

    required_keys = {"video_path", "window_start", "window_end"}
    if not required_keys.issubset(features.columns):
        return False

    source_keys = source[list(required_keys)].drop_duplicates()
    feature_keys = features[list(required_keys)].drop_duplicates()
    merged = source_keys.merge(
        feature_keys,
        on=list(required_keys),
        how="left",
        indicator=True,
    )
    return bool((merged["_merge"] == "both").all())


@app.function(
    image=vjepa_image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 16,
)
def train_fallen_vjepa_models(
    windows_csv: str = f"{DATASET_ROOT}/windows_fallen_state_gmd_le2i_horizon2s.csv",
    vjepa_windows_csv: str = f"{DATASET_ROOT}/vjepa_windows_fallen_state_horizon2s.csv",
    vjepa_output_dir: str = f"{DATASET_ROOT}/vjepa_latents_fallen_state_horizon2s",
    output_suffix: str = "fallen_state_horizon2s",
    epochs: int = 5,
    batch_size: int = 32,
    extract_batch_size: int = 2,
) -> dict:
    import sys
    from pathlib import Path

    if not _metadata_is_complete(windows_csv, vjepa_windows_csv):
        _run_script(
            [
                sys.executable,
                f"{PACKAGE_REMOTE_ROOT}/scripts/extract_vjepa_latents.py",
                "--windows-csv",
                windows_csv,
                "--output-csv",
                vjepa_windows_csv,
                "--output-dir",
                vjepa_output_dir,
                "--model-name",
                "facebook/vjepa2-vitl-fpc64-256",
                "--batch-size",
                str(extract_batch_size),
                "--device",
                "cuda",
            ]
        )
        volume.commit()
    else:
        print(f"Using existing V-JEPA features: {vjepa_windows_csv}", flush=True)

    baseline_metrics = f"{DATASET_ROOT}/outputs/vjepa_baseline_{output_suffix}_metrics.json"
    predictive_metrics = f"{DATASET_ROOT}/outputs/vjepa_latent_predictive_{output_suffix}_metrics.json"

    _run_script(
        [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/train_vjepa_predictive.py",
            "--windows-csv",
            vjepa_windows_csv,
            "--checkpoint",
            f"{DATASET_ROOT}/outputs/vjepa_baseline_{output_suffix}.pt",
            "--metrics",
            baseline_metrics,
            "--model",
            "baseline",
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
            "0",
        ]
    )
    volume.commit()

    _run_script(
        [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/train_vjepa_predictive.py",
            "--windows-csv",
            vjepa_windows_csv,
            "--checkpoint",
            f"{DATASET_ROOT}/outputs/vjepa_latent_predictive_{output_suffix}.pt",
            "--metrics",
            predictive_metrics,
            "--model",
            "predictive",
            "--predictive-loss-weight",
            "0.2",
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
            "0",
        ]
    )
    volume.commit()

    return {
        "vjepa_windows_csv": vjepa_windows_csv,
        "baseline_metrics": baseline_metrics,
        "predictive_metrics": predictive_metrics,
    }


@app.function(
    image=pose_image,
    volumes={DATA_ROOT: volume},
    gpu="L4",
    timeout=60 * 60 * 16,
)
def train_fallen_pose_transformer(
    windows_csv: str = f"{DATASET_ROOT}/windows_fallen_state_gmd_le2i_horizon2s.csv",
    pose_windows_csv: str = f"{DATASET_ROOT}/pose_windows_fallen_state_horizon2s_rtmpose.csv",
    pose_output_dir: str = f"{DATASET_ROOT}/rtmpose_features_fallen_state_horizon2s",
    output_suffix: str = "fallen_state_horizon2s",
    epochs: int = 5,
    batch_size: int = 32,
    chunk_size: int = 1,
) -> str:
    import pandas as pd
    import sys
    from pathlib import Path

    if not _metadata_is_complete(windows_csv, pose_windows_csv):
        total_videos = int(pd.read_csv(windows_csv)["video_path"].nunique())
        failed_chunks = []
        for start in range(0, total_videos, chunk_size):
            count = min(chunk_size, total_videos - start)
            cmd = [
                sys.executable,
                f"{PACKAGE_REMOTE_ROOT}/scripts/extract_rtmpose_features.py",
                "--windows-csv",
                windows_csv,
                "--output-csv",
                pose_windows_csv,
                "--output-dir",
                pose_output_dir,
                "--pose2d",
                "human",
                "--device",
                "cuda",
                "--video-start",
                str(start),
                "--video-count",
                str(count),
            ]
            print(f"Extracting RTMPose chunk: videos {start}..{start + count - 1}", flush=True)
            try:
                _run_script(cmd)
                volume.commit()
            except Exception as exc:
                print(
                    f"Skipping failed RTMPose chunk videos {start}..{start + count - 1}: {exc}",
                    flush=True,
                )
                failed_chunks.append((start, count))
        if failed_chunks:
            print(f"Skipped failed RTMPose chunks: {failed_chunks}", flush=True)
    else:
        print(f"Using existing RTMPose features: {pose_windows_csv}", flush=True)

    metrics = f"{DATASET_ROOT}/outputs/pose_transformer_{output_suffix}_metrics.json"
    _run_script(
        [
            sys.executable,
            f"{PACKAGE_REMOTE_ROOT}/scripts/train_pose_baseline.py",
            "--windows-csv",
            pose_windows_csv,
            "--feature-col",
            "pose_feature_path",
            "--checkpoint",
            f"{DATASET_ROOT}/outputs/pose_transformer_{output_suffix}.pt",
            "--metrics",
            metrics,
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
    return metrics


@app.local_entrypoint()
def main(
    model: str = "all",
    epochs: int = 5,
    batch_size: int = 32,
    windows_csv: str = f"{DATASET_ROOT}/windows_fallen_state_gmd_le2i_horizon2s.csv",
    vjepa_windows_csv: str = f"{DATASET_ROOT}/vjepa_windows_fallen_state_horizon2s.csv",
    vjepa_output_dir: str = f"{DATASET_ROOT}/vjepa_latents_fallen_state_horizon2s",
    pose_windows_csv: str = f"{DATASET_ROOT}/pose_windows_fallen_state_horizon2s_rtmpose.csv",
    pose_output_dir: str = f"{DATASET_ROOT}/rtmpose_features_fallen_state_horizon2s",
    output_suffix: str = "fallen_state_horizon2s",
) -> None:
    if model in {"all", "vjepa"}:
        call = train_fallen_vjepa_models.spawn(
            windows_csv=windows_csv,
            vjepa_windows_csv=vjepa_windows_csv,
            vjepa_output_dir=vjepa_output_dir,
            output_suffix=output_suffix,
            epochs=epochs,
            batch_size=batch_size,
        )
        print(f"Spawned fallen-state V-JEPA baseline + predictive pipeline: {call.object_id}", flush=True)
    if model in {"all", "pose"}:
        call = train_fallen_pose_transformer.spawn(
            windows_csv=windows_csv,
            pose_windows_csv=pose_windows_csv,
            pose_output_dir=pose_output_dir,
            output_suffix=output_suffix,
            epochs=epochs,
            batch_size=batch_size,
        )
        print(f"Spawned fallen-state pose Transformer pipeline: {call.object_id}", flush=True)
