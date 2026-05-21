from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_RTMPOSE_SCRIPT = Path(__file__).parent / "scripts" / "extract_rtmpose_features.py"

app = modal.App(f"{APP_NAME}-extract-pose-l4")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "openmim",
        "numpy<2.0",
        "pandas<3.0",
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
    .add_local_file(
        LOCAL_RTMPOSE_SCRIPT,
        f"{PACKAGE_REMOTE_ROOT}/scripts/extract_rtmpose_features.py",
        copy=True,
    )
    .env({"PYTHONPATH": PACKAGE_REMOTE_ROOT})
)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="L4",
    timeout=60 * 60 * 12,
)
def extract_rtmpose_pose_features(
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
    output_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    output_dir: str = f"{DATASET_ROOT}/rtmpose_features",
    pose2d: str = "human",
    max_videos: int | None = None,
    video_start: int = 0,
    video_count: int | None = None,
) -> str:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/extract_rtmpose_features.py",
        "--windows-csv",
        windows_csv,
        "--output-csv",
        output_csv,
        "--output-dir",
        output_dir,
        "--pose2d",
        pose2d,
        "--device",
        "cuda",
        "--video-start",
        str(video_start),
    ]
    if video_count is not None:
        cmd.extend(["--video-count", str(video_count)])
    if max_videos is not None:
        cmd.extend(["--max-videos", str(max_videos)])

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()
    return output_csv


@app.local_entrypoint()
def main(
    max_videos: int | None = None,
    video_start: int = 0,
    video_count: int | None = None,
) -> None:
    extract_rtmpose_pose_features.remote(
        max_videos=max_videos,
        video_start=video_start,
        video_count=video_count,
    )
