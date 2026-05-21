from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"

app = modal.App(f"{APP_NAME}-check-windows")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "opencv-python-headless", "pandas")
    .add_local_dir(
        LOCAL_PACKAGE_DIR,
        f"{PACKAGE_REMOTE_ROOT}/fall_anticipation_cv",
        copy=True,
    )
    .env({"PYTHONPATH": PACKAGE_REMOTE_ROOT})
)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    timeout=60 * 30,
)
def check_windows(windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv") -> dict:
    import json

    import numpy as np
    import pandas as pd

    windows = pd.read_csv(windows_csv)
    positive = windows[windows["y"] == 1].copy()
    positive["frames_until_fall"] = (
        positive["fall_start_frame"] - positive["target_frame"]
    )
    positive["seconds_until_fall"] = (
        positive["frames_until_fall"] / positive["target_fps"]
    )

    summary = {
        "windows_csv": windows_csv,
        "num_windows": int(len(windows)),
        "target_fps_values": sorted(windows["target_fps"].dropna().unique().tolist()),
        "obs_len_values": sorted(windows["obs_len"].dropna().unique().tolist()),
        "k_frames_values": sorted(windows["k_frames"].dropna().unique().tolist()),
        "window_duration_seconds": sorted(
            (windows["obs_len"] / windows["target_fps"]).dropna().unique().tolist()
        ),
        "horizon_seconds": sorted(
            (windows["k_frames"] / windows["target_fps"]).dropna().unique().tolist()
        ),
        "by_dataset": {},
        "positive_timing_by_dataset": {},
        "positive_timing_violations": int(
            (
                (positive["frames_until_fall"] <= 0)
                | (positive["frames_until_fall"] > positive["k_frames"])
            ).sum()
        ),
    }

    for dataset, group in windows.groupby("dataset", dropna=False):
        implied_original_fps = group["sample_interval"] * group["target_fps"]
        summary["by_dataset"][str(dataset)] = {
            "num_windows": int(len(group)),
            "class_counts": {
                str(k): int(v) for k, v in group["y"].value_counts().items()
            },
            "sample_interval_values": sorted(
                round(float(v), 6) for v in group["sample_interval"].unique()
            ),
            "implied_original_fps_values": sorted(
                round(float(v), 3) for v in implied_original_fps.unique()
            ),
            "num_unique_videos": int(group["video_path"].nunique()),
        }

    for dataset, group in positive.groupby("dataset", dropna=False):
        summary["positive_timing_by_dataset"][str(dataset)] = {
            "num_positive": int(len(group)),
            "frames_until_fall_min": int(group["frames_until_fall"].min()),
            "frames_until_fall_max": int(group["frames_until_fall"].max()),
            "frames_until_fall_values": sorted(
                int(v) for v in group["frames_until_fall"].unique()
            ),
            "seconds_until_fall_min": float(np.round(group["seconds_until_fall"].min(), 3)),
            "seconds_until_fall_max": float(np.round(group["seconds_until_fall"].max(), 3)),
        }

    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv") -> None:
    check_windows.remote(windows_csv=windows_csv)
