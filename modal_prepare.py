from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"

app = modal.App(f"{APP_NAME}-prepare")
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


def _counts(series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    timeout=60 * 60 * 2,
)
def prepare_windows(
    output_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
) -> dict:
    import json
    from pathlib import Path

    from fall_anticipation_cv.data import (
        build_window_dataframe,
        load_all_labels,
        validate_windows,
    )

    labels = load_all_labels(DATA_ROOT)
    existing_labels = labels[labels["video_exists"]].copy()
    windows = build_window_dataframe(labels)
    validate_windows(windows)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_path, index=False)
    volume.commit()

    summary = {
        "output_csv": output_csv,
        "num_label_rows": int(len(labels)),
        "num_existing_label_rows": int(len(existing_labels)),
        "label_rows_by_dataset": _counts(labels["dataset"]),
        "existing_label_rows_by_dataset": _counts(existing_labels["dataset"]),
        "num_windows": int(len(windows)),
        "windows_by_class": _counts(windows["y"]),
        "windows_by_dataset": _counts(windows["dataset"]),
        "windows_by_dataset_and_class": {
            str(dataset): {str(label): int(count) for label, count in counts.items()}
            for dataset, counts in windows.groupby("dataset")["y"]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict(orient="index")
            .items()
        },
        "num_unique_videos_with_windows": int(windows["video_path"].nunique()),
        "num_unique_positive_videos_with_windows": int(
            windows.loc[windows["y"] == 1, "video_path"].nunique()
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(output_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv") -> None:
    prepare_windows.remote(output_csv=output_csv)
