from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"

app = modal.App(f"{APP_NAME}-prepare-fallen")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "opencv-python-headless", "pandas", "scikit-learn")
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
def prepare_fallen_windows(
    output_csv: str = f"{DATASET_ROOT}/windows_fallen_state_gmd_le2i_horizon2s.csv",
    horizon_sec: float = 2.0,
    sample_seed: int = 42,
    assign_splits: bool = True,
    include_caucafall: bool = False,
) -> dict:
    import json
    from pathlib import Path

    from fall_anticipation_cv.data import (
        assign_group_splits,
        build_window_dataframe,
        load_all_labels,
        validate_windows,
    )

    labels = load_all_labels(
        DATA_ROOT,
        include_le2i=True,
        include_caucafall=include_caucafall,
        include_oops=False,
        le2i_event_label="fallen",
    )
    existing_labels = labels[labels["video_exists"]].copy()
    windows = build_window_dataframe(
        labels,
        horizon_sec=horizon_sec,
        positive_label="fallen",
        exclude_labels={"fall"},
    )
    if assign_splits:
        windows = assign_group_splits(windows, random_state=sample_seed)
    validate_windows(windows)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_path, index=False)
    volume.commit()

    summary = {
        "output_csv": output_csv,
        "task": "fallen_state_prediction",
        "positive_label": "fallen",
        "le2i_positive_event": "fall_end_frame",
        "excluded_labels": ["fall"],
        "horizon_sec": horizon_sec,
        "sample_seed": sample_seed,
        "assign_splits": assign_splits,
        "include_caucafall": include_caucafall,
        "num_label_rows": int(len(labels)),
        "num_existing_label_rows": int(len(existing_labels)),
        "label_rows_by_dataset": _counts(labels["dataset"]),
        "existing_label_rows_by_dataset": _counts(existing_labels["dataset"]),
        "label_rows_by_dataset_and_label": {
            str(dataset): {str(label): int(count) for label, count in counts.items()}
            for dataset, counts in labels.groupby("dataset")["label_name"]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict(orient="index")
            .items()
        },
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
    if "split" in windows.columns:
        summary["windows_by_split_and_class"] = {
            str(split): {str(label): int(count) for label, count in counts.items()}
            for split, counts in windows.groupby("split")["y"]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict(orient="index")
            .items()
        }
        summary["windows_by_dataset_split_and_class"] = {
            f"{dataset}:{split}": {
                str(label): int(count) for label, count in counts.items()
            }
            for (dataset, split), counts in windows.groupby(["dataset", "split"])[
                "y"
            ]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict(orient="index")
            .items()
        }

    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(
    output_csv: str = f"{DATASET_ROOT}/windows_fallen_state_gmd_le2i_horizon2s.csv",
    horizon_sec: float = 2.0,
    sample_seed: int = 42,
    assign_splits: bool = True,
    include_caucafall: bool = False,
) -> None:
    prepare_fallen_windows.remote(
        output_csv=output_csv,
        horizon_sec=horizon_sec,
        sample_seed=sample_seed,
        assign_splits=assign_splits,
        include_caucafall=include_caucafall,
    )
