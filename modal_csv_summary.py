from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
VOLUME_NAME = "final_project_dataset"

app = modal.App(f"{APP_NAME}-csv-summary")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas<3.0")


def _counts(df, columns: list[str]) -> dict:
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return {}
    grouped = df.groupby(existing, dropna=False).size().reset_index(name="count")
    return {
        "|".join(str(row[column]) for column in existing): int(row["count"])
        for _, row in grouped.iterrows()
    }


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 5)
def summarize_csv(path: str) -> dict:
    import json

    import pandas as pd

    df = pd.read_csv(path)
    summary = {
        "path": path,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "by_dataset": _counts(df, ["dataset"]),
        "by_dataset_split": _counts(df, ["dataset", "split"]),
        "by_class": _counts(df, ["y"]),
        "by_dataset_class": _counts(df, ["dataset", "y"]),
        "by_dataset_split_class": _counts(df, ["dataset", "split", "y"]),
        "by_split_class": _counts(df, ["split", "y"]),
        "unique_videos": int(df["video_path"].nunique()) if "video_path" in df else None,
        "duplicate_window_keys": None,
    }
    key_columns = [
        column
        for column in ["video_path", "window_start", "window_end", "y"]
        if column in df.columns
    ]
    if key_columns:
        summary["duplicate_window_keys"] = int(df.duplicated(key_columns).sum())

    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(path: str) -> None:
    summarize_csv.remote(path)
