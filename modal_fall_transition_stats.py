from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"

app = modal.App(f"{APP_NAME}-fall-transition-stats")
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


def _stats(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}

    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "pct_le_1s": float((arr <= 1.0).mean()),
        "pct_le_2s": float((arr <= 2.0).mean()),
        "pct_gt_2s": float((arr > 2.0).mean()),
    }


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    timeout=60 * 30,
)
def fall_transition_stats() -> dict:
    import json

    import pandas as pd

    from fall_anticipation_cv.data import (
        load_caucafall_labels,
        load_gmd_labels,
        load_le2i_labels,
    )

    gmd = load_gmd_labels(DATA_ROOT)
    gmd_existing = gmd[gmd["video_exists"]].copy()
    gmd_pairs = []
    for video_path, group in gmd_existing.groupby("video_path"):
        fall_rows = group[group["label_name"] == "fall"].sort_values("start")
        fallen_rows = group[group["label_name"] == "fallen"].sort_values("start")
        if fall_rows.empty or fallen_rows.empty:
            continue

        for _, fallen_row in fallen_rows.iterrows():
            prior_falls = fall_rows[fall_rows["start"] <= fallen_row["start"]]
            if prior_falls.empty:
                continue
            fall_row = prior_falls.iloc[-1]
            duration = float(fallen_row["start"]) - float(fall_row["start"])
            if duration >= 0:
                gmd_pairs.append(
                    {
                        "video_path": video_path,
                        "fall_start": float(fall_row["start"]),
                        "fallen_start": float(fallen_row["start"]),
                        "duration_sec": duration,
                    }
                )

    le2i = load_le2i_labels(DATA_ROOT)
    le2i_existing = le2i[le2i["video_exists"]].copy()
    le2i_durations = (
        (le2i_existing["end_frame"] - le2i_existing["start_frame"])
        / le2i_existing["original_fps"]
    ).astype(float)

    caucafall = load_caucafall_labels(DATA_ROOT)
    caucafall_existing = caucafall[caucafall["video_exists"]].copy()
    caucafall_label_counts = (
        caucafall_existing["label_name"].value_counts().sort_index().to_dict()
        if not caucafall_existing.empty
        else {}
    )
    caucafall_pairs = []
    if not caucafall_existing.empty and "fallen" in set(caucafall_existing["label_name"]):
        for video_path, group in caucafall_existing.groupby("video_path"):
            fall_rows = group[group["label_name"] == "fall"].sort_values("start")
            fallen_rows = group[group["label_name"] == "fallen"].sort_values("start")
            if fall_rows.empty or fallen_rows.empty:
                continue
            for _, fallen_row in fallen_rows.iterrows():
                prior_falls = fall_rows[fall_rows["start"] <= fallen_row["start"]]
                if prior_falls.empty:
                    continue
                fall_row = prior_falls.iloc[-1]
                duration = float(fallen_row["start"]) - float(fall_row["start"])
                if duration >= 0:
                    caucafall_pairs.append(duration)
    else:
        fall_rows = caucafall_existing[caucafall_existing["label_name"] == "fall"]
        caucafall_pairs = (fall_rows["end"] - fall_rows["start"]).astype(float).tolist()

    result = {
        "definition": {
            "GMDCSA24": "fallen.start - nearest prior fall.start within the same video",
            "LE2I": "(fall_end_frame - fall_start_frame) / video_fps",
            "CAUCAFall": (
                "fallen.start - fall.start if explicit fallen rows exist; otherwise "
                "fall row end - start as a transition-duration proxy"
            ),
        },
        "GMDCSA24": {
            "num_existing_label_rows": int(len(gmd_existing)),
            "label_counts": gmd_existing["label_name"].value_counts().sort_index().to_dict(),
            "num_paired_fall_to_fallen_events": int(len(gmd_pairs)),
            "duration_seconds": _stats([row["duration_sec"] for row in gmd_pairs]),
            "shortest_examples": sorted(gmd_pairs, key=lambda row: row["duration_sec"])[:5],
            "longest_examples": sorted(gmd_pairs, key=lambda row: row["duration_sec"], reverse=True)[:5],
        },
        "LE2I": {
            "num_existing_label_rows": int(len(le2i_existing)),
            "duration_seconds": _stats(le2i_durations.tolist()),
            "fps_counts": le2i_existing["original_fps"].round(3).value_counts().sort_index().to_dict(),
        },
        "CAUCAFall": {
            "num_existing_label_rows": int(len(caucafall_existing)),
            "label_counts": caucafall_label_counts,
            "uses_proxy_fall_segment_duration": "fallen" not in set(caucafall_existing["label_name"]),
            "duration_seconds": _stats(caucafall_pairs),
        },
    }

    all_values = []
    all_values.extend([row["duration_sec"] for row in gmd_pairs])
    all_values.extend(le2i_durations.tolist())
    all_values.extend(caucafall_pairs)
    result["combined"] = {"duration_seconds": _stats(all_values)}

    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main() -> None:
    fall_transition_stats.remote()
