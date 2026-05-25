from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"

app = modal.App(f"{APP_NAME}-audit-new-sources")
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
def audit_new_sources() -> dict:
    import glob
    import json
    import os
    from pathlib import Path

    import pandas as pd

    from fall_anticipation_cv.data import (
        HORIZON_SEC,
        K_FRAMES,
        OBS_LEN,
        OBS_SEC,
        POSITIVE_LABEL,
        STRIDE,
        TARGET_FPS,
        dataset_root,
        get_video_info,
    )

    root = dataset_root(DATA_ROOT)
    labels_dir = root / "labels"
    label_map = pd.read_csv(labels_dir / "label2id.csv")
    id_to_label = dict(zip(label_map["id"], label_map["label"]))
    normalize = {"walking": "walk"}

    def first_existing(paths: list[Path]) -> Path:
        for path in paths:
            if path.exists():
                return path
        return paths[0]

    source_specs = [
        ("of-syn", labels_dir / "of-syn.csv", root / "of-syn"),
        (
            "OOPs",
            first_existing(
                [
                    labels_dir / "OOPs.csv",
                    labels_dir / "OOPS.csv",
                    labels_dir / "oops.csv",
                ]
            ),
            first_existing([root / "OOPs", root / "OOPS", root / "oops"]),
        ),
    ]

    def resolve_video(source_root: Path, rel_path: str) -> str | None:
        rel = str(rel_path).strip()
        candidates = []
        path = source_root / rel
        if path.suffix:
            candidates.append(path)
        else:
            for suffix in [".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"]:
                candidates.append(path.with_suffix(suffix))

        # Some OOPs labels use "falls/..." while the shared folder may also
        # expose files directly under OOPs. Try both layouts.
        if rel.startswith("falls/"):
            tail = rel.split("/", 1)[1]
            tail_path = source_root / tail
            for suffix in [".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"]:
                candidates.append(tail_path.with_suffix(suffix))

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        matches = glob.glob(str(source_root / f"{rel}.*"))
        if not matches and rel.startswith("falls/"):
            matches = glob.glob(str(source_root / f"{rel.split('/', 1)[1]}.*"))
        return matches[0] if matches else None

    results = {
        "config": {
            "target_fps": TARGET_FPS,
            "obs_len_frames": OBS_LEN,
            "obs_sec": OBS_SEC,
            "horizon_sec": HORIZON_SEC,
            "positive_label": POSITIVE_LABEL,
            "note": "fall rows whose onset is <= observation length cannot create positive anticipation windows",
        },
        "sources": {},
    }

    def estimate_row_windows(row, video_duration_sec: float) -> tuple[int, int]:
        label_name = str(row["label_name"])
        action_start = float(row["start"])
        n_sampled_frames = int((video_duration_sec * TARGET_FPS) + 0.999999)
        if n_sampled_frames <= OBS_LEN + K_FRAMES:
            return 0, 0

        action_start_sampled = int(round(action_start * TARGET_FPS))
        if action_start_sampled <= OBS_LEN:
            return 0, 0

        positive = 0
        negative = 0
        for target_frame in range(OBS_LEN, n_sampled_frames - K_FRAMES, STRIDE):
            if label_name == POSITIVE_LABEL:
                if target_frame >= action_start_sampled:
                    continue
                if target_frame < action_start_sampled <= target_frame + K_FRAMES:
                    positive += 1
                else:
                    negative += 1
            else:
                if target_frame + K_FRAMES > action_start_sampled:
                    continue
                negative += 1
        return negative, positive

    for source_name, csv_path, source_root in source_specs:
        if not csv_path.exists():
            results["sources"][source_name] = {"error": f"missing {csv_path}"}
            continue
        if not source_root.exists():
            results["sources"][source_name] = {"error": f"missing {source_root}"}
            continue

        labels = pd.read_csv(csv_path)
        labels["dataset"] = source_name
        labels["label_name"] = labels["label"].map(id_to_label).fillna(labels["label"])
        labels["label_name"] = labels["label_name"].astype(str).str.strip().str.lower()
        labels["label_name"] = labels["label_name"].replace(normalize)
        labels["video_path"] = labels["path"].apply(
            lambda rel_path: resolve_video(source_root, rel_path)
        )
        labels["video_exists"] = labels["video_path"].apply(
            lambda path: path is not None and os.path.exists(str(path))
        )
        labels["split_group"] = labels["path"].astype(str)

        existing = labels[labels["video_exists"]].copy()
        video_durations = existing.groupby("video_path")["end"].max().to_dict()

        video_records = []
        for video_path, group in existing.groupby("video_path"):
            fall_rows = group[group["label_name"] == POSITIVE_LABEL]
            earliest_fall_start = None
            if not fall_rows.empty:
                earliest_fall_start = float(fall_rows["start"].min())
            video_records.append(
                {
                    "video_path": video_path,
                    "source_path": str(group["path"].iloc[0]),
                    "duration_sec_from_labels": float(group["end"].max()),
                    "has_fall_label": bool(not fall_rows.empty),
                    "earliest_fall_start_sec": earliest_fall_start,
                    "fall_onset_after_obs": bool(
                        earliest_fall_start is not None
                        and earliest_fall_start > OBS_SEC
                    ),
                }
            )

        videos = pd.DataFrame(video_records)
        fall_label_rows = existing[existing["label_name"] == POSITIVE_LABEL].copy()
        early_fall_rows = fall_label_rows[
            fall_label_rows["start"].astype(float) <= OBS_SEC
        ]
        usable_fall_rows = fall_label_rows[
            fall_label_rows["start"].astype(float) > OBS_SEC
        ]

        per_row_counts = []
        for row in existing.to_dict(orient="records"):
            duration = float(video_durations.get(row["video_path"], row["end"]))
            negative, positive = estimate_row_windows(row, duration)
            per_row_counts.append(
                {
                    "video_path": row["video_path"],
                    "path": row["path"],
                    "label_name": row["label_name"],
                    "negative_windows": negative,
                    "positive_windows": positive,
                }
            )
        row_windows = pd.DataFrame(per_row_counts)

        if not row_windows.empty:
            # Mirror the training window deduplication key as closely as possible
            # without opening videos. For estimates, deduplicate only by source
            # video/path and class contribution.
            raw_negative_windows = int(row_windows["negative_windows"].sum())
            raw_positive_windows = int(row_windows["positive_windows"].sum())
        else:
            raw_negative_windows = 0
            raw_positive_windows = 0

        source_summary = {
            "csv_path": str(csv_path),
            "source_root": str(source_root),
            "label_rows": int(len(labels)),
            "existing_label_rows": int(len(existing)),
            "label_rows_by_name": _counts(labels["label_name"]),
            "existing_label_rows_by_name": _counts(existing["label_name"]),
            "raw_videos_existing": int(existing["video_path"].nunique()),
            "raw_videos_with_fall_label": int(
                existing.loc[
                    existing["label_name"] == POSITIVE_LABEL, "video_path"
                ].nunique()
            ),
            "raw_fall_videos_with_onset_after_obs": int(
                videos.loc[videos["fall_onset_after_obs"], "video_path"].nunique()
            )
            if not videos.empty
            else 0,
            "fall_label_rows": int(len(fall_label_rows)),
            "fall_label_rows_start_le_obs": int(len(early_fall_rows)),
            "fall_label_rows_start_gt_obs": int(len(usable_fall_rows)),
            "estimated_windows_total_before_dedup": int(
                raw_negative_windows + raw_positive_windows
            ),
            "estimated_windows_negative_before_dedup": raw_negative_windows,
            "estimated_windows_positive_before_dedup": raw_positive_windows,
            "videos_with_estimated_positive_windows": int(
                row_windows.loc[
                    row_windows["positive_windows"] > 0, "video_path"
                ].nunique()
            )
            if not row_windows.empty
            else 0,
            "estimated_windows_by_label_name_before_dedup": {}
            if row_windows.empty
            else {
                str(label): {
                    "negative": int(group["negative_windows"].sum()),
                    "positive": int(group["positive_windows"].sum()),
                    "total": int(
                        group["negative_windows"].sum()
                        + group["positive_windows"].sum()
                    ),
                }
                for label, group in row_windows.groupby("label_name")
            },
        }

        if not videos.empty:
            source_summary["fall_start_sec_summary"] = (
                {}
                if videos["earliest_fall_start_sec"].dropna().empty
                else {
                    "min": float(videos["earliest_fall_start_sec"].dropna().min()),
                    "median": float(
                        videos["earliest_fall_start_sec"].dropna().median()
                    ),
                    "max": float(videos["earliest_fall_start_sec"].dropna().max()),
                }
            )

        results["sources"][source_name] = source_summary

    print(json.dumps(results, indent=2))
    return results


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    timeout=60 * 30,
)
def audit_oops_fps(max_videos: int | None = None) -> dict:
    import json
    import os
    from pathlib import Path

    import pandas as pd

    from fall_anticipation_cv.data import (
        TARGET_FPS,
        dataset_root,
        first_existing_path,
        get_video_info,
        resolve_csv_video_path,
    )

    root = dataset_root(DATA_ROOT)
    labels_dir = root / "labels"
    csv_path = first_existing_path(
        [labels_dir / "OOPs.csv", labels_dir / "OOPS.csv", labels_dir / "oops.csv"]
    )
    oops_root = first_existing_path([root / "OOPs", root / "OOPS", root / "oops"])
    labels = pd.read_csv(csv_path)
    labels["video_path"] = labels["path"].apply(
        lambda path: resolve_csv_video_path(path, oops_root, alternate_prefixes=("falls",))
    )
    videos = [
        path
        for path in sorted(labels["video_path"].dropna().unique())
        if os.path.exists(path)
    ]
    if max_videos is not None:
        videos = videos[:max_videos]

    records = []
    for video_path in videos:
        info = get_video_info(video_path)
        if info is None:
            continue
        fps, n_frames = info
        records.append(
            {
                "video_path": video_path,
                "fps": float(fps),
                "n_frames": int(n_frames),
                "duration_sec": float(n_frames / fps),
                "sample_interval_to_10fps": float(max(1.0, fps / TARGET_FPS)),
            }
        )

    fps = pd.DataFrame(records)
    summary = {
        "num_videos_checked": int(len(videos)),
        "num_videos_with_valid_metadata": int(len(fps)),
        "fps_counts": {
            str(k): int(v)
            for k, v in fps["fps"].round(6).value_counts().sort_index().items()
        }
        if not fps.empty
        else {},
        "fps_summary": {}
        if fps.empty
        else {
            "min": float(fps["fps"].min()),
            "median": float(fps["fps"].median()),
            "max": float(fps["fps"].max()),
        },
        "sample_interval_summary": {}
        if fps.empty
        else {
            "min": float(fps["sample_interval_to_10fps"].min()),
            "median": float(fps["sample_interval_to_10fps"].median()),
            "max": float(fps["sample_interval_to_10fps"].max()),
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(action: str = "sources", max_videos: int | None = None) -> None:
    if action == "sources":
        audit_new_sources.remote()
    elif action == "oops-fps":
        audit_oops_fps.remote(max_videos=max_videos)
    else:
        raise ValueError("action must be 'sources' or 'oops-fps'")
