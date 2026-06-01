from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TARGET_FPS = 10
OBS_SEC = 1.6
HORIZON_SEC = 1.0
STRIDE_SEC = 0.2

OBS_LEN = int(OBS_SEC * TARGET_FPS)
K_FRAMES = int(HORIZON_SEC * TARGET_FPS)
STRIDE = int(STRIDE_SEC * TARGET_FPS)

POSITIVE_LABEL = "fall"
NEGATIVE_LABELS = {
    "walk",
    "walking",
    "sitting",
    "standing",
    "sit_down",
    "stand_up",
    "lie_down",
    "lying",
    "other",
}
EXCLUDE_LABELS = {"fallen"}
DEFAULT_LE2I_FPS = 25.0
VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mov",
    ".mkv",
    ".wmv",
}

LABEL_NORMALIZATION = {
    "walking": "walk",
}


def dataset_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    nested_root = root / "final_project_dataset"

    if nested_root.exists():
        return nested_root

    return root


def resolve_gmd_video_path(label_path: str, data_root: str | Path) -> str | None:
    gmd_root = dataset_root(data_root) / "GMDCSA24"
    rel_path = str(label_path).replace("Subject_", "Subject ")
    candidate = gmd_root / f"{rel_path}.mp4"

    if candidate.exists():
        return str(candidate)

    matches = glob.glob(str(gmd_root / f"{rel_path}.*"))
    return matches[0] if matches else None


def first_existing_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def resolve_csv_video_path(
    label_path: str,
    source_root: str | Path,
    alternate_prefixes: tuple[str, ...] = (),
) -> str | None:
    source_root = Path(source_root)
    rel_path = str(label_path).strip()
    candidate_rel_paths = [rel_path]

    for prefix in alternate_prefixes:
        prefix = prefix.strip("/")
        if rel_path.startswith(f"{prefix}/"):
            candidate_rel_paths.append(rel_path.split("/", 1)[1])

    for candidate_rel_path in candidate_rel_paths:
        candidate = source_root / candidate_rel_path
        candidates = [candidate] if candidate.suffix else []
        candidates.extend(candidate.with_suffix(suffix) for suffix in VIDEO_EXTENSIONS)

        for path in candidates:
            if path.exists():
                return str(path)

        matches = sorted(
            path
            for path in source_root.glob(f"{candidate_rel_path}.*")
            if path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if matches:
            return str(matches[0])

    return None


def load_label_map(labels_dir: Path) -> dict:
    label_map = pd.read_csv(labels_dir / "label2id.csv")
    return dict(zip(label_map["id"], label_map["label"]))


def normalize_label_name(label_name: object) -> str:
    normalized = str(label_name).strip().lower()
    return LABEL_NORMALIZATION.get(normalized, normalized)


def load_gmd_labels(data_root: str | Path) -> pd.DataFrame:
    data_root = dataset_root(data_root)
    labels_dir = data_root / "labels"
    matched_csv = labels_dir / "GMDCSA24_matched.csv"
    raw_csv = labels_dir / "GMDCSA24.csv"

    labels = pd.read_csv(matched_csv if matched_csv.exists() else raw_csv)
    id_to_label = load_label_map(labels_dir)

    labels["label_name"] = labels["label"].map(id_to_label).apply(normalize_label_name)

    resolved_video_paths = labels["path"].apply(
        lambda path: resolve_gmd_video_path(path, data_root)
    )
    if "video_path" not in labels.columns:
        labels["video_path"] = resolved_video_paths
    else:
        labels["video_path"] = labels["video_path"].where(
            labels["video_path"].apply(
                lambda path: isinstance(path, str) and os.path.exists(path)
            ),
            resolved_video_paths,
        )

    labels["video_exists"] = labels["video_path"].apply(
        lambda path: path is not None and os.path.exists(path)
    )
    labels["split_group"] = labels["subject"].apply(
        lambda subject: f"GMDCSA24:{subject}"
    )
    return labels


def load_oops_labels(data_root: str | Path) -> pd.DataFrame:
    data_root = dataset_root(data_root)
    labels_dir = data_root / "labels"
    csv_path = first_existing_path(
        [
            labels_dir / "OOPs.csv",
            labels_dir / "OOPS.csv",
            labels_dir / "oops.csv",
        ]
    )
    oops_root = first_existing_path(
        [
            data_root / "OOPs",
            data_root / "OOPS",
            data_root / "oops",
        ]
    )

    if not csv_path.exists() or not oops_root.exists():
        return pd.DataFrame()

    labels = pd.read_csv(csv_path)
    id_to_label = load_label_map(labels_dir)
    labels["label_name"] = labels["label"].map(id_to_label).apply(normalize_label_name)
    labels["dataset"] = "OOPs"
    labels["video_path"] = labels["path"].apply(
        lambda path: resolve_csv_video_path(path, oops_root, alternate_prefixes=("falls",))
    )
    labels["video_exists"] = labels["video_path"].apply(
        lambda path: path is not None and os.path.exists(path)
    )
    labels["subject"] = labels["path"].astype(str)
    labels["split_group"] = labels["path"].astype(str).apply(lambda path: f"OOPs:{path}")
    return labels


def _caucafall_action_filename(label_path: str) -> str:
    stem = Path(str(label_path).strip()).stem
    # Label paths encode subject at the end, e.g. FallForwardS1.
    import re

    action = re.sub(r"S\d+$", "", stem)
    explicit_names = {
        "FallBackwards": "Fall backwards",
        "FallForward": "Fall forward",
        "FallLeft": "Fall left",
        "FallRight": "Fall right",
        "FallSitting": "Fall sitting",
        "Hop": "Hop",
        "Kneel": "Kneel",
        "Pickupobject": "Pick up object",
        "PickUpObject": "Pick up object",
        "SitDown": "Sit down",
        "Walk": "Walk",
    }
    if action in explicit_names:
        return explicit_names[action]

    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", action).strip()
    return spaced or action


def resolve_caucafall_video_path(label_path: str, data_root: str | Path) -> str | None:
    data_root = dataset_root(data_root)
    caucafall_root = first_existing_path(
        [
            data_root / "CAUCAFall" / "CAUCAFall" / "videos",
            data_root / "CAUCAFall" / "videos",
            data_root / "caucafall" / "videos",
        ]
    )
    if not caucafall_root.exists():
        return None

    subject_match = None
    import re

    path_stem = Path(str(label_path).strip()).stem
    match = re.search(r"S(\d+)$", path_stem)
    if match is not None:
        subject_match = match.group(1)

    subject_dirs = (
        [caucafall_root / f"Subject.{subject_match}"]
        if subject_match is not None
        else sorted(caucafall_root.glob("Subject.*"))
    )
    action_filename = _caucafall_action_filename(label_path)

    for subject_dir in subject_dirs:
        if not subject_dir.exists():
            continue
        for suffix in VIDEO_EXTENSIONS:
            candidate = subject_dir / f"{action_filename}{suffix}"
            if candidate.exists():
                return str(candidate)

        lower_action = action_filename.lower().replace(" ", "")
        matches = sorted(
            path
            for path in subject_dir.iterdir()
            if path.suffix.lower() in VIDEO_EXTENSIONS
            and path.stem.lower().replace(" ", "") == lower_action
        )
        if matches:
            return str(matches[0])

    return None


def load_caucafall_labels(data_root: str | Path) -> pd.DataFrame:
    data_root = dataset_root(data_root)
    labels_dir = data_root / "labels"
    csv_path = first_existing_path(
        [
            labels_dir / "caucafall.csv",
            labels_dir / "CAUCAFall.csv",
            labels_dir / "CAUCAFALL.csv",
        ]
    )
    if not csv_path.exists():
        return pd.DataFrame()

    labels = pd.read_csv(csv_path)
    id_to_label = load_label_map(labels_dir)
    labels["label_name"] = labels["label"].map(id_to_label).apply(normalize_label_name)
    labels["dataset"] = "caucafall"
    labels["video_path"] = labels["path"].apply(
        lambda path: resolve_caucafall_video_path(path, data_root)
    )
    labels["video_exists"] = labels["video_path"].apply(
        lambda path: isinstance(path, str) and os.path.exists(path)
    )
    labels["split_group"] = labels["subject"].apply(
        lambda subject: f"caucafall:{subject}"
    )
    return labels


def parse_le2i_annotation(annotation_path: str | Path) -> tuple[int, int] | None:
    lines = [
        line.strip()
        for line in Path(annotation_path).read_text().splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        return None

    try:
        return int(float(lines[0])), int(float(lines[1]))
    except ValueError:
        return None


def resolve_le2i_video_path(annotation_path: str | Path) -> str | None:
    annotation_path = Path(annotation_path)
    videos_dir = annotation_path.parent.parent / "Videos"
    if not videos_dir.exists():
        return None

    matches = sorted(
        path
        for path in videos_dir.glob(f"{annotation_path.stem}.*")
        if path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return str(matches[0]) if matches else None


def load_le2i_labels(
    data_root: str | Path,
    event_label: str = POSITIVE_LABEL,
) -> pd.DataFrame:
    le2i_root = dataset_root(data_root) / "le2i"
    if not le2i_root.exists():
        return pd.DataFrame()

    records = []
    annotation_paths = sorted(le2i_root.glob("**/Annotation_files/*.txt"))
    for annotation_path in annotation_paths:
        fall_bounds = parse_le2i_annotation(annotation_path)
        if fall_bounds is None:
            continue

        fall_start_frame, fall_end_frame = fall_bounds
        video_path = resolve_le2i_video_path(annotation_path)
        video_info = get_video_info(video_path) if video_path else None
        fps = video_info[0] if video_info is not None else DEFAULT_LE2I_FPS
        subset = annotation_path.parent.parent.name
        video_stem = annotation_path.stem
        event_label_name = normalize_label_name(event_label)
        event_start_frame = (
            fall_end_frame if event_label_name == "fallen" else fall_start_frame
        )

        records.append(
            {
                "path": f"{subset}/{video_stem}",
                "label": 2 if event_label_name == "fallen" else 1,
                "label_name": event_label_name,
                "start": event_start_frame / fps,
                "end": fall_end_frame / fps,
                "start_frame": event_start_frame,
                "end_frame": fall_end_frame,
                "original_fps": fps,
                "subject": f"le2i:{subset}:{video_stem}",
                "cam": 1,
                "dataset": "le2i",
                "video_path": video_path,
                "annotation_path": str(annotation_path),
                "split_group": f"le2i:{subset}:{video_stem}",
            }
        )

    labels = pd.DataFrame(records)
    if labels.empty:
        return labels

    labels["video_exists"] = labels["video_path"].apply(
        lambda path: path is not None and os.path.exists(path)
    )
    return labels


def load_all_labels(
    data_root: str | Path,
    include_le2i: bool = True,
    include_caucafall: bool = False,
    include_oops: bool = False,
    le2i_event_label: str = POSITIVE_LABEL,
) -> pd.DataFrame:
    label_frames = [load_gmd_labels(data_root)]
    if include_le2i:
        le2i_labels = load_le2i_labels(data_root, event_label=le2i_event_label)
        if not le2i_labels.empty:
            label_frames.append(le2i_labels)
    if include_caucafall:
        caucafall_labels = load_caucafall_labels(data_root)
        if not caucafall_labels.empty:
            label_frames.append(caucafall_labels)
    if include_oops:
        oops_labels = load_oops_labels(data_root)
        if not oops_labels.empty:
            label_frames.append(oops_labels)

    return pd.concat(label_frames, ignore_index=True, sort=False)


def get_video_info(video_path: str) -> tuple[float, int] | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or n_frames <= 0:
        return None

    return fps, n_frames


def build_windows_for_row(
    row: pd.Series,
    horizon_sec: float = HORIZON_SEC,
    positive_label: str = POSITIVE_LABEL,
    negative_labels: set[str] | None = None,
    exclude_labels: set[str] | None = None,
) -> list[dict]:
    k_frames = int(horizon_sec * TARGET_FPS)
    positive_label = normalize_label_name(positive_label)
    negative_labels = NEGATIVE_LABELS if negative_labels is None else negative_labels
    exclude_labels = EXCLUDE_LABELS if exclude_labels is None else exclude_labels
    label_name = str(row.get("label_name", "")).strip().lower()

    if label_name in {"", "nan"} or label_name in exclude_labels:
        return []

    if label_name != positive_label and label_name not in negative_labels:
        return []

    video_path = row.get("video_path", None)
    if video_path is None or pd.isna(video_path) or not os.path.exists(str(video_path)):
        return []

    video_info = get_video_info(str(video_path))
    if video_info is None:
        return []

    original_fps, n_original_frames = video_info
    sample_interval = max(1.0, original_fps / TARGET_FPS)
    n_sampled_frames = int(np.ceil(n_original_frames / sample_interval))

    if n_sampled_frames <= OBS_LEN + k_frames:
        return []

    action_start_sampled = None
    if label_name == positive_label:
        if "start_frame" in row and not pd.isna(row["start_frame"]):
            action_start_original = int(row["start_frame"])
        else:
            action_start_original = int(float(row["start"]) * original_fps)
        action_start_sampled = int(round(action_start_original / sample_interval))

        if action_start_sampled <= OBS_LEN:
            return []
    else:
        if "start" in row and not pd.isna(row["start"]):
            if "start_frame" in row and not pd.isna(row["start_frame"]):
                action_start_original = int(row["start_frame"])
            else:
                action_start_original = int(float(row["start"]) * original_fps)
            action_start_sampled = int(round(action_start_original / sample_interval))
            if action_start_sampled <= OBS_LEN:
                return []

    samples = []
    for target_frame in range(OBS_LEN, n_sampled_frames - k_frames, STRIDE):
        if label_name == positive_label:
            y = int(target_frame < action_start_sampled <= target_frame + k_frames)
            if target_frame >= action_start_sampled:
                continue
        else:
            if (
                action_start_sampled is not None
                and target_frame + k_frames > action_start_sampled
            ):
                continue
            y = 0

        samples.append(
            {
                "video_path": str(video_path),
                "label_name": label_name,
                "subject": row["subject"],
                "cam": row["cam"],
                "dataset": row.get("dataset", None),
                "split_group": row.get("split_group", row["subject"]),
                "window_start": target_frame - OBS_LEN,
                "window_end": target_frame,
                "target_frame": target_frame,
                "obs_len": OBS_LEN,
                "k_frames": k_frames,
                "target_fps": TARGET_FPS,
                "sample_interval": sample_interval,
                "y": y,
                "fall_start_frame": (
                    action_start_sampled if action_start_sampled is not None else np.nan
                ),
            }
        )

    return samples


def build_window_dataframe(
    labels: pd.DataFrame,
    horizon_sec: float = HORIZON_SEC,
    positive_label: str = POSITIVE_LABEL,
    exclude_labels: set[str] | None = None,
) -> pd.DataFrame:
    all_samples = []
    for _, row in labels.iterrows():
        all_samples.extend(
            build_windows_for_row(
                row,
                horizon_sec=horizon_sec,
                positive_label=positive_label,
                exclude_labels=exclude_labels,
            )
        )

    columns = [
        "video_path",
        "label_name",
        "subject",
        "cam",
        "dataset",
        "split_group",
        "window_start",
        "window_end",
        "target_frame",
        "obs_len",
        "k_frames",
        "target_fps",
        "sample_interval",
        "y",
        "fall_start_frame",
    ]
    windows = pd.DataFrame(all_samples, columns=columns)
    dedupe_columns = [
        "video_path",
        "window_start",
        "window_end",
        "target_frame",
        "y",
        "dataset",
    ]
    return windows.drop_duplicates(subset=dedupe_columns, keep="first").reset_index(
        drop=True
    )


def sample_oops_negative_windows(
    windows: pd.DataFrame,
    negative_to_positive_ratio: float = 3.0,
    random_state: int = 42,
) -> pd.DataFrame:
    oops_mask = windows["dataset"].astype(str).str.lower() == "oops"
    oops_positive = windows[oops_mask & (windows["y"] == 1)]
    oops_negative = windows[oops_mask & (windows["y"] == 0)]

    target_negative_count = int(round(len(oops_positive) * negative_to_positive_ratio))
    if (
        oops_negative.empty
        or len(oops_positive) == 0
        or len(oops_negative) <= target_negative_count
    ):
        return windows.reset_index(drop=True)

    label_names = sorted(oops_negative["label_name"].dropna().unique())
    remaining_quota = target_negative_count
    selected_parts = []
    rng = np.random.default_rng(random_state)

    for index, label_name in enumerate(label_names):
        label_windows = oops_negative[oops_negative["label_name"] == label_name]
        labels_left = len(label_names) - index
        label_quota = min(len(label_windows), int(np.ceil(remaining_quota / labels_left)))
        if label_quota <= 0:
            continue

        group_sizes = label_windows.groupby("video_path").size().sort_index()
        per_video_quota = max(1, int(np.ceil(label_quota / len(group_sizes))))
        label_parts = []
        for _, video_windows in label_windows.groupby("video_path", sort=True):
            take = min(len(video_windows), per_video_quota)
            label_parts.append(
                video_windows.sample(
                    n=take,
                    random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
            )

        sampled_label = pd.concat(label_parts, ignore_index=False)
        if len(sampled_label) > label_quota:
            sampled_label = sampled_label.sample(
                n=label_quota,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )

        selected_parts.append(sampled_label)
        remaining_quota -= len(sampled_label)

    selected_negative = pd.concat(selected_parts, ignore_index=False)
    selected_indices = set(selected_negative.index)
    keep_mask = (~oops_mask) | (windows["y"] == 1) | windows.index.isin(selected_indices)
    return windows[keep_mask].reset_index(drop=True)


def assign_group_splits(
    windows: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    from sklearn.model_selection import GroupShuffleSplit

    split_total = train_size + val_size + test_size
    if not np.isclose(split_total, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    group_column = "split_group" if "split_group" in windows.columns else "subject"
    split_windows = windows.copy()
    split_windows["split"] = ""

    for dataset_index, (_, dataset_windows) in enumerate(
        split_windows.groupby("dataset", sort=True)
    ):
        dataset_indices = dataset_windows.index.to_numpy()
        groups = dataset_windows[group_column]
        unique_groups = groups.nunique()

        if unique_groups < 3:
            split_windows.loc[dataset_indices, "split"] = "train"
            continue

        test_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state + dataset_index,
        )
        train_val_pos, test_pos = next(
            test_splitter.split(dataset_windows, dataset_windows["y"], groups)
        )

        train_val = dataset_windows.iloc[train_val_pos]
        val_fraction_of_train_val = val_size / (train_size + val_size)
        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_fraction_of_train_val,
            random_state=random_state + 100 + dataset_index,
        )
        train_pos, val_pos = next(
            val_splitter.split(
                train_val,
                train_val["y"],
                train_val[group_column],
            )
        )

        split_windows.loc[train_val.iloc[train_pos].index, "split"] = "train"
        split_windows.loc[train_val.iloc[val_pos].index, "split"] = "val"
        split_windows.loc[dataset_windows.iloc[test_pos].index, "split"] = "test"

    return split_windows.reset_index(drop=True)


def validate_windows(windows: pd.DataFrame) -> None:
    if windows.empty:
        raise ValueError(
            "No training windows were created. Check that video paths exist under "
            "the dataset root, labels include supported label names, and fall clips "
            "start after the observation window."
        )

    required_columns = {
        "video_path",
        "window_start",
        "window_end",
        "obs_len",
        "sample_interval",
        "y",
        "subject",
    }
    missing_columns = required_columns.difference(windows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Window metadata is missing required columns: {missing}")


def split_by_subject(
    windows: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    val_random_state: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import GroupShuffleSplit

    if "split" in windows.columns:
        train = windows[windows["split"] == "train"].reset_index(drop=True)
        val = windows[windows["split"] == "val"].reset_index(drop=True)
        test = windows[windows["split"] == "test"].reset_index(drop=True)
        if not train.empty and not val.empty and not test.empty:
            return train, val, test

    group_column = "split_group" if "split_group" in windows.columns else "subject"
    groups = windows[group_column]
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_val_idx, test_idx = next(gss.split(windows, windows["y"], groups))

    train_val = windows.iloc[train_val_idx].reset_index(drop=True)
    test = windows.iloc[test_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=val_random_state,
    )
    train_idx, val_idx = next(
        gss_val.split(train_val, train_val["y"], train_val[group_column])
    )

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    return train, val, test


def load_video_window(
    video_path: str,
    window_start: int,
    window_end: int,
    sample_interval: float,
    resize: tuple[int, int] = (224, 224),
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []

    for sampled_idx in range(window_start, window_end):
        original_idx = int(round(sampled_idx * sample_interval))
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_idx)
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame.astype(np.float32) / 255.0)

    cap.release()
    return frames


class FallWindowDataset:
    def __init__(self, windows_df: pd.DataFrame, resize: tuple[int, int] = (224, 224)):
        self.df = windows_df.reset_index(drop=True)
        self.resize = resize

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        row = self.df.iloc[idx]
        frames = load_video_window(
            video_path=row["video_path"],
            window_start=int(row["window_start"]),
            window_end=int(row["window_end"]),
            sample_interval=float(row["sample_interval"]),
            resize=self.resize,
        )

        if len(frames) < row["obs_len"]:
            if len(frames) == 0:
                frames = [
                    np.zeros((self.resize[1], self.resize[0], 3), dtype=np.float32)
                ]

            while len(frames) < row["obs_len"]:
                frames.append(frames[-1].copy())

        frames = frames[: int(row["obs_len"])]
        video = np.stack(frames, axis=0)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        label = torch.tensor(row["y"], dtype=torch.long)

        return video, label
