import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


TARGET_FPS = 10
OBS_SEC = 1.6
HORIZON_SEC = 0.5
STRIDE_SEC = 0.2

OBS_LEN = int(OBS_SEC * TARGET_FPS)
K_FRAMES = int(HORIZON_SEC * TARGET_FPS)
STRIDE = int(STRIDE_SEC * TARGET_FPS)

POSITIVE_LABEL = "fall"
NEGATIVE_LABELS = {
    "walk",
    "sitting",
    "standing",
    "sit_down",
    "stand_up",
    "lie_down",
    "lying",
    "other",
}
EXCLUDE_LABELS = {"fallen"}
LE2I_FPS = 25.0


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


def load_gmd_labels(data_root: str | Path) -> pd.DataFrame:
    data_root = dataset_root(data_root)
    labels_dir = data_root / "labels"
    matched_csv = labels_dir / "GMDCSA24_matched.csv"
    raw_csv = labels_dir / "GMDCSA24.csv"

    labels = pd.read_csv(matched_csv if matched_csv.exists() else raw_csv)
    label_map = pd.read_csv(labels_dir / "label2id.csv")
    id_to_label = dict(zip(label_map["id"], label_map["label"]))

    labels["label_name"] = labels["label"].map(id_to_label)

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

    matches = sorted(videos_dir.glob(f"{annotation_path.stem}.*"))
    return str(matches[0]) if matches else None


def load_le2i_labels(data_root: str | Path) -> pd.DataFrame:
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
        subset = annotation_path.parent.parent.name
        video_stem = annotation_path.stem

        records.append(
            {
                "path": f"{subset}/{video_stem}",
                "label": 1,
                "label_name": POSITIVE_LABEL,
                "start": fall_start_frame / LE2I_FPS,
                "end": fall_end_frame / LE2I_FPS,
                "start_frame": fall_start_frame,
                "end_frame": fall_end_frame,
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


def load_all_labels(data_root: str | Path, include_le2i: bool = True) -> pd.DataFrame:
    label_frames = [load_gmd_labels(data_root)]
    if include_le2i:
        le2i_labels = load_le2i_labels(data_root)
        if not le2i_labels.empty:
            label_frames.append(le2i_labels)

    return pd.concat(label_frames, ignore_index=True, sort=False)


def get_video_info(video_path: str) -> tuple[float, int] | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or n_frames <= 0:
        return None

    return fps, n_frames


def build_windows_for_row(row: pd.Series) -> list[dict]:
    label_name = str(row.get("label_name", "")).strip().lower()

    if label_name in {"", "nan"} or label_name in EXCLUDE_LABELS:
        return []

    if label_name != POSITIVE_LABEL and label_name not in NEGATIVE_LABELS:
        return []

    video_path = row.get("video_path", None)
    if video_path is None or pd.isna(video_path) or not os.path.exists(str(video_path)):
        return []

    video_info = get_video_info(str(video_path))
    if video_info is None:
        return []

    original_fps, n_original_frames = video_info
    sample_interval = max(1, int(round(original_fps / TARGET_FPS)))
    n_sampled_frames = int(np.ceil(n_original_frames / sample_interval))

    if n_sampled_frames <= OBS_LEN + K_FRAMES:
        return []

    action_start_sampled = None
    if label_name == POSITIVE_LABEL:
        if "start_frame" in row and not pd.isna(row["start_frame"]):
            action_start_original = int(row["start_frame"])
        else:
            action_start_original = int(float(row["start"]) * original_fps)
        action_start_sampled = int(action_start_original / sample_interval)

        if action_start_sampled <= OBS_LEN:
            return []
    else:
        if "start" in row and not pd.isna(row["start"]):
            if "start_frame" in row and not pd.isna(row["start_frame"]):
                action_start_original = int(row["start_frame"])
            else:
                action_start_original = int(float(row["start"]) * original_fps)
            action_start_sampled = int(action_start_original / sample_interval)
            if action_start_sampled <= OBS_LEN:
                return []

    samples = []
    for target_frame in range(OBS_LEN, n_sampled_frames - K_FRAMES, STRIDE):
        if label_name == POSITIVE_LABEL:
            y = int(target_frame < action_start_sampled <= target_frame + K_FRAMES)
            if target_frame >= action_start_sampled:
                continue
        else:
            if (
                action_start_sampled is not None
                and target_frame + K_FRAMES > action_start_sampled
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
                "k_frames": K_FRAMES,
                "target_fps": TARGET_FPS,
                "sample_interval": sample_interval,
                "y": y,
                "fall_start_frame": (
                    action_start_sampled if action_start_sampled is not None else np.nan
                ),
            }
        )

    return samples


def build_window_dataframe(labels: pd.DataFrame) -> pd.DataFrame:
    all_samples = []
    for _, row in labels.iterrows():
        all_samples.extend(build_windows_for_row(row))

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
    return pd.DataFrame(all_samples, columns=columns)


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
    sample_interval: int,
    resize: tuple[int, int] = (224, 224),
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []

    for sampled_idx in range(window_start, window_end):
        original_idx = int(sampled_idx * sample_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_idx)
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame.astype(np.float32) / 255.0)

    cap.release()
    return frames


class FallWindowDataset(Dataset):
    def __init__(self, windows_df: pd.DataFrame, resize: tuple[int, int] = (224, 224)):
        self.df = windows_df.reset_index(drop=True)
        self.resize = resize

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        frames = load_video_window(
            video_path=row["video_path"],
            window_start=int(row["window_start"]),
            window_end=int(row["window_end"]),
            sample_interval=int(row["sample_interval"]),
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
