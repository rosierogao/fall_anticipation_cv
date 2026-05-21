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
HORIZON_SEC = 1.0
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


def resolve_gmd_video_path(label_path: str, data_root: str | Path) -> str | None:
    gmd_root = Path(data_root) / "GMDCSA24"
    rel_path = str(label_path).replace("Subject_", "Subject ")
    candidate = gmd_root / f"{rel_path}.mp4"

    if candidate.exists():
        return str(candidate)

    matches = glob.glob(str(gmd_root / f"{rel_path}.*"))
    return matches[0] if matches else None


def load_gmd_labels(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    labels = pd.read_csv(data_root / "labels" / "GMDCSA24.csv")
    label_map = pd.read_csv(data_root / "labels" / "label2id.csv")
    id_to_label = dict(zip(label_map["id"], label_map["label"]))

    labels["label_name"] = labels["label"].map(id_to_label)
    labels["video_path"] = labels["path"].apply(
        lambda path: resolve_gmd_video_path(path, data_root)
    )
    labels["video_exists"] = labels["video_path"].apply(
        lambda path: path is not None and os.path.exists(path)
    )
    return labels


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

    fall_start_sampled = None
    if label_name == POSITIVE_LABEL:
        fall_start_original = int(float(row["start"]) * original_fps)
        fall_start_sampled = int(fall_start_original / sample_interval)

        if fall_start_sampled <= OBS_LEN:
            return []

    samples = []
    for target_frame in range(OBS_LEN, n_sampled_frames - K_FRAMES, STRIDE):
        if label_name == POSITIVE_LABEL:
            y = int(target_frame < fall_start_sampled <= target_frame + K_FRAMES)
            if target_frame >= fall_start_sampled:
                continue
        else:
            y = 0

        samples.append(
            {
                "video_path": str(video_path),
                "label_name": label_name,
                "subject": row["subject"],
                "cam": row["cam"],
                "dataset": row.get("dataset", None),
                "window_start": target_frame - OBS_LEN,
                "window_end": target_frame,
                "target_frame": target_frame,
                "obs_len": OBS_LEN,
                "k_frames": K_FRAMES,
                "target_fps": TARGET_FPS,
                "sample_interval": sample_interval,
                "y": y,
                "fall_start_frame": (
                    fall_start_sampled if fall_start_sampled is not None else np.nan
                ),
            }
        )

    return samples


def build_window_dataframe(labels: pd.DataFrame) -> pd.DataFrame:
    all_samples = []
    for _, row in labels.iterrows():
        all_samples.extend(build_windows_for_row(row))
    return pd.DataFrame(all_samples)


def split_by_subject(
    windows: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    val_random_state: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = windows["subject"]
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
        gss_val.split(train_val, train_val["y"], train_val["subject"])
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

