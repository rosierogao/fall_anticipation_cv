from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PoseWindowDataset(Dataset):
    """Dataset for pre-extracted pose feature windows.

    The metadata CSV must contain a label column and a path column pointing to a
    NumPy feature file. Supported feature shapes are [T, F] and [T, J, C].
    """

    def __init__(
        self,
        windows_df: pd.DataFrame,
        feature_col: str = "pose_feature_path",
        label_col: str = "y",
        normalize: bool = True,
        add_velocity: bool = True,
    ):
        self.df = windows_df.reset_index(drop=True)
        self.feature_col = feature_col
        self.label_col = label_col
        self.normalize = normalize
        self.add_velocity = add_velocity

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        feature_path = Path(row[self.feature_col])

        features = np.load(feature_path).astype(np.float32)
        features = prepare_pose_features(
            features,
            normalize=self.normalize,
            add_velocity=self.add_velocity,
        )

        features = torch.from_numpy(features)
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        return features, label


def prepare_pose_features(
    features: np.ndarray,
    normalize: bool = True,
    add_velocity: bool = True,
) -> np.ndarray:
    """Prepare pose sequences for temporal modeling.

    RTMPose stores each frame as keypoints with [x, y, confidence]. The raw x/y
    coordinates are camera-position dependent, so by default we center each
    frame on the detected person and scale by the person's pose bounding box.
    """

    features = features.astype(np.float32)
    if features.ndim == 1:
        features = features[None, :]

    if features.ndim == 2:
        if features.shape[1] % 3 != 0:
            return features
        pose = features.reshape(features.shape[0], features.shape[1] // 3, 3)
    elif features.ndim == 3 and features.shape[-1] >= 3:
        pose = features[..., :3]
    else:
        return features.reshape(features.shape[0], -1)

    coords = pose[..., :2].copy()
    confidence = pose[..., 2:3].copy()

    if normalize:
        coords = _normalize_pose_coordinates(coords, confidence[..., 0])

    prepared = np.concatenate([coords, confidence], axis=-1)
    if add_velocity:
        velocity = np.zeros_like(coords)
        velocity[1:] = coords[1:] - coords[:-1]
        prepared = np.concatenate([prepared, velocity], axis=-1)

    return prepared.reshape(prepared.shape[0], -1).astype(np.float32)


def _normalize_pose_coordinates(
    coords: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.05,
) -> np.ndarray:
    normalized = np.zeros_like(coords, dtype=np.float32)

    for frame_idx in range(coords.shape[0]):
        frame_coords = coords[frame_idx]
        frame_conf = confidence[frame_idx]
        valid = frame_conf > confidence_threshold
        if not np.any(valid):
            continue

        valid_coords = frame_coords[valid]
        center = _person_center(frame_coords, frame_conf, valid_coords)
        centered = frame_coords - center

        span = valid_coords.max(axis=0) - valid_coords.min(axis=0)
        scale = float(max(span.max(), 1.0))
        normalized[frame_idx] = centered / scale
        normalized[frame_idx, ~valid] = 0.0

    return normalized


def _person_center(
    frame_coords: np.ndarray,
    frame_conf: np.ndarray,
    valid_coords: np.ndarray,
) -> np.ndarray:
    # COCO/RTMPose body keypoint convention: left/right shoulders 5/6, hips 11/12.
    torso_indices = [5, 6, 11, 12]
    torso_valid = [
        idx
        for idx in torso_indices
        if idx < len(frame_conf) and frame_conf[idx] > 0.05
    ]
    if torso_valid:
        return frame_coords[torso_valid].mean(axis=0)
    return valid_coords.mean(axis=0)


def collate_pose_windows(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    feature_dim = sequences[0].shape[1]
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(sequences), max_len, feature_dim, dtype=torch.float32)
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.shape[0]] = seq

    return padded, torch.stack(labels), lengths
