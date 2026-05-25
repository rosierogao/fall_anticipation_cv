from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from fall_anticipation_cv.pose_data import prepare_pose_features


class PoseVJEPALatentWindowDataset(Dataset):
    """Dataset for windows with both RTMPose and V-JEPA features."""

    def __init__(
        self,
        windows_df: pd.DataFrame,
        pose_feature_col: str = "pose_feature_path",
        vjepa_feature_col: str = "vjepa_feature_path",
        label_col: str = "y",
        normalize_pose: bool = True,
        add_velocity: bool = True,
    ):
        self.df = windows_df.reset_index(drop=True)
        self.pose_feature_col = pose_feature_col
        self.vjepa_feature_col = vjepa_feature_col
        self.label_col = label_col
        self.normalize_pose = normalize_pose
        self.add_velocity = add_velocity

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        pose_features = prepare_pose_features(
            np.load(Path(row[self.pose_feature_col])),
            normalize=self.normalize_pose,
            add_velocity=self.add_velocity,
        )
        vjepa_features = np.load(Path(row[self.vjepa_feature_col]))["observed_latents"]

        pose = torch.from_numpy(pose_features.astype(np.float32))
        vjepa = torch.from_numpy(vjepa_features.astype(np.float32))
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        return pose, label, vjepa


def collate_pose_vjepa_windows(batch):
    poses, labels, vjepa_latents = zip(*batch)
    lengths = torch.tensor([pose.shape[0] for pose in poses], dtype=torch.long)
    pose_dim = poses[0].shape[1]
    vjepa_dim = vjepa_latents[0].shape[1]
    max_len = int(lengths.max().item())

    pose_padded = torch.zeros(len(poses), max_len, pose_dim, dtype=torch.float32)
    vjepa_padded = torch.zeros(len(poses), max_len, vjepa_dim, dtype=torch.float32)

    for idx, (pose, vjepa) in enumerate(zip(poses, vjepa_latents)):
        target_len = pose.shape[0]
        pose_padded[idx, :target_len] = pose
        vjepa_padded[idx, :target_len] = _resize_temporal_sequence(vjepa, target_len)

    return pose_padded, torch.stack(labels), vjepa_padded, lengths


def _resize_temporal_sequence(sequence: torch.Tensor, target_len: int) -> torch.Tensor:
    if sequence.shape[0] == target_len:
        return sequence
    if sequence.shape[0] == 1:
        return sequence.expand(target_len, -1)

    resized = F.interpolate(
        sequence.T.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(0).T
