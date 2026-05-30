from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fall_anticipation_cv.pose_data import prepare_pose_features


class PosePredictiveWindowDataset(Dataset):
    """Dataset for observed pose windows with future-pose supervision."""

    def __init__(
        self,
        windows_df: pd.DataFrame,
        feature_col: str = "pose_predictive_feature_path",
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

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        features = np.load(Path(row[self.feature_col]))
        observed = prepare_pose_features(
            features["observed_pose"].astype(np.float32),
            normalize=self.normalize,
            add_velocity=self.add_velocity,
        )
        future = prepare_pose_features(
            features["future_pose"].astype(np.float32),
            normalize=self.normalize,
            add_velocity=self.add_velocity,
        )
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        return torch.from_numpy(observed), label, torch.from_numpy(future)


def collate_pose_predictive_windows(batch):
    observed, labels, future = zip(*batch)
    observed_lengths = torch.tensor([item.shape[0] for item in observed], dtype=torch.long)
    future_lengths = torch.tensor([item.shape[0] for item in future], dtype=torch.long)

    feature_dim = observed[0].shape[1]
    max_obs_len = int(observed_lengths.max().item())
    max_future_len = int(future_lengths.max().item())

    observed_padded = torch.zeros(
        len(observed),
        max_obs_len,
        feature_dim,
        dtype=torch.float32,
    )
    future_padded = torch.zeros(
        len(future),
        max_future_len,
        feature_dim,
        dtype=torch.float32,
    )

    for idx, item in enumerate(observed):
        observed_padded[idx, : item.shape[0]] = item
    for idx, item in enumerate(future):
        future_padded[idx, : item.shape[0]] = item

    return observed_padded, torch.stack(labels), future_padded, observed_lengths
