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
    ):
        self.df = windows_df.reset_index(drop=True)
        self.feature_col = feature_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        feature_path = Path(row[self.feature_col])

        features = np.load(feature_path).astype(np.float32)
        if features.ndim == 1:
            features = features[None, :]
        elif features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        features = torch.from_numpy(features)
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        return features, label


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

