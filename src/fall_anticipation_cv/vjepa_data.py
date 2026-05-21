from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VJEPALatentWindowDataset(Dataset):
    """Dataset for pre-extracted V-JEPA latent anticipation windows."""

    def __init__(
        self,
        windows_df: pd.DataFrame,
        feature_col: str = "vjepa_feature_path",
        label_col: str = "y",
    ):
        self.df = windows_df.reset_index(drop=True)
        self.feature_col = feature_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feature_path = Path(row[self.feature_col])
        features = np.load(feature_path)

        observed = torch.from_numpy(
            features["observed_latents"].astype(np.float32)
        )
        future = torch.from_numpy(features["future_latents"].astype(np.float32))
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        return observed, label, future


def collate_vjepa_latent_windows(batch):
    observed, labels, future = zip(*batch)
    observed_lengths = torch.tensor([item.shape[0] for item in observed], dtype=torch.long)
    future_lengths = torch.tensor([item.shape[0] for item in future], dtype=torch.long)

    latent_dim = observed[0].shape[1]
    max_obs_len = int(observed_lengths.max().item())
    max_future_len = int(future_lengths.max().item())

    observed_padded = torch.zeros(
        len(observed),
        max_obs_len,
        latent_dim,
        dtype=torch.float32,
    )
    future_padded = torch.zeros(
        len(future),
        max_future_len,
        latent_dim,
        dtype=torch.float32,
    )

    for idx, item in enumerate(observed):
        observed_padded[idx, : item.shape[0]] = item
    for idx, item in enumerate(future):
        future_padded[idx, : item.shape[0]] = item

    return observed_padded, torch.stack(labels), future_padded, observed_lengths
