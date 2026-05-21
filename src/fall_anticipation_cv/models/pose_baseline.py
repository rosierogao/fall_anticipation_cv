import torch
import torch.nn as nn


class PoseGRUBaseline(nn.Module):
    """GRU classifier for pre-extracted pose feature sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            _, hidden = self.encoder(x)
        else:
            lengths = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.encoder(packed)

        return self.classifier(hidden[-1])


class PoseTransformerBaseline(nn.Module):
    """Transformer classifier for pre-extracted pose feature sequences."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        from fall_anticipation_cv.models.temporal_transformer import (
            TemporalTransformerClassifier,
        )

        self.temporal_classifier = TemporalTransformerClassifier(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.temporal_classifier(x, lengths)

