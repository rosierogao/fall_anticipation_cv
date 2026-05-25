import torch
import torch.nn as nn


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
