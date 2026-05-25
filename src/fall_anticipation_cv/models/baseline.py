import torch
import torch.nn as nn


class VideoCNNTransformerBaseline(nn.Module):
    """Frame CNN encoder followed by a temporal Transformer encoder."""

    def __init__(
        self,
        num_classes: int = 2,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        from fall_anticipation_cv.models.temporal_transformer import (
            TemporalTransformerClassifier,
        )

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.temporal_classifier = TemporalTransformerClassifier(
            input_dim=128,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        features = self.frame_encoder(x)
        features = features.view(batch_size, seq_len, 128)
        return self.temporal_classifier(features)
