import torch
import torch.nn as nn

from fall_anticipation_cv.models.temporal_transformer import (
    TemporalTransformerClassifier,
)


class PoseVJEPAFusionTransformer(nn.Module):
    """Project pose and V-JEPA tokens to 256 dims, concatenate, classify."""

    def __init__(
        self,
        pose_dim: int,
        vjepa_dim: int,
        projection_dim: int = 256,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.pose_projection = nn.Linear(pose_dim, projection_dim)
        self.vjepa_projection = nn.Linear(vjepa_dim, projection_dim)
        self.classifier = TemporalTransformerClassifier(
            input_dim=projection_dim * 2,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(
        self,
        pose_features: torch.Tensor,
        vjepa_latents: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pose_tokens = self.pose_projection(pose_features)
        vjepa_tokens = self.vjepa_projection(vjepa_latents)
        fused_tokens = torch.cat([pose_tokens, vjepa_tokens], dim=-1)
        return self.classifier(fused_tokens, lengths)
