from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fall_anticipation_cv.models.temporal_transformer import TemporalTransformerClassifier


DEFAULT_POSE_PREDICTIVE_LOSS_WEIGHT = 0.2


@dataclass
class PosePredictiveOutput:
    logits: torch.Tensor
    predicted_future_pose: torch.Tensor
    loss: torch.Tensor | None = None
    classification_loss: torch.Tensor | None = None
    predictive_loss: torch.Tensor | None = None


class FuturePosePredictor(nn.Module):
    """Predict a future pose-feature sequence from observed pose features."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
        future_steps: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.future_steps = future_steps
        self.temporal_model = TemporalTransformerClassifier(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=d_model,
        )
        self.output_projection = nn.Linear(d_model, future_steps * input_dim)

    def forward(
        self,
        observed_pose: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.temporal_model(observed_pose, lengths)
        future = self.output_projection(context)
        return future.view(observed_pose.shape[0], self.future_steps, self.input_dim)


class PoseSeq2SeqPredictiveModel(nn.Module):
    """Pose anticipation model: observed pose -> future pose -> fall classifier.

    This is intentionally close to the fall-prediction paper's setup: the model
    first predicts future pose features, then classifies from the final predicted
    future pose state.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
        future_steps: int = 10,
        predictive_loss_weight: float = DEFAULT_POSE_PREDICTIVE_LOSS_WEIGHT,
        num_classes: int = 2,
    ):
        super().__init__()
        self.predictive_loss_weight = predictive_loss_weight
        self.future_predictor = FuturePosePredictor(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            future_steps=future_steps,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        observed_pose: torch.Tensor,
        future_pose: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> PosePredictiveOutput:
        predicted_future = self.future_predictor(observed_pose, lengths)
        logits = self.classifier(predicted_future[:, -1])

        classification_loss = None
        predictive_loss = None
        total_loss = None

        if labels is not None:
            classification_loss = F.cross_entropy(
                logits,
                labels,
                weight=class_weights,
            )
            total_loss = classification_loss

        if future_pose is not None:
            predictive_loss = F.mse_loss(predicted_future, future_pose)
            weighted_predictive_loss = self.predictive_loss_weight * predictive_loss
            total_loss = (
                weighted_predictive_loss
                if total_loss is None
                else total_loss + weighted_predictive_loss
            )

        return PosePredictiveOutput(
            logits=logits,
            predicted_future_pose=predicted_future,
            loss=total_loss,
            classification_loss=classification_loss,
            predictive_loss=predictive_loss,
        )
