from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fall_anticipation_cv.models.temporal_transformer import (
    TemporalTransformerClassifier,
)


DEFAULT_PREDICTIVE_LOSS_WEIGHT = 0.2


@dataclass
class VJEPAAnticipationOutput:
    logits: torch.Tensor
    predicted_future_latent: torch.Tensor
    loss: torch.Tensor | None = None
    classification_loss: torch.Tensor | None = None
    predictive_loss: torch.Tensor | None = None


class FutureLatentPredictor(nn.Module):
    """Predict future V-JEPA latent states from observed latent states."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
        future_steps: int = 1,
    ):
        super().__init__()
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
        self.input_dim = input_dim

    def forward(
        self,
        observed_latents: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.temporal_model(observed_latents, lengths)
        future = self.output_projection(context)
        return future.view(observed_latents.shape[0], self.future_steps, self.input_dim)


class VJEPALatentPredictiveBaseline(nn.Module):
    """Anticipation model over frozen V-JEPA latents with a joint objective.

    Inputs are precomputed or frozen-encoder V-JEPA latent sequences with shape
    [B, T, D]. If a raw V-JEPA encoder is used, keep it outside this module and
    pass its pooled temporal latents here.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
        future_steps: int = 1,
        predictive_loss_weight: float = DEFAULT_PREDICTIVE_LOSS_WEIGHT,
        predictive_loss: str = "cosine",
        num_classes: int = 2,
    ):
        super().__init__()
        self.predictive_loss_weight = predictive_loss_weight
        self.predictive_loss = predictive_loss
        self.classifier = TemporalTransformerClassifier(
            input_dim=latent_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.future_predictor = FutureLatentPredictor(
            input_dim=latent_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            future_steps=future_steps,
        )

    def compute_predictive_loss(
        self,
        predicted_future: torch.Tensor,
        target_future: torch.Tensor,
    ) -> torch.Tensor:
        if self.predictive_loss == "mse":
            return F.mse_loss(predicted_future, target_future)
        if self.predictive_loss != "cosine":
            raise ValueError(f"Unsupported predictive loss: {self.predictive_loss}")

        predicted = F.normalize(predicted_future.flatten(1), dim=1)
        target = F.normalize(target_future.flatten(1), dim=1)
        return 1.0 - F.cosine_similarity(predicted, target, dim=1).mean()

    def forward(
        self,
        observed_latents: torch.Tensor,
        future_latents: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> VJEPAAnticipationOutput:
        logits = self.classifier(observed_latents, lengths)
        predicted_future = self.future_predictor(observed_latents, lengths)

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

        if future_latents is not None:
            predictive_loss = self.compute_predictive_loss(
                predicted_future,
                future_latents,
            )
            weighted_predictive_loss = self.predictive_loss_weight * predictive_loss
            total_loss = (
                weighted_predictive_loss
                if total_loss is None
                else total_loss + weighted_predictive_loss
            )

        return VJEPAAnticipationOutput(
            logits=logits,
            predicted_future_latent=predicted_future,
            loss=total_loss,
            classification_loss=classification_loss,
            predictive_loss=predictive_loss,
        )

