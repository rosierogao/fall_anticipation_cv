from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch.utils.data import DataLoader


class BatchForwardFn(Protocol):
    def __call__(self, batch: object, model: torch.nn.Module) -> torch.Tensor: ...


@dataclass(frozen=True)
class TrainResult:
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float
    test_predictions: list[int]
    test_labels: list[int]
    class_weights: torch.Tensor


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(torch.long)
    num_classes = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    weights = total / (num_classes * counts.clamp_min(1.0))
    return weights


def binary_classification_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict:
    tn = sum(1 for y, y_hat in zip(labels, predictions) if y == 0 and y_hat == 0)
    fp = sum(1 for y, y_hat in zip(labels, predictions) if y == 0 and y_hat == 1)
    fn = sum(1 for y, y_hat in zip(labels, predictions) if y == 1 and y_hat == 0)
    tp = sum(1 for y, y_hat in zip(labels, predictions) if y == 1 and y_hat == 1)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
    }


def default_video_forward(batch: object, model: torch.nn.Module) -> torch.Tensor:
    videos, _labels = batch
    return model(videos)


def default_pose_forward(batch: object, model: torch.nn.Module) -> torch.Tensor:
    features, _labels, lengths = batch
    return model(features, lengths)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    forward_fn: Callable[[object, torch.nn.Module], torch.Tensor],
) -> tuple[float, float]:
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training"):
        batch = _move_batch_to_device(batch, device)
        labels = _extract_labels(batch)

        optimizer.zero_grad()
        logits = forward_fn(batch, model)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    forward_fn: Callable[[object, torch.nn.Module], torch.Tensor],
) -> tuple[float, float, list[int], list[int]]:
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions_all: list[int] = []
    labels_all: list[int] = []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = _move_batch_to_device(batch, device)
        labels = _extract_labels(batch)

        logits = forward_fn(batch, model)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        predictions_all.extend(predictions.cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())

    return total_loss / total, correct / total, predictions_all, labels_all


def _extract_labels(batch: object) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[1]
    raise TypeError("Unsupported batch type")


def _move_batch_to_device(batch: object, device: torch.device) -> object:
    if isinstance(batch, tuple):
        moved = []
        for item in batch:
            if hasattr(item, "to"):
                moved.append(item.to(device))
            else:
                moved.append(item)
        return tuple(moved)
    if isinstance(batch, list):
        return tuple(item.to(device) if hasattr(item, "to") else item for item in batch)
    raise TypeError("Unsupported batch type")
