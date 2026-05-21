from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class EvalResult:
    loss: float
    accuracy: float
    predictions: list[int]
    labels: list[int]


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in tqdm(loader, desc="Training"):
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(videos)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * videos.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> EvalResult:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for videos, labels in tqdm(loader, desc="Evaluating"):
        videos = videos.to(device)
        labels = labels.to(device)

        logits = model(videos)
        loss = criterion(logits, labels)

        total_loss += loss.item() * videos.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return EvalResult(
        loss=total_loss / total,
        accuracy=correct / total,
        predictions=all_preds,
        labels=all_labels,
    )

