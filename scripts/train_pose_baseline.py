from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pose-feature GRU baseline.")
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument("--feature-col", default="pose_feature_path")
    parser.add_argument("--checkpoint", default="outputs/pose_gru_baseline.pt")
    parser.add_argument("--metrics", default="outputs/pose_gru_metrics.json")
    parser.add_argument(
        "--model",
        choices=["gru", "transformer"],
        default="transformer",
        help="Temporal model to train on pose features.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def infer_input_dim(windows: pd.DataFrame, feature_col: str) -> int:
    import numpy as np

    feature_path = Path(windows.iloc[0][feature_col])
    features = np.load(feature_path)
    if features.ndim == 1:
        return int(features.shape[0])
    if features.ndim == 2:
        return int(features.shape[1])
    return int(np.prod(features.shape[1:]))


def make_loader(
    windows: pd.DataFrame,
    feature_col: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    from torch.utils.data import DataLoader

    from fall_anticipation_cv.pose_data import PoseWindowDataset, collate_pose_windows

    return DataLoader(
        PoseWindowDataset(windows, feature_col=feature_col),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pose_windows,
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels, lengths in tqdm(loader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(features, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    import torch
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels, lengths in tqdm(loader, desc="Evaluating"):
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(features, lengths)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / total, correct / total, all_predictions, all_labels


def positive_metrics(labels: list[int], predictions: list[int]) -> dict:
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


def main() -> None:
    args = parse_args()

    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from fall_anticipation_cv.data import split_by_subject
    from fall_anticipation_cv.models.pose_baseline import (
        PoseGRUBaseline,
        PoseTransformerBaseline,
    )

    checkpoint = Path(args.checkpoint)
    metrics_path = Path(args.metrics)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)
    input_dim = infer_input_dim(train_df, args.feature_col)

    train_loader = make_loader(
        train_df, args.feature_col, args.batch_size, True, args.num_workers
    )
    val_loader = make_loader(
        val_df, args.feature_col, args.batch_size, False, args.num_workers
    )
    test_loader = make_loader(
        test_df, args.feature_col, args.batch_size, False, args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "gru":
        model = PoseGRUBaseline(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        ).to(device)
        model_name = "pose_gru_baseline"
    else:
        model = PoseTransformerBaseline(
            input_dim=input_dim,
            d_model=args.hidden_dim,
            num_layers=args.num_layers,
        ).to(device)
        model_name = "pose_transformer_baseline"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_dim": input_dim,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                checkpoint,
            )
            print(f"Saved best pose baseline checkpoint: {checkpoint}")

    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_acc, predictions, labels = evaluate(
        model, test_loader, criterion, device
    )

    metrics = {
        "model": model_name,
        "windows_csv": args.windows_csv,
        "feature_col": args.feature_col,
        "checkpoint": str(checkpoint),
        "input_dim": input_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        **positive_metrics(labels, predictions),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
