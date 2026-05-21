from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.models.pose_baseline import (
    PoseGRUBaseline,
    PoseTransformerBaseline,
)
from fall_anticipation_cv.training_common import (
    binary_classification_metrics,
    compute_class_weights,
    default_pose_forward,
    evaluate,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pose-feature baseline.")
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


def main() -> None:
    args = parse_args()

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
    class_weights = compute_class_weights(
        torch.tensor(train_df["y"].to_numpy(), dtype=torch.long)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            default_pose_forward,
        )
        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            default_pose_forward,
        )
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
                    "class_weights": class_weights.detach().cpu().tolist(),
                },
                checkpoint,
            )
            print(f"Saved best pose baseline checkpoint: {checkpoint}")

    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_acc, predictions, labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
        default_pose_forward,
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
        "class_weights": class_weights.detach().cpu().tolist(),
        **binary_classification_metrics(labels, predictions),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
