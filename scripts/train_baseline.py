from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import FallWindowDataset, split_by_subject
from fall_anticipation_cv.models.baseline import VideoCNNTransformerBaseline
from fall_anticipation_cv.training_common import (
    binary_classification_metrics,
    compute_class_weights,
    default_video_forward,
    evaluate,
    train_one_epoch,
)


def split_windows(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in windows.columns:
        split = windows["split"].astype(str).str.lower()
        return (
            windows[split == "train"].copy(),
            windows[split == "val"].copy(),
            windows[split == "test"].copy(),
        )
    return split_by_subject(windows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a video fall-anticipation model.")
    parser.add_argument("--windows-csv", required=True, help="Window metadata CSV.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/video_cnn_transformer.pt",
        help="Path for the best validation checkpoint.",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/video_cnn_transformer_metrics.json",
        help="Path for the training metrics JSON.",
    )
    parser.add_argument(
        "--model",
        choices=["transformer"],
        default="transformer",
        help="Video model to train.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint path if it exists.",
    )
    return parser.parse_args()


def make_loader(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        FallWindowDataset(df, resize=(224, 224)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    last_checkpoint_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_last{checkpoint_path.suffix}"
    )
    metrics_path = Path(args.metrics)
    history_path = metrics_path.with_name(f"{metrics_path.stem}_history.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)
    train_df, val_df, test_df = split_windows(windows)

    train_loader = make_loader(train_df, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_df, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_df, args.batch_size, False, args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoCNNTransformerBaseline(num_classes=2).to(device)
    model_name = "video_cnn_transformer_baseline"

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
    epoch_history = []
    start_epoch = 0
    resume_path = None
    if args.resume and last_checkpoint_path.exists():
        resume_path = last_checkpoint_path
    elif args.resume and checkpoint_path.exists():
        resume_path = checkpoint_path

    if resume_path is not None:
        saved = torch.load(resume_path, map_location=device)
        checkpoint_model_name = saved.get("model_name")
        if checkpoint_model_name and checkpoint_model_name != model_name:
            raise ValueError(
                f"Checkpoint model {checkpoint_model_name!r} does not match "
                f"requested model {model_name!r}."
            )
        model.load_state_dict(saved["model_state_dict"])
        if "optimizer_state_dict" in saved:
            optimizer.load_state_dict(saved["optimizer_state_dict"])
        best_val_loss = float(saved.get("val_loss", best_val_loss))
        best_val_acc = float(saved.get("val_acc", best_val_acc))
        best_epoch = int(saved.get("epoch", best_epoch))
        start_epoch = int(saved.get("last_epoch", best_epoch)) + 1
        print(
            f"Resuming from {resume_path} at epoch "
            f"{start_epoch + 1}/{args.epochs}."
        )

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            default_video_forward,
        )
        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            default_video_forward,
        )

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

        epoch_history.append(
            {
                "epoch": epoch,
                "epoch_1_indexed": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        history_path.write_text(json.dumps(epoch_history, indent=2) + "\n")
        print(f"Saved epoch history: {history_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "model_name": model_name,
                },
                checkpoint_path,
            )
            print(f"Saved best baseline checkpoint: {checkpoint_path}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": best_epoch,
                "last_epoch": epoch,
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "class_weights": class_weights.detach().cpu().tolist(),
                "model_name": model_name,
            },
            last_checkpoint_path,
        )
        print(f"Saved last baseline checkpoint: {last_checkpoint_path}")

    saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_acc, predictions, labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
        default_video_forward,
    )

    metrics = {
        "model": model_name,
        "windows_csv": args.windows_csv,
        "checkpoint": str(checkpoint_path),
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
