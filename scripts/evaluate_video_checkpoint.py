from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import FallWindowDataset, split_by_subject
from fall_anticipation_cv.models.baseline import (
    SimpleVideoCNN,
    VideoCNNTransformerBaseline,
)
from fall_anticipation_cv.training_common import (
    binary_classification_metrics,
    compute_class_weights,
    default_video_forward,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved video checkpoint.")
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--model", choices=["cnn", "transformer"], default="transformer")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def make_loader(
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        FallWindowDataset(df, resize=(224, 224)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main() -> None:
    args = parse_args()

    windows = pd.read_csv(args.windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "transformer":
        model = VideoCNNTransformerBaseline(num_classes=2).to(device)
        model_name = "video_cnn_transformer_baseline"
    else:
        model = SimpleVideoCNN(num_classes=2).to(device)
        model_name = "simple_video_cnn"

    saved = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])

    class_weights = compute_class_weights(
        torch.tensor(train_df["y"].to_numpy(), dtype=torch.long)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    val_loader = make_loader(val_df, args.batch_size, args.num_workers)
    test_loader = make_loader(test_df, args.batch_size, args.num_workers)

    val_loss, val_acc, val_predictions, val_labels = evaluate(
        model,
        val_loader,
        criterion,
        device,
        default_video_forward,
    )
    test_loss, test_acc, test_predictions, test_labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
        default_video_forward,
    )

    metrics = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "windows_csv": args.windows_csv,
        "checkpoint_epoch": int(saved.get("epoch", -1)),
        "checkpoint_val_loss": float(saved.get("val_loss", float("nan"))),
        "checkpoint_val_acc": float(saved.get("val_acc", float("nan"))),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "val_metrics": binary_classification_metrics(val_labels, val_predictions),
        "test_metrics": binary_classification_metrics(test_labels, test_predictions),
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
