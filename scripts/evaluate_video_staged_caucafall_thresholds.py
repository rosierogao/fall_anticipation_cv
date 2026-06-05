from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import FallWindowDataset, split_by_subject
from fall_anticipation_cv.models.baseline import VideoCNNTransformerBaseline
from evaluate_thresholds import classification_metrics_from_probs, f_beta, tune_thresholds


def split_windows(windows: pd.DataFrame):
    if "split" in windows.columns:
        split = windows["split"].astype(str).str.lower()
        return (
            windows[split == "train"].copy(),
            windows[split == "val"].copy(),
            windows[split == "test"].copy(),
        )
    return split_by_subject(windows)


def add_balanced_and_f2(metrics: dict) -> dict:
    cm = metrics["confusion_matrix"]
    tn = cm["true_negative"]
    fp = cm["false_positive"]
    precision = metrics["positive_precision"]
    recall = metrics["positive_recall"]
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        **metrics,
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "positive_f2": float(f_beta(precision, recall, beta=2.0)),
    }


@torch.no_grad()
def collect_probs(df: pd.DataFrame, model, device: torch.device, batch_size: int):
    loader = DataLoader(
        FallWindowDataset(df, resize=(224, 224)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    labels_all = []
    probs_all = []
    for videos, labels in loader:
        videos = videos.to(device)
        logits = model(videos)
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
    return labels_all, probs_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument("--windows-csv", default="windows_staged_caucafall.csv")
    parser.add_argument("--checkpoint", default="outputs/video_cnn_transformer_staged_caucafall.pt")
    parser.add_argument("--output", default="outputs/video_cnn_transformer_staged_caucafall_threshold_metrics.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-recall", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    windows_csv = data_root / args.windows_csv
    checkpoint = data_root / args.checkpoint
    output = data_root / args.output

    windows = pd.read_csv(windows_csv)
    train_df, val_df, test_df = split_windows(windows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoCNNTransformerBaseline(num_classes=2).to(device)
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    val_labels, val_probs = collect_probs(val_df, model, device, args.batch_size)
    test_labels, test_probs = collect_probs(test_df, model, device, args.batch_size)
    tuned = tune_thresholds(val_labels, val_probs, target_recall=args.target_recall)

    test_default = add_balanced_and_f2(classification_metrics_from_probs(test_labels, test_probs, 0.5))
    test_f2 = add_balanced_and_f2(classification_metrics_from_probs(test_labels, test_probs, tuned["best_f2"]["threshold"]))
    test_bal = add_balanced_and_f2(classification_metrics_from_probs(test_labels, test_probs, tuned["best_balanced_accuracy"]["threshold"]))

    result = {
        "dataset": "staged_gmdcsa24_le2i_caucafall",
        "model": "video_cnn_transformer_baseline",
        "windows_csv": str(windows_csv),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(saved.get("epoch", -1)),
        "checkpoint_val_loss": float(saved.get("val_loss", float("nan"))),
        "checkpoint_val_acc": float(saved.get("val_acc", float("nan"))),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "validation_best_f2": tuned["best_f2"],
        "validation_best_balanced_accuracy": tuned["best_balanced_accuracy"],
        "test_default_threshold_0_5": test_default,
        "test_at_validation_best_f2_threshold": test_f2,
        "test_at_validation_best_balanced_accuracy_threshold": test_bal,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved metrics: {output}")


if __name__ == "__main__":
    main()
