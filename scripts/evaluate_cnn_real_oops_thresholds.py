from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_thresholds import (
    classification_metrics_from_probs,
    f_beta,
    predict_video,
    tune_thresholds,
)


def add_derived_metrics(metrics: dict) -> dict:
    cm = metrics["confusion_matrix"]
    tn = cm["true_negative"]
    fp = cm["false_positive"]
    fn = cm["false_negative"]
    tp = cm["true_positive"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics["specificity"] = float(specificity)
    metrics["balanced_accuracy"] = float((recall + specificity) / 2.0)
    metrics["positive_f2"] = f_beta(
        metrics["positive_precision"],
        metrics["positive_recall"],
        beta=2.0,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-recall", type=float, default=0.75)
    parser.add_argument("--windows-csv", default="windows_real_oops_balanced_split.csv")
    parser.add_argument("--checkpoint", default="outputs/video_cnn_transformer_real_oops.pt")
    parser.add_argument("--dataset-name", default="GMDCSA24 + LE2I + OOPs")
    parser.add_argument("--note", default="Closest available expanded CNN Transformer run. This checkpoint was not retrained after CAUCAFall was added.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_labels, val_probs, test_labels, test_probs = predict_video(
        windows_csv=data_root / args.windows_csv,
        checkpoint=data_root / args.checkpoint,
        batch_size=args.batch_size,
        device=device,
    )
    tuned = tune_thresholds(val_labels, val_probs, target_recall=args.target_recall)
    f2_threshold = tuned["best_f2"]["threshold"]
    balanced_threshold = tuned["best_balanced_accuracy"]["threshold"]

    results = {
        "dataset": args.dataset_name,
        "note": args.note,
        "model": "video_cnn_transformer",
        "windows_csv": str(data_root / args.windows_csv),
        "checkpoint": str(data_root / args.checkpoint),
        "validation_best_f2": tuned["best_f2"],
        "validation_best_balanced_accuracy": tuned["best_balanced_accuracy"],
        "test_default_threshold_0_5": add_derived_metrics(
            classification_metrics_from_probs(test_labels, test_probs, 0.5)
        ),
        "test_at_validation_best_f2_threshold": add_derived_metrics(
            classification_metrics_from_probs(test_labels, test_probs, f2_threshold)
        ),
        "test_at_validation_best_balanced_accuracy_threshold": add_derived_metrics(
            classification_metrics_from_probs(test_labels, test_probs, balanced_threshold)
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved CNN threshold metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
