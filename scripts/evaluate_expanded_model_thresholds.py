from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_thresholds import (
    classification_metrics_from_probs,
    f_beta,
    predict_pose,
    predict_vjepa,
    tune_thresholds,
)


MODEL_SPECS = [
    {
        "task": "fall_anticipation",
        "name": "pose_transformer",
        "kind": "pose",
        "windows_csv": "pose_windows_staged_caucafall_oops_rtmpose.csv",
        "checkpoint": "outputs/pose_transformer_staged_caucafall_oops_fall_anticipation.pt",
    },
    {
        "task": "fall_anticipation",
        "name": "vjepa_baseline",
        "kind": "vjepa",
        "windows_csv": "vjepa_windows_staged_caucafall_oops.csv",
        "checkpoint": "outputs/vjepa_baseline_staged_caucafall_oops_fall_anticipation.pt",
    },
    {
        "task": "fall_anticipation",
        "name": "vjepa_predictive",
        "kind": "vjepa",
        "windows_csv": "vjepa_windows_staged_caucafall_oops.csv",
        "checkpoint": "outputs/vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation.pt",
    },
]


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


def evaluate_spec(
    spec: dict,
    data_root: Path,
    batch_size: int,
    target_recall: float,
    device: torch.device,
) -> dict:
    windows_csv = data_root / spec["windows_csv"]
    checkpoint = data_root / spec["checkpoint"]
    if spec["kind"] == "pose":
        val_labels, val_probs, test_labels, test_probs = predict_pose(
            windows_csv=windows_csv,
            checkpoint=checkpoint,
            batch_size=batch_size,
            device=device,
        )
    elif spec["kind"] == "vjepa":
        val_labels, val_probs, test_labels, test_probs = predict_vjepa(
            windows_csv=windows_csv,
            checkpoint=checkpoint,
            batch_size=batch_size,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported model kind: {spec['kind']}")

    tuned = tune_thresholds(val_labels, val_probs, target_recall=target_recall)
    f2_threshold = tuned["best_f2"]["threshold"]
    balanced_threshold = tuned["best_balanced_accuracy"]["threshold"]

    return {
        "task": spec["task"],
        "model": spec["name"],
        "windows_csv": str(windows_csv),
        "checkpoint": str(checkpoint),
        "validation_best_f2": tuned["best_f2"],
        "validation_best_balanced_accuracy": tuned["best_balanced_accuracy"],
        "test_default_threshold_0_5": add_derived_metrics(
            classification_metrics_from_probs(test_labels, test_probs, 0.5)
        ),
        "test_at_validation_best_f2_threshold": add_derived_metrics(
            classification_metrics_from_probs(test_labels, test_probs, f2_threshold)
        ),
        "test_at_validation_best_balanced_accuracy_threshold": add_derived_metrics(
            classification_metrics_from_probs(
                test_labels,
                test_probs,
                balanced_threshold,
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-recall", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "dataset": "GMDCSA24 + LE2I + CAUCAFall + OOPs",
        "threshold_policy": (
            "Select thresholds on validation split by positive-class F2 and "
            "balanced accuracy; report test metrics at each threshold."
        ),
        "models": [],
    }
    for spec in MODEL_SPECS:
        print(f"Evaluating {spec['task']} / {spec['name']}", flush=True)
        results["models"].append(
            evaluate_spec(
                spec,
                data_root=data_root,
                batch_size=args.batch_size,
                target_recall=args.target_recall,
                device=device,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved threshold metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
