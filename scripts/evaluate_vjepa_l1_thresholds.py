from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_expanded_model_thresholds import add_derived_metrics
from evaluate_thresholds import (
    classification_metrics_from_probs,
    predict_vjepa,
    tune_thresholds,
)


MODEL_SPECS = [
    {
        "dataset": "staged",
        "name": "vjepa_baseline",
        "windows_csv": "vjepa_windows_staged_caucafall_joined.csv",
        "checkpoint": "outputs/vjepa_baseline_staged_caucafall_fall_anticipation_l1.pt",
    },
    {
        "dataset": "staged",
        "name": "vjepa_predictive_l1",
        "windows_csv": "vjepa_windows_staged_caucafall_joined.csv",
        "checkpoint": "outputs/vjepa_latent_predictive_staged_caucafall_fall_anticipation_l1.pt",
    },
    {
        "dataset": "staged_plus_oops",
        "name": "vjepa_baseline",
        "windows_csv": "vjepa_windows_staged_caucafall_oops.csv",
        "checkpoint": "outputs/vjepa_baseline_staged_caucafall_oops_fall_anticipation_l1.pt",
    },
    {
        "dataset": "staged_plus_oops",
        "name": "vjepa_predictive_l1",
        "windows_csv": "vjepa_windows_staged_caucafall_oops.csv",
        "checkpoint": "outputs/vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation_l1.pt",
    },
]


def evaluate_spec(
    spec: dict,
    data_root: Path,
    batch_size: int,
    target_recall: float,
    device: torch.device,
) -> dict:
    val_labels, val_probs, test_labels, test_probs = predict_vjepa(
        windows_csv=data_root / spec["windows_csv"],
        checkpoint=data_root / spec["checkpoint"],
        batch_size=batch_size,
        device=device,
    )
    tuned = tune_thresholds(val_labels, val_probs, target_recall=target_recall)
    f2_threshold = tuned["best_f2"]["threshold"]
    balanced_threshold = tuned["best_balanced_accuracy"]["threshold"]

    return {
        "dataset": spec["dataset"],
        "model": spec["name"],
        "windows_csv": str(data_root / spec["windows_csv"]),
        "checkpoint": str(data_root / spec["checkpoint"]),
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
        "experiment": "V-JEPA L1 latent-regression predictive loss",
        "threshold_policy": (
            "Select thresholds on validation split by positive-class F2 and "
            "balanced accuracy; report test metrics at each threshold."
        ),
        "models": [],
    }
    for spec in MODEL_SPECS:
        print(f"Evaluating {spec['dataset']} / {spec['name']}", flush=True)
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
    print(f"Saved L1 threshold metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
