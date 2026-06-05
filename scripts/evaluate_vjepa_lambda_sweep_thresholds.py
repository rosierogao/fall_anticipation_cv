from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate_thresholds import classification_metrics_from_probs, f_beta, tune_thresholds
from fall_anticipation_cv.models.vjepa_predictive import VJEPABaseline, VJEPALatentPredictiveModel
from fall_anticipation_cv.vjepa_data import VJEPALatentWindowDataset, collate_vjepa_latent_windows


def split_windows(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in windows.columns:
        split = windows["split"].astype(str).str.lower()
        return (
            windows[split == "train"].copy(),
            windows[split == "val"].copy(),
            windows[split == "test"].copy(),
        )
    raise ValueError("Expected windows CSV to contain an explicit split column.")


def add_derived(metrics: dict) -> dict:
    cm = metrics["confusion_matrix"]
    tn = cm["true_negative"]
    fp = cm["false_positive"]
    fn = cm["false_negative"]
    tp = cm["true_positive"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics["specificity"] = float(specificity)
    metrics["balanced_accuracy"] = float((recall + specificity) / 2.0)
    metrics["positive_f2"] = float(
        f_beta(metrics["positive_precision"], metrics["positive_recall"], beta=2.0)
    )
    return metrics


@torch.no_grad()
def predict_vjepa(
    windows_csv: Path,
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[float], list[int], list[float]]:
    windows = pd.read_csv(windows_csv)
    _train_df, val_df, test_df = split_windows(windows)
    saved = torch.load(checkpoint, map_location=device)

    model_name = saved.get("model_name", "")
    if model_name == "vjepa_baseline":
        model = VJEPABaseline(
            latent_dim=int(saved["latent_dim"]),
            d_model=256,
            num_layers=1,
            dropout=float(saved.get("dropout", 0.1)),
        ).to(device)
    else:
        model = VJEPALatentPredictiveModel(
            latent_dim=int(saved["latent_dim"]),
            d_model=256,
            num_layers=1,
            future_steps=int(saved["future_steps"]),
            predictive_loss_weight=float(saved.get("predictive_loss_weight", 0.2)),
            predictive_loss=str(saved.get("predictive_loss", "cosine")),
            dropout=float(saved.get("dropout", 0.1)),
        ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    def make_loader(df: pd.DataFrame) -> DataLoader:
        return DataLoader(
            VJEPALatentWindowDataset(df, feature_col="vjepa_feature_path"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_vjepa_latent_windows,
        )

    def collect(df: pd.DataFrame) -> tuple[list[int], list[float]]:
        labels_all: list[int] = []
        probs_all: list[float] = []
        for observed, labels, _future, lengths in make_loader(df):
            output = model(observed.to(device), lengths=lengths.to(device))
            probs = torch.softmax(output.logits, dim=1)[:, 1]
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
        return labels_all, probs_all

    val_labels, val_probs = collect(val_df)
    test_labels, test_probs = collect(test_df)
    return val_labels, val_probs, test_labels, test_probs


def model_specs(data_root: Path) -> list[dict]:
    windows_csv = data_root / "vjepa_windows_staged_caucafall_oops.csv"
    outputs = data_root / "outputs"
    return [
        {
            "model": "V-JEPA Baseline",
            "lambda": 0.0,
            "windows_csv": windows_csv,
            "checkpoint": outputs / "vjepa_baseline_staged_caucafall_oops_fall_anticipation.pt",
        },
        {
            "model": "V-JEPA Predictive lambda=0.05",
            "lambda": 0.05,
            "windows_csv": windows_csv,
            "checkpoint": outputs / "vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation_lambda_0p05.pt",
        },
        {
            "model": "V-JEPA Predictive lambda=0.1",
            "lambda": 0.1,
            "windows_csv": windows_csv,
            "checkpoint": outputs / "vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation_lambda_0p1.pt",
        },
        {
            "model": "V-JEPA Predictive lambda=0.2",
            "lambda": 0.2,
            "windows_csv": windows_csv,
            "checkpoint": outputs / "vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation.pt",
        },
        {
            "model": "V-JEPA Predictive lambda=0.5",
            "lambda": 0.5,
            "windows_csv": windows_csv,
            "checkpoint": outputs / "vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation_lambda_0p5.pt",
        },
    ]


def evaluate_spec(spec: dict, batch_size: int, target_recall: float, device: torch.device) -> dict:
    val_labels, val_probs, test_labels, test_probs = predict_vjepa(
        spec["windows_csv"], spec["checkpoint"], batch_size, device
    )
    tuned = tune_thresholds(val_labels, val_probs, target_recall=target_recall)
    f2_threshold = tuned["best_f2"]["threshold"]
    bal_threshold = tuned["best_balanced_accuracy"]["threshold"]
    return {
        "model": spec["model"],
        "lambda": spec["lambda"],
        "checkpoint": str(spec["checkpoint"]),
        "validation_best_f2": tuned["best_f2"],
        "validation_best_balanced_accuracy": tuned["best_balanced_accuracy"],
        "test_at_validation_best_f2_threshold": add_derived(
            classification_metrics_from_probs(test_labels, test_probs, f2_threshold)
        ),
        "test_at_validation_best_balanced_accuracy_threshold": add_derived(
            classification_metrics_from_probs(test_labels, test_probs, bal_threshold)
        ),
    }


def print_table(title: str, rows: list[dict], metric_key: str) -> None:
    print("\n" + title)
    print("| Model | Lambda | Threshold | Test Acc | Balanced Acc | Precision | Recall | F1 | F2 | TN | FP | FN | TP |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        m = row[metric_key]
        cm = m["confusion_matrix"]
        print(
            f"| {row['model']} | {row['lambda']:.2f} | {m['threshold']:.3f} | "
            f"{m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | "
            f"{m['positive_precision']:.3f} | {m['positive_recall']:.3f} | "
            f"{m['positive_f1']:.3f} | {m['positive_f2']:.3f} | "
            f"{cm['true_negative']} | {cm['false_positive']} | "
            f"{cm['false_negative']} | {cm['true_positive']} |"
        )


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
    rows = [
        evaluate_spec(spec, args.batch_size, args.target_recall, device)
        for spec in model_specs(data_root)
    ]
    results = {
        "dataset": "GMDCSA24 + LE2I + CAUCAFall + OOPs",
        "windows_csv": str(data_root / "vjepa_windows_staged_caucafall_oops.csv"),
        "threshold_policy": "Thresholds selected on validation split by F2 or balanced accuracy, then reported on test split.",
        "models": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print_table("F2-Tuned Thresholds", rows, "test_at_validation_best_f2_threshold")
    print_table("Balanced-Accuracy-Tuned Thresholds", rows, "test_at_validation_best_balanced_accuracy_threshold")
    print(f"\nSaved lambda sweep threshold metrics: {output}")


if __name__ == "__main__":
    main()
