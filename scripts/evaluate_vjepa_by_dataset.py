from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.models.vjepa_predictive import VJEPALatentPredictiveModel
from fall_anticipation_cv.training_common import binary_classification_metrics
from fall_anticipation_cv.vjepa_data import (
    VJEPALatentWindowDataset,
    collate_vjepa_latent_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V-JEPA predictive checkpoints by source dataset."
    )
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["GMDCSA24", "le2i"],
        help="Dataset names to report from the held-out test split.",
    )
    return parser.parse_args()


def metrics_from_probs(
    labels: list[int],
    probs: list[float],
    threshold: float,
) -> dict:
    predictions = [int(prob >= threshold) for prob in probs]
    metrics = binary_classification_metrics(labels, predictions)
    metrics["accuracy"] = (
        sum(int(label == pred) for label, pred in zip(labels, predictions))
        / len(labels)
        if labels
        else 0.0
    )
    metrics["threshold"] = float(threshold)
    metrics["support"] = len(labels)
    metrics["positive_support"] = sum(labels)
    metrics["negative_support"] = len(labels) - sum(labels)
    return metrics


def load_model(checkpoint: Path, device: torch.device) -> VJEPALatentPredictiveModel:
    saved = torch.load(checkpoint, map_location=device)
    model = VJEPALatentPredictiveModel(
        latent_dim=int(saved["latent_dim"]),
        d_model=int(saved.get("d_model", 256)),
        num_heads=int(saved.get("num_heads", 4)),
        num_layers=int(saved.get("num_layers", 1)),
        dropout=float(saved.get("dropout", 0.2)),
        future_steps=int(saved["future_steps"]),
        predictive_loss_weight=float(saved.get("predictive_loss_weight", 0.2)),
        predictive_loss=str(saved.get("predictive_loss", "cosine")),
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_group(
    model: VJEPALatentPredictiveModel,
    df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    loader = DataLoader(
        VJEPALatentWindowDataset(df, feature_col="vjepa_feature_path"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_vjepa_latent_windows,
    )

    labels_all: list[int] = []
    probs_all: list[float] = []
    for observed, labels, _future, lengths in loader:
        output = model(observed.to(device), lengths=lengths.to(device))
        probs = torch.softmax(output.logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
    return labels_all, probs_all


def evaluate_checkpoint(
    model_name: str,
    windows_csv: Path,
    checkpoint: Path,
    datasets: list[str],
    batch_size: int,
    threshold: float,
    device: torch.device,
) -> dict:
    windows = pd.read_csv(windows_csv)
    _train_df, _val_df, test_df = split_by_subject(windows)
    model = load_model(checkpoint, device)

    dataset_results = {}
    for dataset in datasets:
        dataset_df = test_df[test_df["dataset"].astype(str) == dataset].copy()
        labels, probs = predict_group(model, dataset_df, batch_size, device)
        dataset_results[dataset] = metrics_from_probs(labels, probs, threshold)

    return {
        "model": model_name,
        "windows_csv": str(windows_csv),
        "checkpoint": str(checkpoint),
        "test_rows": int(len(test_df)),
        "datasets": dataset_results,
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = {
        "non_expanded_vjepa_predictive": {
            "windows_csv": data_root / "vjepa_windows.csv",
            "checkpoint": data_root / "outputs/vjepa_latent_predictive.pt",
        },
        "unstaged_expanded_vjepa_predictive": {
            "windows_csv": data_root / "vjepa_windows_real_oops.csv",
            "checkpoint": data_root / "outputs/vjepa_latent_predictive_real_oops.pt",
        },
    }

    results = {
        "threshold": float(args.threshold),
        "split": "held-out test split from split_by_subject",
        "device": str(device),
        "models": {},
    }
    for model_name, spec in runs.items():
        print(f"Evaluating {model_name}", flush=True)
        results["models"][model_name] = evaluate_checkpoint(
            model_name=model_name,
            windows_csv=spec["windows_csv"],
            checkpoint=spec["checkpoint"],
            datasets=args.datasets,
            batch_size=args.batch_size,
            threshold=args.threshold,
            device=device,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
