from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.models.pose_baseline import PoseTransformerBaseline
from fall_anticipation_cv.pose_data import PoseWindowDataset, collate_pose_windows
from fall_anticipation_cv.training_common import binary_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pose Transformer checkpoint by source dataset."
    )
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument(
        "--windows-csv",
        default="/data/final_project_dataset/pose_windows_rtmpose.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default="/data/final_project_dataset/outputs/pose_transformer_normalized.pt",
    )
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


@torch.no_grad()
def predict_group(
    model: PoseTransformerBaseline,
    df: pd.DataFrame,
    feature_col: str,
    normalize_pose: bool,
    add_velocity: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    loader = DataLoader(
        PoseWindowDataset(
            df,
            feature_col=feature_col,
            normalize=normalize_pose,
            add_velocity=add_velocity,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pose_windows,
    )

    labels_all: list[int] = []
    probs_all: list[float] = []
    for features, labels, lengths in loader:
        logits = model(features.to(device), lengths.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
    return labels_all, probs_all


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows = pd.read_csv(args.windows_csv)
    _train_df, _val_df, test_df = split_by_subject(windows)

    saved = torch.load(args.checkpoint, map_location=device)
    model = PoseTransformerBaseline(input_dim=int(saved["input_dim"])).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    normalize_pose = bool(saved.get("normalize_pose", True))
    add_velocity = bool(saved.get("add_velocity", True))
    feature_col = "pose_feature_path"

    dataset_results = {}
    for dataset in args.datasets:
        dataset_df = test_df[test_df["dataset"].astype(str) == dataset].copy()
        labels, probs = predict_group(
            model,
            dataset_df,
            feature_col=feature_col,
            normalize_pose=normalize_pose,
            add_velocity=add_velocity,
            batch_size=args.batch_size,
            device=device,
        )
        dataset_results[dataset] = metrics_from_probs(labels, probs, args.threshold)

    results = {
        "model": "pose_transformer_normalized",
        "windows_csv": str(args.windows_csv),
        "checkpoint": str(args.checkpoint),
        "threshold": float(args.threshold),
        "split": "held-out test split from split_by_subject",
        "device": str(device),
        "test_rows": int(len(test_df)),
        "datasets": dataset_results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
