from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import FallWindowDataset, split_by_subject
from fall_anticipation_cv.models.baseline import VideoCNNTransformerBaseline
from fall_anticipation_cv.models.pose_baseline import PoseTransformerBaseline
from fall_anticipation_cv.models.vjepa_predictive import VJEPALatentPredictiveModel
from fall_anticipation_cv.pose_data import PoseWindowDataset, collate_pose_windows
from fall_anticipation_cv.training_common import binary_classification_metrics
from fall_anticipation_cv.vjepa_data import (
    VJEPALatentWindowDataset,
    collate_vjepa_latent_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune positive-class thresholds.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["video", "pose", "vjepa"],
        default=["video", "pose", "vjepa"],
    )
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.75,
        help="Also report the highest-F1 threshold satisfying this validation recall.",
    )
    return parser.parse_args()


def classification_metrics_from_probs(
    labels: list[int],
    probs: list[float],
    threshold: float,
) -> dict:
    predictions = [int(prob >= threshold) for prob in probs]
    metrics = binary_classification_metrics(labels, predictions)
    metrics["threshold"] = float(threshold)
    metrics["accuracy"] = (
        sum(int(y == pred) for y, pred in zip(labels, predictions)) / len(labels)
        if labels
        else 0.0
    )
    return metrics


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


def candidate_thresholds(probs: list[float]) -> list[float]:
    candidates = {0.5}
    candidates.update(float(i / 100) for i in range(1, 100))
    candidates.update(float(prob) for prob in probs)
    return sorted(candidates)


def tune_thresholds(
    labels: list[int],
    probs: list[float],
    target_recall: float,
) -> dict:
    best_f2 = None
    best_target = None

    for threshold in candidate_thresholds(probs):
        metrics = classification_metrics_from_probs(labels, probs, threshold)
        precision = metrics["positive_precision"]
        recall = metrics["positive_recall"]
        f1 = metrics["positive_f1"]
        f2 = f_beta(precision, recall, beta=2.0)
        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f2": float(f2),
            "accuracy": float(metrics["accuracy"]),
            "confusion_matrix": metrics["confusion_matrix"],
        }
        if best_f2 is None or (candidate["f2"], candidate["f1"]) > (
            best_f2["f2"],
            best_f2["f1"],
        ):
            best_f2 = candidate
        if recall >= target_recall and (
            best_target is None
            or (candidate["f1"], candidate["precision"], candidate["threshold"])
            > (
                best_target["f1"],
                best_target["precision"],
                best_target["threshold"],
            )
        ):
            best_target = candidate

    return {
        "best_f2": best_f2,
        "best_with_target_recall": best_target,
        "target_recall": target_recall,
    }


@torch.no_grad()
def predict_video(
    windows_csv: Path,
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[float], list[int], list[float]]:
    windows = pd.read_csv(windows_csv)
    _train_df, val_df, test_df = split_by_subject(windows)

    model = VideoCNNTransformerBaseline(num_classes=2).to(device)
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    def make_loader(df: pd.DataFrame) -> DataLoader:
        return DataLoader(
            FallWindowDataset(df, resize=(224, 224)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def collect(df: pd.DataFrame) -> tuple[list[int], list[float]]:
        labels_all = []
        probs_all = []
        for videos, labels in make_loader(df):
            videos = videos.to(device)
            logits = model(videos)
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
        return labels_all, probs_all

    val_labels, val_probs = collect(val_df)
    test_labels, test_probs = collect(test_df)
    return val_labels, val_probs, test_labels, test_probs


@torch.no_grad()
def predict_pose(
    windows_csv: Path,
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[float], list[int], list[float]]:
    windows = pd.read_csv(windows_csv)
    _train_df, val_df, test_df = split_by_subject(windows)
    saved = torch.load(checkpoint, map_location=device)

    model = PoseTransformerBaseline(input_dim=int(saved["input_dim"])).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    normalize_pose = bool(saved.get("normalize_pose", True))
    add_velocity = bool(saved.get("add_velocity", True))

    def make_loader(df: pd.DataFrame) -> DataLoader:
        return DataLoader(
            PoseWindowDataset(
                df,
                feature_col="pose_feature_path",
                normalize=normalize_pose,
                add_velocity=add_velocity,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pose_windows,
        )

    def collect(df: pd.DataFrame) -> tuple[list[int], list[float]]:
        labels_all = []
        probs_all = []
        for features, labels, lengths in make_loader(df):
            logits = model(features.to(device), lengths.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
        return labels_all, probs_all

    val_labels, val_probs = collect(val_df)
    test_labels, test_probs = collect(test_df)
    return val_labels, val_probs, test_labels, test_probs


@torch.no_grad()
def predict_vjepa(
    windows_csv: Path,
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[float], list[int], list[float]]:
    windows = pd.read_csv(windows_csv)
    _train_df, val_df, test_df = split_by_subject(windows)
    saved = torch.load(checkpoint, map_location=device)

    model = VJEPALatentPredictiveModel(
        latent_dim=int(saved["latent_dim"]),
        d_model=256,
        num_layers=1,
        future_steps=int(saved["future_steps"]),
        predictive_loss_weight=float(saved.get("predictive_loss_weight", 0.2)),
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
        labels_all = []
        probs_all = []
        for observed, labels, _future, lengths in make_loader(df):
            output = model(observed.to(device), lengths=lengths.to(device))
            probs = torch.softmax(output.logits, dim=1)[:, 1]
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
        return labels_all, probs_all

    val_labels, val_probs = collect(val_df)
    test_labels, test_probs = collect(test_df)
    return val_labels, val_probs, test_labels, test_probs


def evaluate_model(
    name: str,
    data_root: Path,
    batch_size: int,
    target_recall: float,
    device: torch.device,
) -> dict:
    if name == "video":
        val_labels, val_probs, test_labels, test_probs = predict_video(
            data_root / "windows_gmdcsa24.csv",
            data_root / "outputs/video_cnn_transformer_baseline.pt",
            batch_size=8,
            device=device,
        )
        display_name = "video_cnn_transformer_baseline"
    elif name == "pose":
        val_labels, val_probs, test_labels, test_probs = predict_pose(
            data_root / "pose_windows_rtmpose.csv",
            data_root / "outputs/pose_transformer_normalized.pt",
            batch_size=batch_size,
            device=device,
        )
        display_name = "pose_transformer_normalized"
    elif name == "vjepa":
        val_labels, val_probs, test_labels, test_probs = predict_vjepa(
            data_root / "vjepa_windows.csv",
            data_root / "outputs/vjepa_latent_predictive.pt",
            batch_size=batch_size,
            device=device,
        )
        display_name = "vjepa_latent_predictive"
    else:
        raise ValueError(f"Unsupported model: {name}")

    tuned = tune_thresholds(val_labels, val_probs, target_recall=target_recall)
    default_threshold = classification_metrics_from_probs(test_labels, test_probs, 0.5)

    test_at_best_f2 = classification_metrics_from_probs(
        test_labels,
        test_probs,
        tuned["best_f2"]["threshold"],
    )
    target_choice = tuned["best_with_target_recall"]
    test_at_target = (
        classification_metrics_from_probs(
            test_labels,
            test_probs,
            target_choice["threshold"],
        )
        if target_choice is not None
        else None
    )

    return {
        "model": display_name,
        "validation_threshold_selection": tuned,
        "test_default_threshold_0_5": default_threshold,
        "test_at_best_val_f2_threshold": test_at_best_f2,
        "test_at_target_recall_threshold": test_at_target,
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "threshold_policy": (
            "Select threshold on validation set by maximum positive-class F2; "
            "also report best-F1 threshold among thresholds with validation "
            f"recall >= {args.target_recall:.2f}."
        ),
        "models": {},
    }
    for model_name in args.models:
        print(f"Evaluating thresholds for {model_name}", flush=True)
        results["models"][model_name] = evaluate_model(
            model_name,
            data_root=data_root,
            batch_size=args.batch_size,
            target_recall=args.target_recall,
            device=device,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved threshold metrics: {output_path}")


if __name__ == "__main__":
    main()
