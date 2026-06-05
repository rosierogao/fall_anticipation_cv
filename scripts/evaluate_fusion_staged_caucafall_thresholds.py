from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate_thresholds import classification_metrics_from_probs, f_beta, tune_thresholds
from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.fusion_data import PoseVJEPALatentWindowDataset, collate_pose_vjepa_windows
from fall_anticipation_cv.models.pose_vjepa_fusion import PoseVJEPAFusionTransformer


def split_windows(windows: pd.DataFrame):
    if "split" in windows.columns:
        split = windows["split"].astype(str).str.lower()
        return (
            windows[split == "train"].copy(),
            windows[split == "val"].copy(),
            windows[split == "test"].copy(),
        )
    return split_by_subject(windows)


def load_fusion_windows(vjepa_windows_csv: Path, pose_windows_csv: Path) -> pd.DataFrame:
    windows = pd.read_csv(vjepa_windows_csv)
    if "pose_feature_path" in windows.columns:
        return windows
    pose_windows = pd.read_csv(pose_windows_csv)
    join_cols = ["video_path", "window_start", "window_end"]
    pose_feature_map = pose_windows[join_cols + ["pose_feature_path"]].drop_duplicates(subset=join_cols)
    merged = windows.merge(pose_feature_map, on=join_cols, how="left")
    missing = int(merged["pose_feature_path"].isna().sum())
    if missing:
        print(f"Dropping {missing} windows without matched pose features.", flush=True)
        merged = merged.dropna(subset=["pose_feature_path"])
    return merged


def add_derived_metrics(metrics: dict) -> dict:
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
def collect_probs(df: pd.DataFrame, model, device: torch.device, saved: dict, batch_size: int):
    dataset = PoseVJEPALatentWindowDataset(
        df,
        pose_feature_col="pose_feature_path",
        vjepa_feature_col="vjepa_feature_path",
        normalize_pose=bool(saved.get("normalize_pose", True)),
        add_velocity=bool(saved.get("add_velocity", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_pose_vjepa_windows,
    )
    labels_all = []
    probs_all = []
    for pose_features, labels, vjepa_latents, lengths in loader:
        logits = model(
            pose_features.to(device),
            vjepa_latents.to(device),
            lengths.to(device),
        )
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
    return labels_all, probs_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument("--windows-csv", default="vjepa_windows_staged_caucafall_joined.csv")
    parser.add_argument("--pose-windows-csv", default="pose_windows_staged_caucafall_joined_rtmpose.csv")
    parser.add_argument("--checkpoint", default="outputs/pose_vjepa_fusion_staged_caucafall.pt")
    parser.add_argument("--output", default="outputs/pose_vjepa_fusion_staged_caucafall_threshold_metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-recall", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    windows_csv = data_root / args.windows_csv
    pose_windows_csv = data_root / args.pose_windows_csv
    checkpoint = data_root / args.checkpoint
    output = data_root / args.output

    windows = load_fusion_windows(windows_csv, pose_windows_csv)
    windows = windows.dropna(subset=["pose_feature_path", "vjepa_feature_path"])
    train_df, val_df, test_df = split_windows(windows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device)
    model = PoseVJEPAFusionTransformer(
        pose_dim=int(saved["pose_dim"]),
        vjepa_dim=int(saved["vjepa_dim"]),
        projection_dim=int(saved.get("projection_dim", 256)),
        d_model=int(saved.get("d_model", 256)),
        num_heads=int(saved.get("num_heads", 4)),
        num_layers=int(saved.get("num_layers", 1)),
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    val_labels, val_probs = collect_probs(val_df, model, device, saved, args.batch_size)
    test_labels, test_probs = collect_probs(test_df, model, device, saved, args.batch_size)
    tuned = tune_thresholds(val_labels, val_probs, target_recall=args.target_recall)

    result = {
        "dataset": "staged_gmdcsa24_le2i_caucafall",
        "model": "pose_vjepa_fusion_transformer",
        "windows_csv": str(windows_csv),
        "pose_windows_csv": str(pose_windows_csv),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(saved.get("epoch", -1)),
        "checkpoint_val_loss": float(saved.get("val_loss", float("nan"))),
        "checkpoint_val_acc": float(saved.get("val_acc", float("nan"))),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "validation_best_f2": tuned["best_f2"],
        "validation_best_balanced_accuracy": tuned["best_balanced_accuracy"],
        "test_default_threshold_0_5": add_derived_metrics(classification_metrics_from_probs(test_labels, test_probs, 0.5)),
        "test_at_validation_best_f2_threshold": add_derived_metrics(classification_metrics_from_probs(test_labels, test_probs, tuned["best_f2"]["threshold"])),
        "test_at_validation_best_balanced_accuracy_threshold": add_derived_metrics(classification_metrics_from_probs(test_labels, test_probs, tuned["best_balanced_accuracy"]["threshold"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved fusion threshold metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
