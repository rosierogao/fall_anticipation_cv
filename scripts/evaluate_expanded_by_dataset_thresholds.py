
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate_thresholds import classification_metrics_from_probs, f_beta
from fall_anticipation_cv.models.pose_baseline import PoseTransformerBaseline
from fall_anticipation_cv.models.vjepa_predictive import VJEPABaseline, VJEPALatentPredictiveModel
from fall_anticipation_cv.pose_data import PoseWindowDataset, collate_pose_windows
from fall_anticipation_cv.vjepa_data import VJEPALatentWindowDataset, collate_vjepa_latent_windows

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
        "name": "vjepa_predictive",
        "kind": "vjepa",
        "windows_csv": "vjepa_windows_staged_caucafall_oops.csv",
        "checkpoint": "outputs/vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation.pt",
    },
]


def split_frames(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in windows.columns:
        split = windows["split"].astype(str).str.lower()
        return windows[split == "train"].copy(), windows[split == "val"].copy(), windows[split == "test"].copy()
    from fall_anticipation_cv.data import split_by_subject
    return split_by_subject(windows)


def add_derived(metrics: dict) -> dict:
    cm = metrics["confusion_matrix"]
    tn, fp = cm["true_negative"], cm["false_positive"]
    fn, tp = cm["false_negative"], cm["true_positive"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics["specificity"] = float(specificity)
    metrics["balanced_accuracy"] = float((recall + specificity) / 2.0)
    metrics["positive_f2"] = f_beta(metrics["positive_precision"], metrics["positive_recall"], beta=2.0)
    metrics["support"] = int(tn + fp + fn + tp)
    metrics["positive_support"] = int(tp + fn)
    metrics["negative_support"] = int(tn + fp)
    return metrics


@torch.no_grad()
def predict_pose_group(df: pd.DataFrame, checkpoint: Path, batch_size: int, device: torch.device):
    saved = torch.load(checkpoint, map_location=device)
    model = PoseTransformerBaseline(input_dim=int(saved["input_dim"])).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    loader = DataLoader(
        PoseWindowDataset(
            df,
            feature_col="pose_feature_path",
            normalize=bool(saved.get("normalize_pose", True)),
            add_velocity=bool(saved.get("add_velocity", True)),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pose_windows,
    )
    labels_all, probs_all = [], []
    for features, labels, lengths in loader:
        logits = model(features.to(device), lengths.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
    return labels_all, probs_all


@torch.no_grad()
def predict_vjepa_group(df: pd.DataFrame, checkpoint: Path, batch_size: int, device: torch.device):
    saved = torch.load(checkpoint, map_location=device)
    model_cls = VJEPABaseline if saved.get("model_name") == "vjepa_baseline" else VJEPALatentPredictiveModel
    if model_cls is VJEPABaseline:
        model = model_cls(
            latent_dim=int(saved["latent_dim"]),
            d_model=int(saved.get("d_model", 256)),
            num_heads=int(saved.get("num_heads", 4)),
            num_layers=int(saved.get("num_layers", 1)),
            dropout=float(saved.get("dropout", 0.2)),
        ).to(device)
    else:
        model = model_cls(
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
    loader = DataLoader(
        VJEPALatentWindowDataset(df, feature_col="vjepa_feature_path"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_vjepa_latent_windows,
    )
    labels_all, probs_all = [], []
    for observed, labels, _future, lengths in loader:
        output = model(observed.to(device), lengths=lengths.to(device))
        logits = output.logits if hasattr(output, "logits") else output
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
    return labels_all, probs_all


def threshold_lookup(thresholds_path: Path):
    data = json.loads(thresholds_path.read_text())
    lookup = {}
    for item in data["models"]:
        lookup[(item["task"], item["model"])] = {
            "f2": item["validation_best_f2"]["threshold"],
            "balanced_accuracy": item["validation_best_balanced_accuracy"]["threshold"],
        }
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument("--thresholds", default="outputs/expanded_staged_caucafall_oops_f2_and_balanced_threshold_metrics.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", default=["GMDCSA24", "le2i", "caucafall", "OOPs"])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = threshold_lookup(data_root / args.thresholds)
    results = {"split": "existing split column when present; fallback to split_by_subject", "device": str(device), "models": []}

    for spec in MODEL_SPECS:
        key = (spec["task"], spec["name"])
        windows = pd.read_csv(data_root / spec["windows_csv"])
        _train_df, _val_df, test_df = split_frames(windows)
        item = {"task": spec["task"], "model": spec["name"], "datasets": {}}
        for ds in args.datasets:
            group = test_df[test_df["dataset"].astype(str) == ds].copy()
            if group.empty:
                continue
            checkpoint = data_root / spec["checkpoint"]
            if spec["kind"] == "pose":
                labels, probs = predict_pose_group(group, checkpoint, args.batch_size, device)
            elif spec["kind"] == "vjepa":
                labels, probs = predict_vjepa_group(group, checkpoint, args.batch_size, device)
            else:
                raise ValueError(spec["kind"])
            item["datasets"][ds] = {}
            for policy, threshold in thresholds[key].items():
                item["datasets"][ds][policy] = add_derived(classification_metrics_from_probs(labels, probs, threshold))
        results["models"].append(item)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
