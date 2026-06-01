from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate_thresholds import (
    classification_metrics_from_probs,
    f_beta,
    predict_pose,
    predict_vjepa,
    tune_thresholds,
)
from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.models.pose_predictive import PoseSeq2SeqPredictiveModel
from fall_anticipation_cv.pose_predictive_data import (
    PosePredictiveWindowDataset,
    collate_pose_predictive_windows,
)


MODEL_SPECS = [
    {
        "task": "fall_anticipation",
        "name": "pose_transformer",
        "kind": "pose",
        "windows_csv": "pose_windows_staged_caucafall_joined_rtmpose.csv",
        "checkpoint": "outputs/pose_transformer_staged_caucafall_fall_anticipation.pt",
    },
    {
        "task": "fall_anticipation",
        "name": "pose_predictive",
        "kind": "pose_predictive",
        "windows_csv": "pose_predictive_windows_rtmpose.csv",
        "checkpoint": "outputs/pose_seq2seq_predictive_staged_caucafall_fall_anticipation.pt",
    },
    {
        "task": "fall_anticipation",
        "name": "vjepa_baseline",
        "kind": "vjepa",
        "windows_csv": "vjepa_windows_staged_caucafall_joined.csv",
        "checkpoint": "outputs/vjepa_baseline_staged_caucafall_fall_anticipation.pt",
    },
    {
        "task": "fall_anticipation",
        "name": "vjepa_predictive",
        "kind": "vjepa",
        "windows_csv": "vjepa_windows_staged_caucafall_joined.csv",
        "checkpoint": "outputs/vjepa_latent_predictive_staged_caucafall_fall_anticipation.pt",
    },
    {
        "task": "fallen_state",
        "name": "pose_transformer",
        "kind": "pose",
        "windows_csv": "pose_windows_fallen_state_staged_caucafall_horizon2s_rtmpose.csv",
        "checkpoint": "outputs/pose_transformer_fallen_state_staged_caucafall_horizon2s.pt",
    },
    {
        "task": "fallen_state",
        "name": "pose_predictive",
        "kind": "pose_predictive",
        "windows_csv": "pose_predictive_windows_fallen_state_staged_caucafall_horizon2s_rtmpose.csv",
        "checkpoint": "outputs/pose_seq2seq_predictive_fallen_state_staged_caucafall_horizon2s.pt",
    },
    {
        "task": "fallen_state",
        "name": "vjepa_baseline",
        "kind": "vjepa",
        "windows_csv": "vjepa_windows_fallen_state_staged_caucafall_horizon2s.csv",
        "checkpoint": "outputs/vjepa_baseline_fallen_state_staged_caucafall_horizon2s.pt",
    },
    {
        "task": "fallen_state",
        "name": "vjepa_predictive",
        "kind": "vjepa",
        "windows_csv": "vjepa_windows_fallen_state_staged_caucafall_horizon2s.csv",
        "checkpoint": "outputs/vjepa_latent_predictive_fallen_state_staged_caucafall_horizon2s.pt",
    },
]


@torch.no_grad()
def predict_pose_predictive(
    windows_csv: Path,
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[float], list[int], list[float]]:
    import pandas as pd

    windows = pd.read_csv(windows_csv)
    _train_df, val_df, test_df = split_by_subject(windows)
    saved = torch.load(checkpoint, map_location=device)

    model = PoseSeq2SeqPredictiveModel(
        input_dim=int(saved["input_dim"]),
        d_model=int(saved.get("d_model", 128)),
        num_heads=int(saved.get("num_heads", 4)),
        num_layers=int(saved.get("num_layers", 1)),
        dropout=float(saved.get("dropout", 0.3)),
        future_steps=int(saved["future_steps"]),
        predictive_loss_weight=float(saved.get("predictive_loss_weight", 0.2)),
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    normalize_pose = bool(saved.get("normalize_pose", True))
    add_velocity = bool(saved.get("add_velocity", True))

    def make_loader(df):
        return DataLoader(
            PosePredictiveWindowDataset(
                df,
                feature_col="pose_predictive_feature_path",
                normalize=normalize_pose,
                add_velocity=add_velocity,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pose_predictive_windows,
        )

    def collect(df) -> tuple[list[int], list[float]]:
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
    elif spec["kind"] == "pose_predictive":
        val_labels, val_probs, test_labels, test_probs = predict_pose_predictive(
            windows_csv=windows_csv,
            checkpoint=checkpoint,
            batch_size=batch_size,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported model kind: {spec['kind']}")

    tuned = tune_thresholds(val_labels, val_probs, target_recall=target_recall)
    best_f2_threshold = tuned["best_f2"]["threshold"]
    test_default = classification_metrics_from_probs(test_labels, test_probs, 0.5)
    test_default["positive_f2"] = f_beta(
        test_default["positive_precision"],
        test_default["positive_recall"],
        beta=2.0,
    )
    test_tuned = classification_metrics_from_probs(
        test_labels,
        test_probs,
        best_f2_threshold,
    )
    test_tuned["positive_f2"] = f_beta(
        test_tuned["positive_precision"],
        test_tuned["positive_recall"],
        beta=2.0,
    )
    best_balanced_accuracy_threshold = tuned["best_balanced_accuracy"]["threshold"]
    test_balanced_tuned = classification_metrics_from_probs(
        test_labels,
        test_probs,
        best_balanced_accuracy_threshold,
    )
    test_balanced_tuned["positive_f2"] = f_beta(
        test_balanced_tuned["positive_precision"],
        test_balanced_tuned["positive_recall"],
        beta=2.0,
    )

    return {
        "task": spec["task"],
        "model": spec["name"],
        "windows_csv": str(windows_csv),
        "checkpoint": str(checkpoint),
        "validation_best_f2": tuned["best_f2"],
        "validation_best_balanced_accuracy": tuned["best_balanced_accuracy"],
        "test_default_threshold_0_5": test_default,
        "test_at_validation_best_f2_threshold": test_tuned,
        "test_at_validation_best_balanced_accuracy_threshold": test_balanced_tuned,
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
        "threshold_policy": (
            "For each model, select the threshold that maximizes positive-class "
            "F2 on the validation split, then report test metrics at that "
            "threshold."
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved threshold metrics: {output_path}", flush=True)


if __name__ == "__main__":
    main()
