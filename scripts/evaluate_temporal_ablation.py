from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.models.pose_baseline import PoseTransformerBaseline
from fall_anticipation_cv.models.vjepa_predictive import (
    VJEPABaseline,
    VJEPALatentPredictiveModel,
)
from fall_anticipation_cv.pose_data import PoseWindowDataset, collate_pose_windows
from fall_anticipation_cv.training_common import binary_classification_metrics
from fall_anticipation_cv.vjepa_data import (
    VJEPALatentWindowDataset,
    collate_vjepa_latent_windows,
)


MODEL_SPEC_SETS = {
    "staged_caucafall": {
        "description": "staged GMDCSA24 + le2i + caucafall",
        "datasets": ["GMDCSA24", "le2i", "caucafall"],
        "models": [
            {
                "name": "pose_transformer",
                "kind": "pose",
                "windows_csv": "pose_windows_staged_caucafall_joined_rtmpose.csv",
                "checkpoint": "outputs/pose_transformer_staged_caucafall_fall_anticipation.pt",
            },
            {
                "name": "vjepa_baseline",
                "kind": "vjepa",
                "windows_csv": "vjepa_windows_staged_caucafall_joined.csv",
                "checkpoint": "outputs/vjepa_baseline_staged_caucafall_fall_anticipation.pt",
            },
            {
                "name": "vjepa_predictive",
                "kind": "vjepa",
                "windows_csv": "vjepa_windows_staged_caucafall_joined.csv",
                "checkpoint": "outputs/vjepa_latent_predictive_staged_caucafall_fall_anticipation.pt",
            },
        ],
    },
    "staged_caucafall_oops": {
        "description": "expanded GMDCSA24 + le2i + caucafall + OOPs",
        "datasets": ["GMDCSA24", "le2i", "caucafall", "OOPs"],
        "models": [
            {
                "name": "pose_transformer",
                "kind": "pose",
                "windows_csv": "pose_windows_staged_caucafall_oops_rtmpose.csv",
                "checkpoint": "outputs/pose_transformer_staged_caucafall_oops_fall_anticipation.pt",
            },
            {
                "name": "vjepa_baseline",
                "kind": "vjepa",
                "windows_csv": "vjepa_windows_staged_caucafall_oops.csv",
                "checkpoint": "outputs/vjepa_baseline_staged_caucafall_oops_fall_anticipation.pt",
            },
            {
                "name": "vjepa_predictive",
                "kind": "vjepa",
                "windows_csv": "vjepa_windows_staged_caucafall_oops.csv",
                "checkpoint": "outputs/vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation.pt",
            },
        ],
    },
}


@dataclass
class BatchPredictions:
    labels: list[int]
    probs: list[float]
    logits: list[float]
    datasets: list[str]
    lengths: list[int]


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return (*self.dataset[index], index)


def collate_pose_indexed(batch):
    features, labels, indices = zip(*batch)
    padded, labels, lengths = collate_pose_windows(list(zip(features, labels)))
    return padded, labels, lengths, torch.tensor(indices, dtype=torch.long)


def collate_vjepa_indexed(batch):
    observed, labels, future, indices = zip(*batch)
    observed, labels, future, lengths = collate_vjepa_latent_windows(
        list(zip(observed, labels, future))
    )
    return observed, labels, future, lengths, torch.tensor(indices, dtype=torch.long)


def metrics_from_probs(labels: list[int], probs: list[float], threshold: float) -> dict:
    predictions = [int(prob >= threshold) for prob in probs]
    metrics = binary_classification_metrics(labels, predictions)
    metrics["accuracy"] = (
        sum(int(label == pred) for label, pred in zip(labels, predictions)) / len(labels)
        if labels
        else 0.0
    )
    metrics["threshold"] = float(threshold)
    metrics["support"] = len(labels)
    metrics["positive_support"] = int(sum(labels))
    metrics["negative_support"] = int(len(labels) - sum(labels))
    return metrics


def tune_threshold(labels: list[int], probs: list[float]) -> dict:
    best = None
    for threshold in sorted({0.5, *[i / 100 for i in range(1, 100)], *probs}):
        metrics = metrics_from_probs(labels, probs, threshold)
        precision = metrics["positive_precision"]
        recall = metrics["positive_recall"]
        f2 = (
            0.0
            if precision == 0.0 and recall == 0.0
            else 5.0 * precision * recall / ((4.0 * precision) + recall)
        )
        metrics["positive_f2"] = float(f2)
        if best is None or (metrics["positive_f2"], metrics["positive_f1"]) > (
            best["positive_f2"],
            best["positive_f1"],
        ):
            best = metrics
    return best


def load_pose_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[PoseTransformerBaseline, bool, bool]:
    saved = torch.load(checkpoint, map_location=device)
    model = PoseTransformerBaseline(
        input_dim=int(saved["input_dim"]),
        d_model=int(saved.get("d_model", 128)),
        num_heads=int(saved.get("num_heads", 4)),
        num_layers=int(saved.get("num_layers", 1)),
        dropout=float(saved.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return (
        model,
        bool(saved.get("normalize_pose", True)),
        bool(saved.get("add_velocity", True)),
    )


def load_vjepa_model(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location=device)
    model_name = str(saved.get("model_name", ""))
    if model_name == "vjepa_baseline":
        model = VJEPABaseline(
            latent_dim=int(saved["latent_dim"]),
            d_model=int(saved.get("d_model", 256)),
            num_heads=int(saved.get("num_heads", 4)),
            num_layers=int(saved.get("num_layers", 1)),
            dropout=float(saved.get("dropout", 0.2)),
        ).to(device)
    else:
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


def remove_time_step(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    time_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ablated = torch.zeros_like(sequences)
    ablated_lengths = torch.clamp(lengths - 1, min=1)
    for row_idx, length in enumerate(lengths.tolist()):
        valid_length = int(length)
        if time_index >= valid_length:
            ablated[row_idx, :valid_length] = sequences[row_idx, :valid_length]
            ablated_lengths[row_idx] = valid_length
            continue
        kept = torch.cat(
            [
                sequences[row_idx, :time_index],
                sequences[row_idx, time_index + 1 : valid_length],
            ],
            dim=0,
        )
        if kept.shape[0] == 0:
            kept = sequences[row_idx, time_index : time_index + 1]
        ablated[row_idx, : kept.shape[0]] = kept
        ablated_lengths[row_idx] = kept.shape[0]
    return ablated, ablated_lengths


def mask_time_step(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    time_index: int,
    mask_value: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ablated = sequences.clone()
    active_mask = lengths > time_index
    if not bool(active_mask.any()):
        return ablated, lengths

    if mask_value == "zero":
        ablated[active_mask, time_index] = 0.0
    elif mask_value == "sequence_mean":
        for row_idx in torch.where(active_mask)[0].tolist():
            valid = sequences[row_idx, : int(lengths[row_idx].item())]
            ablated[row_idx, time_index] = valid.mean(dim=0)
    elif mask_value == "previous":
        for row_idx in torch.where(active_mask)[0].tolist():
            source_index = max(0, time_index - 1)
            ablated[row_idx, time_index] = sequences[row_idx, source_index]
    else:
        raise ValueError(f"Unsupported mask value: {mask_value}")

    return ablated, lengths.clone()


def collect_with_ablation(
    loader: DataLoader,
    model,
    windows_df: pd.DataFrame,
    device: torch.device,
    forward_fn: Callable,
    ablation_method: str,
    mask_value: str,
) -> tuple[BatchPredictions, dict[int, list[dict]]]:
    labels_all: list[int] = []
    probs_all: list[float] = []
    logits_all: list[float] = []
    datasets_all: list[str] = []
    lengths_all: list[int] = []
    ablations: dict[int, list[dict]] = {}

    for batch in loader:
        sequences, labels, *rest = batch
        if len(rest) == 2:
            lengths, indices = rest
        else:
            _future, lengths, indices = rest

        sequences = sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        logits = forward_fn(model, sequences, lengths)
        probs = torch.softmax(logits, dim=1)[:, 1]
        pos_logits = logits[:, 1]

        labels_cpu = labels.cpu().numpy().astype(int)
        probs_cpu = probs.cpu().numpy()
        logits_cpu = pos_logits.cpu().numpy()
        lengths_cpu = lengths.cpu().numpy().astype(int)
        indices_cpu = indices.cpu().numpy().astype(int)
        dataset_values = [
            str(windows_df.iloc[index].get("dataset", "unknown")) for index in indices_cpu
        ]

        labels_all.extend(labels_cpu.tolist())
        probs_all.extend(probs_cpu.tolist())
        logits_all.extend(logits_cpu.tolist())
        lengths_all.extend(lengths_cpu.tolist())
        datasets_all.extend(dataset_values)

        max_len = int(lengths.max().item())
        for time_index in range(max_len):
            active_mask = lengths > time_index
            if not bool(active_mask.any()):
                continue
            if ablation_method == "delete":
                ablated_sequences, ablated_lengths = remove_time_step(
                    sequences,
                    lengths,
                    time_index,
                )
            elif ablation_method == "mask":
                ablated_sequences, ablated_lengths = mask_time_step(
                    sequences,
                    lengths,
                    time_index,
                    mask_value,
                )
            else:
                raise ValueError(f"Unsupported ablation method: {ablation_method}")
            ablated_logits = forward_fn(model, ablated_sequences, ablated_lengths)
            ablated_probs = torch.softmax(ablated_logits, dim=1)[:, 1]
            ablated_pos_logits = ablated_logits[:, 1]

            for row_idx in torch.where(active_mask)[0].cpu().numpy().astype(int):
                length = int(lengths_cpu[row_idx])
                ablations.setdefault(time_index, []).append(
                    {
                        "label": int(labels_cpu[row_idx]),
                        "dataset": dataset_values[row_idx],
                        "length": length,
                        "relative_position": (
                            float(time_index / (length - 1)) if length > 1 else 0.0
                        ),
                        "baseline_prob": float(probs_cpu[row_idx]),
                        "ablated_prob": float(ablated_probs[row_idx].item()),
                        "delta_prob": float(
                            probs_cpu[row_idx] - ablated_probs[row_idx].item()
                        ),
                        "baseline_logit": float(logits_cpu[row_idx]),
                        "ablated_logit": float(ablated_pos_logits[row_idx].item()),
                        "delta_logit": float(
                            logits_cpu[row_idx] - ablated_pos_logits[row_idx].item()
                        ),
                    }
                )

    return (
        BatchPredictions(
            labels=labels_all,
            probs=probs_all,
            logits=logits_all,
            datasets=datasets_all,
            lengths=lengths_all,
        ),
        ablations,
    )


def summarize_records(records: list[dict]) -> dict:
    if not records:
        return {
            "support": 0,
            "mean_delta_prob": None,
            "mean_abs_delta_prob": None,
            "mean_delta_logit": None,
            "mean_abs_delta_logit": None,
        }
    delta_prob = np.array([record["delta_prob"] for record in records], dtype=np.float64)
    delta_logit = np.array([record["delta_logit"] for record in records], dtype=np.float64)
    return {
        "support": int(len(records)),
        "positive_support": int(sum(record["label"] for record in records)),
        "mean_relative_position": float(
            np.mean([record["relative_position"] for record in records])
        ),
        "mean_delta_prob": float(delta_prob.mean()),
        "mean_abs_delta_prob": float(np.abs(delta_prob).mean()),
        "mean_delta_logit": float(delta_logit.mean()),
        "mean_abs_delta_logit": float(np.abs(delta_logit).mean()),
    }


def summarize_ablations(
    ablations: dict[int, list[dict]],
    datasets: list[str],
) -> tuple[list[dict], list[dict]]:
    frame_rows: list[dict] = []
    dataset_rows: list[dict] = []
    for time_index in sorted(ablations):
        records = ablations[time_index]
        row = {"time_index": int(time_index), **summarize_records(records)}
        frame_rows.append(row)

        for dataset in datasets:
            dataset_records = [
                record
                for record in records
                if record["dataset"].lower() == dataset.lower()
            ]
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "time_index": int(time_index),
                    **summarize_records(dataset_records),
                }
            )
    return frame_rows, dataset_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_pose(
    spec: dict,
    data_root: Path,
    batch_size: int,
    device: torch.device,
    dataset_names: list[str],
    ablation_method: str,
    mask_value: str,
) -> dict:
    windows = pd.read_csv(data_root / spec["windows_csv"])
    _train_df, val_df, test_df = split_by_subject(windows)
    model, normalize_pose, add_velocity = load_pose_model(
        data_root / spec["checkpoint"],
        device,
    )

    val_loader = DataLoader(
        PoseWindowDataset(
            val_df,
            feature_col="pose_feature_path",
            normalize=normalize_pose,
            add_velocity=add_velocity,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pose_windows,
    )
    val_labels, val_probs = collect_probs(
        val_loader,
        model,
        device,
        lambda model, features, lengths: model(features, lengths),
    )

    test_dataset = IndexedDataset(
        PoseWindowDataset(
            test_df,
            feature_col="pose_feature_path",
            normalize=normalize_pose,
            add_velocity=add_velocity,
        )
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pose_indexed,
    )
    predictions, ablations = collect_with_ablation(
        test_loader,
        model,
        test_df,
        device,
        lambda model, features, lengths: model(features, lengths),
        ablation_method,
        mask_value,
    )
    frame_rows, dataset_rows = summarize_ablations(ablations, dataset_names)
    return model_result(spec, val_labels, val_probs, predictions, frame_rows, dataset_rows)


def evaluate_vjepa(
    spec: dict,
    data_root: Path,
    batch_size: int,
    device: torch.device,
    dataset_names: list[str],
    ablation_method: str,
    mask_value: str,
) -> dict:
    windows = pd.read_csv(data_root / spec["windows_csv"])
    _train_df, val_df, test_df = split_by_subject(windows)
    model = load_vjepa_model(data_root / spec["checkpoint"], device)

    val_loader = DataLoader(
        VJEPALatentWindowDataset(val_df, feature_col="vjepa_feature_path"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_vjepa_latent_windows,
    )
    val_labels, val_probs = collect_probs(
        val_loader,
        model,
        device,
        lambda model, observed, lengths: model(observed, lengths=lengths).logits,
    )

    test_dataset = IndexedDataset(
        VJEPALatentWindowDataset(test_df, feature_col="vjepa_feature_path")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_vjepa_indexed,
    )
    predictions, ablations = collect_with_ablation(
        test_loader,
        model,
        test_df,
        device,
        lambda model, observed, lengths: model(observed, lengths=lengths).logits,
        ablation_method,
        mask_value,
    )
    frame_rows, dataset_rows = summarize_ablations(ablations, dataset_names)
    return model_result(spec, val_labels, val_probs, predictions, frame_rows, dataset_rows)


@torch.no_grad()
def collect_probs(
    loader: DataLoader,
    model,
    device: torch.device,
    forward_fn: Callable,
) -> tuple[list[int], list[float]]:
    labels_all: list[int] = []
    probs_all: list[float] = []
    for batch in loader:
        sequences, labels, *rest = batch
        lengths = rest[-1]
        logits = forward_fn(model, sequences.to(device), lengths.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy().astype(int).tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
    return labels_all, probs_all


def model_result(
    spec: dict,
    val_labels: list[int],
    val_probs: list[float],
    predictions: BatchPredictions,
    frame_rows: list[dict],
    dataset_rows: list[dict],
) -> dict:
    threshold_metrics = tune_threshold(val_labels, val_probs)
    threshold = float(threshold_metrics["threshold"])
    dataset_metrics = {}
    for dataset in sorted(set(predictions.datasets), key=str.lower):
        labels = [
            label
            for label, ds in zip(predictions.labels, predictions.datasets)
            if ds.lower() == dataset.lower()
        ]
        probs = [
            prob
            for prob, ds in zip(predictions.probs, predictions.datasets)
            if ds.lower() == dataset.lower()
        ]
        dataset_metrics[dataset] = metrics_from_probs(labels, probs, threshold)

    top_positive = sorted(
        frame_rows,
        key=lambda row: row["mean_delta_prob"]
        if row["mean_delta_prob"] is not None
        else -float("inf"),
        reverse=True,
    )[:5]
    top_absolute = sorted(
        frame_rows,
        key=lambda row: row["mean_abs_delta_prob"]
        if row["mean_abs_delta_prob"] is not None
        else -float("inf"),
        reverse=True,
    )[:5]

    return {
        "model": spec["name"],
        "windows_csv": spec["windows_csv"],
        "checkpoint": spec["checkpoint"],
        "validation_threshold_max_f2": threshold_metrics,
        "test_metrics_at_validation_threshold": metrics_from_probs(
            predictions.labels,
            predictions.probs,
            threshold,
        ),
        "test_metrics_by_dataset_at_validation_threshold": dataset_metrics,
        "test_support_by_observed_length": {
            str(length): int(predictions.lengths.count(length))
            for length in sorted(set(predictions.lengths))
        },
        "top_time_steps_by_mean_positive_probability_drop": top_positive,
        "top_time_steps_by_mean_absolute_probability_change": top_absolute,
        "frame_rows": frame_rows,
        "dataset_frame_rows": dataset_rows,
    }


def write_report(results: dict, output_dir: Path) -> None:
    lines = [
        "# Temporal Ablation Results",
        "",
        f"Ablation method: {results['ablation_method']}. "
        "The reported value is the change in positive-class probability on the "
        "held-out test split.",
        "",
    ]
    for model in results["models"]:
        lines.extend(
            [
                f"## {model['model']}",
                "",
                f"Validation F2 threshold: {model['validation_threshold_max_f2']['threshold']:.6f}",
                "",
                "Top time steps by mean positive-probability drop:",
                "",
                "| rank | time_index | mean_relative_position | mean_delta_prob | support |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for rank, row in enumerate(
            model["top_time_steps_by_mean_positive_probability_drop"],
            start=1,
        ):
            lines.append(
                "| "
                f"{rank} | {row['time_index']} | "
                f"{row['mean_relative_position']:.3f} | "
                f"{row['mean_delta_prob']:.6f} | {row['support']} |"
            )
        lines.append("")
    (output_dir / "temporal_ablation_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leave-one-time-step-out temporal ablation on staged models."
    )
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--model-set",
        choices=sorted(MODEL_SPEC_SETS),
        default="staged_caucafall",
        help="Checkpoint/window set to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset filter. Defaults to the selected model set datasets.",
    )
    parser.add_argument(
        "--ablation-method",
        choices=["delete", "mask"],
        default="delete",
        help="Delete the time step or mask it in place while preserving length.",
    )
    parser.add_argument(
        "--mask-value",
        choices=["zero", "sequence_mean", "previous"],
        default="zero",
        help="Replacement value for --ablation-method mask.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec_set = MODEL_SPEC_SETS[args.model_set]
    selected_datasets = args.datasets or spec_set["datasets"]

    results = {
        "dataset": spec_set["description"],
        "model_set": args.model_set,
        "dataset_filter": selected_datasets,
        "split": "held-out test split from split_by_subject",
        "ablation_policy": (
            "For each observed pose frame or V-JEPA temporal latent token, "
            "rerun the frozen model after applying the requested ablation and "
            "aggregate the drop in positive-class probability/logit from the "
            "intact sequence."
        ),
        "ablation_method": args.ablation_method,
        "mask_value": args.mask_value if args.ablation_method == "mask" else None,
        "device": str(device),
        "models": [],
    }
    for spec in spec_set["models"]:
        print(f"Evaluating temporal ablation for {spec['name']}", flush=True)
        if spec["kind"] == "pose":
            model_result_data = evaluate_pose(
                spec,
                data_root=data_root,
                batch_size=args.batch_size,
                device=device,
                dataset_names=selected_datasets,
                ablation_method=args.ablation_method,
                mask_value=args.mask_value,
            )
        elif spec["kind"] == "vjepa":
            model_result_data = evaluate_vjepa(
                spec,
                data_root=data_root,
                batch_size=args.batch_size,
                device=device,
                dataset_names=selected_datasets,
                ablation_method=args.ablation_method,
                mask_value=args.mask_value,
            )
        else:
            raise ValueError(f"Unsupported model kind: {spec['kind']}")

        write_csv(
            output_dir / f"{spec['name']}_temporal_ablation.csv",
            model_result_data["frame_rows"],
        )
        write_csv(
            output_dir / f"{spec['name']}_temporal_ablation_by_dataset.csv",
            model_result_data["dataset_frame_rows"],
        )
        results["models"].append(model_result_data)

    compact_results = {
        **results,
        "models": [
            {
                key: value
                for key, value in model.items()
                if key not in {"frame_rows", "dataset_frame_rows"}
            }
            for model in results["models"]
        ],
    }
    (output_dir / "temporal_ablation_summary.json").write_text(
        json.dumps(compact_results, indent=2) + "\n"
    )
    write_report(compact_results, output_dir)
    print(json.dumps(compact_results, indent=2))
    print(f"Saved temporal ablation outputs under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
