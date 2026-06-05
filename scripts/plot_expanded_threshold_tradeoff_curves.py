from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evaluate_expanded_model_thresholds import MODEL_SPECS
from evaluate_thresholds import (
    classification_metrics_from_probs,
    candidate_thresholds,
    predict_pose,
    predict_vjepa,
    tune_thresholds,
)

DISPLAY_NAMES = {
    "pose_transformer": "Pose Transformer",
    "vjepa_baseline": "V-JEPA Baseline",
    "vjepa_predictive": "V-JEPA Predictive",
}

COLORS = {
    "pose_transformer": "#9E1B1B",
    "vjepa_baseline": "#2B8C83",
    "vjepa_predictive": "#4C5BAA",
}

MARKERS = {
    "f2": "o",
    "balanced_accuracy": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot threshold tradeoff curves for expanded staged+unstaged models."
    )
    parser.add_argument("--data-root", default="/data/final_project_dataset")
    parser.add_argument("--output-dir", default="/data/final_project_dataset/outputs/figures")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-recall", type=float, default=0.75)
    return parser.parse_args()


def metrics_for_thresholds(labels: list[int], probs: list[float]) -> list[dict]:
    rows = []
    for threshold in candidate_thresholds(probs):
        metrics = classification_metrics_from_probs(labels, probs, threshold)
        cm = metrics["confusion_matrix"]
        tn = cm["true_negative"]
        fp = cm["false_positive"]
        fn = cm["false_negative"]
        tp = cm["true_positive"]
        recall = metrics["positive_recall"]
        precision = metrics["positive_precision"]
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "false_positive_rate": float(fpr),
                "specificity": float(specificity),
                "f1": float(metrics["positive_f1"]),
                "accuracy": float(metrics["accuracy"]),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    rows.sort(key=lambda row: (row["recall"], row["precision"]))
    return rows


def nearest_row(rows: list[dict], threshold: float) -> dict:
    return min(rows, key=lambda row: abs(row["threshold"] - threshold))


def collect_model(spec: dict, data_root: Path, batch_size: int, device: torch.device):
    windows_csv = data_root / spec["windows_csv"]
    checkpoint = data_root / spec["checkpoint"]
    if spec["kind"] == "pose":
        return predict_pose(windows_csv, checkpoint, batch_size, device)
    if spec["kind"] == "vjepa":
        return predict_vjepa(windows_csv, checkpoint, batch_size, device)
    raise ValueError(f"Unsupported model kind: {spec['kind']}")


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_curves(model_results: list[dict], output_dir: Path) -> None:
    fig_pr, ax_pr = plt.subplots(figsize=(7.2, 4.6))
    fig_rf, ax_rf = plt.subplots(figsize=(7.2, 4.6))

    for result in model_results:
        key = result["model_key"]
        rows = result["curve_rows"]
        label = DISPLAY_NAMES[key]
        color = COLORS[key]
        recalls = [row["recall"] for row in rows]
        precisions = [row["precision"] for row in rows]
        fprs = [row["false_positive_rate"] for row in rows]

        ax_pr.plot(recalls, precisions, color=color, linewidth=2.2, label=label)
        ax_rf.plot(fprs, recalls, color=color, linewidth=2.2, label=label)

        marker_specs = [
            ("f2", "F2-opt", result["validation_best_f2"]["threshold"]),
            (
                "balanced_accuracy",
                "BalAcc-opt",
                result["validation_best_balanced_accuracy"]["threshold"],
            ),
        ]
        for marker_key, marker_label, threshold in marker_specs:
            row = nearest_row(rows, threshold)
            ax_pr.scatter(
                [row["recall"]],
                [row["precision"]],
                marker=MARKERS[marker_key],
                s=72,
                color=color,
                edgecolor="white",
                linewidth=0.9,
                zorder=5,
            )
            ax_rf.scatter(
                [row["false_positive_rate"]],
                [row["recall"]],
                marker=MARKERS[marker_key],
                s=72,
                color=color,
                edgecolor="white",
                linewidth=0.9,
                zorder=5,
            )

    style_axes(ax_pr, "Recall", "Precision")
    ax_pr.set_title("Expanded Fall Anticipation: Precision-Recall Tradeoff", fontsize=13, weight="bold")
    ax_pr.legend(loc="lower left", frameon=False, fontsize=9)
    ax_pr.text(
        0.99,
        0.03,
        "circle: validation F2 threshold\nsquare: validation balanced-accuracy threshold",
        transform=ax_pr.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    style_axes(ax_rf, "False Positive Rate", "Recall")
    ax_rf.set_title("Expanded Fall Anticipation: Recall vs. False Positive Rate", fontsize=13, weight="bold")
    ax_rf.legend(loc="upper left", frameon=False, fontsize=9)
    ax_rf.text(
        0.99,
        0.03,
        "circle: validation F2 threshold\nsquare: validation balanced-accuracy threshold",
        transform=ax_rf.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fig, stem in [
        (fig_pr, "expanded_precision_recall_curve"),
        (fig_rf, "expanded_recall_fpr_curve"),
    ]:
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_results = []
    curve_csv = output_dir / "expanded_threshold_tradeoff_curve_points.csv"
    marker_json = output_dir / "expanded_threshold_tradeoff_markers.json"

    for spec in MODEL_SPECS:
        print(f"Collecting probabilities for {spec['name']}", flush=True)
        val_labels, val_probs, test_labels, test_probs = collect_model(
            spec, data_root, args.batch_size, device
        )
        tuned = tune_thresholds(val_labels, val_probs, target_recall=args.target_recall)
        rows = metrics_for_thresholds(test_labels, test_probs)
        model_results.append(
            {
                "model_key": spec["name"],
                "curve_rows": rows,
                "validation_best_f2": tuned["best_f2"],
                "validation_best_balanced_accuracy": tuned["best_balanced_accuracy"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with curve_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "threshold",
                "precision",
                "recall",
                "false_positive_rate",
                "specificity",
                "f1",
                "accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ],
        )
        writer.writeheader()
        for result in model_results:
            for row in result["curve_rows"]:
                writer.writerow({"model": result["model_key"], **row})

    markers = {}
    for result in model_results:
        rows = result["curve_rows"]
        markers[result["model_key"]] = {
            "validation_best_f2": result["validation_best_f2"],
            "test_point_at_validation_best_f2": nearest_row(
                rows, result["validation_best_f2"]["threshold"]
            ),
            "validation_best_balanced_accuracy": result[
                "validation_best_balanced_accuracy"
            ],
            "test_point_at_validation_best_balanced_accuracy": nearest_row(
                rows, result["validation_best_balanced_accuracy"]["threshold"]
            ),
        }
    marker_json.write_text(json.dumps(markers, indent=2) + "\n")

    plot_curves(model_results, output_dir)
    print(f"Saved curve figures and data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
