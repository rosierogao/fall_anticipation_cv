from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.models.vjepa_predictive import (
    DEFAULT_PREDICTIVE_LOSS_WEIGHT,
    VJEPABaseline,
    VJEPALatentPredictiveModel,
)
from fall_anticipation_cv.training_common import (
    binary_classification_metrics,
    compute_class_weights,
)
from fall_anticipation_cv.vjepa_data import (
    VJEPALatentWindowDataset,
    collate_vjepa_latent_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a V-JEPA latent model.")
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument("--feature-col", default="vjepa_feature_path")
    parser.add_argument("--checkpoint", default="outputs/vjepa_latent_predictive.pt")
    parser.add_argument("--metrics", default="outputs/vjepa_latent_predictive_metrics.json")
    parser.add_argument(
        "--model",
        choices=["baseline", "predictive"],
        default="predictive",
        help="Train classification-only V-JEPA baseline or predictive-loss model.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--predictive-loss-weight",
        type=float,
        default=DEFAULT_PREDICTIVE_LOSS_WEIGHT,
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def make_loader(
    windows: pd.DataFrame,
    feature_col: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        VJEPALatentWindowDataset(windows, feature_col=feature_col),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_vjepa_latent_windows,
    )


def infer_shapes(windows: pd.DataFrame, feature_col: str) -> tuple[int, int]:
    import numpy as np

    features = np.load(Path(windows.iloc[0][feature_col]))
    observed = features["observed_latents"]
    future = features["future_latents"]
    return int(observed.shape[-1]), int(future.shape[0])


def run_epoch(model, loader, optimizer, device, class_weights, use_predictive_loss: bool):
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_pred_loss = 0.0
    correct = 0
    total = 0

    for observed, labels, future, lengths in tqdm(loader, desc="Training"):
        observed = observed.to(device)
        labels = labels.to(device)
        future = future.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        if use_predictive_loss:
            output = model(
                observed,
                future_latents=future,
                labels=labels,
                lengths=lengths,
                class_weights=class_weights,
            )
        else:
            output = model(
                observed,
                labels=labels,
                lengths=lengths,
                class_weights=class_weights,
            )
        output.loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += output.loss.item() * batch_size
        total_cls_loss += output.classification_loss.item() * batch_size
        pred_loss = 0.0 if output.predictive_loss is None else output.predictive_loss.item()
        total_pred_loss += pred_loss * batch_size
        predictions = torch.argmax(output.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return (
        total_loss / total,
        total_cls_loss / total,
        total_pred_loss / total,
        correct / total,
    )


@torch.no_grad()
def evaluate(model, loader, device, class_weights, use_predictive_loss: bool):
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_pred_loss = 0.0
    correct = 0
    total = 0
    predictions_all = []
    labels_all = []

    for observed, labels, future, lengths in tqdm(loader, desc="Evaluating"):
        observed = observed.to(device)
        labels = labels.to(device)
        future = future.to(device)
        lengths = lengths.to(device)

        if use_predictive_loss:
            output = model(
                observed,
                future_latents=future,
                labels=labels,
                lengths=lengths,
                class_weights=class_weights,
            )
        else:
            output = model(
                observed,
                labels=labels,
                lengths=lengths,
                class_weights=class_weights,
            )
        predictions = torch.argmax(output.logits, dim=1)

        batch_size = labels.size(0)
        total_loss += output.loss.item() * batch_size
        total_cls_loss += output.classification_loss.item() * batch_size
        pred_loss = 0.0 if output.predictive_loss is None else output.predictive_loss.item()
        total_pred_loss += pred_loss * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size
        predictions_all.extend(predictions.cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())

    return (
        total_loss / total,
        total_cls_loss / total,
        total_pred_loss / total,
        correct / total,
        predictions_all,
        labels_all,
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    metrics_path = Path(args.metrics)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)
    latent_dim, future_steps = infer_shapes(train_df, args.feature_col)

    train_loader = make_loader(
        train_df,
        args.feature_col,
        args.batch_size,
        True,
        args.num_workers,
    )
    val_loader = make_loader(
        val_df,
        args.feature_col,
        args.batch_size,
        False,
        args.num_workers,
    )
    test_loader = make_loader(
        test_df,
        args.feature_col,
        args.batch_size,
        False,
        args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_predictive_loss = args.model == "predictive"
    if use_predictive_loss:
        model = VJEPALatentPredictiveModel(
            latent_dim=latent_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            future_steps=future_steps,
            predictive_loss_weight=args.predictive_loss_weight,
        ).to(device)
        model_name = "vjepa_latent_predictive"
    else:
        model = VJEPABaseline(
            latent_dim=latent_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
        ).to(device)
        model_name = "vjepa_baseline"

    class_weights = compute_class_weights(
        torch.tensor(train_df["y"].to_numpy(), dtype=torch.long)
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_cls_loss, train_pred_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            use_predictive_loss,
        )
        val_loss, val_cls_loss, val_pred_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            device,
            class_weights,
            use_predictive_loss,
        )
        print(
            "Train loss: "
            f"{train_loss:.4f} | cls: {train_cls_loss:.4f} | "
            f"pred: {train_pred_loss:.4f} | acc: {train_acc:.4f}"
        )
        print(
            "Val loss:   "
            f"{val_loss:.4f} | cls: {val_cls_loss:.4f} | "
            f"pred: {val_pred_loss:.4f} | acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latent_dim": latent_dim,
                    "future_steps": future_steps,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "predictive_loss_weight": args.predictive_loss_weight,
                    "model_name": model_name,
                },
                checkpoint_path,
            )
            print(f"Saved best V-JEPA checkpoint: {checkpoint_path}")

    saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_cls_loss, test_pred_loss, test_acc, predictions, labels = evaluate(
        model,
        test_loader,
        device,
        class_weights,
        use_predictive_loss,
    )

    metrics = {
        "model": model_name,
        "windows_csv": args.windows_csv,
        "feature_col": args.feature_col,
        "checkpoint": str(checkpoint_path),
        "latent_dim": latent_dim,
        "future_steps": future_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_classification_loss": test_cls_loss,
        "test_predictive_loss": test_pred_loss,
        "test_acc": test_acc,
        "class_weights": class_weights.detach().cpu().tolist(),
        "predictive_loss_weight": args.predictive_loss_weight if use_predictive_loss else 0.0,
        **binary_classification_metrics(labels, predictions),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
