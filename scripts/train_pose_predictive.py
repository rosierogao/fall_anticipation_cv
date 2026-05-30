from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.models.pose_predictive import (
    DEFAULT_POSE_PREDICTIVE_LOSS_WEIGHT,
    PoseSeq2SeqPredictiveModel,
)
from fall_anticipation_cv.pose_predictive_data import (
    PosePredictiveWindowDataset,
    collate_pose_predictive_windows,
)
from fall_anticipation_cv.training_common import (
    binary_classification_metrics,
    compute_class_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train pose future-prediction + fall-classification model."
    )
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument("--feature-col", default="pose_predictive_feature_path")
    parser.add_argument("--checkpoint", default="outputs/pose_predictive.pt")
    parser.add_argument("--metrics", default="outputs/pose_predictive_metrics.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--predictive-loss-weight", type=float, default=DEFAULT_POSE_PREDICTIVE_LOSS_WEIGHT)
    parser.add_argument("--lr-plateau-patience", type=int, default=2)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--raw-pose",
        action="store_true",
        help="Use raw RTMPose coordinates instead of centered/scaled pose features.",
    )
    parser.add_argument(
        "--no-velocity",
        action="store_true",
        help="Do not append frame-to-frame pose velocity features.",
    )
    return parser.parse_args()


def make_loader(
    windows: pd.DataFrame,
    feature_col: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    normalize: bool,
    add_velocity: bool,
) -> DataLoader:
    return DataLoader(
        PosePredictiveWindowDataset(
            windows,
            feature_col=feature_col,
            normalize=normalize,
            add_velocity=add_velocity,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pose_predictive_windows,
    )


def infer_shapes(
    windows: pd.DataFrame,
    feature_col: str,
    normalize: bool,
    add_velocity: bool,
) -> tuple[int, int]:
    dataset = PosePredictiveWindowDataset(
        windows.head(1),
        feature_col=feature_col,
        normalize=normalize,
        add_velocity=add_velocity,
    )
    observed, _label, future = dataset[0]
    return int(observed.shape[-1]), int(future.shape[0])


def run_epoch(model, loader, optimizer, device, class_weights):
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
        output = model(
            observed,
            future_pose=future,
            labels=labels,
            lengths=lengths,
            class_weights=class_weights,
        )
        output.loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += output.loss.item() * batch_size
        total_cls_loss += output.classification_loss.item() * batch_size
        total_pred_loss += output.predictive_loss.item() * batch_size
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
def evaluate(model, loader, device, class_weights):
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

        output = model(
            observed,
            future_pose=future,
            labels=labels,
            lengths=lengths,
            class_weights=class_weights,
        )
        predictions = torch.argmax(output.logits, dim=1)

        batch_size = labels.size(0)
        total_loss += output.loss.item() * batch_size
        total_cls_loss += output.classification_loss.item() * batch_size
        total_pred_loss += output.predictive_loss.item() * batch_size
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
    history_path = metrics_path.with_name(f"{metrics_path.stem}_history.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)
    normalize_pose = not args.raw_pose
    add_velocity = not args.no_velocity
    input_dim, future_steps = infer_shapes(
        train_df,
        args.feature_col,
        normalize=normalize_pose,
        add_velocity=add_velocity,
    )

    train_loader = make_loader(
        train_df,
        args.feature_col,
        args.batch_size,
        True,
        args.num_workers,
        normalize_pose,
        add_velocity,
    )
    val_loader = make_loader(
        val_df,
        args.feature_col,
        args.batch_size,
        False,
        args.num_workers,
        normalize_pose,
        add_velocity,
    )
    test_loader = make_loader(
        test_df,
        args.feature_col,
        args.batch_size,
        False,
        args.num_workers,
        normalize_pose,
        add_velocity,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseSeq2SeqPredictiveModel(
        input_dim=input_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        future_steps=future_steps,
        predictive_loss_weight=args.predictive_loss_weight,
    ).to(device)

    class_weights = compute_class_weights(
        torch.tensor(train_df["y"].to_numpy(), dtype=torch.long)
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
        min_lr=args.min_lr,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    epoch_history = []
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_cls_loss, train_pred_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
        )
        val_loss, val_cls_loss, val_pred_loss, val_acc, _val_preds, _val_labels = evaluate(
            model,
            val_loader,
            device,
            class_weights,
        )
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        print(
            "Train loss: "
            f"{train_loss:.4f} | cls: {train_cls_loss:.4f} | "
            f"pose pred: {train_pred_loss:.4f} | acc: {train_acc:.4f}"
        )
        print(
            "Val loss:   "
            f"{val_loss:.4f} | cls: {val_cls_loss:.4f} | "
            f"pose pred: {val_pred_loss:.4f} | acc: {val_acc:.4f} | lr: {current_lr:.2e}"
        )

        epoch_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_classification_loss": train_cls_loss,
                "train_predictive_loss": train_pred_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_classification_loss": val_cls_loss,
                "val_predictive_loss": val_pred_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
        )
        history_path.write_text(json.dumps(epoch_history, indent=2) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_name": "pose_seq2seq_predictive",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_dim": input_dim,
                    "d_model": args.d_model,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "future_steps": future_steps,
                    "predictive_loss_weight": args.predictive_loss_weight,
                    "normalize_pose": normalize_pose,
                    "add_velocity": add_velocity,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_weights": class_weights.detach().cpu().tolist(),
                },
                checkpoint_path,
            )
            print(f"Saved best pose predictive checkpoint: {checkpoint_path}")

    saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_cls_loss, test_pred_loss, test_acc, predictions, labels = evaluate(
        model,
        test_loader,
        device,
        class_weights,
    )

    metrics = {
        "model": "pose_seq2seq_predictive",
        "windows_csv": args.windows_csv,
        "feature_col": args.feature_col,
        "checkpoint": str(checkpoint_path),
        "input_dim": input_dim,
        "future_steps": future_steps,
        "normalize_pose": normalize_pose,
        "add_velocity": add_velocity,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "predictive_loss_weight": args.predictive_loss_weight,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_classification_loss": test_cls_loss,
        "test_predictive_loss": test_pred_loss,
        "test_acc": test_acc,
        "class_weights": class_weights.detach().cpu().tolist(),
        "history_path": str(history_path),
        **binary_classification_metrics(labels, predictions),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
