from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.models.pose_baseline import PoseTransformerBaseline
from fall_anticipation_cv.training_common import (
    binary_classification_metrics,
    compute_auc_pr,
    compute_class_weights,
    default_pose_forward,
    evaluate,
    evaluate_with_proba,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pose transformer for full-clip fall classification."
    )
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument("--feature-col", default="pose_feature_path")
    parser.add_argument("--checkpoint", default="outputs/pose_classification.pt")
    parser.add_argument("--metrics", default="outputs/pose_classification_metrics.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Cap sequence length via uniform subsampling. Positional encoding sized to match.",
    )
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


def infer_input_dim(
    windows: pd.DataFrame,
    feature_col: str,
    normalize: bool,
    add_velocity: bool,
) -> int:
    import numpy as np
    from fall_anticipation_cv.pose_data import prepare_pose_features

    feature_path = Path(windows.iloc[0][feature_col])
    features = prepare_pose_features(
        np.load(feature_path),
        normalize=normalize,
        add_velocity=add_velocity,
    )
    return int(features.shape[1])


def make_loader(
    windows: pd.DataFrame,
    feature_col: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    normalize: bool,
    add_velocity: bool,
    max_seq_len: int | None = None,
) -> DataLoader:
    from fall_anticipation_cv.pose_data import PoseWindowDataset, collate_pose_windows

    return DataLoader(
        PoseWindowDataset(
            windows,
            feature_col=feature_col,
            normalize=normalize,
            add_velocity=add_velocity,
            max_seq_len=max_seq_len,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pose_windows,
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training"):
        features, labels, lengths = (t.to(device) for t in batch)
        optimizer.zero_grad()
        logits = model(features, lengths)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    metrics_path = Path(args.metrics)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)
    normalize_pose = not args.raw_pose
    add_velocity = not args.no_velocity
    input_dim = infer_input_dim(
        train_df,
        args.feature_col,
        normalize=normalize_pose,
        add_velocity=add_velocity,
    )

    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Input dim: {input_dim}")

    train_loader = make_loader(train_df, args.feature_col, args.batch_size, True, args.num_workers, normalize_pose, add_velocity, args.max_seq_len)
    val_loader = make_loader(val_df, args.feature_col, args.batch_size, False, args.num_workers, normalize_pose, add_velocity, args.max_seq_len)
    test_loader = make_loader(test_df, args.feature_col, args.batch_size, False, args.num_workers, normalize_pose, add_velocity, args.max_seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PoseTransformerBaseline(
        input_dim=input_dim,
        d_model=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    class_weights = compute_class_weights(
        torch.tensor(train_df["y"].to_numpy(), dtype=torch.long)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 20
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch + 1}/{args.epochs}  (lr={current_lr:.2e})")

        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip,
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, default_pose_forward,
        )
        scheduler.step()

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_dim": input_dim,
                    "normalize_pose": normalize_pose,
                    "add_velocity": add_velocity,
                    "d_model": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_weights": class_weights.detach().cpu().tolist(),
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs (patience={args.patience}).")
                break

    saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_acc, predictions, labels, probas = evaluate_with_proba(
        model, test_loader, criterion, device, default_pose_forward,
    )
    auc_pr = compute_auc_pr(labels, probas)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | AUC-PR: {auc_pr:.4f}")

    metrics = {
        "model": "pose_classification",
        "windows_csv": args.windows_csv,
        "feature_col": args.feature_col,
        "checkpoint": str(checkpoint_path),
        "input_dim": input_dim,
        "d_model": args.hidden_dim,
        "num_layers": args.num_layers,
        "normalize_pose": normalize_pose,
        "add_velocity": add_velocity,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "auc_pr": auc_pr,
        "class_weights": class_weights.detach().cpu().tolist(),
        **binary_classification_metrics(labels, predictions),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
