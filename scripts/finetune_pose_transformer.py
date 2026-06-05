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
        description=(
            "Fine-tune a pretrained PoseTransformerBaseline by updating only "
            "the transformer encoder and classifier head while optionally "
            "keeping the input projection frozen as a fixed feature extractor."
        )
    )
    parser.add_argument("--pretrained-checkpoint", required=True,
                        help="Path to pretrained .pt checkpoint.")
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument("--feature-col", default="pose_feature_path")
    parser.add_argument("--checkpoint", default="outputs/pose_transformer_finetuned.pt")
    parser.add_argument("--metrics", default="outputs/pose_transformer_finetuned_metrics.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of transformer layers. Defaults to pretrained value.")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Transformer d_model. Defaults to pretrained value.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate (lower than initial training).")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled).")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor for cross-entropy loss.")
    parser.add_argument("--patience", type=int, default=4,
                        help="Early stopping patience in epochs without val_loss improvement.")
    parser.add_argument("--freeze-projection", action="store_true",
                        help="Freeze input projection, fine-tuning only the transformer and head.")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _infer_arch_from_state_dict(state_dict: dict) -> dict[str, int]:
    d_model = int(state_dict["temporal_classifier.input_projection.weight"].shape[0])
    num_layers = sum(
        1
        for k in state_dict
        if k.startswith("temporal_classifier.encoder.layers.")
        and k.endswith(".self_attn.in_proj_weight")
    )
    return {"d_model": d_model, "num_layers": max(num_layers, 1)}


def make_loader(
    windows: pd.DataFrame,
    feature_col: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    normalize: bool,
    add_velocity: bool,
) -> DataLoader:
    from fall_anticipation_cv.pose_data import PoseWindowDataset, collate_pose_windows

    return DataLoader(
        PoseWindowDataset(
            windows,
            feature_col=feature_col,
            normalize=normalize,
            add_velocity=add_velocity,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pose_windows,
    )


def finetune_one_epoch(
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
        features, labels, lengths = _move_to_device(batch, device)

        optimizer.zero_grad()
        logits = model(features, lengths)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def _move_to_device(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels, lengths = batch
    return features.to(device), labels.to(device), lengths.to(device)


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    metrics_path = Path(args.metrics)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    pretrained_path = Path(args.pretrained_checkpoint)
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    saved = torch.load(pretrained_path, map_location=device)
    arch = _infer_arch_from_state_dict(saved["model_state_dict"])
    input_dim: int = int(saved["input_dim"])
    normalize_pose: bool = bool(saved.get("normalize_pose", True))
    add_velocity: bool = bool(saved.get("add_velocity", True))

    d_model = args.hidden_dim if args.hidden_dim is not None else arch["d_model"]
    num_layers = args.num_layers if args.num_layers is not None else arch["num_layers"]

    print(f"Pretrained arch: d_model={arch['d_model']}, num_layers={arch['num_layers']}, input_dim={input_dim}")
    print(f"Fine-tune arch:  d_model={d_model}, num_layers={num_layers}")

    windows = pd.read_csv(args.windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)

    train_loader = make_loader(train_df, args.feature_col, args.batch_size, True, args.num_workers, normalize_pose, add_velocity)
    val_loader = make_loader(val_df, args.feature_col, args.batch_size, False, args.num_workers, normalize_pose, add_velocity)
    test_loader = make_loader(test_df, args.feature_col, args.batch_size, False, args.num_workers, normalize_pose, add_velocity)

    model = PoseTransformerBaseline(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
        dropout=args.dropout,
    ).to(device)

    if d_model != arch["d_model"]:
        # All tensor shapes differ when d_model changes — train from scratch.
        print(f"d_model changed ({arch['d_model']} → {d_model}) — initializing from scratch.")
    elif num_layers != arch["num_layers"]:
        # Same d_model, different depth — partial load (new layers get random init).
        missing, _ = model.load_state_dict(saved["model_state_dict"], strict=False)
        print(f"Partial weight load — missing keys: {missing}")
    else:
        model.load_state_dict(saved["model_state_dict"], strict=True)

    if args.freeze_projection:
        for param in model.temporal_classifier.input_projection.parameters():
            param.requires_grad = False
        print("Input projection frozen.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,}")

    class_weights = compute_class_weights(
        torch.tensor(train_df["y"].to_numpy(), dtype=torch.long)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
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

        train_loss, train_acc = finetune_one_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, default_pose_forward
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
                    "d_model": d_model,
                    "num_layers": num_layers,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "pretrained_checkpoint": str(pretrained_path),
                    "freeze_projection": args.freeze_projection,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs (patience={args.patience}).")
                break

    best_saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_saved["model_state_dict"])
    test_loss, test_acc, predictions, labels, probas = evaluate_with_proba(
        model, test_loader, criterion, device, default_pose_forward
    )
    auc_pr = compute_auc_pr(labels, probas)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | AUC-PR: {auc_pr:.4f}")

    metrics = {
        "model": "pose_transformer_finetuned",
        "pretrained_checkpoint": str(pretrained_path),
        "windows_csv": args.windows_csv,
        "feature_col": args.feature_col,
        "checkpoint": str(checkpoint_path),
        "input_dim": input_dim,
        "d_model": d_model,
        "num_layers": num_layers,
        "normalize_pose": normalize_pose,
        "add_velocity": add_velocity,
        "freeze_projection": args.freeze_projection,
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
