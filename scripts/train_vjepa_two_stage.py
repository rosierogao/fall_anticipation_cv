"""Two-stage V-JEPA latent training: pre-train on OOPs, fine-tune on surveillance.

Stage 1 trains the full model on a large, diverse dataset (e.g. OOPs).
Stage 2 loads the Stage-1 checkpoint and fine-tunes on a smaller target dataset
(e.g. GMDCSA24 + le2i) with a lower learning rate and fewer epochs.

Usage:
    python scripts/train_vjepa_two_stage.py \
        --pretrain-csv /data/.../vjepa_windows_real_oops.csv \
        --finetune-csv /data/.../vjepa_windows.csv \
        --checkpoint outputs/vjepa_two_stage.pt \
        --metrics   outputs/vjepa_two_stage_metrics.json
"""
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
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage V-JEPA latent training. "
            "Stage 1 pre-trains on --pretrain-csv (e.g. OOPs). "
            "Stage 2 fine-tunes on --finetune-csv (e.g. GMDCSA24 + le2i)."
        )
    )
    # ── data ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--pretrain-csv",
        required=True,
        help="Windows CSV for Stage 1 (large/diverse dataset, e.g. OOPs).",
    )
    parser.add_argument(
        "--finetune-csv",
        required=True,
        help="Windows CSV for Stage 2 (target surveillance dataset).",
    )
    parser.add_argument("--feature-col", default="vjepa_feature_path")
    parser.add_argument("--checkpoint", default="outputs/vjepa_two_stage.pt")
    parser.add_argument("--metrics", default="outputs/vjepa_two_stage_metrics.json")
    # ── architecture ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        choices=["baseline", "predictive"],
        default="predictive",
        help="Classification-only baseline or predictive-loss model.",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    # ── stage 1 ─────────────────────────────────────────────────────────────
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument(
        "--pretrain-predictive-loss-weight",
        type=float,
        default=DEFAULT_PREDICTIVE_LOSS_WEIGHT,
    )
    # ── stage 2 ─────────────────────────────────────────────────────────────
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-batch-size", type=int, default=32)
    parser.add_argument(
        "--finetune-predictive-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Predictive loss weight for Stage 2. "
            "Defaults to 0 (classification-only fine-tuning)."
        ),
    )
    parser.add_argument(
        "--freeze-body",
        action="store_true",
        help=(
            "Freeze the transformer body during Stage 2, "
            "updating only the final classifier linear layer. "
            "Reduces overfitting risk when fine-tune data is small."
        ),
    )
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


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor,
    use_predictive_loss: bool,
    predictive_loss_weight: float,
) -> tuple[float, float, float, float]:
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

    return total_loss / total, total_cls_loss / total, total_pred_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    use_predictive_loss: bool,
) -> tuple[float, float, float, float, list[int], list[int]]:
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


def build_model(
    model_type: str,
    latent_dim: int,
    d_model: int,
    num_layers: int,
    future_steps: int,
    predictive_loss_weight: float,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    if model_type == "predictive":
        model = VJEPALatentPredictiveModel(
            latent_dim=latent_dim,
            d_model=d_model,
            num_layers=num_layers,
            future_steps=future_steps,
            predictive_loss_weight=predictive_loss_weight,
        ).to(device)
        return model, "vjepa_latent_predictive_two_stage"
    model = VJEPABaseline(
        latent_dim=latent_dim,
        d_model=d_model,
        num_layers=num_layers,
    ).to(device)
    return model, "vjepa_baseline_two_stage"


def run_stage(
    stage_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int,
    use_predictive_loss: bool,
    predictive_loss_weight: float,
    stage1_checkpoint: Path,
) -> dict:
    """Train for one stage and return the best-val-loss epoch metrics."""
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    stage_history = []

    for epoch in range(epochs):
        print(f"\n[{stage_name}] Epoch {epoch + 1}/{epochs}")
        train_loss, train_cls_loss, train_pred_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            use_predictive_loss,
            predictive_loss_weight,
        )
        val_loss, val_cls_loss, val_pred_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            device,
            class_weights,
            use_predictive_loss,
        )
        print(
            f"  Train loss: {train_loss:.4f} | cls: {train_cls_loss:.4f} | "
            f"pred: {train_pred_loss:.4f} | acc: {train_acc:.4f}"
        )
        print(
            f"  Val   loss: {val_loss:.4f} | cls: {val_cls_loss:.4f} | "
            f"pred: {val_pred_loss:.4f} | acc: {val_acc:.4f}"
        )
        stage_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_cls_loss": train_cls_loss,
                "train_pred_loss": train_pred_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_cls_loss": val_cls_loss,
                "val_pred_loss": val_pred_loss,
                "val_acc": val_acc,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), stage1_checkpoint)
            print(f"  Saved best checkpoint: {stage1_checkpoint}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": stage_history,
    }


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    metrics_path = Path(args.metrics)
    stage1_checkpoint = checkpoint_path.with_suffix(".stage1.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_predictive_loss = args.model == "predictive"

    # ── load data ────────────────────────────────────────────────────────────
    pretrain_windows = pd.read_csv(args.pretrain_csv)
    finetune_windows = pd.read_csv(args.finetune_csv)

    # Drop duplicate windows (same video/start/end with different feature paths) that
    # can arise when feature extraction was re-run after the source CSV was regenerated.
    _dedup_keys = ["video_path", "window_start", "window_end"]
    n_before = len(finetune_windows)
    finetune_windows = finetune_windows.drop_duplicates(subset=_dedup_keys).reset_index(drop=True)
    if len(finetune_windows) < n_before:
        print(f"Dropped {n_before - len(finetune_windows)} duplicate windows from finetune CSV.")

    pretrain_train, pretrain_val, pretrain_test = split_by_subject(pretrain_windows)
    finetune_train, finetune_val, finetune_test = split_by_subject(finetune_windows)

    latent_dim, future_steps = infer_shapes(pretrain_train, args.feature_col)
    print(f"Latent dim: {latent_dim}  Future steps: {future_steps}")
    print(
        f"Stage 1 — train: {len(pretrain_train):,}  "
        f"val: {len(pretrain_val):,}  test: {len(pretrain_test):,}"
    )
    print(
        f"Stage 2 — train: {len(finetune_train):,}  "
        f"val: {len(finetune_val):,}  test: {len(finetune_test):,}"
    )

    pretrain_train_loader = make_loader(
        pretrain_train, args.feature_col, args.pretrain_batch_size, True, args.num_workers
    )
    pretrain_val_loader = make_loader(
        pretrain_val, args.feature_col, args.pretrain_batch_size, False, args.num_workers
    )

    finetune_train_loader = make_loader(
        finetune_train, args.feature_col, args.finetune_batch_size, True, args.num_workers
    )
    finetune_val_loader = make_loader(
        finetune_val, args.feature_col, args.finetune_batch_size, False, args.num_workers
    )
    finetune_test_loader = make_loader(
        finetune_test, args.feature_col, args.finetune_batch_size, False, args.num_workers
    )

    # ── build model ──────────────────────────────────────────────────────────
    model, model_name = build_model(
        args.model,
        latent_dim,
        args.d_model,
        args.num_layers,
        future_steps,
        args.pretrain_predictive_loss_weight,
        device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}  Parameters: {n_params:,}")

    # ── stage 1: pre-train ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 1 — pre-train on {args.pretrain_csv}")
    print(f"  epochs={args.pretrain_epochs}  lr={args.pretrain_lr}  "
          f"predictive_loss_weight={args.pretrain_predictive_loss_weight}")
    print(f"{'='*60}")

    pretrain_class_weights = compute_class_weights(
        torch.tensor(pretrain_train["y"].to_numpy(), dtype=torch.long)
    ).to(device)

    stage1_optimizer = optim.Adam(
        model.parameters(),
        lr=args.pretrain_lr,
        weight_decay=args.weight_decay,
    )

    stage1_metrics = run_stage(
        stage_name="Stage 1",
        model=model,
        train_loader=pretrain_train_loader,
        val_loader=pretrain_val_loader,
        optimizer=stage1_optimizer,
        device=device,
        class_weights=pretrain_class_weights,
        epochs=args.pretrain_epochs,
        use_predictive_loss=use_predictive_loss,
        predictive_loss_weight=args.pretrain_predictive_loss_weight,
        stage1_checkpoint=stage1_checkpoint,
    )

    print(
        f"\nStage 1 best: epoch {stage1_metrics['best_epoch'] + 1}  "
        f"val_loss={stage1_metrics['best_val_loss']:.4f}  "
        f"val_acc={stage1_metrics['best_val_acc']:.4f}"
    )

    # ── stage 2: fine-tune ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 2 — fine-tune on {args.finetune_csv}")
    print(f"  epochs={args.finetune_epochs}  lr={args.finetune_lr}  "
          f"predictive_loss_weight={args.finetune_predictive_loss_weight}  "
          f"freeze_body={args.freeze_body}")
    print(f"{'='*60}")

    # Load the best Stage 1 weights before starting Stage 2.
    model.load_state_dict(torch.load(stage1_checkpoint, map_location=device))

    # Update predictive loss weight for stage 2 on the model.
    if use_predictive_loss:
        model.predictive_loss_weight = args.finetune_predictive_loss_weight

    if args.freeze_body:
        # Freeze everything except the final classifier linear layer so that
        # Stage 2 only shifts the decision boundary, not the representation.
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.classifier.parameters():
            param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Body frozen. Trainable parameters: {n_trainable:,} / {n_params:,}")

    use_finetune_predictive = use_predictive_loss and args.finetune_predictive_loss_weight > 0.0

    finetune_class_weights = compute_class_weights(
        torch.tensor(finetune_train["y"].to_numpy(), dtype=torch.long)
    ).to(device)

    stage2_optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )

    stage2_metrics = run_stage(
        stage_name="Stage 2",
        model=model,
        train_loader=finetune_train_loader,
        val_loader=finetune_val_loader,
        optimizer=stage2_optimizer,
        device=device,
        class_weights=finetune_class_weights,
        epochs=args.finetune_epochs,
        use_predictive_loss=use_finetune_predictive,
        predictive_loss_weight=args.finetune_predictive_loss_weight,
        stage1_checkpoint=checkpoint_path,
    )

    print(
        f"\nStage 2 best: epoch {stage2_metrics['best_epoch'] + 1}  "
        f"val_loss={stage2_metrics['best_val_loss']:.4f}  "
        f"val_acc={stage2_metrics['best_val_acc']:.4f}"
    )

    # ── final evaluation on finetune test set ────────────────────────────────
    # run_stage saves a raw state_dict; load it back for evaluation.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_cls_loss, test_pred_loss, test_acc, predictions, labels = evaluate(
        model,
        finetune_test_loader,
        device,
        finetune_class_weights,
        use_finetune_predictive,
    )
    print(
        f"\nTest — loss: {test_loss:.4f} | cls: {test_cls_loss:.4f} | "
        f"pred: {test_pred_loss:.4f} | acc: {test_acc:.4f}"
    )

    # ── save full checkpoint with metadata ───────────────────────────────────
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "latent_dim": latent_dim,
            "future_steps": future_steps,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "pretrain_class_weights": pretrain_class_weights.detach().cpu().tolist(),
            "finetune_class_weights": finetune_class_weights.detach().cpu().tolist(),
            "stage2_best_epoch": stage2_metrics["best_epoch"],
            "stage2_val_loss": stage2_metrics["best_val_loss"],
            "stage2_val_acc": stage2_metrics["best_val_acc"],
        },
        checkpoint_path,
    )

    metrics = {
        "model": model_name,
        "pretrain_csv": args.pretrain_csv,
        "finetune_csv": args.finetune_csv,
        "feature_col": args.feature_col,
        "checkpoint": str(checkpoint_path),
        "latent_dim": latent_dim,
        "future_steps": future_steps,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "freeze_body": args.freeze_body,
        "stage1": {
            "epochs": args.pretrain_epochs,
            "lr": args.pretrain_lr,
            "batch_size": args.pretrain_batch_size,
            "predictive_loss_weight": args.pretrain_predictive_loss_weight,
            "class_weights": pretrain_class_weights.detach().cpu().tolist(),
            **{k: v for k, v in stage1_metrics.items() if k != "history"},
            "history": stage1_metrics["history"],
        },
        "stage2": {
            "epochs": args.finetune_epochs,
            "lr": args.finetune_lr,
            "batch_size": args.finetune_batch_size,
            "predictive_loss_weight": args.finetune_predictive_loss_weight,
            "class_weights": finetune_class_weights.detach().cpu().tolist(),
            **{k: v for k, v in stage2_metrics.items() if k != "history"},
            "history": stage2_metrics["history"],
        },
        "test_loss": test_loss,
        "test_classification_loss": test_cls_loss,
        "test_predictive_loss": test_pred_loss,
        "test_acc": test_acc,
        **binary_classification_metrics(labels, predictions),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
