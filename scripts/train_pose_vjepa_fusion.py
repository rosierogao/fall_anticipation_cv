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
from fall_anticipation_cv.fusion_data import (
    PoseVJEPALatentWindowDataset,
    collate_pose_vjepa_windows,
)
from fall_anticipation_cv.models.pose_vjepa_fusion import PoseVJEPAFusionTransformer
from fall_anticipation_cv.training_common import (
    binary_classification_metrics,
    compute_class_weights,
    evaluate,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pose + V-JEPA fusion model.")
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument(
        "--pose-windows-csv",
        default=None,
        help=(
            "Optional pose metadata CSV to merge when --windows-csv has "
            "V-JEPA features but no pose_feature_path column."
        ),
    )
    parser.add_argument("--pose-feature-col", default="pose_feature_path")
    parser.add_argument("--vjepa-feature-col", default="vjepa_feature_path")
    parser.add_argument("--checkpoint", default="outputs/pose_vjepa_fusion.pt")
    parser.add_argument("--metrics", default="outputs/pose_vjepa_fusion_metrics.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


def fusion_forward(batch: object, model: torch.nn.Module) -> torch.Tensor:
    pose_features, _labels, vjepa_latents, lengths = batch
    return model(pose_features, vjepa_latents, lengths)


def split_windows(windows: pd.DataFrame):
    if "split" in windows.columns:
        split = windows["split"].astype(str).str.lower()
        return (
            windows[split == "train"].copy(),
            windows[split == "val"].copy(),
            windows[split == "test"].copy(),
        )
    return split_by_subject(windows)


def infer_input_dims(
    windows: pd.DataFrame,
    pose_feature_col: str,
    vjepa_feature_col: str,
    normalize_pose: bool,
    add_velocity: bool,
) -> tuple[int, int]:
    dataset = PoseVJEPALatentWindowDataset(
        windows.head(1),
        pose_feature_col=pose_feature_col,
        vjepa_feature_col=vjepa_feature_col,
        normalize_pose=normalize_pose,
        add_velocity=add_velocity,
    )
    pose, _label, vjepa = dataset[0]
    return int(pose.shape[1]), int(vjepa.shape[1])


def make_loader(
    windows: pd.DataFrame,
    pose_feature_col: str,
    vjepa_feature_col: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    normalize_pose: bool,
    add_velocity: bool,
) -> DataLoader:
    return DataLoader(
        PoseVJEPALatentWindowDataset(
            windows,
            pose_feature_col=pose_feature_col,
            vjepa_feature_col=vjepa_feature_col,
            normalize_pose=normalize_pose,
            add_velocity=add_velocity,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pose_vjepa_windows,
    )


def load_fusion_windows(args: argparse.Namespace) -> pd.DataFrame:
    windows = pd.read_csv(args.windows_csv)
    if args.pose_feature_col in windows.columns:
        return windows

    if args.pose_windows_csv is None:
        raise ValueError(
            f"{args.windows_csv} does not contain {args.pose_feature_col}. "
            "Pass --pose-windows-csv so pose feature paths can be merged in."
        )

    pose_windows = pd.read_csv(args.pose_windows_csv)
    if args.pose_feature_col not in pose_windows.columns:
        raise ValueError(
            f"{args.pose_windows_csv} does not contain {args.pose_feature_col}."
        )

    join_cols = ["video_path", "window_start", "window_end"]
    missing_join_cols = [
        col
        for col in join_cols
        if col not in windows.columns or col not in pose_windows.columns
    ]
    if missing_join_cols:
        raise ValueError(
            "Cannot merge pose and V-JEPA windows; missing join columns: "
            f"{missing_join_cols}"
        )

    pose_feature_map = pose_windows[join_cols + [args.pose_feature_col]].drop_duplicates(
        subset=join_cols
    )
    merged = windows.merge(pose_feature_map, on=join_cols, how="left")
    missing_pose = int(merged[args.pose_feature_col].isna().sum())
    if missing_pose:
        print(
            f"Dropping {missing_pose} windows without matched pose features.",
            flush=True,
        )
        merged = merged.dropna(subset=[args.pose_feature_col])
    return merged


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    metrics_path = Path(args.metrics)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    windows = load_fusion_windows(args)
    required_columns = {args.pose_feature_col, args.vjepa_feature_col, "y"}
    missing = required_columns.difference(windows.columns)
    if missing:
        raise ValueError(f"Missing required columns in windows CSV: {sorted(missing)}")
    windows = windows.dropna(subset=[args.pose_feature_col, args.vjepa_feature_col])

    train_df, val_df, test_df = split_windows(windows)
    normalize_pose = not args.raw_pose
    add_velocity = not args.no_velocity
    pose_dim, vjepa_dim = infer_input_dims(
        train_df,
        args.pose_feature_col,
        args.vjepa_feature_col,
        normalize_pose=normalize_pose,
        add_velocity=add_velocity,
    )

    train_loader = make_loader(
        train_df,
        args.pose_feature_col,
        args.vjepa_feature_col,
        args.batch_size,
        True,
        args.num_workers,
        normalize_pose,
        add_velocity,
    )
    val_loader = make_loader(
        val_df,
        args.pose_feature_col,
        args.vjepa_feature_col,
        args.batch_size,
        False,
        args.num_workers,
        normalize_pose,
        add_velocity,
    )
    test_loader = make_loader(
        test_df,
        args.pose_feature_col,
        args.vjepa_feature_col,
        args.batch_size,
        False,
        args.num_workers,
        normalize_pose,
        add_velocity,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("CUDA unavailable; training will run on CPU.", flush=True)

    model = PoseVJEPAFusionTransformer(
        pose_dim=pose_dim,
        vjepa_dim=vjepa_dim,
        projection_dim=args.projection_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)
    num_parameters = sum(param.numel() for param in model.parameters())
    class_weights = compute_class_weights(
        torch.tensor(train_df["y"].to_numpy(), dtype=torch.long)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            fusion_forward,
        )
        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            fusion_forward,
        )
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "pose_dim": pose_dim,
                    "vjepa_dim": vjepa_dim,
                    "projection_dim": args.projection_dim,
                    "d_model": args.d_model,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "num_parameters": num_parameters,
                    "normalize_pose": normalize_pose,
                    "add_velocity": add_velocity,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "model_name": "pose_vjepa_fusion_transformer",
                },
                checkpoint,
            )
            print(f"Saved best pose + V-JEPA fusion checkpoint: {checkpoint}")

    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_acc, predictions, labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
        fusion_forward,
    )

    metrics = {
        "model": "pose_vjepa_fusion_transformer",
        "windows_csv": args.windows_csv,
        "pose_feature_col": args.pose_feature_col,
        "vjepa_feature_col": args.vjepa_feature_col,
        "checkpoint": str(checkpoint),
        "pose_dim": pose_dim,
        "vjepa_dim": vjepa_dim,
        "projection_dim": args.projection_dim,
        "fused_dim": args.projection_dim * 2,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "num_parameters": num_parameters,
        "normalize_pose": normalize_pose,
        "add_velocity": add_velocity,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "class_weights": class_weights.detach().cpu().tolist(),
        **binary_classification_metrics(labels, predictions),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
