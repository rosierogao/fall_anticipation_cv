import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import FallWindowDataset, split_by_subject
from fall_anticipation_cv.models.baseline import SimpleVideoCNN
from fall_anticipation_cv.training import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline SimpleVideoCNN.")
    parser.add_argument("--windows-csv", required=True, help="Window metadata CSV.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/baseline_simple_video_cnn.pt",
        help="Path for the best validation checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def make_loader(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        FallWindowDataset(df, resize=(224, 224)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)

    train_loader = make_loader(train_df, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_df, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_df, args.batch_size, False, args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleVideoCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_result = evaluate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_result.loss:.4f} | Val acc:   {val_result.accuracy:.4f}")

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_result.loss,
                    "val_acc": val_result.accuracy,
                },
                checkpoint_path,
            )
            print(f"Saved best baseline checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_result.loss:.4f}")
    print(f"Test acc:  {test_result.accuracy:.4f}")


if __name__ == "__main__":
    main()

