from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
VOLUME_NAME = "final_project_dataset"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy",
        "pandas",
        "opencv-python-headless",
        "scikit-learn",
        "torch",
        "tqdm",
    )
    .add_local_python_source("fall_anticipation_cv")
)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    timeout=60 * 60 * 2,
)
def prepare_windows(
    output_csv: str = f"{DATA_ROOT}/windows_gmdcsa24.csv",
) -> str:
    from fall_anticipation_cv.data import build_window_dataframe, load_gmd_labels

    labels = load_gmd_labels(DATA_ROOT)
    windows = build_window_dataframe(labels)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_path, index=False)
    volume.commit()

    print(f"Labels: {len(labels)}")
    print(f"Matched videos: {labels['video_exists'].sum()}/{len(labels)}")
    print(f"Windows: {windows.shape}")
    print(windows["y"].value_counts())
    print(f"Saved: {output_csv}")

    return output_csv


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="T4",
    timeout=60 * 60 * 8,
)
def train_baseline(
    windows_csv: str = f"{DATA_ROOT}/windows_gmdcsa24.csv",
    checkpoint_path: str = f"{DATA_ROOT}/outputs/baseline_simple_video_cnn.pt",
    epochs: int = 1,
    batch_size: int = 8,
    num_workers: int = 2,
) -> str:
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from fall_anticipation_cv.data import FallWindowDataset, split_by_subject
    from fall_anticipation_cv.models.baseline import SimpleVideoCNN
    from fall_anticipation_cv.training import evaluate, train_one_epoch

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)

    def make_loader(df, shuffle):
        return DataLoader(
            FallWindowDataset(df, resize=(224, 224)),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)
    test_loader = make_loader(test_df, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleVideoCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
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
                checkpoint,
            )
            volume.commit()
            print(f"Saved best baseline checkpoint: {checkpoint_path}")

    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_result = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_result.loss:.4f}")
    print(f"Test acc:  {test_result.accuracy:.4f}")

    return checkpoint_path


@app.local_entrypoint()
def main(
    prepare: bool = False,
    train: bool = True,
    epochs: int = 1,
    batch_size: int = 8,
) -> None:
    windows_csv = f"{DATA_ROOT}/windows_gmdcsa24.csv"

    if prepare:
        prepare_windows.remote(windows_csv)

    if train:
        train_baseline.remote(
            windows_csv=windows_csv,
            epochs=epochs,
            batch_size=batch_size,
        )

