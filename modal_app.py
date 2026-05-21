from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "matplotlib",
        "numpy",
        "pandas",
        "opencv-python-headless",
        "scikit-learn",
        "torch",
        "tqdm",
    )
    .add_local_dir(
        LOCAL_PACKAGE_DIR,
        f"{PACKAGE_REMOTE_ROOT}/fall_anticipation_cv",
        copy=True,
    )
    .env({"PYTHONPATH": PACKAGE_REMOTE_ROOT})
)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    timeout=60 * 60 * 2,
)
def prepare_windows(
    output_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
) -> str:
    from fall_anticipation_cv.data import (
        build_window_dataframe,
        load_gmd_labels,
        validate_windows,
    )

    labels = load_gmd_labels(DATA_ROOT)
    windows = build_window_dataframe(labels)
    validate_windows(windows)

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
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
    checkpoint_path: str = f"{DATASET_ROOT}/outputs/baseline_simple_video_cnn.pt",
    metrics_path: str = f"{DATASET_ROOT}/outputs/baseline_metrics.json",
    epochs: int = 1,
    batch_size: int = 8,
    num_workers: int = 2,
) -> str:
    import json

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

    labels = test_result.labels
    predictions = test_result.predictions
    true_negative = sum(1 for y, y_hat in zip(labels, predictions) if y == 0 and y_hat == 0)
    false_positive = sum(1 for y, y_hat in zip(labels, predictions) if y == 0 and y_hat == 1)
    false_negative = sum(1 for y, y_hat in zip(labels, predictions) if y == 1 and y_hat == 0)
    true_positive = sum(1 for y, y_hat in zip(labels, predictions) if y == 1 and y_hat == 1)
    positive_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    positive_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    positive_f1 = (
        2 * positive_precision * positive_recall / (positive_precision + positive_recall)
        if positive_precision + positive_recall > 0
        else 0.0
    )

    metrics = {
        "model": "baseline_simple_video_cnn",
        "checkpoint_path": checkpoint_path,
        "windows_csv": windows_csv,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_epoch": int(saved.get("epoch", -1)),
        "val_loss": float(saved.get("val_loss", float("nan"))),
        "val_acc": float(saved.get("val_acc", float("nan"))),
        "test_loss": float(test_result.loss),
        "test_acc": float(test_result.accuracy),
        "test_confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "test_positive_precision": positive_precision,
        "test_positive_recall": positive_recall,
        "test_positive_f1": positive_f1,
        "test_num_negative": sum(1 for y in labels if y == 0),
        "test_num_positive": sum(1 for y in labels if y == 1),
        "test_predicted_negative": sum(1 for y_hat in predictions if y_hat == 0),
        "test_predicted_positive": sum(1 for y_hat in predictions if y_hat == 1),
    }
    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    volume.commit()
    print(f"Saved metrics: {metrics_path}")

    return checkpoint_path


@app.local_entrypoint()
def main(
    prepare: bool = False,
    train: bool = True,
    epochs: int = 1,
    batch_size: int = 8,
) -> None:
    windows_csv = f"{DATASET_ROOT}/windows_gmdcsa24.csv"

    if prepare:
        prepare_windows.remote(windows_csv)

    if train:
        train_baseline.remote(
            windows_csv=windows_csv,
            epochs=epochs,
            batch_size=batch_size,
        )
