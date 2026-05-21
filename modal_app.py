from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_RTMPOSE_SCRIPT = Path(__file__).parent / "scripts" / "extract_rtmpose_features.py"

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

rtmpose_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("openmim", "numpy", "pandas", "torch", "torchvision", "tqdm")
    .run_commands(
        'mim install "mmengine>=0.7.0" "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.3.0"'
    )
    .add_local_dir(
        LOCAL_PACKAGE_DIR,
        f"{PACKAGE_REMOTE_ROOT}/fall_anticipation_cv",
        copy=True,
    )
    .add_local_file(
        LOCAL_RTMPOSE_SCRIPT,
        f"{PACKAGE_REMOTE_ROOT}/scripts/extract_rtmpose_features.py",
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
        load_all_labels,
        validate_windows,
    )

    labels = load_all_labels(DATA_ROOT)
    windows = build_window_dataframe(labels)
    validate_windows(windows)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_path, index=False)
    volume.commit()

    print(f"Labels: {len(labels)}")
    print(f"Matched videos: {labels['video_exists'].sum()}/{len(labels)}")
    print(labels["dataset"].value_counts())
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
    video_model: str = "cnn",
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
    from fall_anticipation_cv.models.baseline import (
        SimpleVideoCNN,
        VideoCNNTransformerBaseline,
    )
    from fall_anticipation_cv.training_common import (
        binary_classification_metrics,
        compute_class_weights,
        default_video_forward,
        evaluate,
        train_one_epoch,
    )

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

    if video_model == "transformer":
        model = VideoCNNTransformerBaseline(num_classes=2).to(device)
        model_name = "video_cnn_transformer_baseline"
    else:
        model = SimpleVideoCNN(num_classes=2).to(device)
        model_name = "simple_video_cnn"

    class_weights = compute_class_weights(
        torch.tensor(train_df["y"].to_numpy(), dtype=torch.long)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            default_video_forward,
        )
        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            default_video_forward,
        )

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "model_name": model_name,
                },
                checkpoint,
            )
            volume.commit()
            print(f"Saved best baseline checkpoint: {checkpoint_path}")

    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_acc, predictions, labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
        default_video_forward,
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc:  {test_acc:.4f}")

    metrics = {
        "model": model_name,
        "checkpoint_path": checkpoint_path,
        "windows_csv": windows_csv,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_epoch": int(saved.get("epoch", -1)),
        "val_loss": float(saved.get("val_loss", float("nan"))),
        "val_acc": float(saved.get("val_acc", float("nan"))),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "class_weights": class_weights.detach().cpu().tolist(),
        "test_num_negative": sum(1 for y in labels if y == 0),
        "test_num_positive": sum(1 for y in labels if y == 1),
        "test_predicted_negative": sum(1 for y_hat in predictions if y_hat == 0),
        "test_predicted_positive": sum(1 for y_hat in predictions if y_hat == 1),
        **binary_classification_metrics(labels, predictions),
    }
    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    volume.commit()
    print(f"Saved metrics: {metrics_path}")

    return checkpoint_path


@app.function(
    image=rtmpose_image,
    volumes={DATA_ROOT: volume},
    gpu="T4",
    timeout=60 * 60 * 12,
)
def extract_rtmpose_pose_features(
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
    output_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    output_dir: str = f"{DATASET_ROOT}/rtmpose_features",
    pose2d: str = "human",
    max_videos: int | None = None,
) -> str:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/extract_rtmpose_features.py",
        "--windows-csv",
        windows_csv,
        "--output-csv",
        output_csv,
        "--output-dir",
        output_dir,
        "--pose2d",
        pose2d,
        "--device",
        "cuda",
    ]
    if max_videos is not None:
        cmd.extend(["--max-videos", str(max_videos)])

    subprocess.run(cmd, check=True)
    volume.commit()
    return output_csv


@app.local_entrypoint()
def main(
    prepare: bool = False,
    train: bool = True,
    extract_pose: bool = False,
    epochs: int = 1,
    batch_size: int = 8,
    video_model: str = "cnn",
    max_pose_videos: int | None = None,
) -> None:
    windows_csv = f"{DATASET_ROOT}/windows_gmdcsa24.csv"

    if prepare:
        prepare_windows.remote(windows_csv)

    if extract_pose:
        extract_rtmpose_pose_features.remote(
            windows_csv=windows_csv,
            max_videos=max_pose_videos,
        )

    if train:
        train_baseline.remote(
            windows_csv=windows_csv,
            epochs=epochs,
            batch_size=batch_size,
            video_model=video_model,
        )
