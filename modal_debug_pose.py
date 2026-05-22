from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"

app = modal.App(f"{APP_NAME}-debug-pose")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "opencv-python-headless==4.10.0.84",
        "pandas<3.0",
        "scikit-learn",
        "torch==2.3.1",
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
    timeout=60 * 30,
)
def debug_pose_checkpoint(
    windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    checkpoint: str = f"{DATASET_ROOT}/outputs/pose_transformer_baseline.pt",
    batch_size: int = 64,
) -> dict:
    import json

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from fall_anticipation_cv.data import split_by_subject
    from fall_anticipation_cv.models.pose_baseline import PoseTransformerBaseline
    from fall_anticipation_cv.pose_data import PoseWindowDataset, collate_pose_windows
    from fall_anticipation_cv.training_common import binary_classification_metrics

    def make_loader(df: pd.DataFrame) -> DataLoader:
        return DataLoader(
            PoseWindowDataset(df, feature_col="pose_feature_path"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pose_windows,
        )

    def summarize(values: list[float]) -> dict:
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
        }

    @torch.no_grad()
    def predict(df: pd.DataFrame, model: torch.nn.Module, device: torch.device) -> dict:
        labels_all = []
        probs_all = []
        logits_all = []
        model.eval()
        for features, labels, lengths in make_loader(df):
            features = features.to(device)
            lengths = lengths.to(device)
            logits = model(features, lengths)
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
            logits_all.extend(logits.cpu().numpy().tolist())

        preds = [int(p >= 0.5) for p in probs_all]
        summary = {
            "rows": int(len(df)),
            "label_counts": {
                str(k): int(v) for k, v in df["y"].value_counts().to_dict().items()
            },
            "threshold_0_5": binary_classification_metrics(labels_all, preds),
            "positive_probability": {
                "all": summarize(probs_all),
                "true_negative_class": summarize(
                    [p for p, y in zip(probs_all, labels_all) if y == 0]
                ),
                "true_positive_class": summarize(
                    [p for p, y in zip(probs_all, labels_all) if y == 1]
                ),
            },
            "logit_mean": np.asarray(logits_all, dtype=np.float64).mean(axis=0).tolist(),
        }
        return summary | {"labels": labels_all, "probs": probs_all}

    def best_threshold(labels: list[int], probs: list[float]) -> dict:
        best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
        for threshold in np.linspace(0.01, 0.99, 99):
            preds = [int(p >= threshold) for p in probs]
            metrics = binary_classification_metrics(labels, preds)
            if metrics["positive_f1"] > best["f1"]:
                best = {
                    "threshold": float(threshold),
                    "f1": float(metrics["positive_f1"]),
                    "precision": float(metrics["positive_precision"]),
                    "recall": float(metrics["positive_recall"]),
                }
        return best

    def feature_quality(df: pd.DataFrame, max_rows_per_class: int = 128) -> dict:
        quality = {}
        for label, group in df.groupby("y"):
            rows = group.head(max_rows_per_class)
            conf_means = []
            zero_fractions = []
            coord_abs_means = []
            for path in rows["pose_feature_path"]:
                features = np.load(path).astype(np.float32)
                if features.ndim == 3 and features.shape[-1] >= 3:
                    coords = features[..., :2]
                    conf = features[..., 2]
                else:
                    flat = features.reshape(features.shape[0], -1)
                    coords = flat[:, : max(0, flat.shape[1] // 3 * 2)]
                    conf = flat[:, 2::3]
                conf_means.append(float(np.mean(conf)))
                zero_fractions.append(float(np.mean(np.isclose(features, 0.0))))
                coord_abs_means.append(float(np.mean(np.abs(coords))))
            quality[str(int(label))] = {
                "sampled_rows": int(len(rows)),
                "confidence_mean": summarize(conf_means),
                "zero_fraction": summarize(zero_fractions),
                "coordinate_abs_mean": summarize(coord_abs_means),
            }
        return quality

    windows = pd.read_csv(windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)
    saved = torch.load(checkpoint, map_location="cpu")
    input_dim = int(saved["input_dim"])
    model = PoseTransformerBaseline(input_dim=input_dim)
    model.load_state_dict(saved["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train = predict(train_df, model, device)
    val = predict(val_df, model, device)
    test = predict(test_df, model, device)
    threshold = best_threshold(val["labels"], val["probs"])
    test_threshold_preds = [int(p >= threshold["threshold"]) for p in test["probs"]]

    report = {
        "windows_csv": windows_csv,
        "checkpoint": checkpoint,
        "checkpoint_epoch_0_indexed": int(saved.get("epoch", -1)),
        "checkpoint_val_loss": float(saved.get("val_loss", float("nan"))),
        "checkpoint_val_acc": float(saved.get("val_acc", float("nan"))),
        "split_rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "train": {k: v for k, v in train.items() if k not in {"labels", "probs"}},
        "val": {k: v for k, v in val.items() if k not in {"labels", "probs"}},
        "test": {k: v for k, v in test.items() if k not in {"labels", "probs"}},
        "best_val_threshold": threshold,
        "test_at_best_val_threshold": binary_classification_metrics(
            test["labels"], test_threshold_preds
        ),
        "feature_quality_train_sample": feature_quality(train_df),
    }
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main() -> None:
    debug_pose_checkpoint.remote()
