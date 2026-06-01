from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-qualitative-cases")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
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
    .add_local_dir(LOCAL_SCRIPTS_DIR, f"{PACKAGE_REMOTE_ROOT}/scripts", copy=True)
    .env({"PYTHONPATH": f"{PACKAGE_REMOTE_ROOT}:{PACKAGE_REMOTE_ROOT}/scripts"})
)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 2,
)
def find_qualitative_cases(
    output_dir: str = f"{DATASET_ROOT}/outputs/qualitative_cases",
    threshold_policy: str = "balanced",
    examples_per_bucket: int = 8,
    frames_per_contact_sheet: int = 8,
    dataset_preset: str = "staged_caucafall_oops",
) -> dict:
    import json
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from evaluate_expanded_model_thresholds import add_derived_metrics
    from evaluate_thresholds import (
        classification_metrics_from_probs,
        tune_thresholds,
    )
    from fall_anticipation_cv.data import split_by_subject
    from fall_anticipation_cv.models.pose_baseline import PoseTransformerBaseline
    from fall_anticipation_cv.models.vjepa_predictive import (
        VJEPABaseline,
        VJEPALatentPredictiveModel,
    )
    from fall_anticipation_cv.pose_data import PoseWindowDataset, collate_pose_windows
    from fall_anticipation_cv.vjepa_data import (
        VJEPALatentWindowDataset,
        collate_vjepa_latent_windows,
    )

    data_root = Path(DATASET_ROOT)
    output_path = Path(output_dir)
    sheets_dir = output_path / "contact_sheets"
    output_path.mkdir(parents=True, exist_ok=True)
    if frames_per_contact_sheet > 0:
        sheets_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    if dataset_preset == "staged_caucafall":
        specs = {
            "pose": {
                "windows_csv": data_root / "pose_windows_staged_caucafall_joined_rtmpose.csv",
                "checkpoint": data_root
                / "outputs/pose_transformer_staged_caucafall_fall_anticipation.pt",
            },
            "vjepa": {
                "windows_csv": data_root / "vjepa_windows_staged_caucafall_joined.csv",
                "checkpoint": data_root
                / "outputs/vjepa_latent_predictive_staged_caucafall_fall_anticipation.pt",
            },
        }
    elif dataset_preset == "staged_caucafall_oops":
        specs = {
            "pose": {
                "windows_csv": data_root / "pose_windows_staged_caucafall_oops_rtmpose.csv",
                "checkpoint": data_root
                / "outputs/pose_transformer_staged_caucafall_oops_fall_anticipation.pt",
            },
            "vjepa": {
                "windows_csv": data_root / "vjepa_windows_staged_caucafall_oops.csv",
                "checkpoint": data_root
                / "outputs/vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation.pt",
            },
        }
    else:
        raise ValueError(
            "dataset_preset must be one of: staged_caucafall, staged_caucafall_oops"
        )

    def threshold_for(labels, probs):
        tuned = tune_thresholds(labels, probs, target_recall=0.75)
        if threshold_policy == "default":
            return 0.5
        if threshold_policy == "f2":
            return float(tuned["best_f2"]["threshold"])
        if threshold_policy == "balanced":
            return float(tuned["best_balanced_accuracy"]["threshold"])
        raise ValueError("threshold_policy must be one of: default, f2, balanced")

    @torch.no_grad()
    def pose_records():
        windows = pd.read_csv(specs["pose"]["windows_csv"])
        _train_df, val_df, test_df = split_by_subject(windows)
        saved = torch.load(specs["pose"]["checkpoint"], map_location=device)
        model = PoseTransformerBaseline(input_dim=int(saved["input_dim"])).to(device)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        normalize_pose = bool(saved.get("normalize_pose", True))
        add_velocity = bool(saved.get("add_velocity", True))

        def collect(df):
            loader = DataLoader(
                PoseWindowDataset(
                    df,
                    feature_col="pose_feature_path",
                    normalize=normalize_pose,
                    add_velocity=add_velocity,
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_pose_windows,
            )
            labels, probs = [], []
            for features, batch_labels, lengths in loader:
                logits = model(features.to(device), lengths.to(device))
                batch_probs = torch.softmax(logits, dim=1)[:, 1]
                labels.extend(batch_labels.numpy().tolist())
                probs.extend(batch_probs.cpu().numpy().tolist())
            return labels, probs

        val_labels, val_probs = collect(val_df)
        test_labels, test_probs = collect(test_df)
        threshold = threshold_for(val_labels, val_probs)
        out = test_df.reset_index(drop=True).copy()
        out["pose_label"] = test_labels
        out["pose_prob"] = test_probs
        out["pose_pred"] = (out["pose_prob"] >= threshold).astype(int)
        return out, threshold, add_derived_metrics(
            classification_metrics_from_probs(test_labels, test_probs, threshold)
        )

    @torch.no_grad()
    def vjepa_records():
        windows = pd.read_csv(specs["vjepa"]["windows_csv"])
        _train_df, val_df, test_df = split_by_subject(windows)
        saved = torch.load(specs["vjepa"]["checkpoint"], map_location=device)
        model_name = saved.get("model_name", "")
        if model_name == "vjepa_baseline":
            model = VJEPABaseline(
                latent_dim=int(saved["latent_dim"]),
                d_model=256,
                num_layers=1,
            ).to(device)
        else:
            model = VJEPALatentPredictiveModel(
                latent_dim=int(saved["latent_dim"]),
                d_model=256,
                num_layers=1,
                future_steps=int(saved["future_steps"]),
                predictive_loss_weight=float(saved.get("predictive_loss_weight", 0.2)),
            ).to(device)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()

        def collect(df):
            loader = DataLoader(
                VJEPALatentWindowDataset(df, feature_col="vjepa_feature_path"),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_vjepa_latent_windows,
            )
            labels, probs = [], []
            for observed, batch_labels, _future, lengths in loader:
                output = model(observed.to(device), lengths=lengths.to(device))
                batch_probs = torch.softmax(output.logits, dim=1)[:, 1]
                labels.extend(batch_labels.numpy().tolist())
                probs.extend(batch_probs.cpu().numpy().tolist())
            return labels, probs

        val_labels, val_probs = collect(val_df)
        test_labels, test_probs = collect(test_df)
        threshold = threshold_for(val_labels, val_probs)
        out = test_df.reset_index(drop=True).copy()
        out["vjepa_label"] = test_labels
        out["vjepa_prob"] = test_probs
        out["vjepa_pred"] = (out["vjepa_prob"] >= threshold).astype(int)
        return out, threshold, add_derived_metrics(
            classification_metrics_from_probs(test_labels, test_probs, threshold)
        )

    def make_contact_sheet(row, prefix):
        if frames_per_contact_sheet <= 0:
            return ""
        video_path = row["video_path"]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ""
        start = int(row["window_start"])
        end = int(row["window_end"])
        if end <= start:
            end = start + 1
        frame_ids = np.linspace(start, end - 1, frames_per_contact_sheet)
        frame_ids = [int(round(x)) for x in frame_ids]
        frames = []
        for frame_id in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.putText(
                frame,
                f"f={frame_id}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(frame)
        cap.release()
        if not frames:
            return ""
        while len(frames) < frames_per_contact_sheet:
            frames.append(np.zeros_like(frames[0]))
        rows = []
        cols = 4
        for idx in range(0, len(frames), cols):
            rows.append(np.concatenate(frames[idx : idx + cols], axis=1))
        sheet = np.concatenate(rows, axis=0)
        safe = (
            f"{prefix}_{row['bucket']}_{row.name}_y{int(row['y'])}"
            f"_vp{row['vjepa_pred']}_pp{row['pose_pred']}.jpg"
        )
        safe = safe.replace("/", "_").replace(" ", "_")
        out_file = sheets_dir / safe
        cv2.imwrite(str(out_file), sheet)
        return str(out_file)

    pose_df, pose_threshold, pose_metrics = pose_records()
    vjepa_df, vjepa_threshold, vjepa_metrics = vjepa_records()

    candidate_join_cols = [
        "video_path",
        "label_name",
        "subject",
        "cam",
        "dataset",
        "split_group",
        "split",
        "window_start",
        "window_end",
        "target_frame",
        "y",
        "fall_start_frame",
    ]
    join_cols = [
        col
        for col in candidate_join_cols
        if col in pose_df.columns and col in vjepa_df.columns
    ]
    pose_df = pose_df.copy()
    vjepa_df = vjepa_df.copy()
    pose_df["_join_occurrence"] = pose_df.groupby(join_cols).cumcount()
    vjepa_df["_join_occurrence"] = vjepa_df.groupby(join_cols).cumcount()
    join_cols = join_cols + ["_join_occurrence"]
    pose_cols = join_cols + ["pose_prob", "pose_pred"]
    merged = vjepa_df.merge(
        pose_df[pose_cols],
        on=join_cols,
        how="inner",
        validate="one_to_one",
    )
    merged["y"] = merged["y"].astype(int)
    merged["vjepa_correct"] = merged["vjepa_pred"].astype(int) == merged["y"]
    merged["pose_correct"] = merged["pose_pred"].astype(int) == merged["y"]
    merged["window_start_sec"] = merged["window_start"] / merged["target_fps"]
    merged["window_end_sec"] = merged["window_end"] / merged["target_fps"]
    merged["target_sec"] = merged["target_frame"] / merged["target_fps"]
    if "fall_start_frame" in merged.columns:
        merged["fall_start_sec"] = merged["fall_start_frame"] / merged["target_fps"]
    merged["vjepa_margin"] = (merged["vjepa_prob"] - vjepa_threshold).abs()
    merged["pose_margin"] = (merged["pose_prob"] - pose_threshold).abs()
    merged["score_gap"] = (merged["vjepa_margin"] + merged["pose_margin"]).astype(float)

    def bucket_name(row):
        if row["vjepa_correct"] and not row["pose_correct"]:
            return "vjepa_correct_pose_wrong"
        if row["pose_correct"] and not row["vjepa_correct"]:
            return "pose_correct_vjepa_wrong"
        if row["vjepa_correct"] and row["pose_correct"]:
            return "both_right"
        return "both_wrong"

    merged["bucket"] = merged.apply(bucket_name, axis=1)
    sort_cols = ["score_gap", "vjepa_margin", "pose_margin"]
    selected = (
        merged.sort_values(sort_cols, ascending=False)
        .groupby("bucket", group_keys=False)
        .head(examples_per_bucket)
        .copy()
    )
    selected["contact_sheet_path"] = [
        make_contact_sheet(row, "case") for _, row in selected.iterrows()
    ]

    bucket_counts = merged["bucket"].value_counts().to_dict()
    full_csv = output_path / f"{dataset_preset}_vjepa_pose_cases_{threshold_policy}_all.csv"
    selected_csv = (
        output_path / f"{dataset_preset}_vjepa_pose_cases_{threshold_policy}_selected.csv"
    )
    merged.to_csv(full_csv, index=False)
    selected.to_csv(selected_csv, index=False)

    summary = {
        "dataset_preset": dataset_preset,
        "threshold_policy": threshold_policy,
        "pose_threshold": pose_threshold,
        "vjepa_threshold": vjepa_threshold,
        "pose_metrics": pose_metrics,
        "vjepa_metrics": vjepa_metrics,
        "joined_test_windows": int(len(merged)),
        "bucket_counts": {k: int(v) for k, v in bucket_counts.items()},
        "all_cases_csv": str(full_csv),
        "selected_cases_csv": str(selected_csv),
        "contact_sheets_dir": str(sheets_dir),
    }
    summary_path = output_path / f"{dataset_preset}_vjepa_pose_cases_{threshold_policy}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    output_dir: str = f"{DATASET_ROOT}/outputs/qualitative_cases",
    threshold_policy: str = "balanced",
    examples_per_bucket: int = 8,
    frames_per_contact_sheet: int = 8,
    dataset_preset: str = "staged_caucafall_oops",
) -> None:
    summary = find_qualitative_cases.remote(
        output_dir=output_dir,
        threshold_policy=threshold_policy,
        examples_per_bucket=examples_per_bucket,
        frames_per_contact_sheet=frames_per_contact_sheet,
        dataset_preset=dataset_preset,
    )
    print(summary)
