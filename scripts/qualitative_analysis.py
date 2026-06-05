"""Qualitative analysis: t-SNE embeddings, temporal attention, pose saliency."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from fall_anticipation_cv.data import split_by_subject
from fall_anticipation_cv.models.pose_baseline import PoseTransformerBaseline
from fall_anticipation_cv.models.vjepa_predictive import (
    VJEPABaseline,
    VJEPALatentPredictiveModel,
)
from fall_anticipation_cv.pose_data import (
    PoseWindowDataset,
    collate_pose_windows,
    prepare_pose_features,
)
from fall_anticipation_cv.vjepa_data import (
    VJEPALatentWindowDataset,
    collate_vjepa_latent_windows,
)

COCO_KEYPOINTS = [
    "nose",
    "L eye",
    "R eye",
    "L ear",
    "R ear",
    "L shoulder",
    "R shoulder",
    "L elbow",
    "R elbow",
    "L wrist",
    "R wrist",
    "L hip",
    "R hip",
    "L knee",
    "R knee",
    "L ankle",
    "R ankle",
]

DATASET_SPECS = {
    "staged_caucafall_oops": {
        "pose": {
            "windows_csv": "pose_windows_staged_caucafall_oops_rtmpose.csv",
            "checkpoint": "outputs/pose_transformer_staged_caucafall_oops_fall_anticipation.pt",
        },
        "vjepa": {
            "windows_csv": "vjepa_windows_staged_caucafall_oops.csv",
            "checkpoint": "outputs/vjepa_latent_predictive_staged_caucafall_oops_fall_anticipation.pt",
        },
    },
    "staged_caucafall": {
        "pose": {
            "windows_csv": "pose_windows_staged_caucafall_joined_rtmpose.csv",
            "checkpoint": "outputs/pose_transformer_staged_caucafall_fall_anticipation.pt",
        },
        "vjepa": {
            "windows_csv": "vjepa_windows_staged_caucafall_joined.csv",
            "checkpoint": "outputs/vjepa_latent_predictive_staged_caucafall_fall_anticipation.pt",
        },
    },
}


# ── Helpers ──────────────────────────────────────────────────────────


def get_temporal_classifier(model):
    if isinstance(model, PoseTransformerBaseline):
        return model.temporal_classifier
    if isinstance(model, (VJEPALatentPredictiveModel, VJEPABaseline)):
        return model.classifier
    raise ValueError(f"Unknown model type: {type(model)}")


def find_balanced_threshold(labels, probs):
    best_threshold, best_ba = 0.5, 0.0
    for t in np.arange(0.01, 1.0, 0.01):
        preds = np.array(probs) >= t
        tp = int(((np.array(labels) == 1) & preds).sum())
        fn = int(((np.array(labels) == 1) & ~preds).sum())
        tn = int(((np.array(labels) == 0) & ~preds).sum())
        fp = int(((np.array(labels) == 0) & preds).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        ba = (recall + spec) / 2
        if ba > best_ba:
            best_ba, best_threshold = ba, float(t)
    return best_threshold


def load_pose_model(checkpoint_path, device):
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PoseTransformerBaseline(input_dim=int(saved["input_dim"])).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, bool(saved.get("normalize_pose", True)), bool(saved.get("add_velocity", True))


def load_vjepa_model(checkpoint_path, device):
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = saved.get("model_name", "")
    if model_name == "vjepa_baseline":
        model = VJEPABaseline(
            latent_dim=int(saved["latent_dim"]), d_model=256, num_layers=1,
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
    return model


# ── Embedding extraction (for t-SNE) ────────────────────────────────


@contextmanager
def capture_cls_embeddings(model):
    embeddings = []
    tc = get_temporal_classifier(model)

    def hook_fn(module, inp, output):
        embeddings.append(output[:, 0].detach().cpu())

    handle = tc.encoder.register_forward_hook(hook_fn)
    yield embeddings
    handle.remove()


@torch.no_grad()
def collect_embeddings_pose(model, loader, device):
    all_labels, all_probs = [], []
    with capture_cls_embeddings(model) as emb_list:
        for features, labels, lengths in loader:
            logits = model(features.to(device), lengths.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    return torch.cat(emb_list, dim=0).numpy(), all_labels, all_probs


@torch.no_grad()
def collect_embeddings_vjepa(model, loader, device):
    all_labels, all_probs = [], []
    with capture_cls_embeddings(model) as emb_list:
        for observed, labels, _future, lengths in loader:
            output = model(observed.to(device), lengths=lengths.to(device))
            probs = torch.softmax(output.logits, dim=1)[:, 1]
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    return torch.cat(emb_list, dim=0).numpy(), all_labels, all_probs


# ── Attention extraction ─────────────────────────────────────────────


@contextmanager
def capture_attention_weights(model):
    """Capture self-attention weights from all MHA layers."""
    attention_maps = []
    tc = get_temporal_classifier(model)
    originals = []
    handles = []

    for module in tc.encoder.modules():
        if not isinstance(module, nn.MultiheadAttention):
            continue
        orig = module.forward
        originals.append((module, orig))

        def make_wrapper(original):
            def wrapper(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                return original(*args, **kwargs)
            return wrapper

        module.forward = make_wrapper(orig)

        def hook_fn(mod, inp, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                attention_maps.append(out[1].detach().cpu())

        handles.append(module.register_forward_hook(hook_fn))

    yield attention_maps

    for h in handles:
        h.remove()
    for mod, orig in originals:
        mod.forward = orig


@torch.no_grad()
def extract_attention_pose(model, features, length, device):
    """Returns CLS→token attention of shape [T] for valid timesteps."""
    features = features.unsqueeze(0).to(device)
    lengths = torch.tensor([length], device=device)
    with capture_attention_weights(model) as attn_list:
        model(features, lengths)
    if not attn_list:
        return np.zeros(length)
    attn = attn_list[-1][0, 0, 1:]  # last layer, CLS row, skip CLS column
    return attn.numpy()[:length]


@torch.no_grad()
def extract_attention_vjepa(model, observed, device):
    """Returns CLS→token attention of shape [T] for valid timesteps."""
    T = observed.shape[0]
    observed = observed.unsqueeze(0).to(device)
    lengths = torch.tensor([T], device=device)
    with capture_attention_weights(model) as attn_list:
        model(observed, lengths=lengths)
    if not attn_list:
        return np.zeros(T)
    attn = attn_list[-1][0, 0, 1:]
    return attn.numpy()[:T]


# ── Pose saliency ───────────────────────────────────────────────────


def compute_pose_saliency(model, features, length, device, target_class=1):
    """Gradient-based saliency: returns [T, num_joints] importance map."""
    x = features[:length].clone().detach().unsqueeze(0).to(device).requires_grad_(True)
    lengths = torch.tensor([length], device=device)
    model.zero_grad()
    logits = model(x, lengths)
    logits[0, target_class].backward()
    grad = x.grad[0].cpu()  # [T, input_dim]

    num_joints = len(COCO_KEYPOINTS)
    input_dim = grad.shape[1]
    if input_dim % num_joints == 0:
        fpj = input_dim // num_joints
        saliency = grad.reshape(length, num_joints, fpj).abs().mean(dim=-1)
    else:
        saliency = grad.abs()
    return saliency.numpy()


# ── Plotting ─────────────────────────────────────────────────────────


def plot_tsne_by_label(
    pose_emb, pose_labels, pose_preds,
    vjepa_emb, vjepa_labels, vjepa_preds,
    output_path,
):
    perp_pose = min(30, len(pose_labels) - 1)
    perp_vjepa = min(30, len(vjepa_labels) - 1)
    pose_2d = TSNE(n_components=2, random_state=42, perplexity=perp_pose).fit_transform(pose_emb)
    vjepa_2d = TSNE(n_components=2, random_state=42, perplexity=perp_vjepa).fit_transform(vjepa_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    pose_labels = np.array(pose_labels)
    vjepa_labels = np.array(vjepa_labels)
    pose_preds = np.array(pose_preds)
    vjepa_preds = np.array(vjepa_preds)

    for ax, pts, labels, preds, title in [
        (axes[0], pose_2d, pose_labels, pose_preds, "Pose Transformer"),
        (axes[1], vjepa_2d, vjepa_labels, vjepa_preds, "V-JEPA"),
    ]:
        correct = labels == preds
        for lv, c, nm in [(0, "#2196F3", "No Fall"), (1, "#F44336", "Fall")]:
            m = (labels == lv) & correct
            ax.scatter(pts[m, 0], pts[m, 1], c=c, alpha=0.4, s=20,
                       label=f"{nm} (correct)", edgecolors="none")
        for lv, c, nm in [(0, "#2196F3", "No Fall"), (1, "#F44336", "Fall")]:
            m = (labels == lv) & ~correct
            ax.scatter(pts[m, 0], pts[m, 1], c=c, alpha=0.9, s=60,
                       label=f"{nm} (error)", edgecolors="black",
                       linewidths=1.0, marker="X")
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("CLS Token Embeddings (t-SNE)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tsne_by_dataset(
    pose_emb, pose_labels, pose_datasets,
    vjepa_emb, vjepa_labels, vjepa_datasets,
    output_path,
):
    perp_pose = min(30, len(pose_labels) - 1)
    perp_vjepa = min(30, len(vjepa_labels) - 1)
    pose_2d = TSNE(n_components=2, random_state=42, perplexity=perp_pose).fit_transform(pose_emb)
    vjepa_2d = TSNE(n_components=2, random_state=42, perplexity=perp_vjepa).fit_transform(vjepa_emb)

    all_ds = sorted(set(pose_datasets) | set(vjepa_datasets))
    cmap = plt.cm.Set2
    ds_colors = {ds: cmap(i / max(len(all_ds) - 1, 1)) for i, ds in enumerate(all_ds)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, pts, labels, datasets, title in [
        (axes[0], pose_2d, np.array(pose_labels), pose_datasets, "Pose Transformer"),
        (axes[1], vjepa_2d, np.array(vjepa_labels), vjepa_datasets, "V-JEPA"),
    ]:
        for ds in all_ds:
            mask = np.array([d == ds for d in datasets])
            fall = mask & (labels == 1)
            nofall = mask & (labels == 0)
            ax.scatter(pts[nofall, 0], pts[nofall, 1], c=[ds_colors[ds]],
                       alpha=0.5, s=20, label=f"{ds} (no fall)", marker="o")
            ax.scatter(pts[fall, 0], pts[fall, 1], c=[ds_colors[ds]],
                       alpha=0.8, s=40, label=f"{ds} (fall)", marker="^")
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("CLS Token Embeddings by Dataset (t-SNE)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attention_comparison(cases, output_path, title_suffix=""):
    """Plot attention for disagreement cases.

    cases: list of dicts with keys pose_attn, vjepa_attn, y, pose_pred, vjepa_pred
    """
    n = len(cases)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.2 * n + 1), squeeze=False)

    for i, case in enumerate(cases):
        for col, key, color, model_name in [
            (0, "pose_attn", "#4CAF50", "Pose"),
            (1, "vjepa_attn", "#2196F3", "V-JEPA"),
        ]:
            ax = axes[i, col]
            attn = case[key]
            ax.bar(range(len(attn)), attn, color=color, alpha=0.7, width=1.0)
            if i == 0:
                ax.set_title(f"{model_name} Attention", fontsize=12)
            pred_key = "pose_pred" if col == 0 else "vjepa_pred"
            correct = int(case[pred_key]) == int(case["y"])
            marker = "✓" if correct else "✗"
            ax.text(
                0.98, 0.92,
                f"y={case['y']}  pred={case[pred_key]} {marker}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6),
            )
            if i == n - 1:
                ax.set_xlabel("Timestep")
            ax.set_ylabel(f"Ex {i+1}", fontsize=9) if col == 0 else None

    fig.suptitle(
        f"Temporal Attention: CLS → Timesteps{title_suffix}", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pose_saliency(saliency_cases, output_path):
    """Plot pose joint saliency heatmaps.

    saliency_cases: list of dicts with keys saliency ([T, J]), y, pose_pred, bucket
    """
    n = len(saliency_cases)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n + 1), squeeze=False)

    for i, case in enumerate(saliency_cases):
        ax = axes[i, 0]
        sal = case["saliency"]
        im = ax.imshow(sal.T, aspect="auto", cmap="hot", interpolation="nearest")
        if sal.shape[1] == len(COCO_KEYPOINTS):
            ax.set_yticks(range(len(COCO_KEYPOINTS)))
            ax.set_yticklabels(COCO_KEYPOINTS, fontsize=7)
        bucket = case.get("bucket", "")
        ax.set_title(f"y={case['y']}  pred={case['pose_pred']}  [{bucket}]", fontsize=10)
        if i == n - 1:
            ax.set_xlabel("Timestep")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.suptitle("Pose Joint Saliency (Gradient Magnitude)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Merge logic ──────────────────────────────────────────────────────


def merge_test_predictions(pose_df, vjepa_df):
    """Join pose and vjepa test DataFrames on shared columns."""
    candidate_join = [
        "video_path", "label_name", "subject", "cam", "dataset",
        "split_group", "split", "window_start", "window_end",
        "target_frame", "y", "fall_start_frame",
    ]
    join_cols = [c for c in candidate_join if c in pose_df.columns and c in vjepa_df.columns]

    pose_df = pose_df.copy()
    vjepa_df = vjepa_df.copy()
    pose_df["_occ"] = pose_df.groupby(join_cols).cumcount()
    vjepa_df["_occ"] = vjepa_df.groupby(join_cols).cumcount()
    jc = join_cols + ["_occ"]

    extra_pose_cols = [c for c in ["pose_feature_path"] if c in pose_df.columns]
    pose_cols = jc + ["pose_prob", "pose_pred"] + extra_pose_cols
    merged = vjepa_df.merge(pose_df[pose_cols], on=jc, how="inner", validate="one_to_one")
    merged["y"] = merged["y"].astype(int)
    merged["vjepa_correct"] = merged["vjepa_pred"].astype(int) == merged["y"]
    merged["pose_correct"] = merged["pose_pred"].astype(int) == merged["y"]

    def bucket(row):
        if row["vjepa_correct"] and not row["pose_correct"]:
            return "vjepa_correct_pose_wrong"
        if row["pose_correct"] and not row["vjepa_correct"]:
            return "pose_correct_vjepa_wrong"
        if row["vjepa_correct"] and row["pose_correct"]:
            return "both_correct"
        return "both_wrong"

    merged["bucket"] = merged.apply(bucket, axis=1)
    return merged


# ── Main ─────────────────────────────────────────────────────────────


def run_analysis(
    data_root: str,
    output_dir: str,
    dataset_preset: str = "staged_caucafall_oops",
    examples_per_bucket: int = 8,
    batch_size: int = 32,
):
    data_root = Path(data_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = DATASET_SPECS[dataset_preset]
    pose_csv = data_root / specs["pose"]["windows_csv"]
    pose_ckpt = data_root / specs["pose"]["checkpoint"]
    vjepa_csv = data_root / specs["vjepa"]["windows_csv"]
    vjepa_ckpt = data_root / specs["vjepa"]["checkpoint"]

    # ── Load models ──
    print("Loading models...", flush=True)
    pose_model, normalize_pose, add_velocity = load_pose_model(pose_ckpt, device)
    vjepa_model = load_vjepa_model(vjepa_ckpt, device)

    # ── Pose: split, embed, predict ──
    print("Collecting pose embeddings...", flush=True)
    pose_windows = pd.read_csv(pose_csv)
    _, pose_val, pose_test = split_by_subject(pose_windows)
    pose_val_loader = DataLoader(
        PoseWindowDataset(pose_val, feature_col="pose_feature_path",
                          normalize=normalize_pose, add_velocity=add_velocity),
        batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pose_windows,
    )
    pose_test_loader = DataLoader(
        PoseWindowDataset(pose_test, feature_col="pose_feature_path",
                          normalize=normalize_pose, add_velocity=add_velocity),
        batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pose_windows,
    )

    # val set for threshold
    _, pose_val_labels, pose_val_probs = collect_embeddings_pose(pose_model, pose_val_loader, device)
    pose_threshold = find_balanced_threshold(pose_val_labels, pose_val_probs)
    print(f"  Pose threshold (balanced): {pose_threshold:.3f}", flush=True)

    # test set
    pose_emb, pose_labels, pose_probs = collect_embeddings_pose(pose_model, pose_test_loader, device)
    pose_preds = (np.array(pose_probs) >= pose_threshold).astype(int)
    pose_datasets = pose_test["dataset"].fillna("unknown").tolist()

    # ── VJEPA: split, embed, predict ──
    print("Collecting V-JEPA embeddings...", flush=True)
    vjepa_windows = pd.read_csv(vjepa_csv)
    _, vjepa_val, vjepa_test = split_by_subject(vjepa_windows)
    vjepa_val_loader = DataLoader(
        VJEPALatentWindowDataset(vjepa_val, feature_col="vjepa_feature_path"),
        batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_vjepa_latent_windows,
    )
    vjepa_test_loader = DataLoader(
        VJEPALatentWindowDataset(vjepa_test, feature_col="vjepa_feature_path"),
        batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_vjepa_latent_windows,
    )

    _, vjepa_val_labels, vjepa_val_probs = collect_embeddings_vjepa(vjepa_model, vjepa_val_loader, device)
    vjepa_threshold = find_balanced_threshold(vjepa_val_labels, vjepa_val_probs)
    print(f"  VJEPA threshold (balanced): {vjepa_threshold:.3f}", flush=True)

    vjepa_emb, vjepa_labels, vjepa_probs = collect_embeddings_vjepa(vjepa_model, vjepa_test_loader, device)
    vjepa_preds = (np.array(vjepa_probs) >= vjepa_threshold).astype(int)
    vjepa_datasets = vjepa_test["dataset"].fillna("unknown").tolist()

    # ── t-SNE plots ──
    print("Running t-SNE...", flush=True)
    tsne_label_path = str(output_path / "tsne_by_label.png")
    plot_tsne_by_label(
        pose_emb, pose_labels, pose_preds,
        vjepa_emb, vjepa_labels, vjepa_preds,
        tsne_label_path,
    )
    tsne_dataset_path = str(output_path / "tsne_by_dataset.png")
    plot_tsne_by_dataset(
        pose_emb, pose_labels, pose_datasets,
        vjepa_emb, vjepa_labels, vjepa_datasets,
        tsne_dataset_path,
    )
    print(f"  Saved t-SNE plots", flush=True)

    # ── Merge test sets for disagreement analysis ──
    print("Merging test sets for disagreement analysis...", flush=True)
    pose_test_df = pose_test.reset_index(drop=True).copy()
    pose_test_df["pose_prob"] = pose_probs
    pose_test_df["pose_pred"] = pose_preds

    vjepa_test_df = vjepa_test.reset_index(drop=True).copy()
    vjepa_test_df["vjepa_prob"] = vjepa_probs
    vjepa_test_df["vjepa_pred"] = vjepa_preds

    merged = merge_test_predictions(pose_test_df, vjepa_test_df)
    bucket_counts = merged["bucket"].value_counts().to_dict()
    print(f"  Bucket counts: {bucket_counts}", flush=True)

    # Score gap for selecting most confident disagreements
    merged["pose_margin"] = (np.array(merged["pose_prob"]) - pose_threshold).__abs__()
    merged["vjepa_margin"] = (np.array(merged["vjepa_prob"]) - vjepa_threshold).__abs__()
    merged["score_gap"] = merged["pose_margin"] + merged["vjepa_margin"]

    # ── Extract attention + saliency for disagreement cases ──
    for bucket_name, title_suffix in [
        ("pose_correct_vjepa_wrong", " (Pose Correct, V-JEPA Wrong)"),
        ("vjepa_correct_pose_wrong", " (V-JEPA Correct, Pose Wrong)"),
        ("both_wrong", " (Both Wrong)"),
    ]:
        bucket_df = merged[merged["bucket"] == bucket_name]
        if bucket_df.empty:
            print(f"  No cases for {bucket_name}, skipping", flush=True)
            continue

        selected = bucket_df.sort_values("score_gap", ascending=False).head(examples_per_bucket)
        print(f"  Processing {len(selected)} {bucket_name} cases...", flush=True)

        attn_cases = []
        saliency_cases = []

        for _, row in selected.iterrows():
            meta = {
                "y": int(row["y"]),
                "pose_pred": int(row["pose_pred"]),
                "vjepa_pred": int(row["vjepa_pred"]),
                "bucket": bucket_name,
            }

            # Load pose features for this window
            pose_feature_path = row.get("pose_feature_path")
            vjepa_feature_path = row.get("vjepa_feature_path")

            # Pose attention + saliency
            if pose_feature_path and Path(str(pose_feature_path)).exists():
                raw = np.load(str(pose_feature_path)).astype(np.float32)
                features = prepare_pose_features(
                    raw, normalize=normalize_pose, add_velocity=add_velocity,
                )
                features_t = torch.from_numpy(features)
                length = features_t.shape[0]

                pose_attn = extract_attention_pose(pose_model, features_t, length, device)
                saliency = compute_pose_saliency(pose_model, features_t, length, device)
                saliency_cases.append({**meta, "saliency": saliency})
            else:
                pose_attn = np.zeros(1)

            # VJEPA attention
            if vjepa_feature_path and Path(str(vjepa_feature_path)).exists():
                data = np.load(str(vjepa_feature_path))
                observed = torch.from_numpy(data["observed_latents"].astype(np.float32))
                vjepa_attn = extract_attention_vjepa(vjepa_model, observed, device)
            else:
                vjepa_attn = np.zeros(1)

            attn_cases.append({**meta, "pose_attn": pose_attn, "vjepa_attn": vjepa_attn})

        safe_name = bucket_name.replace(" ", "_")
        if attn_cases:
            plot_attention_comparison(
                attn_cases,
                str(output_path / f"attention_{safe_name}.png"),
                title_suffix=title_suffix,
            )
        if saliency_cases:
            plot_pose_saliency(
                saliency_cases,
                str(output_path / f"saliency_{safe_name}.png"),
            )

    # ── Summary ──
    summary = {
        "dataset_preset": dataset_preset,
        "pose_threshold": pose_threshold,
        "vjepa_threshold": vjepa_threshold,
        "pose_test_size": len(pose_labels),
        "vjepa_test_size": len(vjepa_labels),
        "merged_test_size": len(merged),
        "bucket_counts": {k: int(v) for k, v in bucket_counts.items()},
        "outputs": {
            "tsne_by_label": tsne_label_path,
            "tsne_by_dataset": tsne_dataset_path,
        },
    }
    summary_file = output_path / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nDone. Summary: {summary_file}")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative analysis: t-SNE, attention, saliency")
    p.add_argument("--data-root", default="/data/final_project_dataset")
    p.add_argument("--output-dir", default="/data/final_project_dataset/outputs/qualitative_analysis")
    p.add_argument("--dataset-preset", default="staged_caucafall_oops",
                    choices=list(DATASET_SPECS.keys()))
    p.add_argument("--examples-per-bucket", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset_preset=args.dataset_preset,
        examples_per_bucket=args.examples_per_bucket,
        batch_size=args.batch_size,
    )
