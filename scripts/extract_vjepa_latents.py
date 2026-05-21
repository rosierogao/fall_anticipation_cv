from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen V-JEPA2 latents for anticipation windows."
    )
    parser.add_argument("--windows-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-name",
        default="facebook/vjepa2-vitl-fpc64-256",
        help="Hugging Face V-JEPA2 checkpoint.",
    )
    parser.add_argument("--feature-col", default="vjepa_feature_path")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args()


def stable_window_id(row_idx: int, video_path: str, window_start: int, target_frame: int) -> str:
    key = f"{row_idx}:{video_path}:{window_start}:{target_frame}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def load_sampled_frames(row, start_frame: int, end_frame: int):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(row["video_path"])
    frames = []
    sample_interval = float(row["sample_interval"])

    for sampled_idx in range(start_frame, end_frame):
        original_idx = int(round(sampled_idx * sample_interval))
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_idx)
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        frames = [np.zeros((256, 256, 3), dtype=np.uint8)]

    while len(frames) < end_frame - start_frame:
        frames.append(frames[-1].copy())

    return np.stack(frames, axis=0)


def pool_temporal_latents(hidden_state, num_frames: int, tubelet_size: int):
    import torch

    temporal_tokens = max(1, num_frames // max(1, tubelet_size))
    batch_size, seq_len, latent_dim = hidden_state.shape

    if seq_len % temporal_tokens != 0:
        return hidden_state.mean(dim=1, keepdim=True)

    spatial_tokens = seq_len // temporal_tokens
    latents = hidden_state.reshape(
        batch_size,
        temporal_tokens,
        spatial_tokens,
        latent_dim,
    )
    return latents.mean(dim=2)


def extract_latent_batch(model, processor, videos, device: str):
    import torch

    inputs = processor(videos, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, skip_predictor=True)

    tubelet_size = int(getattr(model.config, "tubelet_size", 2))
    latents = pool_temporal_latents(
        outputs.last_hidden_state.detach().float().cpu(),
        num_frames=videos[0].shape[0],
        tubelet_size=tubelet_size,
    )
    return latents.numpy()


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoVideoProcessor

    output_dir = Path(args.output_dir)
    latent_dir = output_dir / "window_latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows_csv)
    if args.max_windows is not None:
        windows = windows.head(args.max_windows).copy()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    processor = AutoVideoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(
        args.model_name,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    rows = []
    pending_rows = []
    pending_observed = []
    pending_future = []

    def flush_batch():
        if not pending_rows:
            return

        observed_latents = extract_latent_batch(
            model,
            processor,
            pending_observed,
            device,
        )
        future_latents = extract_latent_batch(
            model,
            processor,
            pending_future,
            device,
        )

        for row, observed, future in zip(
            pending_rows,
            observed_latents,
            future_latents,
        ):
            feature_path = Path(row["_feature_path"])
            np.savez_compressed(
                feature_path,
                observed_latents=observed.astype(np.float16),
                future_latents=future.astype(np.float16),
            )
            output_row = row["_metadata"]
            output_row[args.feature_col] = str(feature_path)
            output_row["vjepa_model_name"] = args.model_name
            output_row["vjepa_latent_dim"] = int(observed.shape[-1])
            output_row["vjepa_observed_tokens"] = int(observed.shape[0])
            output_row["vjepa_future_tokens"] = int(future.shape[0])
            rows.append(output_row)

        pending_rows.clear()
        pending_observed.clear()
        pending_future.clear()

    for row_idx, row in tqdm(windows.iterrows(), total=len(windows), desc="V-JEPA"):
        feature_id = stable_window_id(
            row_idx,
            row["video_path"],
            int(row["window_start"]),
            int(row["target_frame"]),
        )
        feature_path = latent_dir / f"{feature_id}.npz"

        if feature_path.exists():
            output_row = row.to_dict()
            output_row[args.feature_col] = str(feature_path)
            rows.append(output_row)
            continue

        observed_video = load_sampled_frames(
            row,
            int(row["window_start"]),
            int(row["window_end"]),
        )
        future_video = load_sampled_frames(
            row,
            int(row["target_frame"]),
            int(row["target_frame"] + row["k_frames"]),
        )
        pending_rows.append(
            {
                "_metadata": row.to_dict(),
                "_feature_path": str(feature_path),
            }
        )
        pending_observed.append(observed_video)
        pending_future.append(future_video)

        if len(pending_rows) >= args.batch_size:
            flush_batch()

    flush_batch()

    output = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved V-JEPA latent metadata: {output_path}")
    print(f"Saved {len(output)} latent feature files under: {latent_dir}")


if __name__ == "__main__":
    main()
