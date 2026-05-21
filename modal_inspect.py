from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"

app = modal.App(f"{APP_NAME}-inspect")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "opencv-python-headless", "pandas")
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
def inspect_le2i_fps(max_videos: int | None = 20) -> dict:
    import json

    import cv2

    from fall_anticipation_cv.data import load_le2i_labels

    labels = load_le2i_labels(DATA_ROOT)
    rows = labels[labels["video_exists"]].copy()
    if max_videos is not None:
        rows = rows.head(max_videos)

    fps_counts: dict[str, int] = {}
    examples = []
    for _, row in rows.iterrows():
        cap = cv2.VideoCapture(row["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        fps_key = f"{fps:.3f}"
        fps_counts[fps_key] = fps_counts.get(fps_key, 0) + 1
        examples.append(
            {
                "path": row["path"],
                "video_path": row["video_path"],
                "fps": fps,
                "n_frames": n_frames,
                "fall_start_frame": int(row["start_frame"]),
                "fall_end_frame": int(row["end_frame"]),
            }
        )

    summary = {
        "num_le2i_labels": int(len(labels)),
        "num_existing_videos": int(labels["video_exists"].sum()) if not labels.empty else 0,
        "num_checked": int(len(rows)),
        "fps_counts": fps_counts,
        "examples": examples,
    }
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(max_videos: int = 20) -> None:
    inspect_le2i_fps.remote(max_videos=max_videos)
