# Fall Anticipation CV

Computer-vision experiments for anticipating falls from video windows.

The original milestone notebook is preserved at
`notebooks/baseline_milestone_check_in_2.ipynb`. Its current model has been
promoted into the repository as the baseline model:
`fall_anticipation_cv.models.baseline.SimpleVideoCNN`.

## Repository layout

```text
fall_anticipation_cv/
  notebooks/                 # Original exploratory notebooks
  scripts/                   # Command-line training/data utilities
  src/fall_anticipation_cv/   # Reusable Python package
    data.py                  # Label loading, window building, dataset
    training.py              # Train/evaluate loops
    models/baseline.py       # Baseline SimpleVideoCNN model
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare window metadata

```bash
python scripts/prepare_windows.py \
  --data-root /path/to/final_project_dataset \
  --output data/windows_gmdcsa24.csv
```

## Run EDA

```bash
PYTHONPATH=src python scripts/eda.py \
  --data-root /path/to/final_project_dataset \
  --output-dir outputs/eda
```

This writes `label_counts.csv`, `start_zero_stats.csv`, and
`fall_start_time_distribution.png`.

## Train the baseline

```bash
python scripts/train_baseline.py \
  --windows-csv data/windows_gmdcsa24.csv \
  --checkpoint outputs/baseline_simple_video_cnn.pt
```

## Train a Pose-Feature Baseline

Use RTMPose from MMPose to extract pose features from the prepared video-window
metadata:

For local extraction, install the OpenMMLab stack first:

```bash
pip install -r requirements-pose.txt
mim install "mmengine>=0.7.0" "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.3.0"
```

```bash
PYTHONPATH=src python scripts/extract_rtmpose_features.py \
  --windows-csv data/windows_gmdcsa24.csv \
  --output-csv data/pose_windows.csv \
  --output-dir data/rtmpose_features \
  --pose2d human
```

The `human` MMPose alias uses RTMPose-m for human pose estimation. The extractor
saves per-window NumPy arrays and writes their paths to `pose_feature_path`.
Because the videos are on Modal, run extraction remotely with:

```bash
/opt/anaconda3/bin/python -m modal run modal_app.py \
  --extract-pose --no-train
```

For a small smoke test, limit the number of videos:

```bash
/opt/anaconda3/bin/python -m modal run modal_app.py \
  --extract-pose --no-train --max-pose-videos 2
```

```bash
PYTHONPATH=src python scripts/train_pose_baseline.py \
  --windows-csv data/pose_windows.csv \
  --feature-col pose_feature_path \
  --checkpoint outputs/pose_gru_baseline.pt \
  --metrics outputs/pose_gru_metrics.json
```

Feature arrays may be shaped as `[T, F]` or `[T, J, C]`; higher-dimensional
pose arrays are flattened per frame before entering the GRU baseline.

## Run on Modal

The Modal remote runner uses your Modal volume named `final_project_dataset` and
mounts it at `/data`. The dataset files are expected under
`/data/final_project_dataset`.

```python
volume = modal.Volume.from_name("final_project_dataset", create_if_missing=False)
DATA_ROOT = "/data"
label_csv_path = "/data/final_project_dataset/labels/GMDCSA24_matched.csv"
label_map_path = "/data/final_project_dataset/labels/label2id.csv"
```

If `modal` is installed in your Anaconda Python, run:

```bash
/opt/anaconda3/bin/python -m modal run modal_app.py --prepare --epochs 1
```

After the window CSV exists on the volume, you can skip preparation:

```bash
/opt/anaconda3/bin/python -m modal run modal_app.py --epochs 1
```

The baseline checkpoint is written back to the Modal volume at
`/data/final_project_dataset/outputs/baseline_simple_video_cnn.pt`.
Metrics are written to
`/data/final_project_dataset/outputs/baseline_metrics.json`.

Large datasets, generated window CSVs, checkpoints, and run outputs are ignored
by Git by default.
