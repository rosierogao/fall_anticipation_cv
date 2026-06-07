# Fall Anticipation CV

This repository contains the code, experiment artifacts, and CS231n report for a computer-vision project on early fall anticipation from short video clips. The main task is binary prediction: given a 1.6 second observed clip sampled at 10 FPS, predict whether a fall will begin within the next 1.0 second. A secondary experiment predicts future fallen state within a 2.0 second horizon.


## Current Scope

The project compares lightweight temporal heads over several video representations:

- **Video CNN + Transformer**: raw RGB frames encoded with a small CNN, followed by a Transformer temporal classifier.
- **Pose Transformer**: RTMPose keypoint features followed by a Transformer temporal classifier.
- **Pose Predictive**: pose sequence model with an auxiliary future-pose prediction objective.
- **V-JEPA Baseline**: frozen V-JEPA latent features with a Transformer classifier and classification loss only.
- **V-JEPA Predictive**: frozen V-JEPA latent features with a Transformer classifier plus a future-latent prediction head using cosine predictive loss.
- **Pose + V-JEPA Fusion**: pose and V-JEPA features projected to 256 dimensions, concatenated, and classified with a Transformer head.

All final comparisons use class-weighted cross entropy for the binary classification objective. Thresholds are tuned on validation data under two policies: positive-class F2 and balanced accuracy.

## Data Setup

The project uses datasets stored on a Modal volume named `final_project_dataset`, mounted at `/data` in remote jobs. The expected root is:

```text
/data/final_project_dataset
```

Final experiments use these data pools:

- **Staged data**: GMDCSA24 + LE2I + CAUCAFall
- **Expanded staged+unstaged data**: staged data + OOPs unstaged fall videos

Generated windows, features, checkpoints, and large run outputs are stored on Modal or ignored locally. They are not committed to this repository.

## Repository Layout

```text
fall_anticipation_cv/
  src/fall_anticipation_cv/        # Reusable dataset/model/training package
    data.py                       # Window construction and raw-frame datasets
    pose_data.py                  # Pose feature datasets
    vjepa_data.py                 # V-JEPA latent datasets
    fusion_data.py                # Pose + V-JEPA fusion datasets
    training_common.py            # Shared training/evaluation helpers
    models/                       # CNN, pose, V-JEPA, and fusion model definitions
  scripts/                        # Local data prep, extraction, training, evaluation utilities
  modal_*.py                      # Modal entrypoints for remote preparation/training/evaluation
  results/                        # Small JSON/CSV/Markdown summaries used for report tables/plots
  AI_USAGE.md                     # Generative AI usage documentation
```

## Installation

For local utilities and report-related scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For RTMPose extraction, install the OpenMMLab/MMPose stack in addition to the base requirements:

```bash
pip install -r requirements-pose.txt
mim install "mmengine>=0.7.0" "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.3.0"
```

## Modal Workflow

Most expensive jobs were run remotely on Modal. The app definitions are split by task so runs can be resumed or repeated selectively.

Common commands:

```bash
# Prepare fall-anticipation window metadata
/opt/anaconda3/bin/python -m modal run modal_prepare.py

# Prepare fallen-state prediction windows
/opt/anaconda3/bin/python -m modal run modal_prepare_fallen.py

# Extract RTMPose features
/opt/anaconda3/bin/python -m modal run modal_extract_pose_l4.py

# Extract frozen V-JEPA latents
/opt/anaconda3/bin/python -m modal run modal_vjepa.py

# Train feature-head models
/opt/anaconda3/bin/python -m modal run modal_train_feature_heads.py

# Train pose + V-JEPA fusion model
/opt/anaconda3/bin/python -m modal run modal_fusion.py
```

Several `modal_evaluate_*.py` and `modal_tune_*.py` files reproduce threshold tuning and per-dataset evaluation summaries used in the report.

## Local Scripts

Useful local scripts include:

```bash
# Data exploration
PYTHONPATH=src python scripts/eda.py --data-root /path/to/final_project_dataset --output-dir outputs/eda

# Prepare video windows locally, if data are available on disk
PYTHONPATH=src python scripts/prepare_windows.py --data-root /path/to/final_project_dataset --output data/windows.csv

# Train a V-JEPA predictive head from pre-extracted latent features
PYTHONPATH=src python scripts/train_vjepa_predictive.py --help

# Train pose, pose-predictive, or fusion heads
PYTHONPATH=src python scripts/train_pose_baseline.py --help
PYTHONPATH=src python scripts/train_pose_predictive.py --help
PYTHONPATH=src python scripts/train_pose_vjepa_fusion.py --help

# Recreate threshold-tradeoff plots
PYTHONPATH=src python scripts/plot_expanded_threshold_tradeoff_curves.py --help
```

## Results

Small result summaries are committed under [`results/`](results/). These include JSON, CSV, and Markdown outputs for:

- staged fall-anticipation model comparisons
- expanded staged+unstaged model comparisons
- fallen-state secondary-task results
- per-dataset generalization summaries
- predictive-loss ablations
- threshold tradeoff curve points and markers
- temporal ablation summaries

## Notes on Large Files

The repository intentionally excludes raw videos, generated feature arrays, model checkpoints, and large training outputs. Those artifacts live on the Modal volume or local scratch directories and can be regenerated from the scripts above.
