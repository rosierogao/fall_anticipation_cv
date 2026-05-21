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

## Train the baseline

```bash
python scripts/train_baseline.py \
  --windows-csv data/windows_gmdcsa24.csv \
  --checkpoint outputs/baseline_simple_video_cnn.pt
```

Large datasets, generated window CSVs, checkpoints, and run outputs are ignored
by Git by default.
