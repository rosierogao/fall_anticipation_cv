from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"

app = modal.App(f"{APP_NAME}-split-analysis")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pandas", "scikit-learn")
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
    timeout=60 * 10,
)
def analyze_split(
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
) -> None:
    import pandas as pd
    from fall_anticipation_cv.data import split_by_subject

    windows = pd.read_csv(windows_csv)
    train_df, val_df, test_df = split_by_subject(windows)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    # Determine which datasets are present
    all_datasets = sorted(windows["dataset"].dropna().unique()) if "dataset" in windows.columns else ["(unknown)"]

    total_pos = (windows["y"] == 1).sum()
    total_neg = (windows["y"] == 0).sum()
    total = len(windows)

    print(f"\n{'='*70}")
    print(f"Windows CSV: {windows_csv}")
    print(f"Total windows: {total:,}  |  positive: {total_pos:,}  |  negative: {total_neg:,}")
    print(f"Overall positive rate: {total_pos/total*100:.1f}%")
    print(f"{'='*70}")

    for split_name, df in splits.items():
        n = len(df)
        n_pos = (df["y"] == 1).sum()
        n_neg = (df["y"] == 0).sum()
        pos_pct = n_pos / n * 100 if n > 0 else 0

        print(f"\n--- {split_name.upper()} ({n:,} windows total | {n_pos:,} pos / {n_neg:,} neg | {pos_pct:.1f}% positive) ---")

        if "dataset" in df.columns:
            for dataset in all_datasets:
                ds = df[df["dataset"] == dataset]
                if ds.empty:
                    print(f"  {dataset:12s}  (none)")
                    continue
                ds_pos = (ds["y"] == 1).sum()
                ds_neg = (ds["y"] == 0).sum()
                ds_total = len(ds)
                ds_pos_pct = ds_pos / ds_total * 100 if ds_total > 0 else 0
                print(
                    f"  {dataset:12s}  total={ds_total:5,}  pos={ds_pos:4,} ({ds_pos_pct:5.1f}%)  neg={ds_neg:5,}"
                )
        else:
            print(f"  (no dataset column — single dataset)")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (positive windows per split x dataset)")
    print(f"{'':14s}", end="")
    for split_name in splits:
        print(f"  {split_name:>10s}", end="")
    print("   total")

    for dataset in all_datasets:
        print(f"  {dataset:12s}", end="")
        row_total = 0
        for df in splits.values():
            if "dataset" in df.columns:
                ds = df[df["dataset"] == dataset]
                n_pos = (ds["y"] == 1).sum()
            else:
                n_pos = (df["y"] == 1).sum()
            print(f"  {n_pos:>10,}", end="")
            row_total += n_pos
        print(f"  {row_total:>7,}")

    print(f"  {'total':12s}", end="")
    grand_total = 0
    for df in splits.values():
        n_pos = (df["y"] == 1).sum()
        print(f"  {n_pos:>10,}", end="")
        grand_total += n_pos
    print(f"  {grand_total:>7,}")
    print(f"{'='*70}\n")


@app.local_entrypoint()
def main(
    windows_csv: str = f"{DATASET_ROOT}/windows_gmdcsa24.csv",
) -> None:
    analyze_split.remote(windows_csv=windows_csv)
