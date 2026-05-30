from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-merge-features")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(LOCAL_SCRIPTS_DIR, f"{PACKAGE_REMOTE_ROOT}/scripts", copy=True)
)


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 10)
def merge_feature_csvs(
    staged_csv: str,
    oops_csv: str,
    output_csv: str,
) -> str:
    import sys
    import subprocess

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/merge_feature_csvs.py",
        "--input",
        staged_csv,
        "--input",
        oops_csv,
        "--output",
        output_csv,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()
    return output_csv


@app.local_entrypoint()
def main(
    staged_csv: str,
    oops_csv: str,
    output_csv: str,
) -> None:
    merged = merge_feature_csvs.remote(
        staged_csv=staged_csv,
        oops_csv=oops_csv,
        output_csv=output_csv,
    )
    print(f"Merged feature CSV: {merged}", flush=True)
