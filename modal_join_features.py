from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-join-features")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas<3.0")
    .add_local_dir(LOCAL_SCRIPTS_DIR, f"{PACKAGE_REMOTE_ROOT}/scripts", copy=True)
)


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=60 * 10)
def join_feature_csvs(
    windows_csv: str,
    feature_csvs: list[str],
    output_csv: str,
    required_feature_col: str,
) -> str:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/join_feature_csvs.py",
        "--windows-csv",
        windows_csv,
        "--output-csv",
        output_csv,
        "--required-feature-col",
        required_feature_col,
    ]
    for feature_csv in feature_csvs:
        cmd.extend(["--feature-csv", feature_csv])

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()
    return output_csv


@app.local_entrypoint()
def main(
    windows_csv: str,
    feature_csv_1: str,
    feature_csv_2: str,
    output_csv: str,
    required_feature_col: str,
) -> None:
    output = join_feature_csvs.remote(
        windows_csv=windows_csv,
        feature_csvs=[feature_csv_1, feature_csv_2],
        output_csv=output_csv,
        required_feature_col=required_feature_col,
    )
    print(f"Joined feature CSV: {output}", flush=True)
