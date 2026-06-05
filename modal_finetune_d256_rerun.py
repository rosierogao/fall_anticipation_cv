"""One-off re-run of the 4 d256 configs that failed due to the strict=False load bug."""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-finetune-d256")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "opencv-python-headless==4.10.0.84",
        "pandas<3.0",
        "scikit-learn",
        "torch==2.3.1",
        "tqdm",
    )
    .add_local_dir(LOCAL_PACKAGE_DIR, f"{PACKAGE_REMOTE_ROOT}/fall_anticipation_cv", copy=True)
    .add_local_dir(LOCAL_SCRIPTS_DIR, f"{PACKAGE_REMOTE_ROOT}/scripts", copy=True)
    .env({"PYTHONPATH": PACKAGE_REMOTE_ROOT})
)

CONFIGS = [
    {"run_name": "d256_layers1_unfrozen_lr1e-4", "lr": 1e-4, "num_layers": 1, "d_model": 256},
    {"run_name": "d256_layers2_unfrozen_lr1e-4", "lr": 1e-4, "num_layers": 2, "d_model": 256},
    {"run_name": "d256_layers1_unfrozen_lr3e-4", "lr": 3e-4, "num_layers": 1, "d_model": 256},
    {"run_name": "d256_layers2_unfrozen_lr3e-4", "lr": 3e-4, "num_layers": 2, "d_model": 256},
]
for cfg in CONFIGS:
    cfg["checkpoint"] = f"{DATASET_ROOT}/outputs/sweep/{cfg['run_name']}.pt"
    cfg["metrics_out"] = f"{DATASET_ROOT}/outputs/sweep/{cfg['run_name']}_metrics.json"


def _run_script(cmd: list[str]) -> None:
    import subprocess
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=image, volumes={DATA_ROOT: volume}, gpu="H100", timeout=60 * 30)
def finetune_d256(
    run_name: str,
    lr: float,
    num_layers: int,
    d_model: int,
    checkpoint: str,
    metrics_out: str,
    pretrained_checkpoint: str = f"{DATASET_ROOT}/outputs/pose_transformer_normalized.pt",
    windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    epochs: int = 10,
    batch_size: int = 32,
    grad_clip: float = 1.0,
    patience: int = 4,
) -> dict:
    import json
    import sys
    from pathlib import Path

    cmd = [
        sys.executable,
        f"{PACKAGE_REMOTE_ROOT}/scripts/finetune_pose_transformer.py",
        "--pretrained-checkpoint", pretrained_checkpoint,
        "--windows-csv", windows_csv,
        "--checkpoint", checkpoint,
        "--metrics", metrics_out,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--hidden-dim", str(d_model),
        "--lr", str(lr),
        "--label-smoothing", "0.0",
        "--grad-clip", str(grad_clip),
        "--patience", str(patience),
        "--num-workers", "0",
    ]
    _run_script(cmd)
    volume.commit()
    metrics = json.loads(Path(metrics_out).read_text())
    metrics["run_name"] = run_name
    return metrics


@app.local_entrypoint()
def main() -> None:
    print(f"Re-running {len(CONFIGS)} d256 configs in parallel...")
    calls = [finetune_d256.spawn(**cfg) for cfg in CONFIGS]

    results = []
    for cfg, call in zip(CONFIGS, calls):
        try:
            result = call.get()
            results.append(result)
            print(f"  done: {cfg['run_name']}")
        except Exception as exc:
            print(f"  FAILED: {cfg['run_name']} — {exc}")

    if results:
        results_sorted = sorted(results, key=lambda r: r.get("positive_f1", 0.0), reverse=True)
        header = f"{'run_name':<40} {'test_acc':>9} {'recall':>8} {'precision':>10} {'f1':>8} {'auc_pr':>8}"
        print("\n" + "=" * len(header))
        print("D256 RESULTS")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for r in results_sorted:
            cm = r.get("confusion_matrix", {})
            tp = cm.get("true_positive", 0)
            fp = cm.get("false_positive", 0)
            fn = cm.get("false_negative", 0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            print(
                f"{r['run_name']:<40}"
                f" {r.get('test_acc', float('nan')):>9.4f}"
                f" {recall:>8.4f}"
                f" {precision:>10.4f}"
                f" {r.get('positive_f1', 0.0):>8.4f}"
                f" {r.get('auc_pr', float('nan')):>8.4f}"
            )
        print("=" * len(header))
