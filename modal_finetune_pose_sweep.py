"""Hyperparameter sweep for fine-tuning PoseTransformerBaseline.

Grid: lr × freeze_projection × num_layers × d_model
  d_model=128: lr(2) × freeze(2) × layers(2) = 8 runs  (weight transfer from pretrained)
  d_model=256: lr(2) × freeze=False × layers(2) = 4 runs  (d_model mismatch → trains from scratch)
Total: 12 runs. All jobs are spawned in parallel; wall-clock ≈ longest single run.

Usage:
    MODAL_PROFILE=cs231n-final-project modal run modal_finetune_pose_sweep.py
    python modal_finetune_pose_sweep.py --dry-run        # print configs locally, no Modal needed
"""

from __future__ import annotations

import itertools
from pathlib import Path

import modal


APP_NAME = "cs231n-final-project"
DATA_ROOT = "/data"
DATASET_ROOT = f"{DATA_ROOT}/final_project_dataset"
VOLUME_NAME = "final_project_dataset"
PACKAGE_REMOTE_ROOT = "/root/project"
LOCAL_PACKAGE_DIR = Path(__file__).parent / "src" / "fall_anticipation_cv"
LOCAL_SCRIPTS_DIR = Path(__file__).parent / "scripts"

app = modal.App(f"{APP_NAME}-finetune-pose-sweep")
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
    .add_local_dir(
        LOCAL_PACKAGE_DIR,
        f"{PACKAGE_REMOTE_ROOT}/fall_anticipation_cv",
        copy=True,
    )
    .add_local_dir(LOCAL_SCRIPTS_DIR, f"{PACKAGE_REMOTE_ROOT}/scripts", copy=True)
    .env({"PYTHONPATH": PACKAGE_REMOTE_ROOT})
)

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

_SWEEP_GRID = {
    "lr": [1e-4, 3e-4],
    "freeze_projection": [True, False],
    "num_layers": [1, 2],
    "d_model": [128, 256],
}


def _build_configs() -> list[dict]:
    keys = list(_SWEEP_GRID.keys())
    configs = []
    for values in itertools.product(*_SWEEP_GRID.values()):
        cfg = dict(zip(keys, values))
        # d_model=256 is incompatible with pretrained weights — force unfreeze
        if cfg["d_model"] == 256 and cfg["freeze_projection"]:
            continue
        freeze_tag = "frozen" if cfg["freeze_projection"] else "unfrozen"
        lr_tag = f"lr{cfg['lr']:.0e}".replace("-0", "-").replace("+0", "")
        cfg["run_name"] = f"d{cfg['d_model']}_layers{cfg['num_layers']}_{freeze_tag}_{lr_tag}"
        cfg["checkpoint"] = f"{DATASET_ROOT}/outputs/sweep/{cfg['run_name']}.pt"
        cfg["metrics_out"] = f"{DATASET_ROOT}/outputs/sweep/{cfg['run_name']}_metrics.json"
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Remote function — one config per invocation
# ---------------------------------------------------------------------------

def _run_script(cmd: list[str]) -> None:
    import subprocess
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu="H100",
    timeout=60 * 60 * 2,
)
def finetune_one_config(
    run_name: str,
    lr: float,
    freeze_projection: bool,
    num_layers: int,
    d_model: int,
    checkpoint: str,
    metrics_out: str,
    pretrained_checkpoint: str = f"{DATASET_ROOT}/outputs/pose_transformer_normalized.pt",
    windows_csv: str = f"{DATASET_ROOT}/pose_windows_rtmpose.csv",
    epochs: int = 10,
    batch_size: int = 32,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
    patience: int = 4,
) -> dict:
    import json
    import sys
    from pathlib import Path

    if not Path(pretrained_checkpoint).exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")
    if not Path(windows_csv).exists():
        raise FileNotFoundError(f"Missing pose windows CSV: {windows_csv}")

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
        "--label-smoothing", str(label_smoothing),
        "--grad-clip", str(grad_clip),
        "--patience", str(patience),
        "--num-workers", "0",
    ]
    if freeze_projection:
        cmd.append("--freeze-projection")

    _run_script(cmd)
    volume.commit()

    metrics = json.loads(Path(metrics_out).read_text())
    metrics["run_name"] = run_name
    return metrics


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

def _print_sweep_results(results: list[dict]) -> None:
    results_sorted = sorted(results, key=lambda r: r.get("positive_f1", 0.0), reverse=True)

    header = (
        f"{'run_name':<42} {'val_acc':>8} {'test_acc':>9}"
        f" {'recall':>8} {'precision':>10} {'f1':>8} {'auc_pr':>8}"
    )
    print("\n" + "=" * len(header))
    print("SWEEP RESULTS (sorted by F1)")
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
        f1 = r.get("positive_f1", 0.0)
        auc_pr = r.get("auc_pr", float("nan"))
        print(
            f"{r['run_name']:<42}"
            f" {r.get('val_acc', float('nan')):>8.4f}"
            f" {r.get('test_acc', float('nan')):>9.4f}"
            f" {recall:>8.4f}"
            f" {precision:>10.4f}"
            f" {f1:>8.4f}"
            f" {auc_pr:>8.4f}"
        )
    print("=" * len(header))
    best = results_sorted[0]
    print(f"\nBest config: {best['run_name']}")
    print(f"  checkpoint: {best.get('checkpoint', 'N/A')}")


def _print_configs() -> None:
    configs = _build_configs()
    print(f"Sweep grid: {_SWEEP_GRID}")
    print(f"Total configs: {len(configs)}\n")
    print(f"{'#':<3} {'run_name':<42} {'lr':>8}  {'freeze':>6}  {'layers':>6}  {'d_model':>7}")
    print("-" * 75)
    for i, cfg in enumerate(configs, 1):
        freeze_tag = str(cfg["freeze_projection"])
        print(
            f"{i:<3} {cfg['run_name']:<42} {cfg['lr']:>8.0e}"
            f"  {freeze_tag:>6}  {cfg['num_layers']:>6}  {cfg['d_model']:>7}"
        )


if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv:
        _print_configs()
        sys.exit(0)


@app.local_entrypoint()
def main(dry_run: bool = False) -> None:
    configs = _build_configs()

    _print_configs()

    if dry_run:
        print("\nDry run — no jobs submitted.")
        return

    print("\nSpawning all jobs in parallel...")
    calls = [
        finetune_one_config.spawn(
            run_name=cfg["run_name"],
            lr=cfg["lr"],
            freeze_projection=cfg["freeze_projection"],
            num_layers=cfg["num_layers"],
            d_model=cfg["d_model"],
            checkpoint=cfg["checkpoint"],
            metrics_out=cfg["metrics_out"],
        )
        for cfg in configs
    ]
    print(f"Spawned {len(calls)} jobs. Waiting for results...")

    results = []
    for cfg, call in zip(configs, calls):
        try:
            result = call.get()
            results.append(result)
            print(f"  done: {cfg['run_name']}")
        except Exception as exc:
            print(f"  FAILED: {cfg['run_name']} — {exc}")

    if results:
        _print_sweep_results(results)
