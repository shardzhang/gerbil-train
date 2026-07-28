#!/usr/bin/env python3
"""Batch benchmark runner for all GERBIL models.

Usage:
    # Quick test (2 epochs, CPU)
    python run_benchmark.py --epochs 2 --device cpu --models deepfm,din,wnd

    # Full benchmark (5 epochs, MPS for Apple Silicon)
    python run_benchmark.py --epochs 5 --device mps

    # Single model family only
    python run_benchmark.py --epochs 5 --device mps --family ctr

    # Dry-run to list what would execute
    python run_benchmark.py --dry-run

    # Resume from previous run (skip already-completed models)
    python run_benchmark.py --epochs 5 --device mps --resume

Output:
    benchmark_results/benchmark_<timestamp>.csv  -- per-model metrics
    checkpoints/<model>/<timestamp>/              -- run artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.resolve()

MODELS: list[dict] = [
    # ---- Classic CTR (Feature Interaction) ----
    {"name": "fm",           "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.1-fm_train",           "config": "configs/1-fm/experiment.yaml",            "monitor": "val_auc"},
    {"name": "ffm",          "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.1-ffm_train",          "config": "configs/1-ffm/experiment.yaml",           "monitor": "val_auc"},
    {"name": "ftrl",         "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.1-lr_train",         "config": "configs/1-lr/experiment.yaml",          "monitor": "val_auc"},
    {"name": "afm",          "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.3-afm_train",          "config": "configs/3-afm/experiment.yaml",           "monitor": "val_auc"},
    {"name": "ncf",          "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.3-ncf_train",          "config": "configs/3-ncf/experiment.yaml",           "monitor": "val_auc"},
    {"name": "nfm",          "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.3-nfm_train",          "config": "configs/3-nfm/experiment.yaml",           "monitor": "val_auc"},
    {"name": "pnn",          "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.3-pnn_train",          "config": "configs/3-pnn/experiment.yaml",           "monitor": "val_auc"},
    {"name": "wide_and_deep","family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.4-wide_and_deep_train","config": "configs/4-wide_and_deep/experiment.yaml", "monitor": "val_gauc"},
    {"name": "deepfm",       "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.5-deepfm_train",       "config": "configs/5-deepfm/experiment.yaml",        "monitor": "val_gauc"},
    {"name": "xdeepfm",      "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.5-xdeepfm_train",      "config": "configs/5-xdeepfm/experiment.yaml",       "monitor": "val_gauc"},
    {"name": "autoint",      "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.6-autoint_train",      "config": "configs/6-autoint/experiment.yaml",       "monitor": "val_auc"},
    {"name": "dcn",          "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.6-dcn_train",          "config": "configs/6-dcn/experiment.yaml",           "monitor": "val_auc"},
    {"name": "dcnv2",        "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.6-dcnv2_train",        "config": "configs/6-dcnv2/experiment.yaml",         "monitor": "val_auc"},
    {"name": "dlrm",         "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.6-dlrm_train",         "config": "configs/6-dlrm/experiment.yaml",          "monitor": "val_auc"},
    {"name": "fibinet",      "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.6-fibinet_train",      "config": "configs/6-fibinet/experiment.yaml",       "monitor": "val_auc"},
    {"name": "masknet",      "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.6-masknet_train",      "config": "configs/6-masknet/experiment.yaml",       "monitor": "val_auc"},
    {"name": "star",         "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.6-star_train",         "config": "configs/6-star/experiment.yaml",          "monitor": "val_auc"},
    {"name": "gwen_binary",  "family": "ctr",       "task": "binary", "cli": "gerbil_train.cli.2-gwen_binary_train",  "config": "configs/2-gwen_ml1m_binary/experiment.yaml", "monitor": "val_gauc"},

    # ---- Sequential & Interest ----
    {"name": "din",        "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.7-din_train",    "config": "configs/7-din/experiment.yaml",      "monitor": "val_gauc"},
    {"name": "dien",       "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.7-dien_train",   "config": "configs/7-dien/experiment.yaml",     "monitor": "val_gauc"},
    {"name": "dsin",       "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.7-dsin_train",   "config": "configs/7-dsin/experiment.yaml",     "monitor": "val_gauc"},
    {"name": "mimn",       "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.10-mimn_train",  "config": "configs/10-mimn/experiment.yaml",    "monitor": "val_auc"},
    {"name": "sim",        "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.11-sim_train",   "config": "configs/11-sim/experiment.yaml",     "monitor": "val_auc"},
    {"name": "mind",       "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.12-mind_train",  "config": "configs/12-mind/experiment.yaml",    "monitor": "val_auc"},
    {"name": "bst",        "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.13-bst_train",   "config": "configs/13-bst/experiment.yaml",     "monitor": "val_auc"},
    {"name": "eta",        "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.14-eta_train",   "config": "configs/14-eta/experiment.yaml",     "monitor": "val_auc"},
    {"name": "sdim",       "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.15-sdim_train",  "config": "configs/15-sdim/experiment.yaml",    "monitor": "val_auc"},
    {"name": "sasrec",     "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.16-sasrec_train","config": "configs/16-sasrec/experiment.yaml",  "monitor": "val_auc"},
    {"name": "bert4rec",   "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.17-bert4rec_train","config": "configs/17-bert4rec/experiment.yaml","monitor": "val_auc"},
    {"name": "twin",       "family": "sequential", "task": "binary", "cli": "gerbil_train.cli.25-twin_train",  "config": "configs/25-twin/experiment.yaml",    "monitor": "val_auc"},

    # ---- Multi-Task & Multi-Domain ----
    {"name": "esmm",       "family": "multitask",  "task": "binary", "cli": "gerbil_train.cli.20-esmm_train",    "config": "configs/20-esmm/experiment.yaml",     "monitor": "val_auc"},
    {"name": "mmoe",       "family": "multitask",  "task": "binary", "cli": "gerbil_train.cli.21-mmoe_train",    "config": "configs/21-mmoe/experiment.yaml",     "monitor": "val_auc"},
    {"name": "ple",        "family": "multitask",  "task": "binary", "cli": "gerbil_train.cli.22-ple_train",     "config": "configs/22-ple/experiment.yaml",      "monitor": "val_auc"},
    {"name": "gatenet",    "family": "multitask",  "task": "binary", "cli": "gerbil_train.cli.23-gatenet_train", "config": "configs/23-gatenet/experiment.yaml",  "monitor": "val_auc"},
    {"name": "pepnet",     "family": "multitask",  "task": "binary", "cli": "gerbil_train.cli.23-pepnet_train",  "config": "configs/23-pepnet/experiment.yaml",   "monitor": "val_auc"},
    {"name": "adatt",      "family": "multitask",  "task": "binary", "cli": "gerbil_train.cli.24-adatt_train",   "config": "configs/24-adatt/experiment.yaml",    "monitor": "val_auc"},

    # ---- Ranking & Pairwise ----
    {"name": "bpr",        "family": "ranking",    "task": "binary", "cli": "gerbil_train.cli.99-bpr_train",    "config": "configs/99-bpr/experiment.yaml",      "monitor": "val_auc"},
    {"name": "mvdnn",      "family": "ranking",    "task": "binary", "cli": "gerbil_train.cli.99-mvdnn_train",  "config": "configs/99-mvdnn/experiment.yaml",    "monitor": "val_auc"},

    # ---- Recall / Embedding ----
    {"name": "eges",       "family": "recall",     "task": "binary", "cli": "gerbil_train.cli.31-eges_train",       "config": "configs/31-eges/experiment.yaml",      "monitor": "val_auc"},
    {"name": "word2vec",   "family": "recall",     "task": "binary", "cli": "gerbil_train.cli.32-word2vec_train",   "config": "configs/32-word2vec/experiment.yaml",  "monitor": "val_loss"},
    {"name": "node2vec",   "family": "recall",     "task": "binary", "cli": "gerbil_train.cli.33-node2vec_train",   "config": "configs/33-node2vec/experiment.yaml",  "monitor": "val_loss"},
    {"name": "ssl",        "family": "recall",     "task": "binary", "cli": "gerbil_train.cli.34-ssl_train",        "config": "configs/34-ssl/experiment.yaml",       "monitor": "val_loss"},

    # ---- Multi-class ----
    {"name": "youtube_dnn","family": "multi",      "task": "multi",  "cli": "gerbil_train.cli.2-youtube_dnn_train",     "config": "configs/2-youtube_dnn/experiment.yaml",     "monitor": "val_hit@1"},
    {"name": "gwen_multi", "family": "multi",      "task": "multi",  "cli": "gerbil_train.cli.2-gwen_multiclass_train","config": "configs/2-gwen_ml1m_multiclass/experiment.yaml", "monitor": "val_hit@1"},
]

FAMILIES = sorted(set(m["family"] for m in MODELS))

RESULTS_DIR = PROJECT_ROOT / "benchmark_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch benchmark runner for gerbil-train models")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs per model (default: 5)")
    p.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"], help="PyTorch device (default: mps)")
    p.add_argument("--models", type=str, default=None, help="Comma-separated model names to run (default: all)")
    p.add_argument("--family", type=str, default=None, choices=FAMILIES, help="Run only one model family")
    p.add_argument("--dry-run", action="store_true", help="List models without running")
    p.add_argument("--resume", action="store_true", help="Skip models that already have results")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def _collect_metrics_from_log(log_path: Path) -> dict:
    """Parse the exp.log for the best and final test metrics."""
    metrics = {}
    if not log_path.exists():
        return metrics

    text = log_path.read_text()

    best_epoch = None
    for m in re.finditer(r"Saved best model.*Epoch (\d+)", text):
        best_epoch = int(m.group(1))

    final_match = re.search(r"Final test metrics:\s*(\{.*?\})", text)
    if final_match:
        try:
            d = json.loads(final_match.group(1).replace("'", '"'))
            metrics.update(d)
        except json.JSONDecodeError:
            metrics["test_raw"] = final_match.group(1)

    epoch_pattern = re.compile(r"Epoch (\d+).*? loss: ([\d.]+) \| val_loss: ([\d.]+).*?auc: ([\d.]+).*?gauc: ([\d.]+)")
    for m in epoch_pattern.finditer(text):
        epoch_idx = int(m.group(1))
        if epoch_idx == best_epoch:
            metrics["best_epoch"] = epoch_idx
            metrics["best_loss"] = float(m.group(2))
            metrics["best_val_auc"] = float(m.group(4))
            metrics["best_val_gauc"] = float(m.group(5))

    time_pattern = re.findall(r"\|\s*steps/s:\s*([\d.]+)\s*\|\s*time:\s*([\d.]+)", text)
    if time_pattern:
        total_time = sum(float(t) for _, t in time_pattern)
        avg_speed = sum(float(s) for s, _ in time_pattern) / len(time_pattern)
        metrics["total_time_s"] = round(total_time, 1)
        metrics["avg_steps_per_sec"] = round(avg_speed, 2)

    return metrics


def run_one_model(m: dict, args: argparse.Namespace) -> Optional[dict]:
    """Run a single model and return its results dict, or None if skipped/failed."""
    result = {
        "model": m["name"],
        "family": m["family"],
        "task": m["task"],
        "monitor": m["monitor"],
        "device": args.device,
        "epochs": args.epochs,
        "seed": args.seed,
        "status": "unknown",
    }

    run_dir = PROJECT_ROOT / "checkpoints" / m["name"]

    if args.resume:
        existing_runs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir()],
            key=os.path.getmtime, reverse=True,
        ) if run_dir.exists() else []
        for existing in existing_runs:
            metrics = _collect_metrics_from_log(existing / "exp.log")
            if metrics.get("total_time_s"):
                print(f"  [SKIP] {m['name']}: already has complete results in {existing.name}")
                result.update(metrics)
                result["status"] = "cached"
                return result

    print(f"\n{'='*60}")
    print(f"  [{m['family'].upper()}] {m['name']} ({m['task']}, {args.epochs} epochs, device={args.device})")
    print(f"{'='*60}")

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = [
        sys.executable, "-m", m["cli"],
        "--config", str(PROJECT_ROOT / m["config"]),
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.epochs * 1200,  # 20 min per epoch timeout
        )
        elapsed = time.time() - t0
        result["wall_time_s"] = round(elapsed, 1)

        if proc.returncode != 0:
            print(f"  [FAIL] {m['name']}: exit code {proc.returncode}")
            print(f"  stderr: {proc.stderr[-500:]}")
            result["status"] = "failed"
            result["error"] = proc.stderr[-200:] if proc.stderr else f"exit code {proc.returncode}"
            return result

        latest_run = sorted(
            [d for d in run_dir.iterdir() if d.is_dir()],
            key=os.path.getmtime, reverse=True,
        )[0] if run_dir.exists() else None

        if latest_run:
            metrics = _collect_metrics_from_log(latest_run / "exp.log")
            result.update(metrics)
            result["status"] = "completed" if metrics else "no_metrics"

        final_line = [l for l in proc.stdout.strip().split("\n") if "Final test metrics" in l]
        if final_line:
            print(f"  {final_line[-1].strip()}")

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {m['name']}")
        result["status"] = "timeout"
        result["wall_time_s"] = round(time.time() - t0, 1)

    return result


def main():
    args = parse_args()

    selected = MODELS
    if args.models:
        names = set(args.models.split(","))
        selected = [m for m in MODELS if m["name"] in names]
    if args.family:
        selected = [m for m in selected if m["family"] == args.family]

    if not selected:
        print("No models selected.")
        return

    print(f"Models: {len(selected)} | Epochs: {args.epochs} | Device: {args.device} | Seed: {args.seed}")
    print(f"Models by family: {dict((f, len([m for m in selected if m['family']==f])) for f in sorted(set(m['family'] for m in selected)))}")

    if args.dry_run:
        print("\nDry-run mode — would execute:")
        for i, m in enumerate(selected):
            print(f"  {i+1:2d}. {m['name']:15s} [{m['family']:12s}] {m['task']:6s}")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"benchmark_{timestamp}.csv"

    results = []
    for i, m in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] Running {m['name']}...")
        r = run_one_model(m, args)
        if r:
            results.append(r)

        if r and r.get("status") in ("completed", "cached"):
            _write_csv(results, csv_path)

    _write_csv(results, csv_path)

    print(f"\n{'='*60}")
    print(f"Benchmark complete. Results saved to {csv_path}")
    print(f"Summary: {sum(1 for r in results if r.get('status')=='completed')}/{len(results)} completed")
    print(f"{'='*60}")

    _print_summary(results)


def _write_csv(results: list[dict], path: Path):
    if not results:
        return
    fieldnames = ["model", "family", "task", "monitor", "device", "epochs", "seed", "status",
                  "test_auc", "test_gauc", "test_ap", "test_map", "test_mrr",
                  "test_hit@1", "test_hit@10",
                  "best_epoch", "best_loss", "best_val_auc", "best_val_gauc",
                  "total_time_s", "wall_time_s", "avg_steps_per_sec", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


def _print_summary(results: list[dict]):
    completed = [r for r in results if r.get("status") in ("completed", "cached") and "test_auc" in r]
    if not completed:
        print("No completed results to summarize.")
        return

    print("\nModel Ranking by test_auc:")
    completed.sort(key=lambda r: r.get("test_auc", 0), reverse=True)
    print(f"{'Rank':<5} {'Model':<18} {'Family':<12} {'AUC':<8} {'GAUC':<8} {'Time(s)':<10}")
    print("-" * 65)
    for i, r in enumerate(completed):
        auc = r.get("test_auc", "N/A")
        gauc = r.get("test_gauc", "N/A")
        t = r.get("total_time_s", r.get("wall_time_s", "N/A"))
        print(f"{i+1:<5} {r['model']:<18} {r['family']:<12} {str(auc):<8} {str(gauc):<8} {str(t):<10}")


if __name__ == "__main__":
    main()
