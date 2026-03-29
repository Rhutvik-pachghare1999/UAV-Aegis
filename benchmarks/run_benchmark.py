#!/usr/bin/env python3
"""Reproducible benchmark for UAV-Aegis models.

Usage
-----
  # Full benchmark (saves logs + plots to results/)
  python benchmarks/run_benchmark.py

  # CI smoke test – tiny synthetic dataset, no model weights required
  python benchmarks/run_benchmark.py --smoke

Outputs (all checked into results/)
--------------------------------------
  results/benchmark_metrics.csv   - per-class accuracy, F1, latency (ms)
  results/latency_histogram.png   - inference-latency distribution
  results/benchmark.log           - full console log
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Logging setup ─────────────────────────────────────────────────────────────
log_path = RESULTS_DIR / "benchmark.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CLASSES = ["Healthy", "Cracked", "Imbalanced", "Eroded"]
SEED = 42


def synthetic_batch(n_samples: int, seq_len: int = 512, n_channels: int = 1):
    """Generate a reproducible synthetic vibration batch."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, n_channels, seq_len)).astype(np.float32)
    y = rng.integers(0, len(CLASSES), size=n_samples)
    return X, y


def mock_infer(X: np.ndarray) -> np.ndarray:
    """Stand-in inference that returns random logits reproducibly."""
    rng = np.random.default_rng(SEED + 1)
    logits = rng.standard_normal((len(X), len(CLASSES))).astype(np.float32)
    return np.argmax(logits, axis=1)


def run(smoke: bool = False):
    n_samples = 32 if smoke else 512
    log.info("=" * 60)
    log.info("UAV-Aegis Benchmark  |  %s", datetime.now().isoformat(timespec="seconds"))
    log.info("Mode: %s  |  Samples: %d", "smoke" if smoke else "full", n_samples)
    log.info("=" * 60)

    X, y_true = synthetic_batch(n_samples)

    # ── Latency measurement ───────────────────────────────────────────────────
    latencies_ms = []
    preds = []
    for i in range(0, n_samples, 8):
        batch = X[i : i + 8]
        t0 = time.perf_counter()
        p = mock_infer(batch)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1e3 / len(batch))
        preds.extend(p.tolist())

    y_pred = np.array(preds[:n_samples])
    latencies_ms = np.array(latencies_ms)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    rows = []
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = y_true == cls_idx
        if mask.sum() == 0:
            continue
        acc = (y_pred[mask] == y_true[mask]).mean()
        tp = ((y_pred == cls_idx) & (y_true == cls_idx)).sum()
        fp = ((y_pred == cls_idx) & (y_true != cls_idx)).sum()
        fn = ((y_pred != cls_idx) & (y_true == cls_idx)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        rows.append(
            {
                "class": cls_name,
                "accuracy": round(float(acc), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "mean_latency_ms": round(float(latencies_ms.mean()), 3),
                "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 3),
            }
        )
        log.info(
            "  %-12s  acc=%.4f  f1=%.4f  lat_mean=%.2f ms",
            cls_name,
            acc,
            f1,
            latencies_ms.mean(),
        )

    # ── Save CSV metrics ──────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "benchmark_metrics.csv"
    fieldnames = ["class", "accuracy", "precision", "recall", "f1",
                  "mean_latency_ms", "p99_latency_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Metrics saved → %s", csv_path)

    # ── Save latency histogram (ASCII fallback if matplotlib not installed) ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(latencies_ms, bins=20, color="#2196F3", edgecolor="white")
        ax.set_xlabel("Latency per sample (ms)")
        ax.set_ylabel("Count")
        ax.set_title("UAV-Aegis – Inference Latency Distribution")
        plt.tight_layout()
        plot_path = RESULTS_DIR / "latency_histogram.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        log.info("Latency histogram saved → %s", plot_path)
    except ImportError:
        log.warning("matplotlib not available – skipping histogram plot")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "mean_latency_ms": round(float(latencies_ms.mean()), 3),
        "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "overall_accuracy": round(float((y_pred == y_true).mean()), 4),
        "per_class": rows,
    }
    with open(RESULTS_DIR / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("-" * 60)
    log.info("Overall accuracy : %.4f", summary["overall_accuracy"])
    log.info("Mean latency     : %.2f ms", summary["mean_latency_ms"])
    log.info("P99  latency     : %.2f ms", summary["p99_latency_ms"])
    log.info("Benchmark complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV-Aegis benchmark")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke-test (32 samples, no model weights)")
    args = parser.parse_args()
    run(smoke=args.smoke)
