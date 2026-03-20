#!/usr/bin/env python3
"""
Publication-quality visualization for WCCM conference presentation.

Generates figures for CFRP defect localization using GNN (GAT, 19-class
node classification) results.

Modes:
    confusion      - 19x19 confusion matrix heatmap
    f1_comparison  - Per-class F1 bar chart (multi-experiment)
    ood_comparison - Grouped bar: IID vs OOD split performance
    training_curve - Loss / macro-F1 over epochs
    summary_table  - LaTeX-formatted summary table

Usage:
    # Single confusion matrix from a benchmark results directory
    python tools/wccm_visualize.py --mode confusion \\
        --results_dir runs/benchmark_20260226_023845

    # Compare per-class F1 across two runs
    python tools/wccm_visualize.py --mode f1_comparison \\
        --results_dir runs/benchmark_20260226_023845

    # OOD comparison bar chart
    python tools/wccm_visualize.py --mode ood_comparison \\
        --results_dir runs/benchmark_20260226_023845

    # Training curves from legacy CSV files
    python tools/wccm_visualize.py --mode training_curve \\
        --results_dir GNN_hole/GNN2024/GNNmodelcsv \\
        --csv_pattern "loss_data_0.00*.csv"

    # LaTeX summary table
    python tools/wccm_visualize.py --mode summary_table \\
        --results_dir runs/benchmark_20260226_023845

    # Japanese labels
    python tools/wccm_visualize.py --mode confusion --lang ja \\
        --results_dir runs/benchmark_20260226_023845
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Publication-quality defaults
# ---------------------------------------------------------------------------
_RC_PARAMS = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,       # TrueType for editable text in PDF
    "ps.fonttype": 42,
    "axes.grid": False,
    "figure.constrained_layout.use": True,
}

NUM_CLASSES = 19

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

# English labels (class 0 = defect-free, classes 1-18 = layer-location pairs)
CLASS_LABELS_EN = [
    "Defect-free",                  # 0
    "L2 (0\u00b0)",                 # 1
    "L3 (90\u00b0)",                # 2
    "L4 (0\u00b0)",                 # 3
    "L5 (90\u00b0)",                # 4
    "L6 (0\u00b0)",                 # 5
    "L7 (90\u00b0)",                # 6
    "L8 (0\u00b0)",                 # 7
    "L9 (90\u00b0)",                # 8
    "L10 (0\u00b0)",                # 9
    "L11 (90\u00b0)",               # 10
    "L12 (0\u00b0)",                # 11
    "L13 (90\u00b0)",               # 12
    "L14 (0\u00b0)",                # 13
    "L15 (90\u00b0)",               # 14
    "L16 (0\u00b0)",                # 15
    "L17 (90\u00b0)",               # 16
    "L18 (0\u00b0)",                # 17
    "L19 (90\u00b0)",               # 18
]

CLASS_LABELS_JA = [
    "\u6b20\u9665\u306a\u3057",                         # 0
    "L2 (0\u00b0\u5c64)",                               # 1
    "L3 (90\u00b0\u5c64)",                              # 2
    "L4 (0\u00b0\u5c64)",                               # 3
    "L5 (90\u00b0\u5c64)",                              # 4
    "L6 (0\u00b0\u5c64)",                               # 5
    "L7 (90\u00b0\u5c64)",                              # 6
    "L8 (0\u00b0\u5c64)",                               # 7
    "L9 (90\u00b0\u5c64)",                              # 8
    "L10 (0\u00b0\u5c64)",                              # 9
    "L11 (90\u00b0\u5c64)",                             # 10
    "L12 (0\u00b0\u5c64)",                              # 11
    "L13 (90\u00b0\u5c64)",                             # 12
    "L14 (0\u00b0\u5c64)",                              # 13
    "L15 (90\u00b0\u5c64)",                             # 14
    "L16 (0\u00b0\u5c64)",                              # 15
    "L17 (90\u00b0\u5c64)",                             # 16
    "L18 (0\u00b0\u5c64)",                              # 17
    "L19 (90\u00b0\u5c64)",                             # 18
]

SHORT_LABELS = ["DF"] + [f"C{i}" for i in range(1, 19)]

# Split type display names
SPLIT_NAMES_EN = {
    "iid": "IID",
    "defect_size": "OOD (Defect Size)",
    "layer": "OOD (Layer)",
    "property_ood": "OOD (Property)",
}
SPLIT_NAMES_JA = {
    "iid": "IID",
    "defect_size": "OOD (\u6b20\u9665\u30b5\u30a4\u30ba)",
    "layer": "OOD (\u5c64)",
    "property_ood": "OOD (\u7269\u6027)",
}


def _get_labels(lang: str) -> List[str]:
    return CLASS_LABELS_JA if lang == "ja" else CLASS_LABELS_EN


def _get_split_names(lang: str) -> Dict[str, str]:
    return SPLIT_NAMES_JA if lang == "ja" else SPLIT_NAMES_EN


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_metrics_json(path: Path) -> Dict[str, Any]:
    """Load a single metrics.json file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_metrics_files(results_dir: Path) -> List[Path]:
    """Recursively find all metrics.json files under results_dir."""
    return sorted(results_dir.rglob("metrics.json"))


def _load_benchmark_csv(results_dir: Path) -> Optional[pd.DataFrame]:
    """Load benchmark_results.csv if it exists."""
    csv_path = results_dir / "benchmark_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def _load_benchmark_json(results_dir: Path) -> Optional[Dict[str, Any]]:
    """Load benchmark_results.json if it exists."""
    json_path = results_dir / "benchmark_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_loss_csvs(
    results_dir: Path, pattern: str = "loss_data_*.csv"
) -> Dict[str, pd.DataFrame]:
    """Load legacy loss CSV files (Epoch, Training Loss, Validation Loss)."""
    found = sorted(results_dir.glob(pattern))
    result = {}
    for p in found:
        try:
            df = pd.read_csv(p)
            if {"Epoch", "Training Loss", "Validation Loss"}.issubset(df.columns):
                result[p.stem] = df
        except Exception:
            continue
    return result


def _extract_confusion_matrix(metrics: Dict[str, Any]) -> Optional[np.ndarray]:
    """Try to extract or reconstruct a confusion matrix from metrics data.

    The benchmark metrics.json stores per-epoch data and final test results.
    If a 'confusion_matrix' key is present we use it directly; otherwise
    we look for per_class_f1 arrays and return None if nothing is available.
    """
    # Direct confusion matrix (may be stored by newer training scripts)
    for key in ("confusion_matrix", "cm"):
        if key in metrics:
            return np.array(metrics[key])
        test = metrics.get("test", {})
        if key in test:
            return np.array(test[key])

    return None


def _extract_per_class_f1(metrics: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract per-class F1 scores from metrics dict."""
    # Try direct key
    for container in [metrics, metrics.get("test", {}), metrics.get("metrics", {}).get("test", {})]:
        if isinstance(container, dict):
            for key in ("f1_per_class", "per_class_f1"):
                if key in container:
                    return np.array(container[key])
    return None


def _collect_all_run_data(results_dir: Path) -> List[Dict[str, Any]]:
    """Collect metrics from all runs in a results directory.

    Supports:
      - Benchmark directory with sub-run dirs each containing metrics.json
      - Single run directory with metrics.json at the top level
      - benchmark_results.json / benchmark_results.csv at the top level
    """
    runs: List[Dict[str, Any]] = []

    metric_files = _find_metrics_files(results_dir)
    for mf in metric_files:
        data = _load_metrics_json(mf)
        run_id = data.get("run_id", mf.parent.name)
        # Infer split_type from run_id
        split_type = "unknown"
        for st in ("iid", "defect_size", "layer", "property_ood"):
            if st in run_id:
                split_type = st
                break
        data["_run_id"] = run_id
        data["_split_type"] = split_type
        data["_path"] = str(mf)
        runs.append(data)

    return runs


# ---------------------------------------------------------------------------
# Mode: confusion
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    cmap: str = "Blues",
) -> None:
    """Render a publication-quality confusion matrix heatmap."""
    n = cm.shape[0]
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        cm_plot = cm.astype(float)
        fmt = "d"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # Annotate cells
    thresh = cm_plot.max() / 2.0 if vmax is None else 0.5
    for i in range(n):
        for j in range(n):
            val = cm[i, j] if not normalize else cm_plot[i, j]
            text = f"{val:{fmt}}" if normalize else f"{int(val)}"
            color = "white" if cm_plot[i, j] > thresh else "black"
            fontsize = 6 if n >= 15 else 8
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fontsize)

    tick_labels = labels[:n] if len(labels) >= n else [str(i) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path))
    plt.close(fig)
    print(f"[SAVED] {output_path}")


def mode_confusion(
    results_dir: Path,
    output_dir: Path,
    lang: str = "en",
    **kwargs: Any,
) -> None:
    """Generate confusion matrix figures from available data.

    Sources (checked in order):
      1. metrics.json with 'confusion_matrix' key
      2. Existing confusion_matrix*.png files (just report their paths)
    """
    labels = _get_labels(lang)
    runs = _collect_all_run_data(results_dir)

    plotted = False
    for run in runs:
        cm = _extract_confusion_matrix(run)
        if cm is not None:
            run_id = run.get("_run_id", "unknown")
            out = output_dir / f"confusion_matrix_{run_id}.pdf"
            title = f"Confusion Matrix ({run_id})"
            plot_confusion_matrix(cm, labels, out, title=title)
            # Also save raw count version
            out_raw = output_dir / f"confusion_matrix_raw_{run_id}.pdf"
            plot_confusion_matrix(cm, labels, out_raw, title=f"{title} (counts)", normalize=False)
            plotted = True

    if not plotted:
        # Try to find pre-generated confusion matrix PNGs
        existing = sorted(results_dir.rglob("confusion_matrix*.png"))
        if existing:
            print(f"[INFO] No confusion matrix data in metrics.json, but found {len(existing)} "
                  f"existing confusion matrix image(s):")
            for p in existing:
                print(f"       {p}")
            print("[INFO] To regenerate as PDF, ensure the training script saves the confusion "
                  "matrix array in metrics.json under the key 'confusion_matrix'.")
        else:
            # Generate a demo confusion matrix for template purposes
            print("[INFO] No confusion matrix data found. Generating a demo template.")
            rng = np.random.RandomState(42)
            cm_demo = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
            for i in range(NUM_CLASSES):
                total = rng.randint(50, 200)
                correct = int(total * rng.uniform(0.6, 0.95))
                cm_demo[i, i] = correct
                remaining = total - correct
                for _ in range(remaining):
                    j = rng.randint(0, NUM_CLASSES)
                    cm_demo[i, j] += 1
            out = output_dir / "confusion_matrix_demo.pdf"
            plot_confusion_matrix(cm_demo, labels, out, title="Confusion Matrix (demo)")


# ---------------------------------------------------------------------------
# Mode: f1_comparison
# ---------------------------------------------------------------------------

def mode_f1_comparison(
    results_dir: Path,
    output_dir: Path,
    lang: str = "en",
    **kwargs: Any,
) -> None:
    """Bar chart of per-class F1 scores across experiments."""
    labels = _get_labels(lang)
    runs = _collect_all_run_data(results_dir)

    run_f1s: Dict[str, np.ndarray] = {}
    for run in runs:
        f1 = _extract_per_class_f1(run)
        if f1 is not None:
            run_f1s[run["_run_id"]] = f1[:NUM_CLASSES]

    if not run_f1s:
        # Try benchmark CSV
        df = _load_benchmark_csv(results_dir)
        if df is not None and "test_macro_f1" in df.columns:
            print("[INFO] No per-class F1 in metrics.json. Using aggregate macro F1 from CSV.")
            # Show aggregate metrics as a simpler bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            metric_cols = [c for c in df.columns if c.startswith("test_")]
            if metric_cols and "run_id" in df.columns:
                plot_df = df.set_index("run_id")[metric_cols].T
                plot_df.plot(kind="bar", ax=ax, width=0.7)
                ax.set_ylabel("Score")
                ax.set_title("Test Metrics Comparison" if lang != "ja" else "\u30c6\u30b9\u30c8\u6307\u6a19\u6bd4\u8f03")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.grid(axis="y", alpha=0.3)
                out = output_dir / "test_metrics_comparison.pdf"
                output_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(out))
                plt.close(fig)
                print(f"[SAVED] {out}")
                return

        print("[WARN] No per-class F1 data found for f1_comparison mode.")
        print("       Ensure metrics.json contains 'f1_per_class' or 'per_class_f1' arrays.")
        return

    n_runs = len(run_f1s)
    x = np.arange(NUM_CLASSES)
    width = 0.8 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_runs, 1)))

    for idx, (run_id, f1_arr) in enumerate(run_f1s.items()):
        offset = (idx - n_runs / 2 + 0.5) * width
        short_id = run_id.split("_")[-1] if len(run_id) > 30 else run_id
        ax.bar(x + offset, f1_arr, width, label=short_id, color=colors[idx], edgecolor="gray", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=9)
    ax.set_xlabel("Class" if lang != "ja" else "\u30af\u30e9\u30b9")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 Score Comparison" if lang != "ja" else "\u30af\u30e9\u30b9\u5225F1\u30b9\u30b3\u30a2\u6bd4\u8f03")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    out = output_dir / "f1_per_class_comparison.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)
    print(f"[SAVED] {out}")


# ---------------------------------------------------------------------------
# Mode: ood_comparison
# ---------------------------------------------------------------------------

def mode_ood_comparison(
    results_dir: Path,
    output_dir: Path,
    lang: str = "en",
    **kwargs: Any,
) -> None:
    """Grouped bar chart: IID vs OOD performance."""
    split_names = _get_split_names(lang)

    # Try benchmark CSV first
    df = _load_benchmark_csv(results_dir)
    if df is None:
        # Aggregate from metrics.json
        runs = _collect_all_run_data(results_dir)
        if not runs:
            print("[WARN] No data found for ood_comparison mode.")
            return
        rows = []
        for run in runs:
            test_data = run.get("test", {})
            row = {
                "run_id": run.get("_run_id", ""),
                "split_type": run.get("_split_type", "unknown"),
                "test_macro_f1": test_data.get("macro_f1"),
                "test_weighted_f1": test_data.get("weighted_f1"),
                "test_balanced_accuracy": test_data.get("balanced_accuracy"),
                "test_mcc": test_data.get("mcc"),
                "test_accuracy": test_data.get("accuracy"),
            }
            rows.append(row)
        df = pd.DataFrame(rows)

    if df.empty or "split_type" not in df.columns:
        print("[WARN] No split_type information available for OOD comparison.")
        return

    # Metrics to compare
    metric_keys = [
        ("test_macro_f1", "Macro F1"),
        ("test_mcc", "MCC"),
        ("test_accuracy", "Accuracy"),
    ]
    available_metrics = [(k, name) for k, name in metric_keys if k in df.columns]
    if not available_metrics:
        print("[WARN] No recognized test metrics in data.")
        return

    # Group by split_type and compute mean +/- std
    split_types_present = [st for st in ("iid", "defect_size", "layer", "property_ood")
                           if st in df["split_type"].values]
    if not split_types_present:
        split_types_present = sorted(df["split_type"].dropna().unique())

    n_splits = len(split_types_present)
    n_metrics = len(available_metrics)
    x = np.arange(n_splits)
    width = 0.8 / max(n_metrics, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_splits * 2.5), 5))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]

    for midx, (mcol, mname) in enumerate(available_metrics):
        means = []
        stds = []
        for st in split_types_present:
            vals = pd.to_numeric(df.loc[df["split_type"] == st, mcol], errors="coerce").dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)

        offset = (midx - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=mname,
            color=colors[midx % len(colors)],
            edgecolor="gray", linewidth=0.5,
        )
        # Value labels on top of bars
        for bar, mean_val in zip(bars, means):
            if mean_val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{mean_val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

    split_display = [split_names.get(st, st) for st in split_types_present]
    ax.set_xticks(x)
    ax.set_xticklabels(split_display, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("IID vs OOD Performance" if lang != "ja" else "IID \u5bfe OOD \u6027\u80fd\u6bd4\u8f03")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    out = output_dir / "ood_comparison.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)
    print(f"[SAVED] {out}")


# ---------------------------------------------------------------------------
# Mode: training_curve
# ---------------------------------------------------------------------------

def mode_training_curve(
    results_dir: Path,
    output_dir: Path,
    lang: str = "en",
    csv_pattern: str = "loss_data_*.csv",
    **kwargs: Any,
) -> None:
    """Plot training/validation loss and macro-F1 curves.

    Supports two data sources:
      1. Legacy CSV files with columns (Epoch, Training Loss, Validation Loss)
      2. metrics.json with 'epochs' array containing per-epoch metrics
    """
    plotted = False

    # --- Source 1: Legacy CSV files ---
    loss_dfs = _load_loss_csvs(results_dir, pattern=csv_pattern)
    if loss_dfs:
        n = len(loss_dfs)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
        for idx, (name, df) in enumerate(loss_dfs.items()):
            epoch = df["Epoch"]
            # Short legend name
            short = name.replace("loss_data_", "").replace("_", " ")
            axes[0].plot(epoch, df["Training Loss"], color=colors[idx],
                         linestyle="-", alpha=0.8, label=f"{short} (train)")
            axes[0].plot(epoch, df["Validation Loss"], color=colors[idx],
                         linestyle="--", alpha=0.8, label=f"{short} (val)")

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss" if lang != "ja" else "\u8a13\u7df4\u30fb\u691c\u8a3c\u640d\u5931")
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].grid(alpha=0.3)
        axes[0].set_yscale("log")

        # Right panel: if any df has an F1 column
        has_f1 = False
        for idx, (name, df) in enumerate(loss_dfs.items()):
            for f1_col in ("Macro F1", "macro_f1", "F1"):
                if f1_col in df.columns:
                    short = name.replace("loss_data_", "").replace("_", " ")
                    axes[1].plot(df["Epoch"], df[f1_col], color=colors[idx],
                                 label=short, alpha=0.8)
                    has_f1 = True
                    break
        if has_f1:
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Macro F1")
            axes[1].set_title("Macro F1 over Epochs" if lang != "ja" else "\u30a8\u30dd\u30c3\u30af\u6bce\u306eMacro F1")
            axes[1].legend(fontsize=7)
            axes[1].grid(alpha=0.3)
        else:
            axes[1].set_visible(False)

        out = output_dir / "training_curves_legacy.pdf"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out))
        plt.close(fig)
        print(f"[SAVED] {out}")
        plotted = True

    # --- Source 2: metrics.json epoch arrays ---
    runs = _collect_all_run_data(results_dir)
    epoch_runs = [(r["_run_id"], r["epochs"]) for r in runs
                  if r.get("epochs") and len(r["epochs"]) > 0]

    if epoch_runs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(epoch_runs), 1)))

        for idx, (run_id, epochs_data) in enumerate(epoch_runs):
            ep = [e.get("epoch", i) for i, e in enumerate(epochs_data)]
            val_loss = [e.get("val_loss") for e in epochs_data]
            macro_f1 = [e.get("macro_f1") for e in epochs_data]

            short = run_id[-30:] if len(run_id) > 30 else run_id

            # Loss
            if any(v is not None for v in val_loss):
                train_loss = [e.get("train_loss") for e in epochs_data]
                if any(v is not None for v in train_loss):
                    axes[0].plot(ep, train_loss, color=colors[idx],
                                 linestyle="-", alpha=0.7, label=f"{short} (train)")
                axes[0].plot(ep, val_loss, color=colors[idx],
                             linestyle="--", alpha=0.7, label=f"{short} (val)")

            # F1
            if any(v is not None for v in macro_f1):
                axes[1].plot(ep, macro_f1, color=colors[idx], alpha=0.7, label=short)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss" if lang != "ja" else "\u8a13\u7df4\u30fb\u691c\u8a3c\u640d\u5931")
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro F1")
        axes[1].set_title("Macro F1 over Epochs" if lang != "ja" else "\u30a8\u30dd\u30c3\u30af\u6bce\u306eMacro F1")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)

        out = output_dir / "training_curves.pdf"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out))
        plt.close(fig)
        print(f"[SAVED] {out}")
        plotted = True

    if not plotted:
        print("[WARN] No training curve data found.")
        print("       Provide either legacy loss CSV files or metrics.json with 'epochs' data.")


# ---------------------------------------------------------------------------
# Mode: summary_table
# ---------------------------------------------------------------------------

def mode_summary_table(
    results_dir: Path,
    output_dir: Path,
    lang: str = "en",
    **kwargs: Any,
) -> None:
    """Generate a LaTeX-formatted summary table."""
    df = _load_benchmark_csv(results_dir)
    if df is None:
        # Build from metrics.json
        runs = _collect_all_run_data(results_dir)
        if not runs:
            print("[WARN] No data found for summary_table mode.")
            return
        rows = []
        for run in runs:
            test = run.get("test", {})
            rows.append({
                "run_id": run.get("_run_id", ""),
                "split_type": run.get("_split_type", "unknown"),
                "test_accuracy": test.get("accuracy"),
                "test_macro_f1": test.get("macro_f1"),
                "test_weighted_f1": test.get("weighted_f1"),
                "test_balanced_accuracy": test.get("balanced_accuracy"),
                "test_mcc": test.get("mcc"),
            })
        df = pd.DataFrame(rows)

    if df.empty:
        print("[WARN] Empty results, no table generated.")
        return

    # Compute per-split aggregate
    split_names = _get_split_names(lang)
    metric_cols = ["test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "test_mcc", "test_accuracy"]
    metric_display = {
        "test_macro_f1": "Macro F1",
        "test_weighted_f1": "Weighted F1",
        "test_balanced_accuracy": "Balanced Acc.",
        "test_mcc": "MCC",
        "test_accuracy": "Accuracy",
    }
    available = [c for c in metric_cols if c in df.columns]

    agg_rows = []
    split_order = ["iid", "defect_size", "layer", "property_ood"]
    for st in split_order:
        subset = df[df.get("split_type", pd.Series()) == st] if "split_type" in df.columns else pd.DataFrame()
        if subset.empty:
            continue
        row = {"Split": split_names.get(st, st)}
        for col in available:
            vals = pd.to_numeric(subset[col], errors="coerce").dropna()
            if len(vals) > 1:
                row[metric_display[col]] = f"{vals.mean():.4f} $\\pm$ {vals.std():.4f}"
            elif len(vals) == 1:
                row[metric_display[col]] = f"{vals.iloc[0]:.4f}"
            else:
                row[metric_display[col]] = "--"
        agg_rows.append(row)

    # Also include any splits not in the canonical order
    if "split_type" in df.columns:
        extra = set(df["split_type"].dropna().unique()) - set(split_order)
        for st in sorted(extra):
            subset = df[df["split_type"] == st]
            row = {"Split": split_names.get(st, st)}
            for col in available:
                vals = pd.to_numeric(subset[col], errors="coerce").dropna()
                if len(vals) > 1:
                    row[metric_display[col]] = f"{vals.mean():.4f} $\\pm$ {vals.std():.4f}"
                elif len(vals) == 1:
                    row[metric_display[col]] = f"{vals.iloc[0]:.4f}"
                else:
                    row[metric_display[col]] = "--"
            agg_rows.append(row)

    if not agg_rows:
        # Fall back to per-run table without split grouping
        for _, row_data in df.iterrows():
            r = {"Split": str(row_data.get("run_id", ""))}
            for col in available:
                val = row_data.get(col)
                r[metric_display[col]] = f"{val:.4f}" if pd.notna(val) else "--"
            agg_rows.append(r)

    agg_df = pd.DataFrame(agg_rows)

    # Write LaTeX
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "summary_table.tex"

    n_cols = len(agg_df.columns)
    col_fmt = "l" + "c" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Benchmark results: IID and OOD evaluation of GAT-based CFRP defect localization.}",
        r"  \label{tab:benchmark}",
        f"  \\begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        "    " + " & ".join(f"\\textbf{{{c}}}" for c in agg_df.columns) + r" \\",
        r"    \midrule",
    ]
    for _, row in agg_df.iterrows():
        cells = [str(row[c]) for c in agg_df.columns]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    tex_content = "\n".join(lines) + "\n"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"[SAVED] {tex_path}")

    # Also save as CSV for convenience
    csv_path = output_dir / "summary_table.csv"
    agg_df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    # Print to console
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(agg_df.to_string(index=False))
    print()
    print(f"LaTeX source: {tex_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODE_FUNCS = {
    "confusion": mode_confusion,
    "f1_comparison": mode_f1_comparison,
    "ood_comparison": mode_ood_comparison,
    "training_curve": mode_training_curve,
    "summary_table": mode_summary_table,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WCCM conference presentation figure generator for CFRP GNN results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=list(MODE_FUNCS.keys()),
        help="Visualization mode",
    )
    parser.add_argument(
        "--results_dir", required=True, type=str,
        help="Path to benchmark results directory or CSV file directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="tools/wccm_figures",
        help="Output directory for generated figures (default: tools/wccm_figures/)",
    )
    parser.add_argument(
        "--lang", type=str, choices=["en", "ja"], default="en",
        help="Label language: en (English) or ja (Japanese)",
    )
    parser.add_argument(
        "--csv_pattern", type=str, default="loss_data_*.csv",
        help="Glob pattern for legacy loss CSV files (training_curve mode)",
    )

    args = parser.parse_args()

    # Resolve paths relative to repo root if not absolute
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return 1

    # Apply publication-quality matplotlib settings
    plt.rcParams.update(_RC_PARAMS)

    # Check for Japanese font availability
    if args.lang == "ja":
        try:
            import matplotlib.font_manager as fm
            jp_fonts = [f.name for f in fm.fontManager.ttflist
                        if any(kw in f.name.lower() for kw in ("gothic", "mincho", "noto", "ipag", "ipam", "takao", "ume"))]
            if jp_fonts:
                plt.rcParams["font.family"] = jp_fonts[0]
                print(f"[INFO] Using Japanese font: {jp_fonts[0]}")
            else:
                print("[WARN] No Japanese font found. Install fonts-noto-cjk or similar. "
                      "Falling back to default font (Japanese characters may not render).")
        except Exception:
            pass

    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Results dir: {results_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Language: {args.lang}")

    func = MODE_FUNCS[args.mode]
    func(
        results_dir=results_dir,
        output_dir=output_dir,
        lang=args.lang,
        csv_pattern=args.csv_pattern,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
