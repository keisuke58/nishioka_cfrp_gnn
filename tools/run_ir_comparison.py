#!/usr/bin/env python3
"""FEM-only (4ch) vs FEM+synthetic IR (7ch) comparison experiment.

Trains paired models on identical data/splits/seeds and compares metrics.

Usage:
    python tools/run_ir_comparison.py
    python tools/run_ir_comparison.py --seeds 42 123 456 --epochs 150 --split iid
    python tools/run_ir_comparison.py --epochs 50 --seeds 42 --data_usage_ratio 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gnn_common.models import GATModel
from gnn_common.losses import FocalLogitAdjustLoss
from gnn_common.data_utils import prepare_data, group_disjoint_split
from gnn_common.metrics import metrics_from_confusion_matrix, compute_benchmark_metrics
from gnn_common.training_utils import set_seed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default data paths (same as train_multitask.py)
# ---------------------------------------------------------------------------
_DEFAULT_DATA_BASE = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise"
_DEFAULT_LABEL_DIR = "/home/nishioka/GNN/GNN_hole_2026/all_19class_label"
_DEFAULT_COORD_DIR = "/home/nishioka/GNN/GNN_hole/GNN_hole_data"
_MAX_NODES = 13942
_NUM_CLASSES = 19


# ============================================================================
# Data loading helpers
# ============================================================================

def load_coordinates_and_edges():
    """Load shared coordinate and edge data."""
    coord_dir = _DEFAULT_COORD_DIR
    x_coords = np.load(os.path.join(coord_dir, "normalized_x_2layer.npy"))
    y_coords = np.load(os.path.join(coord_dir, "normalized_y_2layer.npy"))
    z_coords = np.load(os.path.join(coord_dir, "normalized_z_2layer.npy"))
    edges = np.load(os.path.join(coord_dir, "hole_edges_2layer_best.npy"))
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    return x_coords, y_coords, z_coords, edge_index


def discover_data_files(seed: int, max_defect: int = 5000):
    """Discover and sample data files, returning (data_file, label_file, data_dir) triples."""
    data_base = _DEFAULT_DATA_BASE
    label_dir = _DEFAULT_LABEL_DIR

    # Noise (defect-free) files
    noise_folder = os.path.join(data_base, "train")
    noise_files = sorted([
        f for f in os.listdir(noise_folder)
        if f.startswith("NoiseDefectFree_") and f.endswith(".npy")
    ]) if os.path.isdir(noise_folder) else []

    # Defect files across splits
    all_defect_files = []
    for split_name in ("train", "val", "test"):
        folder = os.path.join(data_base, split_name)
        if os.path.isdir(folder):
            all_defect_files.extend([
                (f, folder) for f in os.listdir(folder)
                if f.startswith("Defect_L") and f.endswith(".npy")
            ])

    # Reproducible subsample of defect files
    rng = np.random.RandomState(seed)
    num_defect = min(max_defect, len(all_defect_files))
    sel_idx = rng.choice(len(all_defect_files), size=num_defect, replace=False)
    selected_defect = [all_defect_files[i] for i in sel_idx]

    # Build triples
    all_pairs = []
    for nf in noise_files:
        base = os.path.splitext(nf)[0]
        all_pairs.append((nf, f"{base}_19label.npy", noise_folder))
    for df, d_dir in selected_defect:
        base = os.path.splitext(df)[0]
        lf = f"{base}_19label.npy"
        if os.path.exists(os.path.join(label_dir, lf)):
            all_pairs.append((df, lf, d_dir))

    return all_pairs


def split_data(all_pairs, seed: int):
    """Split data into train/val/test using group-disjoint split."""
    train_pairs, val_pairs, test_pairs, train_map, val_map, test_map = (
        group_disjoint_split(
            all_pairs,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            group_key="LBel",
            seed=seed,
        )
    )
    return train_pairs, val_pairs, test_pairs, train_map, val_map, test_map


def build_datasets(
    train_pairs, val_pairs, test_pairs,
    train_map, val_map, test_map,
    x_coords, y_coords, z_coords, edge_index,
    use_synthetic_ir: bool,
    ir_seed: int = 42,
):
    """Prepare PyG datasets for a given IR configuration."""
    label_dir = _DEFAULT_LABEL_DIR

    train_data, class_weights = prepare_data(
        train_pairs, None, label_dir,
        x_coords, y_coords, z_coords, edge_index,
        data_folder_map=train_map,
        max_nodes=_MAX_NODES,
        return_class_weights=True,
        use_synthetic_ir=use_synthetic_ir,
        ir_noise_std=0.05,
        ir_seed=ir_seed,
    )
    val_data = prepare_data(
        val_pairs, None, label_dir,
        x_coords, y_coords, z_coords, edge_index,
        data_folder_map=val_map,
        max_nodes=_MAX_NODES,
        return_class_weights=False,
        use_synthetic_ir=use_synthetic_ir,
        ir_noise_std=0.05,
        ir_seed=ir_seed + 10000,
    )
    test_data = prepare_data(
        test_pairs, None, label_dir,
        x_coords, y_coords, z_coords, edge_index,
        data_folder_map=test_map,
        max_nodes=_MAX_NODES,
        return_class_weights=False,
        use_synthetic_ir=use_synthetic_ir,
        ir_noise_std=0.05,
        ir_seed=ir_seed + 20000,
    )

    return train_data, val_data, test_data, class_weights


# ============================================================================
# Training and evaluation
# ============================================================================

def build_loss_fn(class_prior: torch.Tensor, device: torch.device):
    """Build FocalLogitAdjustLoss from training class prior."""
    return FocalLogitAdjustLoss(
        class_prior=class_prior.to(device),
        tau=3.0,
        gamma=2.0,
    )


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, use_amp=False):
    """Train for one epoch, return average loss."""
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch)
            loss = loss_fn(out, batch.y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes=_NUM_CLASSES):
    """Evaluate model on a dataset, return metrics dict."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    total_loss = 0.0
    num_batches = 0
    all_true = []
    all_pred = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = loss_fn(out, batch.y)
        total_loss += loss.item()
        num_batches += 1

        preds = out.argmax(dim=1)
        probs = F.softmax(out, dim=1)

        for t, p in zip(batch.y.view(-1), preds.view(-1)):
            cm[t, p] += 1

        all_true.extend(batch.y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    cm_np = cm.cpu().numpy()
    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)
    all_probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, num_classes))

    benchmark = compute_benchmark_metrics(
        all_labels=all_true_np,
        all_preds=all_pred_np,
        num_classes=num_classes,
        all_probs=all_probs_np if len(all_probs_np) > 0 else None,
    )
    benchmark["val_loss"] = total_loss / max(num_batches, 1)

    # Per-class f1 from confusion matrix
    cm_metrics = metrics_from_confusion_matrix(cm_np)
    benchmark["f1_per_class"] = cm_metrics["f1_per_class"].tolist()

    return benchmark


def run_single_experiment(
    input_channels: int,
    use_synthetic_ir: bool,
    train_data,
    val_data,
    test_data,
    class_weights: torch.Tensor,
    seed: int,
    epochs: int,
    patience: int = 50,
    batch_size: int = 64,
    hidden_channels: int = 64,
    num_heads: int = 4,
    lr: float = 0.002,
    dropout: float = 0.2,
) -> Dict[str, Any]:
    """Train a single GATModel and return test metrics."""
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    train_loader = PyGDataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = PyGDataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    ) if val_data else None
    test_loader = PyGDataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    ) if test_data else None

    # Model
    model = GATModel(
        hidden_channels=hidden_channels,
        num_classes=_NUM_CLASSES,
        num_heads=num_heads,
        dropout=dropout,
        input_channels=input_channels,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    # Loss
    all_train_labels = []
    for d in train_data:
        all_train_labels.extend(d.y.tolist())
    all_train_labels = np.array(all_train_labels)
    class_counts = np.bincount(all_train_labels, minlength=_NUM_CLASSES).astype(np.float64)
    class_prior = torch.tensor(class_counts / class_counts.sum(), dtype=torch.float32)
    loss_fn = build_loss_fn(class_prior, device)

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-6,
    )

    # Training loop
    best_macro_f1 = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    t_start = time.time()
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, None, device
        )

        # Validation
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            macro_f1 = val_metrics["macro_f1"]
            scheduler.step(macro_f1)

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 25 == 0:
                logger.info(
                    "  [ch=%d] Epoch %d/%d | train_loss=%.4f | val_macro_f1=%.4f (best=%.4f@%d)",
                    input_channels, epoch, epochs, train_loss, macro_f1, best_macro_f1, best_epoch,
                )

            if patience_counter >= patience:
                logger.info(
                    "  [ch=%d] Early stopping at epoch %d (best=%d, f1=%.4f)",
                    input_channels, epoch, best_epoch, best_macro_f1,
                )
                break

    train_time = time.time() - t_start

    # Load best model for test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Test evaluation
    test_metrics = {}
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, loss_fn, device)

    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_val_macro_f1"] = best_macro_f1
    test_metrics["train_time_sec"] = train_time
    test_metrics["num_params"] = num_params

    return test_metrics


# ============================================================================
# Main comparison runner
# ============================================================================

def run_comparison(args) -> List[Dict[str, Any]]:
    """Run the full 4ch vs 7ch comparison across all seeds."""
    logger.info("=" * 70)
    logger.info("IR Fusion Comparison Experiment")
    logger.info("  Seeds: %s", args.seeds)
    logger.info("  Epochs: %d", args.epochs)
    logger.info("  Split: %s", args.split)
    logger.info("  Data usage ratio: %.2f", args.data_usage_ratio)
    logger.info("=" * 70)

    x_coords, y_coords, z_coords, edge_index = load_coordinates_and_edges()
    logger.info("Loaded coordinates and edges")

    all_results = []

    for seed in args.seeds:
        logger.info("\n%s", "-" * 60)
        logger.info("Seed: %d", seed)
        logger.info("-" * 60)

        # Discover and split data (same split for both configurations)
        all_pairs = discover_data_files(seed=42)  # Fixed discovery seed for consistency
        train_pairs, val_pairs, test_pairs, train_map, val_map, test_map = split_data(
            all_pairs, seed=seed,
        )

        # Apply data_usage_ratio
        if args.data_usage_ratio < 1.0:
            r = args.data_usage_ratio
            train_pairs = train_pairs[:max(1, int(len(train_pairs) * r))]
            val_pairs = val_pairs[:max(1, int(len(val_pairs) * r))]
            test_pairs = test_pairs[:max(1, int(len(test_pairs) * r))]
            logger.info("data_usage_ratio=%.2f => train=%d, val=%d, test=%d",
                        r, len(train_pairs), len(val_pairs), len(test_pairs))

        logger.info("Split: train=%d, val=%d, test=%d",
                     len(train_pairs), len(val_pairs), len(test_pairs))

        # ---------- Configuration A: FEM-only (4ch) ----------
        logger.info("\n>>> Training FEM-only (4ch) model, seed=%d", seed)
        train_4, val_4, test_4, cw_4 = build_datasets(
            train_pairs, val_pairs, test_pairs,
            train_map, val_map, test_map,
            x_coords, y_coords, z_coords, edge_index,
            use_synthetic_ir=False,
            ir_seed=seed,
        )
        logger.info("  FEM-only datasets: train=%d, val=%d, test=%d",
                     len(train_4), len(val_4) if val_4 else 0, len(test_4) if test_4 else 0)

        metrics_4ch = run_single_experiment(
            input_channels=4,
            use_synthetic_ir=False,
            train_data=train_4,
            val_data=val_4,
            test_data=test_4,
            class_weights=cw_4,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            hidden_channels=args.hidden_channels,
            lr=args.lr,
        )
        all_results.append({
            "seed": seed,
            "config": "FEM_4ch",
            "input_channels": 4,
            **{k: v for k, v in metrics_4ch.items() if k != "f1_per_class"},
            **{f"f1_class_{i}": f1 for i, f1 in enumerate(metrics_4ch.get("f1_per_class", []))},
        })
        logger.info("  4ch test: macro_f1=%.4f, mcc=%.4f, acc=%.4f",
                     metrics_4ch.get("macro_f1", 0), metrics_4ch.get("mcc", 0),
                     metrics_4ch.get("accuracy", 0))

        # ---------- Configuration B: FEM+IR (7ch) ----------
        logger.info("\n>>> Training FEM+synthetic IR (7ch) model, seed=%d", seed)
        train_7, val_7, test_7, cw_7 = build_datasets(
            train_pairs, val_pairs, test_pairs,
            train_map, val_map, test_map,
            x_coords, y_coords, z_coords, edge_index,
            use_synthetic_ir=True,
            ir_seed=seed,
        )
        logger.info("  IR fusion datasets: train=%d, val=%d, test=%d",
                     len(train_7), len(val_7) if val_7 else 0, len(test_7) if test_7 else 0)

        metrics_7ch = run_single_experiment(
            input_channels=7,
            use_synthetic_ir=True,
            train_data=train_7,
            val_data=val_7,
            test_data=test_7,
            class_weights=cw_7,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            hidden_channels=args.hidden_channels,
            lr=args.lr,
        )
        all_results.append({
            "seed": seed,
            "config": "FEM_IR_7ch",
            "input_channels": 7,
            **{k: v for k, v in metrics_7ch.items() if k != "f1_per_class"},
            **{f"f1_class_{i}": f1 for i, f1 in enumerate(metrics_7ch.get("f1_per_class", []))},
        })
        logger.info("  7ch test: macro_f1=%.4f, mcc=%.4f, acc=%.4f",
                     metrics_7ch.get("macro_f1", 0), metrics_7ch.get("mcc", 0),
                     metrics_7ch.get("accuracy", 0))

    return all_results


def save_results_csv(results: List[Dict[str, Any]], output_path: str):
    """Save results list to CSV."""
    if not results:
        logger.warning("No results to save.")
        return

    # Collect all keys
    all_keys = []
    for r in results:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    logger.info("Results saved to %s", output_path)


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a summary comparing 4ch vs 7ch averaged over seeds."""
    import pandas as pd

    df = pd.DataFrame(results)

    metric_cols = ["macro_f1", "mcc", "accuracy", "weighted_f1", "balanced_accuracy"]
    available_cols = [c for c in metric_cols if c in df.columns]

    print("\n" + "=" * 80)
    print("IR FUSION COMPARISON SUMMARY")
    print("=" * 80)

    # Per-seed results
    print("\n--- Per-seed results ---")
    display_cols = ["seed", "config", "input_channels"] + available_cols + ["best_epoch", "train_time_sec"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False, float_format="%.4f"))

    # Aggregated comparison
    print("\n--- Aggregated comparison (mean +/- std) ---")
    agg = df.groupby("config")[available_cols].agg(["mean", "std"])
    for config_name in ["FEM_4ch", "FEM_IR_7ch"]:
        if config_name not in agg.index:
            continue
        row = agg.loc[config_name]
        parts = []
        for col in available_cols:
            m = row[(col, "mean")]
            s = row[(col, "std")]
            parts.append(f"{col}={m:.4f}+/-{s:.4f}")
        print(f"  {config_name:12s}: {', '.join(parts)}")

    # Delta
    if "FEM_4ch" in agg.index and "FEM_IR_7ch" in agg.index:
        print("\n--- Delta (7ch - 4ch) ---")
        for col in available_cols:
            m4 = agg.loc["FEM_4ch", (col, "mean")]
            m7 = agg.loc["FEM_IR_7ch", (col, "mean")]
            delta = m7 - m4
            sign = "+" if delta >= 0 else ""
            print(f"  {col:25s}: {sign}{delta:.4f}")

    # Per-class F1 comparison (averaged over seeds)
    class_f1_cols = [c for c in df.columns if c.startswith("f1_class_")]
    if class_f1_cols:
        print("\n--- Per-class F1 comparison (mean over seeds) ---")
        print(f"  {'Class':>8s}  {'FEM_4ch':>10s}  {'FEM_IR_7ch':>10s}  {'Delta':>10s}")
        df_4 = df[df["config"] == "FEM_4ch"]
        df_7 = df[df["config"] == "FEM_IR_7ch"]
        for col in sorted(class_f1_cols, key=lambda c: int(c.split("_")[-1])):
            cls_id = col.split("_")[-1]
            m4 = df_4[col].mean() if col in df_4.columns else float("nan")
            m7 = df_7[col].mean() if col in df_7.columns else float("nan")
            d = m7 - m4
            sign = "+" if d >= 0 else ""
            print(f"  {cls_id:>8s}  {m4:>10.4f}  {m7:>10.4f}  {sign}{d:>9.4f}")

    print("\n" + "=" * 80)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="FEM-only (4ch) vs FEM+synthetic IR (7ch) comparison experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Random seeds for reproducibility (default: [42, 123, 456])")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Maximum training epochs (default: 150)")
    parser.add_argument("--split", type=str, default="iid",
                        help="Split strategy (default: iid)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--hidden_channels", type=int, default=64,
                        help="Hidden channels in GAT (default: 64)")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate (default: 0.002)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: tools/ir_comparison_results.csv)")
    parser.add_argument("--data_usage_ratio", type=float, default=1.0,
                        help="Fraction of data to use, for quick testing (default: 1.0)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.output_csv is None:
        args.output_csv = str(REPO_ROOT / "tools" / "ir_comparison_results.csv")

    results = run_comparison(args)
    save_results_csv(results, args.output_csv)
    print_summary_table(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
