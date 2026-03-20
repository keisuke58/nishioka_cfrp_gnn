"""Two-stage GNN training for CFRP defect detection.

Addresses extreme class imbalance (1:170 defect-to-normal ratio) by splitting
the 19-class node classification into two stages:
  Stage 1: Binary classifier (defect vs no-defect) on full graph
  Stage 2: 18-class location classifier on defect nodes only (much more balanced)

Combined pipeline: predict binary -> classify location -> map back to 19 classes

Usage:
    python tools/train_twostage.py --stage1_epochs 300 --stage2_epochs 200
    python tools/train_twostage.py --stage1_lr 0.002 --stage2_lr 0.001 --batch_size 64
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn_common.models_twostage import (
    BinaryDefectModel,
    LocationClassifierModel,
    TwoStageInference,
    labels_to_binary,
    labels_to_location,
)
from gnn_common.losses import FocalLossLogSoftmax, LogitAdjustLoss, FocalLogitAdjustLoss
from gnn_common.data_utils import (
    prepare_data,
    compute_class_weights,
    group_disjoint_split,
)
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


# ============================================================================
# Data loading (reuses train_multitask.py pattern)
# ============================================================================
def load_data_splits(args):
    """Load and split data into train/val/test, returning PyG Data lists."""
    coord_dir = _DEFAULT_COORD_DIR
    x_coords = np.load(os.path.join(coord_dir, "normalized_x_2layer.npy"))
    y_coords = np.load(os.path.join(coord_dir, "normalized_y_2layer.npy"))
    z_coords = np.load(os.path.join(coord_dir, "normalized_z_2layer.npy"))
    edges = np.load(os.path.join(coord_dir, "hole_edges_2layer_best.npy"))
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    data_base = _DEFAULT_DATA_BASE
    label_dir = _DEFAULT_LABEL_DIR

    # Collect noise (defect-free) files
    noise_folder = os.path.join(data_base, "train")
    noise_files = sorted([
        f for f in os.listdir(noise_folder)
        if f.startswith("NoiseDefectFree_") and f.endswith(".npy")
    ]) if os.path.isdir(noise_folder) else []

    # Collect defect files
    all_defect_files = []
    for split_name in ("train", "val", "test"):
        folder = os.path.join(data_base, split_name)
        if os.path.isdir(folder):
            all_defect_files.extend([
                (f, folder) for f in os.listdir(folder)
                if f.startswith("Defect_L") and f.endswith(".npy")
            ])

    # Select 5000 defect samples (reproducible)
    rng = np.random.RandomState(42)
    num_defect = min(5000, len(all_defect_files))
    sel_idx = rng.choice(len(all_defect_files), size=num_defect, replace=False)
    selected_defect = [all_defect_files[i] for i in sel_idx]

    # Build (data_file, label_file, data_dir) triples
    all_pairs = []
    for nf in noise_files:
        base = os.path.splitext(nf)[0]
        all_pairs.append((nf, f"{base}_19label.npy", noise_folder))
    for df, d_dir in selected_defect:
        base = os.path.splitext(df)[0]
        lf = f"{base}_19label.npy"
        if os.path.exists(os.path.join(label_dir, lf)):
            all_pairs.append((df, lf, d_dir))

    logger.info("Total samples: %d (noise=%d, defect=%d)",
                len(all_pairs), len(noise_files), len(all_pairs) - len(noise_files))

    # Split
    if args.split_manifest is not None:
        with open(args.split_manifest) as f:
            manifest = json.load(f)
        logger.info("Using split manifest: %s", args.split_manifest)
        train_pairs = [tuple(p[:2]) for p in manifest["train"]]
        val_pairs = [tuple(p[:2]) for p in manifest["val"]]
        test_pairs = [tuple(p[:2]) for p in manifest["test"]]
        train_map = {p[0]: p[2] for p in manifest["train"]}
        val_map = {p[0]: p[2] for p in manifest["val"]}
        test_map = {p[0]: p[2] for p in manifest["test"]}
    else:
        train_pairs, val_pairs, test_pairs, train_map, val_map, test_map = group_disjoint_split(
            all_pairs,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            group_key=args.group_key, seed=42,
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

    # Prepare datasets
    train_data, class_weights = prepare_data(
        train_pairs, None, label_dir,
        x_coords, y_coords, z_coords, edge_index,
        data_folder_map=train_map, max_nodes=_MAX_NODES,
        return_class_weights=True,
    )
    val_data = prepare_data(
        val_pairs, None, label_dir,
        x_coords, y_coords, z_coords, edge_index,
        data_folder_map=val_map, max_nodes=_MAX_NODES,
        return_class_weights=False,
    )
    test_data = prepare_data(
        test_pairs, None, label_dir,
        x_coords, y_coords, z_coords, edge_index,
        data_folder_map=test_map, max_nodes=_MAX_NODES,
        return_class_weights=False,
    )

    if train_data is None:
        logger.error("No training data loaded. Exiting.")
        sys.exit(1)

    # Convert size_class to tensor (for compatibility)
    for data_list in [train_data, val_data, test_data]:
        if data_list is None:
            continue
        for d in data_list:
            sc = d.size_class if hasattr(d, "size_class") else -1
            d.size_class = torch.tensor(sc, dtype=torch.long) if not isinstance(sc, torch.Tensor) else sc

    logger.info("Datasets loaded: train=%d, val=%d, test=%d",
                len(train_data), len(val_data) if val_data else 0,
                len(test_data) if test_data else 0)

    return train_data, val_data, test_data


# ============================================================================
# Stage 1: Binary training
# ============================================================================
def train_stage1_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Train binary model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        binary_labels = labels_to_binary(batch.y)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)  # [N, 2]
        loss = criterion(logits, binary_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_stage1(model, loader, criterion, device):
    """Validate binary model. Returns loss, accuracy, precision, recall, f1."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    cm = torch.zeros(2, 2, dtype=torch.long, device=device)

    for batch in loader:
        batch = batch.to(device)
        binary_labels = labels_to_binary(batch.y)
        logits = model(batch)
        loss = criterion(logits, binary_labels)
        total_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=1)
        for t, p in zip(binary_labels.view(-1), preds.view(-1)):
            cm[t, p] += 1

    cm_np = cm.cpu().numpy()
    metrics = metrics_from_confusion_matrix(cm_np)
    avg_loss = total_loss / max(num_batches, 1)

    # Binary-specific metrics
    tp = cm_np[1, 1]
    fp = cm_np[0, 1]
    fn = cm_np[1, 0]
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "val_loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "defect_precision": float(precision),
        "defect_recall": float(recall),
        "defect_f1": float(f1),
        "macro_f1": metrics["macro_f1"],
    }


# ============================================================================
# Stage 2: Location training (defect nodes only)
# ============================================================================
def train_stage2_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Train location model for one epoch (loss on defect nodes only)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_defect_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        # Identify defect nodes
        defect_mask = batch.y > 0
        if not defect_mask.any():
            continue  # skip graphs with no defect nodes

        optimizer.zero_grad(set_to_none=True)
        # Forward on full graph (GNN needs neighborhood context)
        logits = model(batch)  # [N, 18]
        # Loss only on defect nodes, with labels shifted to 0-17
        location_labels = labels_to_location(batch.y[defect_mask])
        loss = criterion(logits[defect_mask], location_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        num_defect_nodes += defect_mask.sum().item()

    return total_loss / max(num_batches, 1), num_defect_nodes


@torch.no_grad()
def validate_stage2(model, loader, criterion, device, num_location_classes=18):
    """Validate location model (defect nodes only)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    cm = torch.zeros(num_location_classes, num_location_classes, dtype=torch.long, device=device)

    for batch in loader:
        batch = batch.to(device)
        defect_mask = batch.y > 0
        if not defect_mask.any():
            continue

        logits = model(batch)
        location_labels = labels_to_location(batch.y[defect_mask])
        loss = criterion(logits[defect_mask], location_labels)
        total_loss += loss.item()
        num_batches += 1

        preds = logits[defect_mask].argmax(dim=1)
        for t, p in zip(location_labels.view(-1), preds.view(-1)):
            cm[t, p] += 1

    cm_np = cm.cpu().numpy()
    metrics = metrics_from_confusion_matrix(cm_np)
    avg_loss = total_loss / max(num_batches, 1)

    return {
        "val_loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_f1_support_only": metrics["macro_f1_support_only"],
        "f1_per_class": metrics["f1_per_class"],
    }


# ============================================================================
# Combined pipeline evaluation
# ============================================================================
@torch.no_grad()
def evaluate_twostage_pipeline(binary_model, location_model, loader, device,
                                binary_threshold=0.5):
    """Evaluate the full two-stage pipeline, mapping predictions to 19-class space."""
    pipeline = TwoStageInference(binary_model, location_model, binary_threshold)
    pipeline.eval()

    all_true = []
    all_pred = []
    cm_19 = torch.zeros(19, 19, dtype=torch.long, device=device)

    for batch in loader:
        batch = batch.to(device)
        preds_19, _ = pipeline(batch)
        true_labels = batch.y

        for t, p in zip(true_labels.view(-1), preds_19.view(-1)):
            cm_19[t, p] += 1

        all_true.extend(true_labels.cpu().numpy().tolist())
        all_pred.extend(preds_19.cpu().numpy().tolist())

    cm_19_np = cm_19.cpu().numpy()
    metrics_19 = metrics_from_confusion_matrix(cm_19_np)

    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)

    benchmark = compute_benchmark_metrics(
        all_labels=all_true_np,
        all_preds=all_pred_np,
        num_classes=19,
    )

    return {
        "cm_19": cm_19_np,
        "metrics_19": metrics_19,
        "benchmark": benchmark,
        "true_labels": all_true,
        "pred_labels": all_pred,
    }


# ============================================================================
# Confusion matrix plot helper
# ============================================================================
def save_confusion_matrix_plot(cm, path, title="Confusion Matrix"):
    """Save confusion matrix as a heatmap PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(8, cm.shape[0] * 0.6), max(6, cm.shape[0] * 0.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(cm.shape[1]), yticklabels=range(cm.shape[0]))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    logger.info("Device: %s", device)

    # Output directory
    output_root = args.output_root or "/home/nishioka/GNN/GNN_hole_2026"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"twostage_{timestamp}"
    if args.run_id:
        run_tag = f"twostage_{args.run_id}_{timestamp}"
    result_dir = os.path.join(output_root, "twostage_results", run_tag)
    os.makedirs(result_dir, exist_ok=True)
    logger.info("Results will be saved to %s", result_dir)

    # --- Load data ---
    train_data, val_data, test_data = load_data_splits(args)

    # --- Compute class statistics ---
    all_train_labels = []
    for d in train_data:
        all_train_labels.extend(d.y.tolist())
    all_train_labels = np.array(all_train_labels)

    # Binary class prior
    binary_labels_train = labels_to_binary(torch.tensor(all_train_labels)).numpy()
    binary_counts = np.bincount(binary_labels_train, minlength=2).astype(np.float64)
    binary_prior = torch.tensor(binary_counts / binary_counts.sum(), dtype=torch.float32)
    logger.info("Binary class distribution: no-defect=%d (%.1f%%), defect=%d (%.1f%%)",
                int(binary_counts[0]), 100 * binary_counts[0] / binary_counts.sum(),
                int(binary_counts[1]), 100 * binary_counts[1] / binary_counts.sum())

    # Location class prior (defect nodes only, classes 1-18 -> 0-17)
    defect_labels_train = all_train_labels[all_train_labels > 0] - 1  # shift to 0-17
    location_counts = np.bincount(defect_labels_train, minlength=18).astype(np.float64)
    location_prior = torch.tensor(location_counts / location_counts.sum(), dtype=torch.float32)
    logger.info("Location class distribution (defect nodes only):")
    for i in range(18):
        if location_counts[i] > 0:
            logger.info("  Class %d (orig %d): %d nodes (%.1f%%)",
                        i, i + 1, int(location_counts[i]),
                        100 * location_counts[i] / location_counts.sum())

    # --- DataLoaders ---
    train_loader = PyGDataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = PyGDataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    ) if val_data else None
    test_loader = PyGDataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    ) if test_data else None

    # ==================================================================
    # STAGE 1: Binary defect detection
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STAGE 1: Binary defect detection (defect vs no-defect)")
    logger.info("=" * 70)

    binary_model = BinaryDefectModel(
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        dropout=args.dropout,
        input_channels=args.input_channels,
    ).to(device)
    num_params = sum(p.numel() for p in binary_model.parameters())
    logger.info("BinaryDefectModel: %d parameters", num_params)

    # Loss: FocalLogitAdjust for binary (handles 1:170 imbalance)
    stage1_loss = FocalLogitAdjustLoss(
        class_prior=binary_prior.to(device),
        tau=args.stage1_tau,
        gamma=args.stage1_gamma,
    )
    logger.info("Stage1 loss: FocalLogitAdjustLoss (tau=%.1f, gamma=%.1f)",
                args.stage1_tau, args.stage1_gamma)

    stage1_optimizer = torch.optim.Adam(
        binary_model.parameters(), lr=args.stage1_lr, weight_decay=5e-4,
    )

    # Scheduler
    total_steps_s1 = args.stage1_epochs * len(train_loader)
    stage1_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        stage1_optimizer, max_lr=args.stage1_lr * 2.0,
        total_steps=total_steps_s1, pct_start=0.1, anneal_strategy="cos",
    )

    # Training CSV
    s1_csv_path = os.path.join(result_dir, "stage1_training_log.csv")
    s1_csv_file = open(s1_csv_path, "w", newline="")
    s1_csv_writer = csv.writer(s1_csv_file)
    s1_csv_writer.writerow([
        "epoch", "train_loss", "val_loss", "accuracy", "defect_precision",
        "defect_recall", "defect_f1", "macro_f1", "lr",
    ])

    best_s1_f1 = 0.0
    patience_counter = 0
    best_s1_epoch = 0

    try:
        for epoch in range(args.stage1_epochs):
            t0 = time.time()
            train_loss = train_stage1_epoch(
                binary_model, train_loader, stage1_loss,
                stage1_optimizer, stage1_scheduler, device,
            )
            dt = time.time() - t0

            val_metrics = {}
            if val_loader is not None:
                val_metrics = validate_stage1(binary_model, val_loader, stage1_loss, device)

            current_lr = stage1_optimizer.param_groups[0]["lr"]
            macro_f1 = val_metrics.get("macro_f1", 0.0)

            s1_csv_writer.writerow([
                epoch, f"{train_loss:.6f}",
                f"{val_metrics.get('val_loss', 0):.6f}",
                f"{val_metrics.get('accuracy', 0):.6f}",
                f"{val_metrics.get('defect_precision', 0):.6f}",
                f"{val_metrics.get('defect_recall', 0):.6f}",
                f"{val_metrics.get('defect_f1', 0):.6f}",
                f"{macro_f1:.6f}",
                f"{current_lr:.8f}",
            ])
            s1_csv_file.flush()

            if epoch % 20 == 0 or epoch == args.stage1_epochs - 1:
                logger.info(
                    "[Stage1] Epoch %d/%d (%.1fs) | loss=%.4f | val_loss=%.4f "
                    "acc=%.4f prec=%.4f rec=%.4f f1=%.4f macro_f1=%.4f",
                    epoch, args.stage1_epochs, dt, train_loss,
                    val_metrics.get("val_loss", 0),
                    val_metrics.get("accuracy", 0),
                    val_metrics.get("defect_precision", 0),
                    val_metrics.get("defect_recall", 0),
                    val_metrics.get("defect_f1", 0),
                    macro_f1,
                )

            # Early stopping on macro_f1
            if macro_f1 > best_s1_f1:
                best_s1_f1 = macro_f1
                best_s1_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": binary_model.state_dict(),
                    "metrics": val_metrics,
                }, os.path.join(result_dir, "stage1_best.pth"))
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logger.info("[Stage1] Early stopping at epoch %d (best=%d, f1=%.4f)",
                            epoch, best_s1_epoch, best_s1_f1)
                break
    finally:
        s1_csv_file.close()

    # Load best Stage 1 model
    s1_ckpt = torch.load(os.path.join(result_dir, "stage1_best.pth"),
                         map_location=device, weights_only=False)
    binary_model.load_state_dict(s1_ckpt["model_state_dict"])
    logger.info("[Stage1] Best model from epoch %d (macro_f1=%.4f)", best_s1_epoch, best_s1_f1)

    # ==================================================================
    # STAGE 2: Location classification (defect nodes only)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STAGE 2: 18-class location classification (defect nodes only)")
    logger.info("=" * 70)

    location_model = LocationClassifierModel(
        hidden_channels=args.hidden_channels,
        num_location_classes=18,
        num_heads=args.num_heads,
        dropout=args.dropout,
        input_channels=args.input_channels,
    ).to(device)
    num_params = sum(p.numel() for p in location_model.parameters())
    logger.info("LocationClassifierModel: %d parameters", num_params)

    # Loss: LogitAdjust for 18-class (much more balanced than 19-class)
    stage2_loss = LogitAdjustLoss(
        class_prior=location_prior.to(device),
        tau=args.stage2_tau,
    )
    logger.info("Stage2 loss: LogitAdjustLoss (tau=%.1f)", args.stage2_tau)

    stage2_optimizer = torch.optim.Adam(
        location_model.parameters(), lr=args.stage2_lr, weight_decay=5e-4,
    )

    total_steps_s2 = args.stage2_epochs * len(train_loader)
    stage2_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        stage2_optimizer, max_lr=args.stage2_lr * 2.0,
        total_steps=total_steps_s2, pct_start=0.1, anneal_strategy="cos",
    )

    # Training CSV
    s2_csv_path = os.path.join(result_dir, "stage2_training_log.csv")
    s2_csv_file = open(s2_csv_path, "w", newline="")
    s2_csv_writer = csv.writer(s2_csv_file)
    s2_csv_writer.writerow([
        "epoch", "train_loss", "val_loss", "accuracy", "macro_f1",
        "macro_f1_support_only", "num_defect_nodes", "lr",
    ])

    best_s2_f1 = 0.0
    patience_counter = 0
    best_s2_epoch = 0

    try:
        for epoch in range(args.stage2_epochs):
            t0 = time.time()
            train_loss, num_defect = train_stage2_epoch(
                location_model, train_loader, stage2_loss,
                stage2_optimizer, stage2_scheduler, device,
            )
            dt = time.time() - t0

            val_metrics = {}
            if val_loader is not None:
                val_metrics = validate_stage2(location_model, val_loader, stage2_loss, device)

            current_lr = stage2_optimizer.param_groups[0]["lr"]
            macro_f1 = val_metrics.get("macro_f1", 0.0)

            s2_csv_writer.writerow([
                epoch, f"{train_loss:.6f}",
                f"{val_metrics.get('val_loss', 0):.6f}",
                f"{val_metrics.get('accuracy', 0):.6f}",
                f"{macro_f1:.6f}",
                f"{val_metrics.get('macro_f1_support_only', 0):.6f}",
                num_defect,
                f"{current_lr:.8f}",
            ])
            s2_csv_file.flush()

            if epoch % 20 == 0 or epoch == args.stage2_epochs - 1:
                logger.info(
                    "[Stage2] Epoch %d/%d (%.1fs) | loss=%.4f | val_loss=%.4f "
                    "acc=%.4f macro_f1=%.4f | defect_nodes=%d",
                    epoch, args.stage2_epochs, dt, train_loss,
                    val_metrics.get("val_loss", 0),
                    val_metrics.get("accuracy", 0),
                    macro_f1, num_defect,
                )

            if macro_f1 > best_s2_f1:
                best_s2_f1 = macro_f1
                best_s2_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": location_model.state_dict(),
                    "metrics": val_metrics,
                }, os.path.join(result_dir, "stage2_best.pth"))
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logger.info("[Stage2] Early stopping at epoch %d (best=%d, f1=%.4f)",
                            epoch, best_s2_epoch, best_s2_f1)
                break
    finally:
        s2_csv_file.close()

    # Load best Stage 2 model
    s2_ckpt = torch.load(os.path.join(result_dir, "stage2_best.pth"),
                         map_location=device, weights_only=False)
    location_model.load_state_dict(s2_ckpt["model_state_dict"])
    logger.info("[Stage2] Best model from epoch %d (macro_f1=%.4f)", best_s2_epoch, best_s2_f1)

    # ==================================================================
    # Combined pipeline evaluation
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EVALUATION: Combined two-stage pipeline on test set")
    logger.info("=" * 70)

    if test_loader is not None:
        # Two-stage pipeline evaluation
        twostage_results = evaluate_twostage_pipeline(
            binary_model, location_model, test_loader, device,
            binary_threshold=args.binary_threshold,
        )

        # Save confusion matrix
        save_confusion_matrix_plot(
            twostage_results["cm_19"],
            os.path.join(result_dir, "twostage_confusion_matrix_19class.png"),
            title="Two-Stage Pipeline: 19-Class Confusion Matrix",
        )

        # Classification report
        report = classification_report(
            twostage_results["true_labels"],
            twostage_results["pred_labels"],
            zero_division=0,
        )
        report_path = os.path.join(result_dir, "twostage_classification_report.txt")
        with open(report_path, "w") as f:
            f.write("Two-Stage Pipeline Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        logger.info("Two-stage classification report:\n%s", report)

        # Per-class F1 comparison
        f1_per_class = twostage_results["metrics_19"]["f1_per_class"]
        logger.info("Per-class F1 scores (19 classes):")
        for i in range(19):
            support = twostage_results["metrics_19"]["support_per_class"][i]
            if support > 0:
                logger.info("  Class %2d: F1=%.4f (support=%d)", i, f1_per_class[i], int(support))

        # Summary metrics
        bm = twostage_results["benchmark"]
        logger.info("Two-stage pipeline metrics:")
        logger.info("  Accuracy:     %.4f", bm["accuracy"])
        logger.info("  Macro F1:     %.4f", bm["macro_f1"])
        logger.info("  Weighted F1:  %.4f", bm["weighted_f1"])
        logger.info("  Balanced Acc: %.4f", bm["balanced_accuracy"])
        logger.info("  MCC:          %.4f", bm["mcc"])

        # Save metrics JSON
        metrics_out = {}
        for k, v in bm.items():
            if isinstance(v, (np.floating, np.integer)):
                metrics_out[k] = float(v)
            elif isinstance(v, np.ndarray):
                metrics_out[k] = v.tolist()
            else:
                metrics_out[k] = v

        metrics_out["stage1_best_epoch"] = best_s1_epoch
        metrics_out["stage1_best_macro_f1"] = best_s1_f1
        metrics_out["stage2_best_epoch"] = best_s2_epoch
        metrics_out["stage2_best_macro_f1"] = best_s2_f1
        metrics_out["binary_threshold"] = args.binary_threshold
        metrics_out["f1_per_class"] = f1_per_class.tolist()

        metrics_path = os.path.join(result_dir, "twostage_test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2)

        # Save args for reproducibility
        args_path = os.path.join(result_dir, "args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2)

        logger.info("All results saved to %s", result_dir)

    else:
        logger.warning("No test data available for evaluation.")


# ============================================================================
# Argparse
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage GNN training: binary detection + location classification"
    )
    # Model architecture (shared between stages)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--input_channels", type=int, default=4,
                        help="Number of input features per node (4=FEM)")

    # Stage 1: Binary detection
    parser.add_argument("--stage1_epochs", type=int, default=300,
                        help="Number of epochs for Stage 1 (binary)")
    parser.add_argument("--stage1_lr", type=float, default=0.002,
                        help="Learning rate for Stage 1")
    parser.add_argument("--stage1_tau", type=float, default=2.0,
                        help="Logit adjustment tau for Stage 1")
    parser.add_argument("--stage1_gamma", type=float, default=2.0,
                        help="Focal loss gamma for Stage 1")

    # Stage 2: Location classification
    parser.add_argument("--stage2_epochs", type=int, default=200,
                        help="Number of epochs for Stage 2 (18-class)")
    parser.add_argument("--stage2_lr", type=float, default=0.001,
                        help="Learning rate for Stage 2")
    parser.add_argument("--stage2_tau", type=float, default=1.0,
                        help="Logit adjustment tau for Stage 2 (lower ok, more balanced)")

    # Inference
    parser.add_argument("--binary_threshold", type=float, default=0.5,
                        help="Probability threshold for defect detection in Stage 1")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=80,
                        help="Early stopping patience (per stage)")

    # Data
    parser.add_argument("--group_key", type=str, default="LBel")
    parser.add_argument("--data_usage_ratio", type=float, default=1.0,
                        help="Fraction of data to use (for quick testing)")
    parser.add_argument("--split_manifest", type=str, default=None,
                        help="Path to JSON split manifest for reproducibility")

    # Output
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
