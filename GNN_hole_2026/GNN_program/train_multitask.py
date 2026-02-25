"""Multi-task GNN training script: location (node-level) + size (graph-level) classification.

Usage:
    torchrun --nproc_per_node=1 GNN_hole_2026/GNN_program/train_multitask.py \
        --hidden_channels 64 --epochs 500 --size_weight 0.5
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gnn_common.models import GATMultiTaskModel
from gnn_common.losses import LogitAdjustLoss, FocalLossLogSoftmax, MultiTaskLoss
from gnn_common.data_utils import (
    prepare_data,
    extract_size_class,
    compute_class_weights,
    create_data_label_pairs,
    group_disjoint_split,
)
from gnn_common.metrics import metrics_from_confusion_matrix, compute_benchmark_metrics
from gnn_common.training_utils import set_seed, setup, cleanup

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
# Default data paths (same as GNN_zscore_sub_noise_defect_free.py)
# ---------------------------------------------------------------------------
_DEFAULT_DATA_BASE = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise"
_DEFAULT_LABEL_DIR = "/home/nishioka/GNN/GNN_hole_2026/all_19class_label"
_DEFAULT_COORD_DIR = "/home/nishioka/GNN/GNN_hole/GNN_hole_data"
_MAX_NODES = 13942


# ============================================================================
# Loss builder
# ============================================================================
def build_location_loss_fn(args, class_prior: torch.Tensor, device: torch.device):
    """Build the location classification loss function."""
    if args.use_logit_adjust:
        base_loss = LogitAdjustLoss(
            class_prior=class_prior.to(device),
            tau=args.logit_adjust_tau,
        )
        logger.info("Location loss: LogitAdjustLoss (tau=%.2f)", args.logit_adjust_tau)
    else:
        weights = compute_class_weights(
            np.zeros(1, dtype=np.int64),  # dummy — we already have class_prior
        )
        base_loss = FocalLossLogSoftmax(weights=weights.to(device), gamma=3.0)
        logger.info("Location loss: FocalLossLogSoftmax (gamma=3.0)")
    return base_loss


# ============================================================================
# Train / validate / evaluate
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch,
                    use_amp=False, grad_clip=1.0):
    """Train for one epoch. Returns dict of average losses."""
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_loss = 0.0
    total_loc_loss = 0.0
    total_size_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        # Ensure size_class is a tensor
        size_labels = batch.size_class
        if not isinstance(size_labels, torch.Tensor):
            size_labels = torch.tensor(size_labels, dtype=torch.long, device=device)
        else:
            size_labels = size_labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(batch, multitask=True)
            loss, loss_dict = criterion(outputs, batch.y, size_labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss_dict["total_loss"].item()
        total_loc_loss += loss_dict["location_loss"].item()
        total_size_loss += (
            loss_dict["size_loss"].item()
            if isinstance(loss_dict["size_loss"], torch.Tensor)
            else loss_dict["size_loss"]
        )
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "total_loss": total_loss / n,
        "location_loss": total_loc_loss / n,
        "size_loss": total_size_loss / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=19, num_size_classes=3,
             world_size=1):
    """Validate and return metrics + confusion matrices."""
    model.eval()

    loc_cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    size_cm = torch.zeros(num_size_classes, num_size_classes, dtype=torch.long, device=device)
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        size_labels = batch.size_class
        if not isinstance(size_labels, torch.Tensor):
            size_labels = torch.tensor(size_labels, dtype=torch.long, device=device)
        else:
            size_labels = size_labels.to(device)

        outputs = model(batch, multitask=True)
        loss, _ = criterion(outputs, batch.y, size_labels)
        total_loss += loss.item()
        num_batches += 1

        # Location CM
        loc_preds = outputs["location"].argmax(dim=1)
        for t, p in zip(batch.y.view(-1), loc_preds.view(-1)):
            loc_cm[t, p] += 1

        # Size CM (mask out -1)
        size_mask = size_labels >= 0
        if size_mask.any():
            size_preds = outputs["size"].argmax(dim=1)
            for t, p in zip(size_labels[size_mask], size_preds[size_mask]):
                size_cm[t, p] += 1

    # DDP all-reduce
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(loc_cm, op=dist.ReduceOp.SUM)
        dist.all_reduce(size_cm, op=dist.ReduceOp.SUM)

    loc_cm_np = loc_cm.cpu().numpy()
    size_cm_np = size_cm.cpu().numpy()

    loc_metrics = metrics_from_confusion_matrix(loc_cm_np)
    size_total = size_cm_np.sum()
    size_acc = float(np.diag(size_cm_np).sum()) / max(size_total, 1)

    avg_loss = total_loss / max(num_batches, 1)

    return {
        "val_loss": avg_loss,
        "macro_f1": loc_metrics["macro_f1"],
        "macro_f1_support_only": loc_metrics["macro_f1_support_only"],
        "accuracy": loc_metrics["accuracy"],
        "weighted_f1": loc_metrics["weighted_f1"],
        "size_accuracy": size_acc,
        "loc_cm": loc_cm_np,
        "size_cm": size_cm_np,
    }


@torch.no_grad()
def evaluate_test(model, loader, criterion, device, num_classes=19, num_size_classes=3,
                  world_size=1):
    """Full test evaluation returning predictions and metrics."""
    model.eval()

    all_filenames = []
    all_true_labels = []
    all_pred_labels = []
    all_true_size = []
    all_pred_size = []
    all_probs = []

    loc_cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    size_cm = torch.zeros(num_size_classes, num_size_classes, dtype=torch.long, device=device)

    for batch in loader:
        batch = batch.to(device)
        size_labels = batch.size_class
        if not isinstance(size_labels, torch.Tensor):
            size_labels = torch.tensor(size_labels, dtype=torch.long, device=device)
        else:
            size_labels = size_labels.to(device)

        outputs = model(batch, multitask=True)

        loc_logits = outputs["location"]
        loc_preds = loc_logits.argmax(dim=1)
        loc_probs = F.softmax(loc_logits, dim=1)

        for t, p in zip(batch.y.view(-1), loc_preds.view(-1)):
            loc_cm[t, p] += 1

        all_true_labels.extend(batch.y.cpu().numpy().tolist())
        all_pred_labels.extend(loc_preds.cpu().numpy().tolist())
        all_probs.append(loc_probs.cpu().numpy())

        # Size predictions per graph
        size_preds = outputs["size"].argmax(dim=1)
        size_mask = size_labels >= 0
        if size_mask.any():
            for t, p in zip(size_labels[size_mask], size_preds[size_mask]):
                size_cm[t, p] += 1

        # Collect per-graph info
        batch_ids = batch.batch if hasattr(batch, "batch") else torch.zeros(batch.y.size(0), dtype=torch.long, device=device)
        num_graphs = int(batch_ids.max().item()) + 1
        for g_idx in range(num_graphs):
            fn = ""
            # Get filename from the Data objects if available
            if hasattr(batch, "_data_list") and batch._data_list is not None:
                if g_idx < len(batch._data_list) and hasattr(batch._data_list[g_idx], "filename"):
                    fn = batch._data_list[g_idx].filename
            elif hasattr(batch, "filename"):
                if isinstance(batch.filename, (list, tuple)):
                    fn = batch.filename[g_idx] if g_idx < len(batch.filename) else ""
                else:
                    fn = str(batch.filename)
            all_filenames.append(fn)
            all_true_size.append(size_labels[g_idx].item())
            all_pred_size.append(size_preds[g_idx].item())

    # DDP all-reduce for confusion matrices
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(loc_cm, op=dist.ReduceOp.SUM)
        dist.all_reduce(size_cm, op=dist.ReduceOp.SUM)

    loc_cm_np = loc_cm.cpu().numpy()
    size_cm_np = size_cm.cpu().numpy()

    all_probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, num_classes))

    # Size arrays for benchmark metrics (filter -1)
    size_labels_arr = np.array(all_true_size)
    size_preds_arr = np.array(all_pred_size)
    valid_mask = size_labels_arr >= 0
    sl_valid = size_labels_arr[valid_mask] if valid_mask.any() else None
    sp_valid = size_preds_arr[valid_mask] if valid_mask.any() else None

    benchmark = compute_benchmark_metrics(
        all_labels=np.array(all_true_labels),
        all_preds=np.array(all_pred_labels),
        num_classes=num_classes,
        all_probs=all_probs_np if len(all_probs_np) > 0 else None,
        size_labels=sl_valid,
        size_preds=sp_valid,
    )

    return {
        "loc_cm": loc_cm_np,
        "size_cm": size_cm_np,
        "benchmark": benchmark,
        "filenames": all_filenames,
        "true_labels": all_true_labels,
        "pred_labels": all_pred_labels,
        "true_size": all_true_size,
        "pred_size": all_pred_size,
    }


# ============================================================================
# Checkpoint helpers
# ============================================================================
def save_checkpoint(path, model, optimizer, scheduler, epoch, args, metrics=None):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, path)


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
    # --- DDP setup ---
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        setup(rank, world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    set_seed(42)

    is_main = rank == 0

    # --- Output directory ---
    output_root = args.output_root or "/home/nishioka/GNN/GNN_hole_2026"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{timestamp}"
    if args.run_id:
        run_tag = f"{args.run_id}_{timestamp}"
    result_dir = os.path.join(output_root, "multitask_results", run_tag)
    if is_main:
        os.makedirs(result_dir, exist_ok=True)
        logger.info("Results will be saved to %s", result_dir)

    # --- Load coordinates & edges ---
    coord_dir = _DEFAULT_COORD_DIR
    x_coords = np.load(os.path.join(coord_dir, "normalized_x_2layer.npy"))
    y_coords = np.load(os.path.join(coord_dir, "normalized_y_2layer.npy"))
    z_coords = np.load(os.path.join(coord_dir, "normalized_z_2layer.npy"))
    edges = np.load(os.path.join(coord_dir, "hole_edges_2layer_best.npy"))
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # --- Discover data files ---
    data_base = _DEFAULT_DATA_BASE
    label_dir = _DEFAULT_LABEL_DIR

    # Collect noise (defect-free) + defect files
    noise_folder = os.path.join(data_base, "train")
    noise_files = sorted([
        f for f in os.listdir(noise_folder)
        if f.startswith("NoiseDefectFree_") and f.endswith(".npy")
    ]) if os.path.isdir(noise_folder) else []

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

    # Label files
    label_files = [
        f for f in os.listdir(label_dir)
        if (f.startswith("Defect_L") or f.startswith("NoiseDefectFree_")) and f.endswith("_19label.npy")
    ]

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

    if is_main:
        logger.info("Total samples: %d (noise=%d, defect=%d)",
                     len(all_pairs), len(noise_files), len(all_pairs) - len(noise_files))

    # --- Split ---
    split_manifest_path = args.split_manifest
    if split_manifest_path is not None:
        with open(split_manifest_path) as f:
            manifest = json.load(f)
        if is_main:
            logger.info("Using split manifest: %s", split_manifest_path)
        train_pairs = [tuple(p[:2]) for p in manifest["train"]]
        val_pairs = [tuple(p[:2]) for p in manifest["val"]]
        test_pairs = [tuple(p[:2]) for p in manifest["test"]]
        train_map = {p[0]: p[2] for p in manifest["train"]}
        val_map = {p[0]: p[2] for p in manifest["val"]}
        test_map = {p[0]: p[2] for p in manifest["test"]}
    else:
        train_pairs, val_pairs, test_pairs, train_map, val_map, test_map = group_disjoint_split(
            all_pairs,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            group_key=args.group_key,
            seed=42,
        )

    # Apply data_usage_ratio
    if args.data_usage_ratio < 1.0:
        r = args.data_usage_ratio
        train_pairs = train_pairs[:max(1, int(len(train_pairs) * r))]
        val_pairs = val_pairs[:max(1, int(len(val_pairs) * r))]
        test_pairs = test_pairs[:max(1, int(len(test_pairs) * r))]
        if is_main:
            logger.info("data_usage_ratio=%.2f => train=%d, val=%d, test=%d",
                         r, len(train_pairs), len(val_pairs), len(test_pairs))

    if is_main:
        logger.info("Split: train=%d, val=%d, test=%d",
                     len(train_pairs), len(val_pairs), len(test_pairs))

    # --- Prepare datasets ---
    train_data, class_weights = prepare_data(
        train_pairs, None, label_dir,
        x_coords, y_coords, z_coords, edge_index,
        data_folder_map=train_map,
        max_nodes=_MAX_NODES,
        return_class_weights=True,
    )
    val_data = prepare_data(
        val_pairs, None, label_dir,
        x_coords, y_coords, z_coords, edge_index,
        data_folder_map=val_map,
        max_nodes=_MAX_NODES,
        return_class_weights=False,
    )
    test_data = prepare_data(
        test_pairs, None, label_dir,
        x_coords, y_coords, z_coords, edge_index,
        data_folder_map=test_map,
        max_nodes=_MAX_NODES,
        return_class_weights=False,
    )

    if train_data is None:
        logger.error("No training data loaded. Exiting.")
        return

    # Convert size_class from int to tensor
    for data_list in [train_data, val_data, test_data]:
        if data_list is None:
            continue
        for d in data_list:
            sc = d.size_class if hasattr(d, "size_class") else -1
            d.size_class = torch.tensor(sc, dtype=torch.long) if not isinstance(sc, torch.Tensor) else sc

    if is_main:
        logger.info("Datasets loaded: train=%d, val=%d, test=%d",
                     len(train_data), len(val_data) if val_data else 0,
                     len(test_data) if test_data else 0)

    # --- DataLoaders ---
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, num_replicas=world_size, rank=rank, shuffle=False
        ) if val_data else None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = PyGDataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=2, pin_memory=True,
    )
    val_loader = PyGDataLoader(
        val_data, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=2, pin_memory=True,
    ) if val_data else None
    test_loader = PyGDataLoader(
        test_data, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    ) if test_data else None

    # --- Model ---
    model = GATMultiTaskModel(
        hidden_channels=args.hidden_channels,
        num_classes=19,
        num_size_classes=3,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info("GATMultiTaskModel: %d parameters", num_params)

    # --- Loss ---
    # Compute class prior from training labels
    all_train_labels = []
    for d in train_data:
        all_train_labels.extend(d.y.tolist())
    all_train_labels = np.array(all_train_labels)
    class_counts = np.bincount(all_train_labels, minlength=19).astype(np.float64)
    class_prior = torch.tensor(class_counts / class_counts.sum(), dtype=torch.float32)

    location_loss_fn = build_location_loss_fn(args, class_prior, device)
    criterion = MultiTaskLoss(
        location_loss_fn=location_loss_fn,
        size_weight=args.size_weight,
    ).to(device)

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=5e-4,
    )

    scheduler = None
    if args.use_onecycle:
        max_lr = args.learning_rate * 2.0
        total_steps = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=0.1, anneal_strategy="cos",
        )
        if is_main:
            logger.info("Scheduler: OneCycleLR (max_lr=%.4f, total_steps=%d)", max_lr, total_steps)

    # --- Resume ---
    start_epoch = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main:
            logger.info("Resumed from %s (epoch %d)", args.resume_from, start_epoch)

    # --- Training log CSV ---
    csv_path = os.path.join(result_dir, "training_log.csv") if is_main else None
    csv_file = None
    csv_writer = None
    if is_main:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "epoch", "train_loss", "train_loc_loss", "train_size_loss",
            "val_loss", "macro_f1", "macro_f1_support_only",
            "size_accuracy", "lr",
        ])

    # --- Training loop ---
    best_macro_f1 = 0.0
    patience_counter = 0
    best_epoch = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            t0 = time.time()
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scheduler if args.use_onecycle else None,
                device, epoch, use_amp=args.use_amp,
            )
            dt_train = time.time() - t0

            # Non-OneCycle scheduler step would go here if needed

            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = validate(
                    model, val_loader, criterion, device,
                    num_classes=19, num_size_classes=3,
                    world_size=world_size,
                )

            current_lr = optimizer.param_groups[0]["lr"]
            macro_f1 = val_metrics.get("macro_f1", 0.0)
            macro_f1_so = val_metrics.get("macro_f1_support_only", 0.0)
            size_acc = val_metrics.get("size_accuracy", 0.0)
            val_loss = val_metrics.get("val_loss", 0.0)

            if is_main:
                csv_writer.writerow([
                    epoch,
                    f"{train_metrics['total_loss']:.6f}",
                    f"{train_metrics['location_loss']:.6f}",
                    f"{train_metrics['size_loss']:.6f}",
                    f"{val_loss:.6f}",
                    f"{macro_f1:.6f}",
                    f"{macro_f1_so:.6f}",
                    f"{size_acc:.4f}",
                    f"{current_lr:.8f}",
                ])
                csv_file.flush()

                if epoch % 10 == 0 or epoch == args.epochs - 1:
                    logger.info(
                        "Epoch %d/%d (%.1fs) | loss=%.4f loc=%.4f size=%.4f | "
                        "val_loss=%.4f macro_f1=%.4f size_acc=%.4f | lr=%.6f",
                        epoch, args.epochs, dt_train,
                        train_metrics["total_loss"],
                        train_metrics["location_loss"],
                        train_metrics["size_loss"],
                        val_loss, macro_f1, size_acc, current_lr,
                    )

            # Early stopping on macro_f1 (location)
            improved = macro_f1 > best_macro_f1
            if improved:
                best_macro_f1 = macro_f1
                best_epoch = epoch
                patience_counter = 0
                if is_main:
                    save_checkpoint(
                        os.path.join(result_dir, "best_model.pth"),
                        model, optimizer, scheduler, epoch, args,
                        metrics={"macro_f1": macro_f1, "size_accuracy": size_acc},
                    )
            else:
                patience_counter += 1

            # Periodic checkpoint
            if is_main and epoch % 100 == 0 and epoch > 0:
                save_checkpoint(
                    os.path.join(result_dir, "latest_model.pth"),
                    model, optimizer, scheduler, epoch, args,
                )

            # Broadcast stop flag
            stop_flag = torch.tensor(
                [1 if patience_counter >= args.patience else 0],
                dtype=torch.long, device=device,
            )
            if world_size > 1:
                dist.broadcast(stop_flag, src=0)

            if stop_flag.item() == 1:
                if is_main:
                    logger.info("Early stopping at epoch %d (best=%d, macro_f1=%.4f)",
                                epoch, best_epoch, best_macro_f1)
                break

    finally:
        if csv_file is not None:
            csv_file.close()

    # --- Save latest model ---
    if is_main:
        save_checkpoint(
            os.path.join(result_dir, "latest_model.pth"),
            model, optimizer, scheduler, epoch if 'epoch' in dir() else 0, args,
        )

    # --- Test evaluation (rank 0 only for output) ---
    if is_main and test_loader is not None:
        logger.info("Running test evaluation...")

        # Load best model
        best_path = os.path.join(result_dir, "best_model.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            raw_model = model.module if isinstance(model, DDP) else model
            raw_model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded best model from epoch %d", ckpt.get("epoch", -1))

        test_results = evaluate_test(
            model, test_loader, criterion, device,
            num_classes=19, num_size_classes=3,
            world_size=1,  # single process for test eval
        )

        # Save confusion matrices
        save_confusion_matrix_plot(
            test_results["loc_cm"],
            os.path.join(result_dir, "test_location_confusion_matrix.png"),
            title="Test Location Confusion Matrix (19 classes)",
        )
        save_confusion_matrix_plot(
            test_results["size_cm"],
            os.path.join(result_dir, "test_size_confusion_matrix.png"),
            title="Test Size Confusion Matrix (3 classes)",
        )

        # Classification report
        from sklearn.metrics import classification_report
        loc_report = classification_report(
            test_results["true_labels"], test_results["pred_labels"],
            zero_division=0,
        )
        report_path = os.path.join(result_dir, "test_location_report.txt")
        with open(report_path, "w") as f:
            f.write(loc_report)
        logger.info("Location classification report:\n%s", loc_report)

        # Predictions CSV
        pred_csv_path = os.path.join(result_dir, "test_predictions.csv")
        with open(pred_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_label", "pred_label", "true_size_class", "pred_size_class"])
            # Write per-graph rows (node-level labels aggregated to graph)
            for i, fn in enumerate(test_results["filenames"]):
                w.writerow([
                    fn,
                    "",  # node-level true labels are in loc_cm
                    "",  # node-level pred labels are in loc_cm
                    test_results["true_size"][i],
                    test_results["pred_size"][i],
                ])

        # Metrics JSON
        metrics_path = os.path.join(result_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            # Convert numpy types for JSON serialization
            bm = {}
            for k, v in test_results["benchmark"].items():
                if isinstance(v, (np.floating, np.integer)):
                    bm[k] = float(v)
                elif isinstance(v, np.ndarray):
                    bm[k] = v.tolist()
                else:
                    bm[k] = v
            bm["best_epoch"] = best_epoch
            bm["best_val_macro_f1"] = best_macro_f1
            json.dump(bm, f, indent=2)

        logger.info("Test metrics: %s", {k: f"{v:.4f}" if isinstance(v, float) else v
                                          for k, v in test_results["benchmark"].items()})
        logger.info("All results saved to %s", result_dir)

    # --- Cleanup ---
    if world_size > 1:
        cleanup(device=device, rank=rank)


# ============================================================================
# Argparse
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-task GNN training: location + size classification"
    )
    # Model
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=4)

    # Training
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)

    # Multi-task
    parser.add_argument("--size_weight", type=float, default=0.5,
                        help="Weight for size classification loss")

    # Loss
    parser.add_argument("--use_logit_adjust", action="store_true", default=True)
    parser.add_argument("--no_logit_adjust", dest="use_logit_adjust", action="store_false")
    parser.add_argument("--logit_adjust_tau", type=float, default=1.0)

    # Scheduler
    parser.add_argument("--use_onecycle", action="store_true", default=True)
    parser.add_argument("--no_onecycle", dest="use_onecycle", action="store_false")

    # AMP
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")

    # Data
    parser.add_argument("--group_key", type=str, default="LBel")
    parser.add_argument("--data_usage_ratio", type=float, default=1.0)

    # Output
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--split_manifest", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
