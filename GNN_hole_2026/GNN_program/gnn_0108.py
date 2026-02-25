#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

# =============================================================================
# 0. Utils
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(local_rank: int):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

# =============================================================================
# 1. Config
# =============================================================================
CONFIG = {
    "TRAIN_DIR": "/home/nishioka/GNN/GNN_hole_2026/Defect_hole_8x8_original_derived_1d_norm_clip_final",
    "LABEL_DIR": "/home/nishioka/GNN/GNN_hole/GNN_19class/Def88_19class_label",
    "COORDS_X": "/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy",
    "COORDS_Y": "/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy",
    "COORDS_Z": "/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_z_2layer.npy",
    "EDGE_INDEX": "/home/nishioka/GNN/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy",
    "BATCH_SIZE": 32,
    "LR": 3e-4,
    "EPOCHS": 50,
    "VERTICES_PER_LAYER": 6971,
    "NUM_CLASSES_HEAD": 10,
    "SEED": 42,
    "IGNORE_INDEX": 255,  # for invalid labels
}

# =============================================================================
# 2. Dataset
# =============================================================================
class CFRPDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

        coords = np.stack(
            [
                np.load(CONFIG["COORDS_X"]),
                np.load(CONFIG["COORDS_Y"]),
                np.load(CONFIG["COORDS_Z"]),
            ],
            axis=1,
        ).astype(np.float32)  # (N, 3)
        self.coords = coords

        # edge_index saved as (E,2) or (2,E). We assume original file is (E,2) then transpose.
        ei = np.load(CONFIG["EDGE_INDEX"])
        if ei.shape[0] == 2:
            edge_index = ei
        else:
            edge_index = ei.T
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        x_val = np.load(os.path.join(self.data_dir, fname)).astype(np.float32)  # (N,)

        # label file: data.npy -> data_19label.npy
        label_fname = fname.replace(".npy", "_19label.npy")
        y_val = np.load(os.path.join(self.label_dir, label_fname)).astype(np.float32)  # (N,19) one-hot

        node_features = np.hstack([self.coords, x_val[:, None]]).astype(np.float32)  # (N,4)
        labels = np.argmax(y_val, axis=1).astype(np.int64)  # 0..18

        data = Data(
            x=torch.from_numpy(node_features),
            y=torch.from_numpy(labels),
            edge_index=self.edge_index,
        )
        return data

# =============================================================================
# 3. Model
# =============================================================================
class DualHeadGATv2(nn.Module):
    def __init__(self):
        super().__init__()
        h, heads = 32, 4
        self.conv1 = GATv2Conv(4, h, heads=heads)
        self.bn1 = nn.BatchNorm1d(h * heads)

        self.conv2 = GATv2Conv(h * heads, h * 2, heads=heads)
        self.bn2 = nn.BatchNorm1d(h * 2 * heads)

        out_dim = CONFIG["NUM_CLASSES_HEAD"]
        self.upper_head = nn.Linear(h * 2 * heads, out_dim)
        self.lower_head = nn.Linear(h * 2 * heads, out_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))

        # IMPORTANT: we return full-node logits; caller will mask by layer
        up_logits = self.upper_head(x)   # (total_nodes, 10)
        low_logits = self.lower_head(x)  # (total_nodes, 10)
        return up_logits, low_logits

# =============================================================================
# 4. Focal Loss with ignore_index
# =============================================================================
class SimpleFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)

    def forward(self, logits, targets):
        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return logits.sum() * 0.0  # safe zero (keeps graph)
        logits = logits[valid]
        targets = targets[valid]

        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1.0 - pt) ** self.gamma * ce).mean()

# =============================================================================
# 5. Label remap (fix for nll_loss assert)
# =============================================================================
def split_and_remap_targets(batch: Data, device: torch.device):
    """
    Raw labels: 0..18 (19 classes)
    Two heads each output 10 classes:
      - lower head expects 0..9 (layers/classes 0..9)
      - upper head expects 0..9 where:
            raw 0    -> 0
            raw 10..18 -> 1..9  (raw-9)
        raw 1..9 are invalid for upper head (ignored)
    Also: if lower contains raw >=10, ignore.
    """
    v = CONFIG["VERTICES_PER_LAYER"]
    IGNORE = CONFIG["IGNORE_INDEX"]

    # node index within each graph in batch
    node_idx = torch.arange(batch.y.size(0), device=device) - batch.ptr[batch.batch]
    mask_lower = node_idx < v
    mask_upper = ~mask_lower

    low_y_raw = batch.y[mask_lower]
    up_y_raw = batch.y[mask_upper]

    # lower: keep 0..9, else ignore
    low_y = torch.where(
        low_y_raw < 10,
        low_y_raw,
        torch.full_like(low_y_raw, IGNORE),
    )

    # upper: raw==0 -> 0, raw>=10 -> raw-9 (=> 1..9), raw 1..9 -> ignore
    up_tmp = torch.where(up_y_raw == 0, torch.zeros_like(up_y_raw), up_y_raw - 9)
    up_y = torch.where(
        (up_y_raw == 0) | (up_y_raw >= 10),
        up_tmp,
        torch.full_like(up_y_raw, IGNORE),
    )

    return mask_upper, mask_lower, up_y_raw, low_y_raw, up_y, low_y

# =============================================================================
# 6. Train (single GPU)
# =============================================================================
def train_single(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Single] device={device}")

    train_dataset = CFRPDataset(CONFIG["TRAIN_DIR"], CONFIG["LABEL_DIR"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = DualHeadGATv2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = SimpleFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=CONFIG["IGNORE_INDEX"]).to(device)

    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            up_logits, low_logits = model(batch)

            mask_upper, mask_lower, up_y_raw, low_y_raw, up_y, low_y = split_and_remap_targets(batch, device)

            up_out = up_logits[mask_upper]
            low_out = low_logits[mask_lower]

            loss = criterion(low_out, low_y) + criterion(up_out, up_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            if args.debug_first_batch and epoch == 1 and step == 0:
                print("low_y_raw min/max:", int(low_y_raw.min()), int(low_y_raw.max()))
                print("up_y_raw  min/max:", int(up_y_raw.min()), int(up_y_raw.max()))
                print("low_y     min/max:", int(low_y[low_y != CONFIG['IGNORE_INDEX']].min() if (low_y != CONFIG['IGNORE_INDEX']).any() else -1),
                      int(low_y[low_y != CONFIG['IGNORE_INDEX']].max() if (low_y != CONFIG['IGNORE_INDEX']).any() else -1))
                print("up_y      min/max:", int(up_y[up_y != CONFIG['IGNORE_INDEX']].min() if (up_y != CONFIG['IGNORE_INDEX']).any() else -1),
                      int(up_y[up_y != CONFIG['IGNORE_INDEX']].max() if (up_y != CONFIG['IGNORE_INDEX']).any() else -1))
                print("mask_lower:", int(mask_lower.sum()), "mask_upper:", int(mask_upper.sum()))

        avg_loss = total_loss / max(1, len(train_loader))
        history.append(avg_loss)
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f}")

    # save
    ckpt = {
        "model": model.state_dict(),
        "config": CONFIG,
        "args": vars(args),
    }
    torch.save(ckpt, args.ckpt_path)
    print(f"Saved checkpoint: {args.ckpt_path}")

# =============================================================================
# 7. Predict (single GPU) - show a few node predictions
# =============================================================================
@torch.no_grad()
def predict_single(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] device={device}")

    dataset = CFRPDataset(CONFIG["TRAIN_DIR"], CONFIG["LABEL_DIR"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = DualHeadGATv2().to(device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    batch = next(iter(loader)).to(device)
    up_logits, low_logits = model(batch)

    v = CONFIG["VERTICES_PER_LAYER"]
    node_idx = torch.arange(batch.y.size(0), device=device) - batch.ptr[batch.batch]
    mask_lower = node_idx < v
    mask_upper = ~mask_lower

    # predicted class per head (0..9)
    pred_low = low_logits[mask_lower].argmax(dim=1).cpu().numpy()
    pred_up  = up_logits[mask_upper].argmax(dim=1).cpu().numpy()

    # map back to 19-class labels for display
    # lower: 0..9 -> 0..9
    pred_low_19 = pred_low
    # upper: 0->0, 1..9 -> 10..18
    pred_up_19 = np.where(pred_up == 0, 0, pred_up + 9)

    y_true = batch.y.cpu().numpy()
    true_low = y_true[mask_lower.cpu().numpy()]
    true_up  = y_true[mask_upper.cpu().numpy()]

    print("\n--- LOWER (first 30 nodes) true vs pred (19-class) ---")
    for i in range(min(30, len(true_low))):
        print(f"{i:04d}: true={int(true_low[i])}  pred={int(pred_low_19[i])}")

    print("\n--- UPPER (first 30 nodes) true vs pred (19-class) ---")
    for i in range(min(30, len(true_up))):
        print(f"{i:04d}: true={int(true_up[i])}  pred={int(pred_up_19[i])}")

    # counts
    from collections import Counter
    c_low = Counter(pred_low_19.tolist())
    c_up  = Counter(pred_up_19.tolist())
    print("\nPred label counts (lower):", dict(sorted(c_low.items())))
    print("Pred label counts (upper):", dict(sorted(c_up.items())))

# =============================================================================
# 8. Main
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=["train", "predict"])
    p.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"])
    p.add_argument("--lr", type=float, default=CONFIG["LR"])
    p.add_argument("--epochs", type=int, default=CONFIG["EPOCHS"])
    p.add_argument("--focal_alpha", type=float, default=0.5)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--ckpt_path", type=str, default="./model_single.pt")
    p.add_argument("--seed", type=int, default=CONFIG["SEED"])
    p.add_argument("--debug_first_batch", action="store_true")
    return p

def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    if args.mode == "train":
        train_single(args)
    else:
        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
        predict_single(args)

if __name__ == "__main__":
    main()
