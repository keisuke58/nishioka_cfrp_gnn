#!/usr/bin/env python3
"""
M4-2: バッチ推論 → CSV出力

フォルダ内の全 .npy に対して推論し、filename, pred_class_counts, has_defect 等をCSVに出力。

Usage:
    python tools/batch_predict.py --input GNN_hole_2026/all_sub_hole_defect_zscore_noise/test --output reports/predictions.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "GNN_hole_2026/GNN_program"))

HOLE_DATA = REPO_ROOT / "GNN_hole/GNN_hole_data"
MAX_NODES = 13942


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input folder with .npy files")
    parser.add_argument("--output", type=str, default="reports/predictions.csv")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    from GNN_zscore_sub_noise_defect_free import GATModel

    model_path = args.model or str(
        REPO_ROOT / "runs/20260116_104929_nogit_dsNDF_ep2000_lr0p001_F10p730/outputs/GNN_model/19classmodel_hole_zscore/GATModel_20260116_104950_Best_Final.pth"
    )
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.load(HOLE_DATA / "normalized_x_2layer.npy")[:MAX_NODES]
    y = np.load(HOLE_DATA / "normalized_y_2layer.npy")[:MAX_NODES]
    z = np.load(HOLE_DATA / "normalized_z_2layer.npy")[:MAX_NODES]
    ei = np.load(HOLE_DATA / "hole_edges_2layer_best.npy")
    if ei.shape[0] != 2:
        ei = ei.T
    edge_index = torch.tensor(ei, dtype=torch.long)

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    hidden = 16
    model = GATModel(hidden_channels=hidden, num_classes=19, dropout=0.0, edge_drop_prob=0.0)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    input_path = Path(args.input)
    files = sorted(input_path.glob("*.npy"))[: args.max_files or 999999]
    rows = []
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    for i in range(0, len(files), args.batch_size):
        batch_files = files[i : i + args.batch_size]
        data_list = []
        for fp in batch_files:
            vals = np.load(fp)[:MAX_NODES]
            feat = np.vstack((x, y, z, vals)).T
            data_list.append(Data(x=torch.tensor(feat, dtype=torch.float), edge_index=edge_index, y=torch.zeros(MAX_NODES, dtype=torch.long)))
        loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
        batch = next(iter(loader)).to(device)
        with torch.no_grad():
            out = model(batch)
        pred = out.argmax(dim=1).cpu().numpy()
        for j, fp in enumerate(batch_files):
            start = j * MAX_NODES
            end = (j + 1) * MAX_NODES
            p = pred[start:end]
            n_defect = int((p > 0).sum())
            rows.append({
                "filename": fp.name,
                "defect_nodes": n_defect,
                "has_defect": n_defect > 0,
                "pred_classes": ",".join(map(str, np.unique(p))),
            })
    df = pd.DataFrame(rows)
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
