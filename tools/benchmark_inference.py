#!/usr/bin/env python3
"""
M0-2: 推論速度の計測（CPU/GPU）

保存済みモデルで1サンプルあたりの推論時間を計測し、inference_benchmark.md を生成する。

Usage:
    python tools/benchmark_inference.py
    python tools/benchmark_inference.py --model runs/.../GATModel_*_Best_Final.pth --warmup 10 --repeat 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# 座標・エッジのパス（GNN_zscore_sub_noise_defect_free.py と同様）
HOLE_DATA = REPO_ROOT / "GNN_hole/GNN_hole_data"
DATA_BASE = REPO_ROOT / "GNN_hole_2026/all_sub_hole_defect_zscore_noise"
LABEL_DIR = REPO_ROOT / "GNN_hole_2026/all_19class_label"
MAX_NODES = 13942


def load_coords_and_edges():
    """座標とエッジをロード"""
    x = np.load(HOLE_DATA / "normalized_x_2layer.npy")[:MAX_NODES]
    y = np.load(HOLE_DATA / "normalized_y_2layer.npy")[:MAX_NODES]
    z = np.load(HOLE_DATA / "normalized_z_2layer.npy")[:MAX_NODES]
    edge_index = np.load(HOLE_DATA / "hole_edges_2layer_best.npy")
    if edge_index.shape[0] == 2:
        pass  # already [2, E]
    else:
        edge_index = edge_index.T  # [E, 2] -> [2, E]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return x, y, z, edge_index


def create_sample_data(data_folder: Path, x, y, z, edge_index) -> "Data":
    """1サンプルの Data を作成"""
    from torch_geometric.data import Data
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"No .npy files in {data_folder}")
    fpath = data_folder / files[0]
    values = np.load(fpath)[:MAX_NODES]
    node_features = np.vstack((x, y, z, values)).T
    x_t = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x_t, edge_index=edge_index)


def find_default_model() -> Path:
    """デフォルトのベストモデルを検索"""
    candidates = [
        REPO_ROOT / "runs/20260116_104929_nogit_dsNDF_ep2000_lr0p001_F10p730/outputs/GNN_model/19classmodel_hole_zscore/GATModel_20260116_104950_Best_Final.pth",
        REPO_ROOT / "runs/20260116_104929_nogit_dsNDF_ep2000_lr0p001/outputs/GNN_model/19classmodel_hole_zscore/GATModel_20260116_104950_Best_Final.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in (REPO_ROOT / "runs").rglob("GATModel_*_Best_Final.pth"):
        return p
    raise FileNotFoundError("No GATModel_*_Best_Final.pth found in runs/")


def run_benchmark(
    model_path: Path,
    data: "Data",
    device: torch.device,
    warmup: int = 10,
    repeat: int = 100,
) -> dict:
    """推論ベンチマークを実行"""
    sys.path.insert(0, str(REPO_ROOT / "GNN_hole_2026/GNN_program"))
    from GNN_zscore_sub_noise_defect_free import GATModel  # noqa: E402

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    # hidden_channels を state から推定（conv1 の out_channels / 4）
    first_key = next(iter(state.keys()))
    if "conv1" in first_key or "conv1.lin_src.weight" in state:
        w = state.get("conv1.lin_src.weight", state.get("conv1.weight", None))
        if w is not None:
            hidden = w.shape[0] // 4
        else:
            hidden = 16
    else:
        hidden = 16
    model = GATModel(hidden_channels=hidden, num_classes=19, dropout=0.0, edge_drop_prob=0.0)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    data = data.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(data)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(repeat):
            t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()
            _ = model(data)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", type=str, default="reports/inference_benchmark.md")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_default_model()
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    x, y, z, edge_index = load_coords_and_edges()
    data_folder = DATA_BASE / "test"
    if not data_folder.exists():
        data_folder = DATA_BASE / "train"
    data = create_sample_data(data_folder, x, y, z, edge_index)

    results = {}
    devices_to_run = []
    if torch.cuda.is_available():
        devices_to_run.append(("GPU", torch.device("cuda")))
    devices_to_run.append(("CPU", torch.device("cpu")))
    for dev_name, device in devices_to_run:
        repeat_actual = min(args.repeat, 20) if device.type == "cpu" else args.repeat
        try:
            r = run_benchmark(model_path, data, device, args.warmup, repeat_actual)
            results[dev_name] = r
            print(f"[{dev_name}] mean={r['mean_ms']:.2f} ms, std={r['std_ms']:.2f} ms")
        except Exception as e:
            print(f"[{dev_name}] Error: {e}")
            results[dev_name] = {"error": str(e)}

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 推論速度ベンチマーク (M0-2)\n\n")
        f.write(f"モデル: `{model_path.name}`\n")
        f.write(f"サンプル: 1 graph, {MAX_NODES} nodes\n")
        f.write(f"Warmup: {args.warmup}, Repeat: {args.repeat}\n\n")
        f.write("| Device | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |\n")
        f.write("|--------|-----------|----------|----------|----------|\n")
        for dev, r in results.items():
            if "error" in r:
                f.write(f"| {dev} | - | - | - | - | ({r['error']})\n")
            else:
                f.write(f"| {dev} | {r['mean_ms']:.2f} | {r['std_ms']:.2f} | {r['min_ms']:.2f} | {r['max_ms']:.2f} |\n")
        f.write("\n* 目標: 1サンプル < 100ms (実運用)\n")
    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
