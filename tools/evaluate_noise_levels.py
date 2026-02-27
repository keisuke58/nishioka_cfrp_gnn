#!/usr/bin/env python3
"""
M1-4: ノイズレベル別評価

テストデータに複数レベルのガウシアンノイズを付加し、各レベルでの macro_f1 を取得。
性能曲線（ノイズ強度 vs F1）をレポート出力。

考察:
- ノイズが強いほど macro_f1 は低下する傾向
- 実運用では計測ノイズの想定範囲内で性能を保証する必要がある
- 本スクリプトで「どのノイズレベルまで許容できるか」を定量化

Usage:
    python tools/evaluate_noise_levels.py
    python tools/evaluate_noise_levels.py --noise_ratios 0 0.05 0.1 0.2 0.5 --max_samples 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "GNN_hole_2026/GNN_program"))

HOLE_DATA = REPO_ROOT / "GNN_hole/GNN_hole_data"
DATA_BASE = REPO_ROOT / "GNN_hole_2026/all_sub_hole_defect_zscore_noise"
LABEL_DIR = REPO_ROOT / "GNN_hole_2026/all_19class_label"
MAX_NODES = 13942
DEFAULT_MODEL = REPO_ROOT / "runs/20260116_104929_nogit_dsNDF_ep2000_lr0p001_F10p730/outputs/GNN_model/19classmodel_hole_zscore/GATModel_20260116_104950_Best_Final.pth"


def add_noise(values: np.ndarray, noise_std_ratio: float, seed: int | None = None) -> np.ndarray:
    """データの std に対する比率でガウシアンノイズを付加"""
    if noise_std_ratio <= 0:
        return values.copy()
    std = float(np.std(values)) + 1e-12
    noise_std = noise_std_ratio * std
    rng = np.random.default_rng(seed)
    return values.astype(np.float64) + rng.normal(0, noise_std, size=values.shape)


def load_model_and_data(model_path: Path):
    """モデル・座標・エッジをロード"""
    import torch
    from GNN_zscore_sub_noise_defect_free import GATModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    w = state.get("conv1.lin_src.weight", state.get("conv1.weight"))
    hidden = (w.shape[0] // 4) if w is not None else 16
    model = GATModel(hidden_channels=hidden, num_classes=19, dropout=0.0, edge_drop_prob=0.0)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    x = np.load(HOLE_DATA / "normalized_x_2layer.npy")[:MAX_NODES]
    y = np.load(HOLE_DATA / "normalized_y_2layer.npy")[:MAX_NODES]
    z = np.load(HOLE_DATA / "normalized_z_2layer.npy")[:MAX_NODES]
    ei = np.load(HOLE_DATA / "hole_edges_2layer_best.npy")
    if ei.shape[0] != 2:
        ei = ei.T
    edge_index = torch.tensor(ei, dtype=torch.long, device=device)
    return model, device, x, y, z, edge_index


def collect_test_pairs(data_dir: Path, label_dir: Path, max_samples: int | None):
    """テストペアを収集"""
    test_dir = data_dir / "test"
    if not test_dir.exists():
        test_dir = data_dir / "train"
    files = sorted([f.name for f in test_dir.glob("*.npy") if not f.name.endswith("_19label.npy")])
    if max_samples:
        files = files[:max_samples]
    pairs = []
    for fn in files:
        base = Path(fn).stem
        lp = label_dir / f"{base}_19label.npy"
        if lp.exists():
            pairs.append((test_dir / fn, lp))
    return pairs


def labels_to_class_ids(labels: np.ndarray) -> np.ndarray:
    """one-hot or class id -> class ids"""
    if len(labels.shape) == 2 and labels.shape[1] > 1:
        return np.argmax(labels, axis=1)
    return labels.flatten()


def run_evaluation(
    model, device, x, y, z, edge_index,
    pairs: list, noise_ratio: float, seed: int = 42,
):
    """指定ノイズレベルで推論し、全予測・ラベルを収集"""
    import torch
    from torch_geometric.data import Data

    all_preds = []
    all_labels = []
    for i, (dp, lp) in enumerate(pairs):
        values = np.load(dp)[:MAX_NODES].astype(np.float64)
        labels = np.load(lp)[:MAX_NODES]
        y_true = labels_to_class_ids(labels)
        values_noisy = add_noise(values, noise_ratio, seed=seed + i)
        feat = np.vstack((x, y, z, values_noisy)).T
        x_t = torch.tensor(feat, dtype=torch.float, device=device)
        data = Data(x=x_t, edge_index=edge_index)
        with torch.no_grad():
            logits = model(data)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(pred)
        all_labels.append(y_true)
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description="M1-4: Noise-level evaluation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--noise_ratios", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="reports/noise_level_evaluation.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    pairs = collect_test_pairs(DATA_BASE, LABEL_DIR, args.max_samples)
    if not pairs:
        print("[ERROR] No test pairs found")
        return 1

    print(f"[INFO] Loading model from {model_path}")
    model, device, x, y, z, edge_index = load_model_and_data(model_path)
    print(f"[INFO] Evaluating {len(pairs)} samples at {len(args.noise_ratios)} noise levels")

    from gnn_common.metrics import metrics_from_confusion_matrix
    from sklearn.metrics import confusion_matrix

    results = []
    for nr in args.noise_ratios:
        preds, labels = run_evaluation(
            model, device, x, y, z, edge_index, pairs, nr, seed=args.seed
        )
        cm = confusion_matrix(labels, preds, labels=list(range(19)))
        m = metrics_from_confusion_matrix(cm)
        results.append({
            "noise_ratio": nr,
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "weighted_f1": m["weighted_f1"],
        })
        print(f"  noise_ratio={nr:.2f}: macro_f1={m['macro_f1']:.4f}")

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# M1-4: ノイズレベル別評価\n\n")
        f.write(f"モデル: `{model_path.name}`\n")
        f.write(f"サンプル数: {len(pairs)}\n")
        f.write(f"ノイズ: データ std に対する比率 (Gaussian)\n\n")
        f.write("| noise_ratio | macro_f1 | accuracy | weighted_f1 |\n")
        f.write("|-------------|----------|----------|-------------|\n")
        for r in results:
            f.write(f"| {r['noise_ratio']:.2f} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['weighted_f1']:.4f} |\n")
        f.write("\n## 考察\n")
        f.write("- ノイズ強度が増すと macro_f1 は低下する傾向\n")
        f.write("- 実運用では計測ノイズの想定範囲（例: 0.05〜0.1）で性能を確認\n")
        f.write("- 目標: ノイズ 0.1 以下で macro_f1 ≥ 0.6 を維持\n")

    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
