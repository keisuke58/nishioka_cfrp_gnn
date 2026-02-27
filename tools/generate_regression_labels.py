#!/usr/bin/env python3
"""
M3-2: 連続パラメータ教師ラベルの生成

メタデータ・ラベル・座標から、回帰ヘッド用の連続真値 (φ, E_ratio, cx, cy, cz, size) を生成。
全サンプルに付与し、CSV と .npy 形式で出力。

Usage:
    python tools/generate_regression_labels.py
    python tools/generate_regression_labels.py --data_dir GNN_hole_2026/all_sub_hole_defect_zscore_noise --output reports/regression_labels.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COORD_DIR = REPO_ROOT / "GNN_hole/GNN_hole_data"
MAX_NODES = 13942

SIZE_MAP = {("2", "2"): 0, ("4", "4"): 1, ("8", "8"): 2}
RE_DEFECT = re.compile(r"Defect_L(\d+)_B(\d+)_el(\d+)_H(\d+)_W(\d+)")
RE_NDF = re.compile(r"NoiseDefectFree_(\d+)")


def load_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """座標をロード"""
    x = np.load(COORD_DIR / "normalized_x_2layer.npy")[:MAX_NODES]
    y = np.load(COORD_DIR / "normalized_y_2layer.npy")[:MAX_NODES]
    z = np.load(COORD_DIR / "normalized_z_2layer.npy")[:MAX_NODES]
    return x, y, z


def extract_size_from_filename(filename: str) -> tuple[int, float]:
    """ファイル名から size_class (0,1,2) と連続 size (H*W の正規化) を抽出"""
    base = Path(filename).stem
    m = RE_DEFECT.match(base)
    if not m:
        return -1, 0.0  # NDF
    h, w = int(m.group(4)), int(m.group(5))
    key = (m.group(4), m.group(5))
    size_class = SIZE_MAP.get(key, -1)
    # 連続 size: H*W を 4~64 の範囲で正規化 (2*2=4, 8*8=64)
    size_cont = (h * w - 4) / 60.0 if (h * w) >= 4 else 0.0  # [0, 1]
    return size_class, float(np.clip(size_cont, 0, 1))


def compute_defect_centroid(
    labels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, float, float]:
    """欠陥ノードの重心 (cx, cy, cz) を計算。正規化座標の平均。欠陥なしは 0,0,0"""
    if len(labels.shape) == 2 and labels.shape[1] > 1:
        y_cls = np.argmax(labels, axis=1)
    else:
        y_cls = labels.flatten()
    defect_mask = y_cls > 0
    if not np.any(defect_mask):
        return 0.0, 0.0, 0.0
    cx = float(np.mean(x[defect_mask]))
    cy = float(np.mean(y[defect_mask]))
    cz = float(np.mean(z[defect_mask]))
    return cx, cy, cz


def process_sample(
    data_path: Path,
    label_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> dict | None:
    """1サンプルの回帰ラベルを計算"""
    if not data_path.exists() or not label_path.exists():
        return None
    try:
        labels = np.load(label_path)[:MAX_NODES]
    except Exception:
        return None
    size_class, size_cont = extract_size_from_filename(data_path.name)
    cx, cy, cz = compute_defect_centroid(labels, x, y, z)
    # 既存データには φ, E_ratio のメタデータがないためデフォルト
    phi = 0.1
    E_ratio = 0.5
    return {
        "data_file": data_path.name,
        "phi": phi,
        "E_ratio": E_ratio,
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "size": size_cont,
        "size_class": size_class,
    }


def collect_pairs(data_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    """データ・ラベルペアを収集"""
    pairs = []
    for sub in ("train", "val", "test"):
        d = data_dir / sub
        if not d.exists():
            continue
        for fp in d.glob("*.npy"):
            if fp.name.endswith("_19label.npy"):
                continue
            base = fp.stem
            lp = label_dir / f"{base}_19label.npy"
            if lp.exists():
                pairs.append((fp, lp))
    if not pairs:
        for fp in data_dir.glob("*.npy"):
            if fp.name.endswith("_19label.npy"):
                continue
            base = fp.stem
            lp = label_dir / f"{base}_19label.npy"
            if lp.exists():
                pairs.append((fp, lp))
    return sorted(pairs, key=lambda p: p[0].name)


def main():
    parser = argparse.ArgumentParser(description="M3-2: Generate regression labels")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="GNN_hole_2026/all_sub_hole_defect_zscore_noise",
        help="Data directory",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default="GNN_hole_2026/all_19class_label",
        help="Label directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/regression_labels.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        default="reports/regression_labels.npy",
        help="Output NPY path (dict: filename -> array)",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    data_dir = REPO_ROOT / args.data_dir
    label_dir = REPO_ROOT / args.label_dir
    if not data_dir.exists():
        print(f"[ERROR] Data dir not found: {data_dir}")
        return 1
    if not label_dir.exists():
        print(f"[ERROR] Label dir not found: {label_dir}")
        return 1

    x, y, z = load_coordinates()
    pairs = collect_pairs(data_dir, label_dir)
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    rows = []
    reg_dict = {}
    for dp, lp in pairs:
        r = process_sample(dp, lp, x, y, z)
        if r is None:
            continue
        rows.append(r)
        reg_dict[r["data_file"]] = np.array(
            [r["phi"], r["E_ratio"], r["cx"], r["cy"], r["cz"], r["size"]],
            dtype=np.float32,
        )

    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / args.output
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {len(df)} rows to {out_csv}")

    out_npy = REPO_ROOT / args.output_npy
    np.save(out_npy, reg_dict, allow_pickle=True)
    print(f"[OK] Wrote regression dict to {out_npy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
