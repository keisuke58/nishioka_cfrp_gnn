#!/usr/bin/env python3
"""
M2-2: FEM多物性・境界ぼかし対応の前処理

既存のDSPSSノード値に対して、欠陥パラメータ(φ, E_ratio, blur_sigma)を適用し、
多物性・境界ぼかしをシミュレートした合成データを生成する。

FEMソルバーが外部の場合は、既存データの「合成的改変」として使用。
FEMが要素ごと物性を渡す場合は literature_notes/fem_preprocess_api_spec.md を参照。

Usage:
    from tools.micro_defect_preprocess import apply_defect_params, load_params_schema
    out = apply_defect_params(values, defect_mask, edge_index, {"phi": 0.1, "E_ratio": 0.5, "blur_sigma": 1.0})
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "literature_notes" / "defect_params_schema.yaml"


def load_params_schema(path: Optional[Path] = None) -> dict:
    """defect_params_schema.yaml を読み込む"""
    p = path or SCHEMA_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def halpin_tsai_E_ratio(phi: float, eta: float = 1.0) -> float:
    """
    Halpin-Tsai 近似で空隙率φから有効弾性係数比 E_eff/E_base を計算。

    E_eff/E_base ≈ (1 + eta*xi*phi) / (1 - xi*phi)
    xi = (E_filler/E_base - 1) / (E_filler/E_base + eta)
    空隙の場合は E_filler→0 として xi ≈ -1/eta

    Args:
        phi: 空隙率 [0, 1]
        eta: 形状因子（デフォルト1）

    Returns:
        E_eff / E_base の近似値 [0, 1]
    """
    if phi <= 0:
        return 1.0
    if phi >= 1:
        return 0.0
    # 簡易: xi = -1/eta とすると
    # E_eff/E = (1 - phi) / (1 + phi/eta) の形
    xi = -1.0 / (eta + 1e-8)
    num = 1.0 + eta * xi * phi
    den = 1.0 - xi * phi
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def effective_E_ratio(phi: Optional[float] = None, E_ratio: Optional[float] = None) -> float:
    """
    φ と E_ratio から有効な E スケール係数を返す。
    phi が指定されていれば Halpin-Tsai を使用、否则は E_ratio をそのまま使用。
    """
    if phi is not None:
        return halpin_tsai_E_ratio(float(phi))
    if E_ratio is not None:
        return float(np.clip(E_ratio, 0.0, 1.0))
    return 0.5  # default


def _build_adjacency(edge_index: np.ndarray, num_nodes: int) -> Dict[int, list]:
    """edge_index [2,E] から隣接リストを構築"""
    adj: Dict[int, list] = {i: [] for i in range(num_nodes)}
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u < num_nodes and v < num_nodes:
            adj[u].append(v)
    return adj


def apply_defect_blur(
    values: np.ndarray,
    defect_mask: np.ndarray,
    edge_index: np.ndarray,
    blur_sigma: float,
    num_iter: int = 3,
) -> np.ndarray:
    """
    欠陥境界でガウシアンぼかしを適用。
    欠陥ノードのうち、非欠陥ノードと隣接する「境界ノード」の値を
    隣接ノードとの重み付き平均で滑らかにする。

    Args:
        values: [N] ノード値
        defect_mask: [N] bool, True=欠陥ノード
        edge_index: [2,E] エッジ
        blur_sigma: ぼかし強度（隣接平均の重み）
        num_iter: 反復回数

    Returns:
        ぼかし適用後の [N] 配列
    """
    out = values.copy().astype(np.float64)
    n = len(values)
    adj = _build_adjacency(edge_index, n)

    for _ in range(num_iter):
        new_out = out.copy()
        for i in range(n):
            if not defect_mask[i]:
                continue
            neighbors = adj[i]
            if not neighbors:
                continue
            # 隣接のうち非欠陥があれば境界ノード
            ext_neighbors = [j for j in neighbors if not defect_mask[j]]
            if not ext_neighbors:
                continue
            # 境界: 欠陥内隣接と非欠陥隣接の加重平均
            w_ext = blur_sigma  # 非欠陥側の重み
            w_int = 1.0
            val_ext = np.mean([out[j] for j in ext_neighbors])
            val_int = out[i]
            n_ext, n_int = len(ext_neighbors), len(neighbors) - len(ext_neighbors)
            if n_int > 0:
                val_int = np.mean([out[j] for j in neighbors if defect_mask[j]])
            new_out[i] = (w_ext * val_ext + w_int * val_int) / (w_ext + w_int)
        out = new_out
    return out.astype(values.dtype)


def apply_defect_params(
    values: np.ndarray,
    defect_mask: np.ndarray,
    edge_index: np.ndarray,
    params: Dict[str, float],
    *,
    scale_non_defect: bool = False,
) -> np.ndarray:
    """
    欠陥パラメータを適用してノード値を修正。

    Args:
        values: [N] DSPSS差分などのノード値
        defect_mask: [N] bool, True=欠陥ノード
        edge_index: [2,E] エッジ
        params: {"phi", "E_ratio", "blur_sigma", "delamination_ratio"} のいずれか
        scale_non_defect: 非欠陥ノードもスケールするか（通常False）

    Returns:
        修正後の [N] 配列
    """
    out = values.copy().astype(np.float64)
    phi = params.get("phi")
    E_ratio = params.get("E_ratio", 0.5)
    blur_sigma = params.get("blur_sigma", 0.0)

    scale = effective_E_ratio(phi, E_ratio)
    # 欠陥ノードの値をスケール（差分が小さくなる方向）
    out[defect_mask] *= scale
    if scale_non_defect:
        out[~defect_mask] *= (1.0 + scale) / 2.0  # 軽微な影響

    if blur_sigma > 0:
        out = apply_defect_blur(out, defect_mask, edge_index, blur_sigma)

    return out.astype(values.dtype)


def process_file(
    data_path: Path,
    label_path: Path,
    edge_index: np.ndarray,
    output_path: Path,
    params: Dict[str, float],
    max_nodes: int = 13942,
) -> bool:
    """
    単一ファイルに apply_defect_params を適用して保存。

    Args:
        data_path: 入力 .npy
        label_path: 対応する _19label.npy（欠陥マスク用）
        edge_index: [2,E]
        output_path: 出力 .npy
        params: 欠陥パラメータ
        max_nodes: 最大ノード数

    Returns:
        成功したか
    """
    try:
        values = np.load(data_path)[:max_nodes]
        labels = np.load(label_path)[:max_nodes]
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            y = np.argmax(labels, axis=1)
        else:
            y = labels.flatten()
        defect_mask = y > 0
        if values.shape[0] != len(defect_mask):
            return False
        out = apply_defect_params(values, defect_mask, edge_index, params)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, out)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="M2-2: 多物性・ぼかし前処理")
    parser.add_argument("--input", type=str, required=True, help="入力データフォルダ")
    parser.add_argument("--labels", type=str, required=True, help="ラベルフォルダ")
    parser.add_argument("--output", type=str, required=True, help="出力フォルダ")
    parser.add_argument("--phi", type=float, default=None)
    parser.add_argument("--E_ratio", type=float, default=0.5)
    parser.add_argument("--blur_sigma", type=float, default=0.0)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    edge_path = REPO_ROOT / "GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy"
    ei = np.load(edge_path)
    if ei.shape[0] != 2:
        ei = ei.T

    params = {}
    if args.phi is not None:
        params["phi"] = args.phi
    params["E_ratio"] = args.E_ratio
    params["blur_sigma"] = args.blur_sigma

    input_dir = Path(args.input)
    label_dir = Path(args.labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted([f for f in input_dir.glob("*.npy") if not f.name.endswith("_19label.npy")])
    if args.max_files:
        data_files = data_files[: args.max_files]
    n_ok = 0
    for fp in data_files:
        base = fp.stem
        label_path = label_dir / f"{base}_19label.npy"
        if not label_path.exists():
            continue
        out_path = output_dir / fp.name
        if process_file(fp, label_path, ei, out_path, params):
            n_ok += 1
    print(f"[OK] Processed {n_ok}/{len(data_files)} files -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
