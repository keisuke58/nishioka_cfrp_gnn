#!/usr/bin/env python3
"""
M2-3: パラメータサンプリング → ミクロ欠陥データ生成

defect_params_schema.yaml に基づき LHS で (phi, E_ratio, blur_sigma) をサンプリングし、
既存データに micro_defect_preprocess を適用して多様な欠陥パターンを生成。

Usage:
    python tools/generate_micro_defect_data.py --n_samples 100 --max_source 50
    python tools/generate_micro_defect_data.py --n_samples 500 --output GNN_hole_2026/all_micro_defect_zscore
"""

from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SCHEMA_PATH = REPO_ROOT / "literature_notes" / "defect_params_schema.yaml"


def load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def latin_hypercube_sample(
    n_samples: int,
    phi_range: tuple[float, float] = (0.0, 0.3),
    E_ratio_range: tuple[float, float] = (0.1, 1.0),
    blur_range: tuple[float, float] = (0.0, 2.0),
    seed: int = 42,
) -> list[dict]:
    """LHS でパラメータをサンプリング"""
    rng = np.random.default_rng(seed)
    # 簡易 LHS: 各次元を n 等分し、各ビンから1点
    n = max(n_samples, 1)
    u = rng.permutation(n) / n + rng.uniform(0, 1 / n, n)
    phi = phi_range[0] + u * (phi_range[1] - phi_range[0])
    u2 = rng.permutation(n) / n + rng.uniform(0, 1 / n, n)
    E_ratio = E_ratio_range[0] + u2 * (E_ratio_range[1] - E_ratio_range[0])
    u3 = rng.permutation(n) / n + rng.uniform(0, 1 / n, n)
    blur = blur_range[0] + u3 * (blur_range[1] - blur_range[0])
    return [
        {"phi": float(phi[i]), "E_ratio": float(E_ratio[i]), "blur_sigma": float(blur[i])}
        for i in range(n)
    ]


def main():
    parser = argparse.ArgumentParser(description="M2-3: Generate micro defect data")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_source", type=int, default=30)
    parser.add_argument("--input", type=str, default="GNN_hole_2026/all_sub_hole_defect_zscore_noise/train")
    parser.add_argument("--labels", type=str, default="GNN_hole_2026/all_19class_label")
    parser.add_argument("--output", type=str, default="GNN_hole_2026/all_micro_defect_zscore/train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    schema = load_schema()
    dp = schema.get("defect_params", {})
    phi_cfg = dp.get("phi", {})
    E_cfg = dp.get("E_ratio", {})
    blur_cfg = dp.get("shape", {}).get("blur_sigma", {})

    phi_range = (phi_cfg.get("min", 0.0), phi_cfg.get("max", 0.3))
    E_range = (E_cfg.get("min", 0.1), E_cfg.get("max", 1.0))
    blur_range = (blur_cfg.get("min", 0.0), min(blur_cfg.get("max", 5.0), 2.0))

    params_list = latin_hypercube_sample(
        args.n_samples, phi_range, E_range, blur_range, args.seed
    )

    input_dir = REPO_ROOT / args.input
    label_dir = REPO_ROOT / args.labels
    output_dir = REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted([f for f in input_dir.glob("*.npy") if not f.name.endswith("_19label.npy")])
    data_files = data_files[: args.max_source]
    if not data_files:
        print("[ERROR] No source data found")
        return 1

    ei = np.load(REPO_ROOT / "GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy")
    if ei.shape[0] != 2:
        ei = ei.T

    from tools.micro_defect_preprocess import process_file

    n_written = 0
    for i, params in enumerate(params_list):
        src = data_files[i % len(data_files)]
        base = src.stem
        lp = label_dir / f"{base}_19label.npy"
        if not lp.exists():
            continue
        out_name = f"{base}_micro_phi{params['phi']:.2f}_E{params['E_ratio']:.2f}_b{params['blur_sigma']:.2f}.npy"
        out_path = output_dir / out_name
        p = {"phi": params["phi"], "E_ratio": params["E_ratio"], "blur_sigma": params["blur_sigma"]}
        if process_file(src, lp, ei, out_path, p):
            n_written += 1

    print(f"[OK] Generated {n_written} files to {output_dir}")
    print(f"  Params sampled: {len(params_list)}, Source files: {len(data_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
