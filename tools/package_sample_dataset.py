#!/usr/bin/env python3
"""
Create a small, shareable sample dataset from the full CFRP GNN dataset.

Selects a stratified sample covering all defect sizes and multiple layers,
then packages it with metadata and a standalone loader.

Usage:
    python tools/package_sample_dataset.py \
        --data_dir GNN_hole_2026/all_sub_hole_defect_zscore \
        --label_dir GNN_hole_2026/all_19class_label \
        --output_dir sample_dataset \
        --n_per_size 5 \
        --n_per_layer 3
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Filename parsing patterns
RE_DEFECT = re.compile(
    r"Defect_L(\d+)_B(\d+)_el(\d+)_H(\d+)_W(\d+)"
)
RE_NOISE_FREE = re.compile(r"NoiseDefectFree_")


def parse_filename(fname: str) -> Dict[str, any]:
    """Extract metadata from a data filename."""
    m = RE_DEFECT.match(fname)
    if m:
        h, w = int(m.group(4)), int(m.group(5))
        if h <= 2 and w <= 2:
            size_class = "small"
        elif h <= 4 and w <= 4:
            size_class = "medium"
        else:
            size_class = "large"
        return {
            "type": "defect",
            "layer": int(m.group(1)),
            "block": int(m.group(2)),
            "element": int(m.group(3)),
            "h": h,
            "w": w,
            "size_class": size_class,
        }
    if RE_NOISE_FREE.match(fname):
        return {"type": "noise_defect_free"}
    return {"type": "unknown"}


def select_sample_subset(
    data_dir: str,
    label_dir: str,
    n_per_size: int = 5,
    n_per_layer: int = 3,
    n_defect_free: int = 5,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Select a stratified sample covering all defect sizes and layers.

    Returns list of (data_filename, label_filename, source_dir).
    """
    rng = np.random.RandomState(seed)

    # Collect all data files with labels
    data_path = Path(data_dir)
    label_path = Path(label_dir)

    all_files: List[Tuple[str, str, str]] = []
    # Handle subdirectory structure
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if subdirs:
        dirs_to_scan = subdirs
    else:
        dirs_to_scan = [data_path]

    for scan_dir in dirs_to_scan:
        for f in scan_dir.iterdir():
            if not f.suffix == ".npy":
                continue
            base = f.stem
            label_fname = f"{base}_19label.npy"
            if (label_path / label_fname).exists():
                all_files.append((f.name, label_fname, str(scan_dir)))

    # Group by (size_class, layer)
    by_size: Dict[str, List] = defaultdict(list)
    defect_free: List = []

    for data_f, label_f, src_dir in all_files:
        meta = parse_filename(data_f)
        if meta["type"] == "defect":
            key = meta["size_class"]
            by_size[key].append((data_f, label_f, src_dir, meta))
        elif meta["type"] == "noise_defect_free":
            defect_free.append((data_f, label_f, src_dir))

    selected = []

    # Select per size class, stratified by layer
    for size_class in ["small", "medium", "large"]:
        candidates = by_size.get(size_class, [])
        if not candidates:
            print(f"[WARN] No candidates for size_class={size_class}")
            continue

        # Group by layer within this size class
        by_layer: Dict[int, List] = defaultdict(list)
        for item in candidates:
            by_layer[item[3]["layer"]].append(item)

        # Select n_per_layer from each layer, up to n_per_size total
        layer_selections = []
        layers = sorted(by_layer.keys())
        for layer in layers:
            layer_cands = by_layer[layer]
            n = min(n_per_layer, len(layer_cands))
            indices = rng.choice(len(layer_cands), size=n, replace=False)
            for idx in indices:
                layer_selections.append(layer_cands[idx])

        # Cap at n_per_size
        if len(layer_selections) > n_per_size:
            indices = rng.choice(len(layer_selections), size=n_per_size, replace=False)
            layer_selections = [layer_selections[i] for i in sorted(indices)]

        for item in layer_selections:
            selected.append((item[0], item[1], item[2]))

    # Add defect-free samples
    if defect_free:
        n = min(n_defect_free, len(defect_free))
        indices = rng.choice(len(defect_free), size=n, replace=False)
        for idx in indices:
            selected.append(defect_free[idx])

    print(f"[INFO] Selected {len(selected)} samples for packaging")
    return selected


def package_dataset(
    selected: List[Tuple[str, str, str]],
    label_dir: str,
    output_dir: str,
) -> None:
    """Copy selected files to output directory with proper structure."""
    out = Path(output_dir)
    data_out = out / "data"
    label_out = out / "labels"
    data_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for data_f, label_f, src_dir in selected:
        # Copy data file
        src_data = Path(src_dir) / data_f
        if src_data.exists():
            shutil.copy2(src_data, data_out / data_f)

        # Copy label file
        src_label = Path(label_dir) / label_f
        if src_label.exists():
            shutil.copy2(src_label, label_out / label_f)

    print(f"[INFO] Copied {len(selected)} data + label files to {out}")


def generate_metadata(
    selected: List[Tuple[str, str, str]],
    output_dir: str,
) -> None:
    """Write metadata.json describing the sample dataset."""
    out = Path(output_dir)

    # Analyze selected files
    size_counts = defaultdict(int)
    layer_counts = defaultdict(int)
    types = defaultdict(int)

    for data_f, _, _ in selected:
        meta = parse_filename(data_f)
        types[meta["type"]] += 1
        if meta["type"] == "defect":
            size_counts[meta["size_class"]] += 1
            layer_counts[meta["layer"]] += 1

    # Try to get node count from first file
    sample_data_path = out / "data" / selected[0][0] if selected else None
    num_nodes = None
    if sample_data_path and sample_data_path.exists():
        arr = np.load(sample_data_path)
        num_nodes = len(arr) if arr.ndim == 1 else arr.shape[0]

    metadata = {
        "dataset_name": "CFRP GNN Sample Dataset",
        "version": "1.0",
        "created_at": _dt.datetime.now().isoformat(),
        "description": "Sample subset of CFRP defect localization dataset for GNN benchmarking",
        "num_samples": len(selected),
        "num_nodes_per_sample": num_nodes,
        "num_features": 4,
        "feature_names": ["x_coord", "y_coord", "z_coord", "DSPSS"],
        "num_classes": 19,
        "label_format": "one-hot encoded [N, 19]",
        "defect_size_distribution": dict(size_counts),
        "layer_distribution": {str(k): v for k, v in sorted(layer_counts.items())},
        "sample_type_counts": dict(types),
        "file_naming_convention": "Defect_L{layer}_B{block}_el{element}_H{height}_W{width}.npy",
        "label_naming_convention": "{basename}_19label.npy",
    }

    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Metadata written to {meta_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a sample CFRP GNN dataset")
    parser.add_argument("--data_dir", type=str,
                        default="GNN_hole_2026/all_sub_hole_defect_zscore",
                        help="Data directory (relative to project root)")
    parser.add_argument("--label_dir", type=str,
                        default="GNN_hole_2026/all_19class_label",
                        help="Label directory (relative to project root)")
    parser.add_argument("--output_dir", type=str, default="sample_dataset",
                        help="Output directory for the sample dataset")
    parser.add_argument("--n_per_size", type=int, default=5,
                        help="Number of samples per defect size class")
    parser.add_argument("--n_per_layer", type=int, default=3,
                        help="Number of samples per layer within each size class")
    parser.add_argument("--n_defect_free", type=int, default=5,
                        help="Number of defect-free samples to include")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible selection")
    args = parser.parse_args()

    data_dir = str(REPO_ROOT / args.data_dir)
    label_dir = str(REPO_ROOT / args.label_dir)
    output_dir = str(REPO_ROOT / args.output_dir)

    selected = select_sample_subset(
        data_dir=data_dir,
        label_dir=label_dir,
        n_per_size=args.n_per_size,
        n_per_layer=args.n_per_layer,
        n_defect_free=args.n_defect_free,
        seed=args.seed,
    )

    package_dataset(selected, label_dir, output_dir)
    generate_metadata(selected, output_dir)

    print(f"\n[DONE] Sample dataset packaged at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
