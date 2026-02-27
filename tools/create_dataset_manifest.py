#!/usr/bin/env python3
"""
M0-3: 既存データのメタデータ一覧化

ファイル名から層・ブロック・サイズを抽出し、dataset_manifest.csv を生成する。

Usage:
    python tools/create_dataset_manifest.py
    python tools/create_dataset_manifest.py --data_dir GNN_hole_2026/all_sub_hole_defect_zscore_noise --output reports/dataset_manifest.csv
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# data_utils の extract 関数と同等のロジック
SIZE_MAP = {('2', '2'): 0, ('4', '4'): 1, ('8', '8'): 2}
RE_DEFECT = re.compile(r"Defect_L(\d+)_B(\d+)_el(\d+)_H(\d+)_W(\d+)")
RE_NDF = re.compile(r"NoiseDefectFree_(\d+)")


def extract_metadata(filename: str) -> dict:
    """ファイル名からメタデータを抽出"""
    base = os.path.splitext(filename)[0]
    row = {
        "data_file": filename,
        "label_file": base + "_19label.npy",
        "layer": -1,
        "block": -1,
        "element": -1,
        "h": -1,
        "w": -1,
        "size_class": -1,
        "size_label": "ndf",
        "type": "defect",
    }
    m = RE_NDF.match(base)
    if m:
        row["type"] = "noise_defect_free"
        row["size_label"] = "ndf"
        return row
    m = RE_DEFECT.match(base)
    if m:
        row["layer"] = int(m.group(1))
        row["block"] = int(m.group(2))
        row["element"] = int(m.group(3))
        row["h"] = int(m.group(4))
        row["w"] = int(m.group(5))
        key = (m.group(4), m.group(5))
        row["size_class"] = SIZE_MAP.get(key, -1)
        row["size_label"] = {0: "small", 1: "medium", 2: "large"}.get(row["size_class"], "other")
        return row
    return row


def collect_files(data_dir: Path) -> list[str]:
    """データディレクトリから全 .npy ファイルを収集"""
    files = []
    for sub in ("train", "val", "test"):
        d = data_dir / sub
        if d.exists():
            for f in d.glob("*.npy"):
                files.append(f.name)
    if not files:
        for f in data_dir.glob("*.npy"):
            files.append(f.name)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Create dataset manifest CSV")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="GNN_hole_2026/all_sub_hole_defect_zscore_noise",
        help="Data directory (relative to repo root)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/dataset_manifest.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    data_path = REPO_ROOT / args.data_dir
    output_path = REPO_ROOT / args.output
    if not data_path.exists():
        print(f"[ERROR] Data dir not found: {data_path}")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = collect_files(data_path)
    rows = [extract_metadata(f) for f in files]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[OK] Wrote {len(df)} rows to {output_path}")
    print(f"  Defect: {len(df[df.type=='defect'])}, NDF: {len(df[df.type=='noise_defect_free'])}")
    print(f"  Size: small={len(df[df.size_label=='small'])}, medium={len(df[df.size_label=='medium'])}, large={len(df[df.size_label=='large'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
