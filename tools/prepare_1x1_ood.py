#!/usr/bin/env python3
"""
Prepare 1x1 micro-defect data for property OOD evaluation.

This script:
1. Checks if 1x1 micro-defect data has been preprocessed (.npy)
2. If only raw .inp files exist, prints preprocessing instructions
3. Creates benchmark_config_1x1_ood.yaml for property_ood split
   - train: H2_W2 + H4_W4 + H8_W8 (existing sizes)
   - test:  H1_W1 only (micro-defect, OOD)
4. Creates file lists (train_files.txt, test_files.txt)
5. Validates that label files exist for each data file
6. Updates benchmark_config.yaml to add property_ood_1x1 split entry

Usage:
    python tools/prepare_1x1_ood.py
    python tools/prepare_1x1_ood.py --data_dir GNN_hole_2026/all_sub_hole_defect_zscore_noise
    python tools/prepare_1x1_ood.py --dry_run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Project root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Where preprocessed data lives (existing sizes)
DEFAULT_DATA_DIR = "GNN_hole_2026/all_sub_hole_defect_zscore_noise"
DEFAULT_LABEL_DIR = "GNN_hole_2026/all_19class_label"

# Raw 1x1 data sources
RAW_1x1_DIRS = [
    "GNN_hole/Defecthole_1x1_Random",
    "GNN_hole/Defecthole_1x1_generated",
]

# Preprocessing pipeline scripts (in execution order)
PREPROCESSING_PIPELINE = [
    "GNN_hole_2026/subtract_hole_no_defect.py",
    "GNN_hole_2026/normalize_all_subtracted_zscore.py",
]

# Size pattern in filenames
SIZE_PATTERN = re.compile(r"H(\d+)_W(\d+)")

# Train sizes (existing data)
TRAIN_SIZES = {("2", "2"), ("4", "4"), ("8", "8")}
# Test size (1x1 micro-defect, OOD)
TEST_SIZE = {("1", "1")}

OUTPUT_DIR_NAME = "splits_1x1_ood"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_size(filename: str) -> Tuple[str, str] | None:
    """Extract (H, W) from filename like Defect_L2_B3_el129_H8_W8.npy."""
    m = SIZE_PATTERN.search(filename)
    if m:
        return (m.group(1), m.group(2))
    return None


def find_npy_files(data_dir: Path) -> List[Path]:
    """Find all .npy data files across subdirectories or flat directory."""
    files = []
    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name in ("train", "val", "test")]
    if subdirs:
        for subdir in sorted(data_dir.iterdir()):
            if subdir.is_dir():
                files.extend(sorted(subdir.glob("*.npy")))
    else:
        files = sorted(data_dir.glob("*.npy"))
    return files


def find_raw_inp_files() -> Dict[str, List[Path]]:
    """Find raw .inp files in known 1x1 directories."""
    result = {}
    for rel_dir in RAW_1x1_DIRS:
        abs_dir = REPO_ROOT / rel_dir
        if abs_dir.exists():
            inp_files = sorted(abs_dir.rglob("*H1_W1*.inp"))
            if inp_files:
                result[rel_dir] = inp_files
    return result


def label_file_for_data(data_filename: str) -> str:
    """Construct expected label filename from data filename.

    Data:  Defect_L2_B3_el129_H8_W8.npy
    Label: Defect_L2_B3_el129_H8_W8_19label.npy
    """
    base = os.path.splitext(data_filename)[0]
    return f"{base}_19label.npy"


# ---------------------------------------------------------------------------
# Step 1 & 2: Check 1x1 data status
# ---------------------------------------------------------------------------

def check_1x1_data_status(data_dir: Path) -> Tuple[List[Path], bool]:
    """Check if preprocessed 1x1 .npy files exist.

    Returns:
        (list of 1x1 npy files found, whether data is ready)
    """
    all_files = find_npy_files(data_dir)
    h1w1_files = [f for f in all_files if extract_size(f.name) == ("1", "1")]

    if h1w1_files:
        print(f"[OK] Found {len(h1w1_files)} preprocessed H1_W1 .npy files in {data_dir}")
        return h1w1_files, True

    # No preprocessed data -- check for raw .inp files
    raw_sources = find_raw_inp_files()
    if raw_sources:
        total_inp = sum(len(v) for v in raw_sources.items())
        print(f"[WARN] No preprocessed H1_W1 .npy files found in {data_dir}")
        print(f"       But found raw .inp files in {len(raw_sources)} source(s):")
        for src, files in raw_sources.items():
            print(f"         {src}: {len(files)} .inp files")
        print()
        print("=" * 70)
        print("PREPROCESSING INSTRUCTIONS")
        print("=" * 70)
        print()
        print("To prepare 1x1 data for OOD evaluation, run the following pipeline:")
        print()
        print("  1. Run FEM simulation on .inp files (Abaqus or equivalent)")
        print("     to generate stress/displacement results.")
        print()
        print("  2. Extract DSPSS features and create per-sample .npy arrays.")
        print()
        print("  3. Subtract defect-free baseline:")
        for script in PREPROCESSING_PIPELINE:
            print(f"       python {script}")
        print()
        print("  4. Place output .npy files into:")
        print(f"       {data_dir}/test/  (for H1_W1 data)")
        print()
        print("  5. Generate 19-class labels for 1x1 data and place into:")
        print(f"       {REPO_ROOT / DEFAULT_LABEL_DIR}/")
        print()
        print("  6. Re-run this script to create the split configuration.")
        print("=" * 70)
        return [], False
    else:
        print(f"[WARN] No H1_W1 data found (neither .npy nor .inp)")
        print(f"       Expected raw data in:")
        for d in RAW_1x1_DIRS:
            print(f"         {REPO_ROOT / d}")
        return [], False


# ---------------------------------------------------------------------------
# Step 3: Create benchmark_config_1x1_ood.yaml
# ---------------------------------------------------------------------------

def create_ood_config(output_path: Path) -> None:
    """Create dedicated benchmark config for 1x1 OOD evaluation."""
    config = {
        "benchmark": {
            "name": "cfrp_gnn_1x1_micro_defect_ood",
            "description": (
                "Property OOD benchmark: train on H2_W2/H4_W4/H8_W8, "
                "test on H1_W1 micro-defect (unseen size class)"
            ),
            "seeds": [42, 84, 126],
            "split_types": [
                {
                    "type": "property_ood",
                    "params": {
                        "train_size_classes": [0, 1, 2],  # small(H2), medium(H4), large(H8)
                        "test_size_classes": [3],          # micro(H1_W1)
                    },
                },
            ],
            "models": [
                {
                    "name": "GAT_base",
                    "args": {
                        "hidden_channels": 16,
                        "learning_rate": 0.002,
                        "epochs": 500,
                        "patience": 100,
                        "batch_size": 64,
                        "dropout": 0.10,
                        "edge_drop": 0.01,
                        "use_amp": True,
                        "use_class_frequency_sampler": True,
                    },
                },
            ],
            "metrics": {
                "primary": "macro_f1",
                "secondary": [
                    "weighted_f1",
                    "balanced_accuracy",
                    "mcc",
                    "top_1_accuracy",
                    "top_3_accuracy",
                    "top_5_accuracy",
                    "mean_distance_error",
                    "auprc",
                ],
            },
            "hardware": {
                "nproc_per_node": 4,
            },
            "data": {
                "data_dir": DEFAULT_DATA_DIR,
                "label_dir": DEFAULT_LABEL_DIR,
                "num_classes": 19,
            },
            "output": {
                "base_dir": "runs",
                "prefix": "bench_1x1_ood",
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"[OK] Created OOD config: {output_path}")


# ---------------------------------------------------------------------------
# Step 4: Create file lists
# ---------------------------------------------------------------------------

def create_file_lists(
    data_dir: Path,
    label_dir: Path,
    output_dir: Path,
) -> Tuple[List[str], List[str], int, int]:
    """Create train_files.txt and test_files.txt based on size.

    Returns:
        (train_files, test_files, label_missing_train, label_missing_test)
    """
    all_files = find_npy_files(data_dir)
    train_files: List[str] = []
    test_files: List[str] = []
    label_missing_train = 0
    label_missing_test = 0

    for fpath in all_files:
        fname = fpath.name
        size = extract_size(fname)
        if size is None:
            continue

        # Check label existence
        expected_label = label_file_for_data(fname)
        label_exists = (label_dir / expected_label).exists()

        rel_path = str(fpath.relative_to(REPO_ROOT))

        if size in TRAIN_SIZES:
            train_files.append(rel_path)
            if not label_exists:
                label_missing_train += 1
        elif size in TEST_SIZE:
            test_files.append(rel_path)
            if not label_exists:
                label_missing_test += 1

    # Write file lists
    output_dir.mkdir(parents=True, exist_ok=True)

    train_list_path = output_dir / "train_files.txt"
    with open(train_list_path, "w") as f:
        f.write(f"# Train files for property_ood_1x1 split\n")
        f.write(f"# Sizes: H2_W2 (small), H4_W4 (medium), H8_W8 (large)\n")
        f.write(f"# Total: {len(train_files)} files\n")
        for fp in sorted(train_files):
            f.write(fp + "\n")
    print(f"[OK] Train file list: {train_list_path} ({len(train_files)} files)")

    test_list_path = output_dir / "test_files.txt"
    with open(test_list_path, "w") as f:
        f.write(f"# Test files for property_ood_1x1 split (OOD)\n")
        f.write(f"# Sizes: H1_W1 (micro-defect, unseen)\n")
        f.write(f"# Total: {len(test_files)} files\n")
        for fp in sorted(test_files):
            f.write(fp + "\n")
    print(f"[OK] Test file list:  {test_list_path} ({len(test_files)} files)")

    return train_files, test_files, label_missing_train, label_missing_test


# ---------------------------------------------------------------------------
# Step 5: Validate labels
# ---------------------------------------------------------------------------

def validate_labels(
    train_files: List[str],
    test_files: List[str],
    label_dir: Path,
    label_missing_train: int,
    label_missing_test: int,
) -> bool:
    """Report label validation results."""
    print()
    print("-" * 50)
    print("LABEL VALIDATION")
    print("-" * 50)
    print(f"  Train files: {len(train_files)}, missing labels: {label_missing_train}")
    print(f"  Test  files: {len(test_files)}, missing labels: {label_missing_test}")

    if label_missing_train > 0:
        print(f"  [WARN] {label_missing_train} train files have no matching label in {label_dir}")
    if label_missing_test > 0:
        print(f"  [WARN] {label_missing_test} test files have no matching label in {label_dir}")

    total_missing = label_missing_train + label_missing_test
    if total_missing == 0:
        print("  [OK] All data files have corresponding labels.")
        return True
    else:
        print(f"  [WARN] {total_missing} total files missing labels.")
        print(f"         Label dir: {label_dir}")
        print(f"         Expected pattern: <data_basename>_19label.npy")
        return False


# ---------------------------------------------------------------------------
# Step 6: Update main benchmark_config.yaml
# ---------------------------------------------------------------------------

def update_main_config(config_path: Path) -> None:
    """Add property_ood_1x1 entry to the main benchmark config if not present."""
    if not config_path.exists():
        print(f"[WARN] Main config not found: {config_path}, skipping update")
        return

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        print(f"[WARN] Empty config: {config_path}, skipping update")
        return

    cfg = raw.get("benchmark", raw)
    split_types = cfg.get("split_types", [])

    # Check if property_ood_1x1 already exists
    for st in split_types:
        if st.get("type") == "property_ood_1x1":
            print(f"[OK] property_ood_1x1 already present in {config_path}")
            return

    # Add the new split type
    new_entry = {
        "type": "property_ood",
        "params": {
            "train_size_classes": [0, 1, 2],
            "test_size_classes": [3],
        },
    }

    # Add a comment-like name field so it's distinguishable from the existing property_ood
    # We add it with a descriptive comment via a tagged name
    # Since YAML doesn't support inline comments well, we use a 'name' sub-field
    new_entry_named = {
        "type": "property_ood",
        "name": "property_ood_1x1",
        "params": {
            "train_size_classes": [0, 1, 2],  # small + medium + large
            "test_size_classes": [3],          # micro (H1_W1)
        },
    }

    split_types.append(new_entry_named)
    cfg["split_types"] = split_types

    if "benchmark" in raw:
        raw["benchmark"] = cfg
    else:
        raw = cfg

    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[OK] Added property_ood_1x1 split to {config_path}")


# ---------------------------------------------------------------------------
# Size breakdown summary
# ---------------------------------------------------------------------------

def print_size_summary(data_dir: Path) -> None:
    """Print breakdown of data files by size class."""
    all_files = find_npy_files(data_dir)
    counts: Dict[Tuple[str, str], int] = {}
    unknown = 0
    for f in all_files:
        size = extract_size(f.name)
        if size:
            counts[size] = counts.get(size, 0) + 1
        else:
            unknown += 1

    print()
    print("-" * 50)
    print("DATA SIZE BREAKDOWN")
    print("-" * 50)
    size_labels = {
        ("1", "1"): "micro  (H1_W1) -- OOD test",
        ("2", "2"): "small  (H2_W2) -- train",
        ("4", "4"): "medium (H4_W4) -- train",
        ("8", "8"): "large  (H8_W8) -- train",
    }
    for size_key in [("1", "1"), ("2", "2"), ("4", "4"), ("8", "8")]:
        label = size_labels.get(size_key, f"H{size_key[0]}_W{size_key[1]}")
        count = counts.get(size_key, 0)
        print(f"  {label}: {count:5d} files")
    if unknown > 0:
        print(f"  unknown size pattern:      {unknown:5d} files")

    # Print other sizes not in the standard set
    other_sizes = {k: v for k, v in counts.items() if k not in size_labels}
    for k, v in sorted(other_sizes.items()):
        print(f"  H{k[0]}_W{k[1]} (other):            {v:5d} files")
    print(f"  {'TOTAL':29s}: {len(all_files):5d} files")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare 1x1 micro-defect data for property OOD evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir", type=str, default=DEFAULT_DATA_DIR,
        help=f"Data directory relative to project root (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--label_dir", type=str, default=DEFAULT_LABEL_DIR,
        help=f"Label directory relative to project root (default: {DEFAULT_LABEL_DIR})",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Check data status without creating config files",
    )
    args = parser.parse_args()

    data_dir = REPO_ROOT / args.data_dir
    label_dir = REPO_ROOT / args.label_dir
    tools_dir = REPO_ROOT / "tools"
    output_dir = tools_dir / OUTPUT_DIR_NAME

    print("=" * 70)
    print("1x1 Micro-Defect OOD Evaluation -- Data Preparation")
    print("=" * 70)
    print(f"  Project root: {REPO_ROOT}")
    print(f"  Data dir:     {data_dir}")
    print(f"  Label dir:    {label_dir}")
    print()

    # Step 1-2: Check 1x1 data status
    h1w1_files, data_ready = check_1x1_data_status(data_dir)

    # Print size breakdown regardless
    print_size_summary(data_dir)

    if args.dry_run:
        print("\n[DRY RUN] Stopping before file creation.")
        if not data_ready:
            print("[INFO] 1x1 data not yet available. See preprocessing instructions above.")
        return 0

    # Step 3: Create dedicated OOD config
    ood_config_path = tools_dir / "benchmark_config_1x1_ood.yaml"
    create_ood_config(ood_config_path)

    # Step 4: Create file lists
    train_files, test_files, lm_train, lm_test = create_file_lists(
        data_dir, label_dir, output_dir,
    )

    # Step 5: Validate labels
    labels_ok = validate_labels(train_files, test_files, label_dir, lm_train, lm_test)

    # Step 6: Update main benchmark config
    main_config_path = tools_dir / "benchmark_config.yaml"
    update_main_config(main_config_path)

    # Final summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  OOD config:      {ood_config_path}")
    print(f"  Train file list: {output_dir / 'train_files.txt'} ({len(train_files)} files)")
    print(f"  Test file list:  {output_dir / 'test_files.txt'} ({len(test_files)} files)")
    print(f"  Labels valid:    {'YES' if labels_ok else 'NO (see warnings above)'}")
    if not data_ready:
        print()
        print("  [ACTION REQUIRED] 1x1 micro-defect data not yet preprocessed.")
        print("  The config and train file list have been created for existing data.")
        print("  Once 1x1 .npy files are placed in the data directory, re-run this script")
        print("  to regenerate file lists, or run the benchmark directly:")
        print(f"    python tools/benchmark.py --config {ood_config_path}")
    else:
        print()
        print("  [READY] To run the 1x1 OOD benchmark:")
        print(f"    python tools/benchmark.py --config {ood_config_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
