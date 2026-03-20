#!/usr/bin/env python3
"""
Master IR Dataset Pipeline
==========================
Downloads, converts, projects, and validates all 3 IR thermography datasets
for FEM×IR fusion (Issue #3).

Usage:
    python run_ir_pipeline.py                    # Full pipeline
    python run_ir_pipeline.py --step download    # Download only
    python run_ir_pipeline.py --step convert     # Convert only
    python run_ir_pipeline.py --step project     # Project onto mesh only
    python run_ir_pipeline.py --step validate    # Validate only
    python run_ir_pipeline.py --dataset zenodo   # Single dataset
"""

import os
import sys
import subprocess
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
IR_ROOT = SCRIPT_DIR.parent
RAW_DIR = IR_ROOT / "raw"
PROCESSED_DIR = IR_ROOT / "processed"
MESH_DIR = IR_ROOT / "processed" / "mesh_projected"

DATASETS = {
    "mendeley_cfrp": {
        "raw_dir": RAW_DIR / "mendeley_cfrp_seg",
        "processed_dir": PROCESSED_DIR / "mendeley_cfrp_seg",
        "converter": "convert_mendeley_cfrp_seg",
        "url": "https://data.mendeley.com/datasets/jrsb4b9yy5/1",
        "description": "CFRP defect segmentation (1034 images + masks)",
    },
    "mendeley_composite": {
        "raw_dir": RAW_DIR / "mendeley_composite_samples",
        "processed_dir": PROCESSED_DIR / "mendeley_composite_samples",
        "converter": "convert_mendeley_composite",
        "url": "https://data.mendeley.com/datasets/v4knrwgj9y/2",
        "description": "Composite academic samples (12 sequences, CFRP+GFRP)",
    },
    "zenodo": {
        "raw_dir": RAW_DIR / "zenodo_step_heating",
        "processed_dir": PROCESSED_DIR / "zenodo_step_heating",
        "converter": "convert_zenodo_step_heating",
        "url": "https://zenodo.org/records/5426793",
        "description": "Step-heating CFRP (36 multi-band, 30ch)",
    },
}

FEM_COORDS_DIR = PROJECT_ROOT / "GNN_hole" / "GNN_hole_data"


def step_download(datasets: list):
    """Step 1: Download datasets."""
    print("\n" + "=" * 60)
    print("  STEP 1: Download Datasets")
    print("=" * 60)

    # Try Zenodo auto-download
    if "zenodo" in datasets:
        ds = DATASETS["zenodo"]
        if not (ds["raw_dir"] / ".downloaded").exists():
            print(f"\n[Zenodo] Attempting auto-download...")
            try:
                import urllib.request
                import json
                api_url = "https://zenodo.org/api/records/5426793"
                with urllib.request.urlopen(api_url, timeout=30) as resp:
                    data = json.loads(resp.read())

                os.makedirs(str(ds["raw_dir"]), exist_ok=True)
                for f in data.get("files", []):
                    fname = f["key"]
                    url = f["links"]["self"]
                    size_mb = f["size"] / 1e6
                    out_path = ds["raw_dir"] / fname
                    if out_path.exists():
                        print(f"  [SKIP] {fname}")
                        continue
                    print(f"  Downloading {fname} ({size_mb:.1f} MB)...")
                    urllib.request.urlretrieve(url, str(out_path))
                    print(f"  [OK] {fname}")
                (ds["raw_dir"] / ".downloaded").touch()
            except Exception as e:
                print(f"  [ERROR] {e}")
        else:
            print(f"  [Zenodo] Already downloaded.")

    # Mendeley requires manual download
    for key in ["mendeley_cfrp", "mendeley_composite"]:
        if key not in datasets:
            continue
        ds = DATASETS[key]
        if (ds["raw_dir"] / ".downloaded").exists():
            print(f"\n[{key}] Already downloaded.")
        else:
            print(f"\n[{key}] Manual download required:")
            print(f"  1. Open: {ds['url']}")
            print(f"  2. Click 'Download' (all files)")
            print(f"  3. Extract to: {ds['raw_dir']}/")
            print(f"  4. Run: touch {ds['raw_dir']}/.downloaded")


def step_convert(datasets: list):
    """Step 2: Convert raw datasets to IR features [H, W, 3]."""
    print("\n" + "=" * 60)
    print("  STEP 2: Convert to IR Features")
    print("=" * 60)

    sys.path.insert(0, str(SCRIPT_DIR))

    for key in datasets:
        ds = DATASETS[key]
        if not ds["raw_dir"].exists() or not (ds["raw_dir"] / ".downloaded").exists():
            print(f"\n[{key}] Raw data not found. Skipping conversion.")
            continue

        print(f"\n[{key}] Converting: {ds['description']}")
        os.makedirs(str(ds["processed_dir"]), exist_ok=True)

        try:
            if key == "mendeley_cfrp":
                from convert_mendeley_cfrp_seg import convert_mendeley_cfrp_seg
                convert_mendeley_cfrp_seg(str(ds["raw_dir"]), str(ds["processed_dir"]))
            elif key == "mendeley_composite":
                from convert_mendeley_composite import convert_mendeley_composite
                convert_mendeley_composite(str(ds["raw_dir"]), str(ds["processed_dir"]))
            elif key == "zenodo":
                from convert_zenodo_step_heating import convert_zenodo_step_heating
                convert_zenodo_step_heating(str(ds["raw_dir"]), str(ds["processed_dir"]))
        except Exception as e:
            print(f"  [ERROR] Conversion failed: {e}")
            import traceback
            traceback.print_exc()


def step_project(datasets: list):
    """Step 3: Project IR features onto FEM mesh nodes."""
    print("\n" + "=" * 60)
    print("  STEP 3: Project onto FEM Mesh")
    print("=" * 60)

    if not FEM_COORDS_DIR.exists():
        print(f"ERROR: FEM coordinates not found at {FEM_COORDS_DIR}")
        return

    sys.path.insert(0, str(SCRIPT_DIR))
    from ir_to_mesh_projector import batch_project_ir_dataset

    for key in datasets:
        ds = DATASETS[key]
        if not ds["processed_dir"].exists():
            print(f"\n[{key}] No processed data. Skipping projection.")
            continue

        ir_files = list(ds["processed_dir"].glob("*_ir.npy"))
        if not ir_files:
            print(f"\n[{key}] No IR feature files found. Skipping.")
            continue

        out_dir = MESH_DIR / key
        print(f"\n[{key}] Projecting {len(ir_files)} files onto mesh...")
        batch_project_ir_dataset(
            str(ds["processed_dir"]),
            str(out_dir),
            str(FEM_COORDS_DIR),
        )


def step_validate(datasets: list):
    """Step 4: Validate converted data."""
    print("\n" + "=" * 60)
    print("  STEP 4: Validation")
    print("=" * 60)

    # Check FEM coordinates
    coord_files = ["hole_x_2layer.npy", "hole_y_2layer.npy", "hole_z_2layer.npy"]
    fem_ok = all((FEM_COORDS_DIR / f).exists() for f in coord_files)
    print(f"\nFEM coordinates: {'OK' if fem_ok else 'MISSING'} ({FEM_COORDS_DIR})")

    if fem_ok:
        x = np.load(str(FEM_COORDS_DIR / "hole_x_2layer.npy"))
        print(f"  Mesh nodes: {len(x)}")

    total_ir_files = 0
    total_mesh_files = 0

    for key in datasets:
        ds = DATASETS[key]
        print(f"\n--- {key} ---")

        # Check raw
        raw_exists = ds["raw_dir"].exists() and (ds["raw_dir"] / ".downloaded").exists()
        print(f"  Raw data: {'OK' if raw_exists else 'NOT DOWNLOADED'}")

        # Check processed IR
        ir_files = list(ds["processed_dir"].glob("*_ir.npy")) if ds["processed_dir"].exists() else []
        print(f"  IR features: {len(ir_files)} files")
        total_ir_files += len(ir_files)

        if ir_files:
            sample = np.load(str(ir_files[0]))
            print(f"  Sample shape: {sample.shape}")
            print(f"  Value range: [{sample.min():.3f}, {sample.max():.3f}]")
            print(f"  Mean/Std: {sample.mean():.3f} / {sample.std():.3f}")

        # Check mesh-projected
        mesh_dir = MESH_DIR / key
        mesh_files = list(mesh_dir.glob("*_ir_mesh.npy")) if mesh_dir.exists() else []
        print(f"  Mesh-projected: {len(mesh_files)} files")
        total_mesh_files += len(mesh_files)

        if mesh_files:
            sample = np.load(str(mesh_files[0]))
            print(f"  Mesh sample shape: {sample.shape}")
            n_nonzero = np.count_nonzero(np.abs(sample).sum(axis=1) > 1e-6)
            print(f"  Non-zero nodes: {n_nonzero}/{len(sample)} ({100*n_nonzero/len(sample):.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total IR feature files: {total_ir_files}")
    print(f"  Total mesh-projected files: {total_mesh_files}")
    print(f"  FEM mesh: {'Ready' if fem_ok else 'Missing'}")

    if total_mesh_files > 0:
        print(f"\n  Ready for GNN training with IR fusion!")
        print(f"  Set in config: input_channels=7, ir.ir_data_dir='{MESH_DIR}'")
    elif total_ir_files > 0:
        print(f"\n  IR features ready. Run --step project to map onto mesh.")
    else:
        print(f"\n  No data converted yet. Download datasets first.")


def main():
    parser = argparse.ArgumentParser(description="IR Dataset Pipeline for FEM×IR Fusion")
    parser.add_argument("--step", choices=["download", "convert", "project", "validate", "all"],
                       default="all", help="Pipeline step to run")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"],
                       default="all", help="Which dataset to process")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"IR Thermography Dataset Pipeline")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Step: {args.step}")

    if args.step in ("download", "all"):
        step_download(datasets)
    if args.step in ("convert", "all"):
        step_convert(datasets)
    if args.step in ("project", "all"):
        step_project(datasets)
    if args.step in ("validate", "all"):
        step_validate(datasets)


if __name__ == "__main__":
    main()
