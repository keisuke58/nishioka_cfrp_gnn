#!/usr/bin/env python3
"""
Reproducible benchmark runner for CFRP GNN defect localization.

Orchestrates a matrix of (split_type x seed) training runs via launch_run.py,
collects standardized metrics, and produces comparison tables.

Usage:
    # Full benchmark from config
    python tools/benchmark.py --config tools/benchmark_config.yaml

    # Single split type with custom seeds
    python tools/benchmark.py --split_types iid --seeds 42 84 126

    # Re-aggregate existing runs (skip training)
    python tools/benchmark.py --collect_only --run_dirs runs/bench_*

    # Dry run (print commands without executing)
    python tools/benchmark.py --config tools/benchmark_config.yaml --dry_run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Project root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.compare_runs import load_run_metrics, create_comparison_table


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_benchmark_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load benchmark configuration from YAML file or return defaults."""
    defaults = {
        "name": "cfrp_gnn_benchmark",
        "seeds": [42],
        "split_types": [{"type": "iid"}],
        "models": [{
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
        }],
        "hardware": {"nproc_per_node": 4},
        "data": {
            "data_dir": "GNN_hole_2026/all_sub_hole_defect_zscore",
            "label_dir": "GNN_hole_2026/all_19class_label",
            "num_classes": 19,
        },
        "output": {"base_dir": "runs", "prefix": "bench"},
    }

    if config_path is None:
        return defaults

    path = Path(config_path)
    if not path.exists():
        print(f"[WARN] Config file not found: {path}, using defaults")
        return defaults

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = raw.get("benchmark", raw)
    # Merge with defaults (shallow)
    for key, val in defaults.items():
        cfg.setdefault(key, val)
    return cfg


# ---------------------------------------------------------------------------
# Run matrix generation
# ---------------------------------------------------------------------------

def generate_run_matrix(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand config into a list of individual run specifications."""
    seeds = config.get("seeds", [42])
    split_types = config.get("split_types", [{"type": "iid"}])
    models = config.get("models", [{}])
    prefix = config.get("output", {}).get("prefix", "bench")

    matrix = []
    for model_cfg in models:
        model_name = model_cfg.get("name", "default")
        model_args = model_cfg.get("args", {})
        for split_cfg in split_types:
            split_type = split_cfg["type"]
            split_params = split_cfg.get("params", {})
            for seed in seeds:
                run_id = f"{prefix}_{split_type}_s{seed}_{model_name}"
                matrix.append({
                    "run_id": run_id,
                    "seed": seed,
                    "split_type": split_type,
                    "split_params": split_params,
                    "model_name": model_name,
                    "model_args": model_args,
                })
    return matrix


# ---------------------------------------------------------------------------
# Split manifest creation
# ---------------------------------------------------------------------------

def create_split_manifest(
    split_type: str,
    seed: int,
    split_params: Dict[str, Any],
    data_config: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Create a split manifest JSON using gnn_common.data_utils.create_ood_split."""
    from gnn_common.data_utils import create_ood_split, create_data_label_pairs

    data_dir = str(REPO_ROOT / data_config["data_dir"])
    label_dir = str(REPO_ROOT / data_config["label_dir"])

    # Gather all data-label pairs from the data directories
    all_pairs_with_folders = []

    # Check for subdirectory structure (train/val/test) or flat
    data_path = Path(data_dir)
    subdirs = [d for d in data_path.iterdir() if d.is_dir() and d.name in ("train", "val", "test")]

    if subdirs:
        # Data is already in subdirs — gather all files across them
        for subdir in sorted(data_path.iterdir()):
            if not subdir.is_dir():
                continue
            files = sorted(f.name for f in subdir.iterdir() if f.suffix == ".npy")
            for fname in files:
                base = os.path.splitext(fname)[0]
                label_fname = f"{base}_19label.npy"
                label_path = Path(label_dir) / label_fname
                if label_path.exists():
                    all_pairs_with_folders.append((fname, label_fname, str(subdir)))
    else:
        # Flat directory
        files = sorted(f.name for f in data_path.iterdir() if f.suffix == ".npy")
        for fname in files:
            base = os.path.splitext(fname)[0]
            label_fname = f"{base}_19label.npy"
            label_path = Path(label_dir) / label_fname
            if label_path.exists():
                all_pairs_with_folders.append((fname, label_fname, data_dir))

    if not all_pairs_with_folders:
        raise RuntimeError(f"No data-label pairs found in {data_dir}")

    # Use create_ood_split for the split
    pairs = [(p[0], p[1]) for p in all_pairs_with_folders]
    folder_map = {p[0]: p[2] for p in all_pairs_with_folders}

    # OOD splits require label_data_folder
    ood_kwargs = dict(split_params)
    if split_type != "iid":
        ood_kwargs["label_data_folder"] = str(REPO_ROOT / data_config["label_dir"])

    train_pairs, val_pairs, test_pairs = create_ood_split(
        pairs=pairs,
        split_type=split_type,
        seed=seed,
        **ood_kwargs,
    )

    # Build manifest with folder info
    manifest = {
        "split_type": split_type,
        "seed": seed,
        "split_params": split_params,
        "train": [[p[0], p[1], folder_map.get(p[0], "")] for p in train_pairs],
        "val": [[p[0], p[1], folder_map.get(p[0], "")] for p in val_pairs],
        "test": [[p[0], p[1], folder_map.get(p[0], "")] for p in test_pairs],
    }

    manifest_path = output_dir / f"split_manifest_{split_type}_s{seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[BENCHMARK] Created split manifest: {manifest_path}")
    print(f"  train={len(manifest['train'])}, val={len(manifest['val'])}, test={len(manifest['test'])}")
    return manifest_path


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------

def execute_single_run(
    run_spec: Dict[str, Any],
    config: Dict[str, Any],
    output_base: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a single benchmark run via launch_run.py."""
    run_id = run_spec["run_id"]
    model_args = run_spec["model_args"]
    hardware = config.get("hardware", {})
    nproc = hardware.get("nproc_per_node", 4)

    # Create split manifest
    manifest_dir = output_base / "manifests"
    data_config = config.get("data", {})

    try:
        manifest_path = create_split_manifest(
            split_type=run_spec["split_type"],
            seed=run_spec["seed"],
            split_params=run_spec["split_params"],
            data_config=data_config,
            output_dir=manifest_dir,
        )
    except Exception as e:
        print(f"[ERROR] Failed to create split manifest for {run_id}: {e}")
        return {"run_id": run_id, "status": "manifest_error", "error": str(e)}

    # Build training args
    training_args = ["--"]
    for key, val in model_args.items():
        arg_key = f"--{key}"
        if isinstance(val, bool):
            if val:
                training_args.append(arg_key)
        else:
            training_args.extend([arg_key, str(val)])

    # Add split manifest
    training_args.extend(["--split_manifest", str(manifest_path)])

    # Build launch_run.py command
    launcher = str(REPO_ROOT / "tools" / "launch_run.py")
    cmd = [
        sys.executable, launcher,
        "--run_id", run_id,
        "--output_base", str(output_base),
        "--nproc_per_node", str(nproc),
    ]
    if dry_run:
        cmd.append("--dry_run")
    cmd.extend(training_args)

    run_dir = output_base / run_id

    print(f"\n{'='*60}")
    print(f"[BENCHMARK] {'DRY RUN: ' if dry_run else ''}Starting run: {run_id}")
    print(f"  split_type={run_spec['split_type']}, seed={run_spec['seed']}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        # Still execute launch_run.py in dry_run mode to show the command
        subprocess.run(cmd, cwd=str(REPO_ROOT))
        return {"run_id": run_id, "run_dir": str(run_dir), "status": "dry_run"}

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "exit_code": result.returncode,
        "status": "completed" if result.returncode == 0 else "failed",
        "split_type": run_spec["split_type"],
        "seed": run_spec["seed"],
        "model_name": run_spec["model_name"],
    }


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect_run_metrics(run_dir: Path) -> Dict[str, Any]:
    """Load metrics from a completed run directory.

    Reuses compare_runs.load_run_metrics() and augments with benchmark metadata.
    """
    metrics = load_run_metrics(run_dir)

    # Try to load the split manifest metadata
    meta_dir = run_dir / "meta"
    run_config_path = meta_dir / "run_config.json"
    if run_config_path.exists():
        with open(run_config_path) as f:
            run_config = json.load(f)
        metrics["run_config"] = run_config

    return metrics


def aggregate_results(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested metric dicts into a single DataFrame."""
    rows = []
    for result in all_results:
        run_info = result.get("run_info", {})
        metrics = result.get("metrics", {})

        summary = metrics.get("summary", {})
        test_metrics = metrics.get("metrics", {}).get("test", {})
        best_metrics = metrics.get("metrics", {}).get("best", {})

        row = {
            "run_id": run_info.get("run_id", metrics.get("run_id", "unknown")),
            "split_type": run_info.get("split_type", ""),
            "seed": run_info.get("seed", ""),
            "model_name": run_info.get("model_name", ""),
            "exit_code": summary.get("exit_code"),
            "best_epoch": best_metrics.get("epoch"),
            "best_macro_f1": best_metrics.get("macro_f1"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_weighted_f1": test_metrics.get("weighted_f1"),
            "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "test_mcc": test_metrics.get("mcc"),
            "test_macro_precision": test_metrics.get("macro_precision"),
            "test_macro_recall": test_metrics.get("macro_recall"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by split_type, then seed
    if not df.empty:
        df = df.sort_values(["split_type", "seed", "model_name"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_comparison_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Write benchmark results as JSON, CSV, and human-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[BENCHMARK] Results CSV: {csv_path}")

    # JSON
    json_path = output_dir / "benchmark_results.json"
    records = df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(
            {
                "generated_at": _dt.datetime.now().isoformat(),
                "num_runs": len(records),
                "results": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    print(f"[BENCHMARK] Results JSON: {json_path}")

    # Human-readable summary
    summary_path = output_dir / "benchmark_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CFRP GNN Benchmark Summary\n")
        f.write(f"Generated: {_dt.datetime.now().isoformat()}\n")
        f.write(f"Total runs: {len(df)}\n")
        f.write("=" * 70 + "\n\n")

        if df.empty:
            f.write("No results to report.\n")
        else:
            # Per split-type summary
            numeric_cols = ["test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "test_mcc"]
            for split_type in df["split_type"].unique():
                subset = df[df["split_type"] == split_type]
                f.write(f"\n--- Split: {split_type} ({len(subset)} runs) ---\n")

                for col in numeric_cols:
                    vals = subset[col].dropna()
                    if len(vals) > 0:
                        f.write(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                                f"min={vals.min():.4f}, max={vals.max():.4f}\n")

            f.write("\n\n--- Full Results Table ---\n")
            f.write(df.to_string(index=False))
            f.write("\n")

    print(f"[BENCHMARK] Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible benchmark runner for CFRP GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to benchmark YAML config file")
    parser.add_argument("--split_types", nargs="+", default=None,
                        help="Override split types (e.g., iid defect_size layer)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seeds (e.g., 42 84 126)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for benchmark results")
    parser.add_argument("--collect_only", action="store_true",
                        help="Skip training, just collect metrics from existing runs")
    parser.add_argument("--run_dirs", nargs="+", default=None,
                        help="Explicit run directories for --collect_only mode")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing training")
    parser.add_argument("--nproc_per_node", type=int, default=None,
                        help="Override GPU count")
    args = parser.parse_args()

    # Load config
    config = load_benchmark_config(args.config)

    # CLI overrides
    if args.split_types:
        config["split_types"] = [{"type": st} for st in args.split_types]
    if args.seeds:
        config["seeds"] = args.seeds
    if args.epochs:
        for model in config.get("models", []):
            model.setdefault("args", {})["epochs"] = args.epochs
    if args.nproc_per_node:
        config.setdefault("hardware", {})["nproc_per_node"] = args.nproc_per_node

    # Output directory
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_name = config.get("output", {}).get("base_dir", "runs")
    output_base = Path(args.output_dir) if args.output_dir else REPO_ROOT / output_base_name / f"benchmark_{ts}"
    output_base.mkdir(parents=True, exist_ok=True)

    # Save config used
    config_out = output_base / "benchmark_config_used.json"
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    if args.collect_only:
        # Collect-only mode
        if not args.run_dirs:
            print("[ERROR] --collect_only requires --run_dirs")
            return 1

        all_results = []
        for rd in args.run_dirs:
            run_dir = Path(rd)
            if not run_dir.exists():
                print(f"[WARN] Run dir not found: {run_dir}")
                continue
            metrics = collect_run_metrics(run_dir)
            # Try to infer run info from run_id
            run_id = run_dir.name
            all_results.append({
                "run_info": {"run_id": run_id},
                "metrics": metrics,
            })

        df = aggregate_results(all_results)
        generate_comparison_report(df, output_base)
        return 0

    # Generate and execute run matrix
    matrix = generate_run_matrix(config)

    print(f"\n[BENCHMARK] Configuration: {config.get('name', 'unnamed')}")
    print(f"[BENCHMARK] Total runs planned: {len(matrix)}")
    print(f"[BENCHMARK] Output directory: {output_base}")
    for i, spec in enumerate(matrix):
        print(f"  [{i+1}] {spec['run_id']} (split={spec['split_type']}, seed={spec['seed']})")
    print()

    # Execute runs
    all_results = []
    for i, run_spec in enumerate(matrix):
        print(f"\n[BENCHMARK] Run {i+1}/{len(matrix)}")
        run_result = execute_single_run(
            run_spec=run_spec,
            config=config,
            output_base=output_base,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            all_results.append({"run_info": run_spec, "metrics": {}})
            continue

        # Collect metrics from completed run
        run_dir = Path(run_result.get("run_dir", ""))
        if run_dir.exists():
            metrics = collect_run_metrics(run_dir)
        else:
            metrics = {}

        all_results.append({
            "run_info": {**run_spec, **run_result},
            "metrics": metrics,
        })

    # Aggregate and report
    if not args.dry_run:
        df = aggregate_results(all_results)
        generate_comparison_report(df, output_base)
    else:
        print(f"\n[DRY RUN] Would write results to: {output_base}")

    print(f"\n[BENCHMARK] Done. Output: {output_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
