#!/usr/bin/env python3
"""
Fine-tuning strategy comparison for CFRP GNN.

Runs 3 strategies (freeze_backbone, differential_lr, higher_lr) × N seeds,
collects metrics, and produces a comparison table.

Usage:
    python tools/finetune_comparison.py \
        --pretrained_from runs/.../best_model.pth \
        --seeds 42 84 126

    python tools/finetune_comparison.py --pretrained_from runs/.../best_model.pth --dry_run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.compare_runs import load_run_metrics
from tools.benchmark import aggregate_results, generate_comparison_report

# Default fine-tuning strategies
DEFAULT_STRATEGIES = [
    {
        "name": "freeze_backbone",
        "args": {
            "freeze_backbone": True,
            "head_lr": 0.001,
            "epochs": 500,
            "patience": 150,
            "use_amp": True,
            "use_class_frequency_sampler": True,
        },
    },
    {
        "name": "differential_lr",
        "args": {
            "backbone_lr": 0.0001,
            "head_lr": 0.001,
            "learning_rate": 0.001,
            "epochs": 1000,
            "patience": 200,
            "use_amp": True,
            "use_class_frequency_sampler": True,
        },
    },
    {
        "name": "higher_lr",
        "args": {
            "learning_rate": 0.0005,
            "epochs": 1000,
            "patience": 200,
            "use_amp": True,
            "use_class_frequency_sampler": True,
        },
    },
]


def load_finetune_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load fine-tune comparison config from YAML."""
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f)
            return raw.get("finetune", raw)

    return {
        "strategies": DEFAULT_STRATEGIES,
        "seeds": [42, 84, 126],
    }


def execute_finetune_run(
    strategy: Dict[str, Any],
    seed: int,
    pretrained_from: str,
    output_base: Path,
    nproc: int = 4,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a single fine-tuning run."""
    name = strategy["name"]
    args = strategy["args"]
    run_id = f"ft_{name}_s{seed}"

    launcher = str(REPO_ROOT / "tools" / "launch_run.py")
    cmd = [
        sys.executable, launcher,
        "--run_id", run_id,
        "--output_base", str(output_base),
        "--nproc_per_node", str(nproc),
    ]
    if dry_run:
        cmd.append("--dry_run")

    # Build training args
    cmd.append("--")
    cmd.extend(["--fine_tune_from", pretrained_from])

    for key, val in args.items():
        arg_key = f"--{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(arg_key)
        else:
            cmd.extend([arg_key, str(val)])

    run_dir = output_base / run_id

    print(f"\n{'='*50}")
    print(f"[FT] {'DRY RUN: ' if dry_run else ''}Strategy: {name}, Seed: {seed}")
    print(f"  cmd: {' '.join(cmd[:12])}...")
    print(f"{'='*50}")

    if dry_run:
        subprocess.run(cmd, cwd=str(REPO_ROOT))
        return {"run_id": run_id, "status": "dry_run", "strategy": name, "seed": seed}

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "exit_code": result.returncode,
        "status": "completed" if result.returncode == 0 else "failed",
        "strategy": name,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning strategy comparison")
    parser.add_argument("--pretrained_from", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (uses finetune section from hpo_config.yaml)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seeds")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--nproc_per_node", type=int, default=4, help="GPUs")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    # Validate pretrained model path
    model_path = Path(args.pretrained_from)
    if not model_path.exists() and not args.dry_run:
        print(f"[ERROR] Pretrained model not found: {model_path}")
        return 1

    config = load_finetune_config(args.config)
    strategies = config.get("strategies", DEFAULT_STRATEGIES)
    seeds = args.seeds or config.get("seeds", [42, 84, 126])

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) if args.output_dir else REPO_ROOT / "runs" / f"ft_comparison_{ts}"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"[FT] Comparing {len(strategies)} strategies × {len(seeds)} seeds = {len(strategies)*len(seeds)} runs")
    print(f"[FT] Pretrained from: {args.pretrained_from}")
    print(f"[FT] Output: {output_base}")

    # Execute all runs
    all_results = []
    for strategy in strategies:
        for seed in seeds:
            run_result = execute_finetune_run(
                strategy=strategy,
                seed=seed,
                pretrained_from=args.pretrained_from,
                output_base=output_base,
                nproc=args.nproc_per_node,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                all_results.append({"run_info": run_result, "metrics": {}})
                continue

            run_dir = Path(run_result.get("run_dir", ""))
            metrics = load_run_metrics(run_dir) if run_dir.exists() else {}
            all_results.append({
                "run_info": {
                    "run_id": run_result["run_id"],
                    "split_type": run_result["strategy"],
                    "seed": run_result["seed"],
                    "model_name": run_result["strategy"],
                },
                "metrics": metrics,
            })

    if not args.dry_run:
        df = aggregate_results(all_results)
        generate_comparison_report(df, output_base)

        # Additional strategy-level summary
        if not df.empty and "split_type" in df.columns:
            print("\n--- Strategy Summary (mean ± std) ---")
            for strategy in df["split_type"].unique():
                subset = df[df["split_type"] == strategy]
                f1_vals = subset["test_macro_f1"].dropna()
                if len(f1_vals) > 0:
                    print(f"  {strategy}: test_macro_f1 = {f1_vals.mean():.4f} ± {f1_vals.std():.4f}")
    else:
        print(f"\n[DRY RUN] Would run {len(strategies)*len(seeds)} fine-tuning runs")

    print(f"\n[FT] Done. Output: {output_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
