#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization for CFRP GNN.

Wraps launch_run.py to run training trials with Optuna-sampled hyperparameters.
Supports SQLite storage for pause/resume and built-in visualization.

Usage:
    # Basic HPO (20 trials)
    python tools/optuna_hpo.py --n_trials 20 --study_name cfrp_hpo_v1

    # Quick exploration (reduced epochs)
    python tools/optuna_hpo.py --n_trials 50 --epochs 200 --patience 50

    # Fine-tuning mode
    python tools/optuna_hpo.py --mode finetune --pretrained_from runs/.../best.pth

    # Resume existing study
    python tools/optuna_hpo.py --study_name cfrp_hpo_v1 --n_trials 10

    # Dry run
    python tools/optuna_hpo.py --n_trials 3 --dry_run

    # Export results from existing study
    python tools/optuna_hpo.py --export_only --study_name cfrp_hpo_v1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.compare_runs import load_run_metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_hpo_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load HPO configuration from YAML or return defaults."""
    defaults = {
        "study_name": "cfrp_hpo",
        "n_trials": 20,
        "direction": "maximize",
        "metric": "macro_f1",
        "pruner": {"type": "median", "n_startup_trials": 5, "n_warmup_steps": 50},
        "sampler": {"type": "tpe", "seed": 42},
        "search_space": {
            "learning_rate": {"type": "log_float", "low": 5e-4, "high": 5e-3},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
            "hidden_channels": {"type": "categorical", "choices": [16, 32, 64]},
            "dropout": {"type": "float", "low": 0.05, "high": 0.25},
            "gamma": {"type": "float", "low": 2.0, "high": 5.0},
            "weight_decay": {"type": "log_float", "low": 1e-4, "high": 1e-3},
            "edge_drop_prob": {"type": "float", "low": 0.0, "high": 0.05},
            "use_logit_adjust": {"type": "categorical", "choices": [True, False]},
        },
        "fixed_params": {
            "epochs": 500,
            "patience": 100,
            "use_amp": True,
            "use_class_frequency_sampler": True,
            "data_usage_ratio": 1.0,
        },
        "hardware": {"nproc_per_node": 4},
        "output": {"base_dir": "runs", "study_db_dir": "runs/hpo_studies"},
    }

    if config_path is None:
        return defaults

    path = Path(config_path)
    if not path.exists():
        print(f"[WARN] Config not found: {path}, using defaults")
        return defaults

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = raw.get("hpo", raw)
    for key, val in defaults.items():
        cfg.setdefault(key, val)
    return cfg


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def sample_hyperparams(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample hyperparameters from the search space using Optuna trial."""
    params = {}
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif ptype == "log_float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif ptype == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search space type: {ptype} for param {name}")
    return params


def sample_finetune_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample fine-tuning specific hyperparameters."""
    strategy = trial.suggest_categorical("ft_strategy", ["freeze_backbone", "differential_lr", "higher_lr"])

    params = {
        "epochs": trial.suggest_categorical("epochs", [500, 1000]),
        "patience": trial.suggest_categorical("patience", [100, 150, 200]),
    }

    if strategy == "freeze_backbone":
        params["freeze_backbone"] = True
        params["head_lr"] = trial.suggest_float("head_lr", 5e-4, 5e-3, log=True)
    elif strategy == "differential_lr":
        params["backbone_lr"] = trial.suggest_float("backbone_lr", 1e-5, 1e-3, log=True)
        params["head_lr"] = trial.suggest_float("head_lr", 5e-4, 5e-3, log=True)
    else:  # higher_lr
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

    return params


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def build_training_args(params: Dict[str, Any], fixed_params: Dict[str, Any]) -> List[str]:
    """Convert param dict to training script CLI args."""
    args = ["--"]
    merged = {**fixed_params, **params}

    for key, val in merged.items():
        arg_key = f"--{key}"
        if isinstance(val, bool):
            if val:
                args.append(arg_key)
        else:
            args.extend([arg_key, str(val)])

    return args


def run_trial_training(
    run_id: str,
    training_args: List[str],
    output_base: Path,
    nproc: int,
    dry_run: bool = False,
    pretrained_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single training run and return metrics."""
    launcher = str(REPO_ROOT / "tools" / "launch_run.py")
    cmd = [
        sys.executable, launcher,
        "--run_id", run_id,
        "--output_base", str(output_base),
        "--nproc_per_node", str(nproc),
    ]
    if dry_run:
        cmd.append("--dry_run")

    # Add fine-tune source if specified
    extra_args = list(training_args)
    if pretrained_from:
        extra_args.extend(["--fine_tune_from", pretrained_from])

    cmd.extend(extra_args)

    run_dir = output_base / run_id

    print(f"\n[HPO] {'DRY RUN: ' if dry_run else ''}Trial run: {run_id}")
    print(f"  cmd: {' '.join(cmd[:10])}...")

    if dry_run:
        subprocess.run(cmd, cwd=str(REPO_ROOT))
        return {"status": "dry_run", "macro_f1": 0.0}

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    if result.returncode != 0:
        print(f"[HPO] Trial {run_id} failed (exit_code={result.returncode})")
        return {"status": "failed", "macro_f1": 0.0}

    # Collect metrics
    if run_dir.exists():
        metrics = load_run_metrics(run_dir)
        best_f1 = metrics.get("metrics", {}).get("best", {}).get("macro_f1")
        if best_f1 is not None:
            return {"status": "completed", "macro_f1": float(best_f1), "metrics": metrics}

    return {"status": "no_metrics", "macro_f1": 0.0}


def create_objective(
    config: Dict[str, Any],
    output_base: Path,
    mode: str = "train",
    pretrained_from: Optional[str] = None,
    dry_run: bool = False,
):
    """Create an Optuna objective function."""
    search_space = config.get("search_space", {})
    fixed_params = config.get("fixed_params", {})
    nproc = config.get("hardware", {}).get("nproc_per_node", 4)
    study_name = config.get("study_name", "hpo")

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        if mode == "finetune":
            params = sample_finetune_params(trial)
        else:
            params = sample_hyperparams(trial, search_space)

        # Build run ID
        run_id = f"hpo_{study_name}_t{trial.number:03d}"

        # Build training args
        training_args = build_training_args(params, fixed_params)

        # Execute training
        result = run_trial_training(
            run_id=run_id,
            training_args=training_args,
            output_base=output_base,
            nproc=nproc,
            dry_run=dry_run,
            pretrained_from=pretrained_from,
        )

        macro_f1 = result.get("macro_f1", 0.0)

        if result["status"] == "failed":
            raise optuna.TrialPruned(f"Training failed for trial {trial.number}")

        return macro_f1

    return objective


# ---------------------------------------------------------------------------
# Study management
# ---------------------------------------------------------------------------

def create_study(config: Dict[str, Any]) -> optuna.Study:
    """Create or load an Optuna study with SQLite storage."""
    study_name = config.get("study_name", "cfrp_hpo")
    direction = config.get("direction", "maximize")
    db_dir = Path(config.get("output", {}).get("study_db_dir", "runs/hpo_studies"))
    db_dir = REPO_ROOT / db_dir
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{study_name}.db"

    storage = f"sqlite:///{db_path}"

    # Pruner
    pruner_cfg = config.get("pruner", {})
    pruner_type = pruner_cfg.get("type", "median")
    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 50),
        )
    else:
        pruner = optuna.pruners.NopPruner()

    # Sampler
    sampler_cfg = config.get("sampler", {})
    sampler_type = sampler_cfg.get("type", "tpe")
    seed = sampler_cfg.get("seed", 42)
    if sampler_type == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_type == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    print(f"[HPO] Study: {study_name} (storage: {db_path})")
    print(f"[HPO] Existing trials: {len(study.trials)}")

    return study


# ---------------------------------------------------------------------------
# Export & visualization
# ---------------------------------------------------------------------------

def export_results(study: optuna.Study, output_dir: Path) -> None:
    """Export study results to CSV, JSON, and generate visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trials DataFrame
    df = study.trials_dataframe()
    csv_path = output_dir / "hpo_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"[HPO] Trials CSV: {csv_path}")

    # Best trial summary
    try:
        best = study.best_trial
        summary = {
            "study_name": study.study_name,
            "generated_at": _dt.datetime.now().isoformat(),
            "n_trials": len(study.trials),
            "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_trial": {
                "number": best.number,
                "value": best.value,
                "params": best.params,
            },
        }
    except ValueError:
        summary = {
            "study_name": study.study_name,
            "generated_at": _dt.datetime.now().isoformat(),
            "n_trials": len(study.trials),
            "best_trial": None,
        }

    json_path = output_dir / "hpo_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[HPO] Summary JSON: {json_path}")

    # Visualizations (matplotlib-based, no plotly dependency)
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_trials) < 2:
        print("[HPO] Not enough completed trials for visualization")
        return

    try:
        import matplotlib.pyplot as plt

        # 1. Optimization history
        fig, ax = plt.subplots(figsize=(10, 5))
        values = [t.value for t in complete_trials]
        numbers = [t.number for t in complete_trials]
        best_so_far = np.maximum.accumulate(values)
        ax.scatter(numbers, values, alpha=0.6, label="Trial value")
        ax.plot(numbers, best_so_far, "r-", linewidth=2, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Macro F1")
        ax.set_title("HPO Optimization History")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "hpo_optimization_history.png", dpi=150)
        plt.close(fig)
        print(f"[HPO] Plot: {output_dir / 'hpo_optimization_history.png'}")

        # 2. Parameter importance (simple correlation-based)
        if len(complete_trials) >= 5:
            param_names = list(complete_trials[0].params.keys())
            importances = {}
            for pname in param_names:
                pvals = []
                fvals = []
                for t in complete_trials:
                    if pname in t.params:
                        pv = t.params[pname]
                        if isinstance(pv, (int, float)):
                            pvals.append(float(pv))
                            fvals.append(t.value)
                if len(pvals) >= 3:
                    corr = abs(np.corrcoef(pvals, fvals)[0, 1])
                    if not np.isnan(corr):
                        importances[pname] = corr

            if importances:
                sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                names = [x[0] for x in sorted_imp]
                vals = [x[1] for x in sorted_imp]

                fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
                ax.barh(names, vals)
                ax.set_xlabel("|Correlation| with Macro F1")
                ax.set_title("Parameter Importance (correlation-based)")
                fig.tight_layout()
                fig.savefig(output_dir / "hpo_param_importance.png", dpi=150)
                plt.close(fig)
                print(f"[HPO] Plot: {output_dir / 'hpo_param_importance.png'}")

    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")

    # Human-readable summary
    txt_path = output_dir / "hpo_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"HPO Summary: {study.study_name}\n")
        f.write(f"Generated: {_dt.datetime.now().isoformat()}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Complete: {len(complete_trials)}\n")
        f.write("=" * 60 + "\n\n")

        if summary.get("best_trial"):
            bt = summary["best_trial"]
            f.write(f"Best Trial #{bt['number']}: macro_f1 = {bt['value']:.4f}\n")
            f.write("Parameters:\n")
            for k, v in bt["params"].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("--- All Trials ---\n")
        for t in sorted(complete_trials, key=lambda x: x.value, reverse=True):
            f.write(f"  Trial {t.number:3d}: macro_f1={t.value:.4f}  {t.params}\n")

    print(f"[HPO] Summary: {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for CFRP GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, default=None, help="HPO config YAML")
    parser.add_argument("--study_name", type=str, default=None, help="Override study name")
    parser.add_argument("--n_trials", type=int, default=None, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per trial")
    parser.add_argument("--patience", type=int, default=None, help="Override patience")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "finetune"],
                        help="HPO mode: train (from scratch) or finetune")
    parser.add_argument("--pretrained_from", type=str, default=None,
                        help="Path to pretrained model for finetune mode")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--nproc_per_node", type=int, default=None, help="Override GPU count")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without training")
    parser.add_argument("--export_only", action="store_true",
                        help="Export results from existing study (no new trials)")
    args = parser.parse_args()

    config = load_hpo_config(args.config)

    # CLI overrides
    if args.study_name:
        config["study_name"] = args.study_name
    if args.n_trials:
        config["n_trials"] = args.n_trials
    if args.epochs:
        config.setdefault("fixed_params", {})["epochs"] = args.epochs
    if args.patience:
        config.setdefault("fixed_params", {})["patience"] = args.patience
    if args.nproc_per_node:
        config.setdefault("hardware", {})["nproc_per_node"] = args.nproc_per_node

    # Output directory
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = config.get("study_name", "hpo")
    output_base = (
        Path(args.output_dir) if args.output_dir
        else REPO_ROOT / config.get("output", {}).get("base_dir", "runs") / f"hpo_{study_name}_{ts}"
    )
    output_base.mkdir(parents=True, exist_ok=True)

    # Save config used
    with open(output_base / "hpo_config_used.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    # Create study
    study = create_study(config)

    if args.export_only:
        export_results(study, output_base)
        return 0

    # Create objective
    objective = create_objective(
        config=config,
        output_base=output_base,
        mode=args.mode,
        pretrained_from=args.pretrained_from,
        dry_run=args.dry_run,
    )

    n_trials = config.get("n_trials", 20)

    print(f"\n[HPO] Starting optimization: {n_trials} trials")
    print(f"[HPO] Study: {study_name}")
    print(f"[HPO] Output: {output_base}")
    print(f"[HPO] Mode: {args.mode}")
    if args.pretrained_from:
        print(f"[HPO] Pretrained from: {args.pretrained_from}")
    print()

    if args.dry_run:
        # Run a few trials in dry-run mode
        for i in range(min(n_trials, 3)):
            trial = study.ask()
            val = objective(trial)
            study.tell(trial, val)
        print(f"\n[DRY RUN] Would run {n_trials} trials. Output: {output_base}")
        return 0

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Export results
    export_results(study, output_base)

    print(f"\n[HPO] Done. Output: {output_base}")
    if study.best_trial:
        print(f"[HPO] Best trial #{study.best_trial.number}: macro_f1={study.best_trial.value:.4f}")
        print(f"[HPO] Best params: {study.best_trial.params}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
