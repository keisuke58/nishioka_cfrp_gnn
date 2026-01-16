#!/usr/bin/env python3
"""
One-command training launcher.

Design goal:
  - User runs a single shell script.
  - This launcher creates a run directory, runs preflight checks,
    starts torchrun training, tees logs, and writes a machine-readable summary.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RE_BEST_MACRO_F1 = re.compile(r"Best Macro F1:\s*([0-9]*\.?[0-9]+)")
RE_FINAL_MODEL = re.compile(r"\[FINAL_INFO\]\s+Final model saved to\s+(.+)$")
RE_OUTPUT_DIR = re.compile(r"\[INFO\]\s+Output directory:\s+(.+)$")
RE_OUTPUT_DIR_RENAMED = re.compile(r"\[INFO\]\s+Output directory renamed to include Macro F1 score:\s+(.+)$")

# Metrics parsing (robust to optional logger prefixes)
RE_LOGGER_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2} [0-9:]{2,8}\]\[[A-Z]+\]\s*")
RE_RANK_PREFIX = re.compile(r"^\[rank\d+\]:")

RE_LAUNCH_CMD = re.compile(r"^\[LAUNCH\]\s+cmd:\s+(.+)$")
RE_CMD_RUN_ID = re.compile(r"(?:^|\s)--run_id\s+(\S+)")

RE_EPOCH_HEADER = re.compile(r"^=== Epoch (\d+)/(\d+) ===$")
RE_EPOCH_INLINE = re.compile(r"^=== Epoch (\d+)/(\d+) ===\s*$")
RE_EPOCH_METRICS = re.compile(
    r"^Train Loss:\s*([0-9]*\.?[0-9]+),\s*Val Loss:\s*([0-9]*\.?[0-9]+),\s*Val Acc:\s*([0-9]*\.?[0-9]+),\s*Macro F1:\s*([0-9]*\.?[0-9]+)"
)
RE_ELAPSED = re.compile(r"^Elapsed Time:\s*([0-9]{2}:[0-9]{2}:[0-9]{2})")
RE_EST_REMAIN = re.compile(r"^Estimated Remaining Time:\s*([0-9]{2}:[0-9]{2}:[0-9]{2}).*\(Avg\s*([0-9]*\.?[0-9]+)s/epoch\)")

RE_BEST_VAL_LOSS_AND_F1 = re.compile(r"^Best Val Loss:\s*([0-9]*\.?[0-9]+),\s*Best Macro F1:\s*([0-9]*\.?[0-9]+)")
RE_BEST_EPOCH = re.compile(r"^Best model was at epoch\s+(\d+)")

RE_TEST_LOSS = re.compile(r"^-+\s*Test Loss:\s*([0-9]*\.?[0-9]+)")
RE_TEST_ACC = re.compile(r"^-+\s*Test Accuracy:\s*([0-9]*\.?[0-9]+)")
RE_TEST_WEIGHTED = re.compile(
    r"^-+\s*Weighted Precision:\s*([0-9]*\.?[0-9]+),\s*Weighted Recall:\s*([0-9]*\.?[0-9]+),\s*Weighted F1-Score:\s*([0-9]*\.?[0-9]+)"
)
RE_TEST_MACRO = re.compile(
    r"^-+\s*Macro Precision:\s*([0-9]*\.?[0-9]+),\s*Macro Recall:\s*([0-9]*\.?[0-9]+),\s*Macro F1-Score:\s*([0-9]*\.?[0-9]+)"
)
RE_TEST_BALANCED_ACC = re.compile(r"^-+\s*Balanced Accuracy:\s*([0-9]*\.?[0-9]+)")
RE_TEST_MCC = re.compile(r"^-+\s*Matthews Correlation Coefficient:\s*([0-9]*\.?[0-9]+)")


def _now_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_cmd_capture(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return p.returncode, p.stdout
    except FileNotFoundError:
        return 127, f"Command not found: {cmd[0]}"


def _git_short_sha(repo_root: Path) -> Optional[str]:
    code, out = _run_cmd_capture(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
    if code != 0:
        return None
    return out.strip() or None


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] {label} not found: {path}")


def _tee_process(cmd: List[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[LAUNCH] cmd: {' '.join(cmd)}\n")
        f.write(f"[LAUNCH] cwd: {cwd}\n")
        f.write(f"[LAUNCH] time: {_dt.datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert p.stdout is not None

        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        p.wait()
        return int(p.returncode or 0)


def _parse_summary_from_log(log_path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "best_macro_f1": None,
        "final_model_path": None,
        "predict_output_dir": None,
    }
    if not log_path.exists():
        return summary

    best_f1: Optional[float] = None
    final_model: Optional[str] = None
    output_dir: Optional[str] = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = RE_BEST_MACRO_F1.search(line)
            if m:
                try:
                    best_f1 = float(m.group(1))
                except ValueError:
                    pass

            m = RE_FINAL_MODEL.search(line)
            if m:
                final_model = m.group(1).strip()

            m = RE_OUTPUT_DIR_RENAMED.search(line)
            if m:
                output_dir = m.group(1).strip()
            else:
                m = RE_OUTPUT_DIR.search(line)
                if m:
                    output_dir = m.group(1).strip()

    summary["best_macro_f1"] = best_f1
    summary["final_model_path"] = final_model
    summary["predict_output_dir"] = output_dir
    return summary


def _normalize_log_line(raw: str) -> str:
    """Remove optional prefixes so regexes can match consistently."""
    line = raw.rstrip("\n")
    line = RE_LOGGER_PREFIX.sub("", line)
    line = RE_RANK_PREFIX.sub("", line)
    return line.strip()


def _parse_metrics_from_log(log_path: Path) -> Dict[str, Any]:
    """Parse per-epoch + final test metrics from train.log.

    Output shape is designed to be stable and easy to consume.
    """
    out: Dict[str, Any] = {
        "generated_at": _dt.datetime.now().isoformat(),
        "source_log": str(log_path),
        "run_id": None,
        "epochs": [],
        "best": {"epoch": None, "val_loss": None, "macro_f1": None},
        "test": {
            "loss": None,
            "accuracy": None,
            "weighted_precision": None,
            "weighted_recall": None,
            "weighted_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "balanced_accuracy": None,
            "mcc": None,
        },
    }
    if not log_path.exists():
        return out

    per_epoch: Dict[int, Dict[str, Any]] = {}
    cur_epoch: Optional[int] = None
    cur_total: Optional[int] = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = _normalize_log_line(raw)
            if not line:
                continue

            m = RE_LAUNCH_CMD.match(line)
            if m and out["run_id"] is None:
                cmd = m.group(1)
                m2 = RE_CMD_RUN_ID.search(cmd)
                if m2:
                    out["run_id"] = m2.group(1)

            m = RE_EPOCH_HEADER.match(line) or RE_EPOCH_INLINE.match(line)
            if m:
                try:
                    cur_epoch = int(m.group(1))
                    cur_total = int(m.group(2))
                except ValueError:
                    cur_epoch = None
                    cur_total = None
                continue

            m = RE_EPOCH_METRICS.match(line)
            if m and cur_epoch is not None:
                try:
                    train_loss = float(m.group(1))
                    val_loss = float(m.group(2))
                    val_acc = float(m.group(3))
                    macro_f1 = float(m.group(4))
                except ValueError:
                    continue

                per_epoch[cur_epoch] = {
                    "epoch": cur_epoch,
                    "total_epochs": cur_total,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "macro_f1": macro_f1,
                }
                continue

            m = RE_ELAPSED.match(line)
            if m and cur_epoch is not None and cur_epoch in per_epoch:
                per_epoch[cur_epoch]["elapsed_time"] = m.group(1)
                continue

            m = RE_EST_REMAIN.match(line)
            if m and cur_epoch is not None and cur_epoch in per_epoch:
                per_epoch[cur_epoch]["estimated_remaining_time"] = m.group(1)
                try:
                    per_epoch[cur_epoch]["avg_sec_per_epoch"] = float(m.group(2))
                except ValueError:
                    pass
                continue

            m = RE_BEST_VAL_LOSS_AND_F1.match(line)
            if m:
                try:
                    out["best"]["val_loss"] = float(m.group(1))
                    out["best"]["macro_f1"] = float(m.group(2))
                except ValueError:
                    pass
                continue

            m = RE_BEST_EPOCH.match(line)
            if m:
                try:
                    out["best"]["epoch"] = int(m.group(1))
                except ValueError:
                    pass
                continue

            m = RE_TEST_LOSS.match(line)
            if m:
                try:
                    out["test"]["loss"] = float(m.group(1))
                except ValueError:
                    pass
                continue

            m = RE_TEST_ACC.match(line)
            if m:
                try:
                    out["test"]["accuracy"] = float(m.group(1))
                except ValueError:
                    pass
                continue

            m = RE_TEST_WEIGHTED.match(line)
            if m:
                try:
                    out["test"]["weighted_precision"] = float(m.group(1))
                    out["test"]["weighted_recall"] = float(m.group(2))
                    out["test"]["weighted_f1"] = float(m.group(3))
                except ValueError:
                    pass
                continue

            m = RE_TEST_MACRO.match(line)
            if m:
                try:
                    out["test"]["macro_precision"] = float(m.group(1))
                    out["test"]["macro_recall"] = float(m.group(2))
                    out["test"]["macro_f1"] = float(m.group(3))
                except ValueError:
                    pass
                continue

            m = RE_TEST_BALANCED_ACC.match(line)
            if m:
                try:
                    out["test"]["balanced_accuracy"] = float(m.group(1))
                except ValueError:
                    pass
                continue

            m = RE_TEST_MCC.match(line)
            if m:
                try:
                    out["test"]["mcc"] = float(m.group(1))
                except ValueError:
                    pass
                continue

    out["epochs"] = [per_epoch[k] for k in sorted(per_epoch.keys())]
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Fully-automatic GNN training launcher")
    parser.add_argument(
        "--metrics_from_log",
        type=str,
        default=None,
        help="Parse an existing train.log and write metrics.json, then exit",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="Output path for metrics.json (default: next to train.log)",
    )
    parser.add_argument("--profile", type=str, default="ndf_recommended", help="Training profile name")
    parser.add_argument("--script", type=str, default=None, help="Override training script path")
    parser.add_argument("--workdir", type=str, default=str(repo_root / "GNN_hole_2026" / "GNN_program"), help="Working directory for training")

    parser.add_argument("--torchrun", type=str, default="/home/nishioka/miniconda3/envs/gnn_final_env/bin/torchrun", help="torchrun executable path")
    parser.add_argument("--nproc_per_node", type=int, default=int(os.environ.get("NPROC_PER_NODE", "4")), help="GPUs per node (torchrun)")

    parser.add_argument("--output_base", type=str, default=str(repo_root / "runs"), help="Base directory for runs")
    parser.add_argument("--run_id", type=str, default=None, help="Run identifier (if omitted, auto-generated)")
    parser.add_argument("--resume", type=str, default="auto", choices=["auto", "off"], help="Resume behavior when run dir exists")
    parser.add_argument("--preflight_only", action="store_true", help="Run preflight checks only")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing training")

    # Pass-through args for the training script
    parser.add_argument("training_args", nargs=argparse.REMAINDER, help="Args after `--` are passed to training script")
    args = parser.parse_args()

    if args.metrics_from_log:
        log_path = Path(args.metrics_from_log).expanduser().resolve()
        _ensure_exists(log_path, "train log")
        out_path = (
            Path(args.metrics_out).expanduser().resolve()
            if args.metrics_out
            else (log_path.parent / "metrics.json")
        )
        metrics = _parse_metrics_from_log(log_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote metrics: {out_path}")
        return 0

    profiles: Dict[str, Dict[str, Any]] = {
        "ndf_recommended": {
            "default_script": str(repo_root / "GNN_hole_2026" / "GNN_program" / "GNN_zscore_sub_noise_defect_free.py"),
        },
    }
    if args.profile not in profiles:
        raise SystemExit(f"[FATAL] Unknown profile: {args.profile}. Available: {sorted(profiles.keys())}")

    script_path = Path(args.script or profiles[args.profile]["default_script"]).resolve()
    workdir = Path(args.workdir).resolve()
    torchrun = Path(args.torchrun).resolve()
    output_base = Path(args.output_base).resolve()

    _ensure_exists(script_path, "training script")
    _ensure_exists(workdir, "workdir")
    _ensure_exists(torchrun, "torchrun")

    sha = _git_short_sha(repo_root) or "nogit"
    run_id = args.run_id or f"{_now_ts()}_{sha}_{args.profile}"

    run_dir = output_base / run_id
    meta_dir = run_dir / "meta"
    logs_dir = run_dir / "logs"
    output_root = run_dir / "outputs"

    run_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    # Basic environment snapshot
    env_snapshot: Dict[str, str] = {}
    for k in [
        "NPROC_PER_NODE",
        "OMP_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "TORCH_DISTRIBUTED_DEBUG",
        "NCCL_DEBUG",
        "NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_TIMEOUT",
    ]:
        if k in os.environ:
            env_snapshot[k] = os.environ[k]

    config = {
        "run_id": run_id,
        "profile": args.profile,
        "repo_root": str(repo_root),
        "git_short_sha": sha,
        "script": str(script_path),
        "workdir": str(workdir),
        "torchrun": str(torchrun),
        "nproc_per_node": int(args.nproc_per_node),
        "output_base": str(output_base),
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "training_args": args.training_args,
        "env": env_snapshot,
        "created_at": _dt.datetime.now().isoformat(),
    }
    (meta_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Preflight: optional file matching check (non-fatal, but recorded)
    preflight_log = logs_dir / "preflight_file_matching.log"
    verify_script = repo_root / "GNN_hole_2026" / "GNN_program" / "verify_file_matching.py"
    if verify_script.exists():
        data_dir = repo_root / "GNN_hole_2026" / "all_sub_hole_defect_zscore_noise" / "train"
        label_dir = repo_root / "GNN_hole_2026" / "all_19class_label"
        if data_dir.exists() and label_dir.exists():
            cmd = [sys.executable, str(verify_script), "--data_dir", str(data_dir), "--label_dir", str(label_dir)]
            code = _tee_process(cmd, cwd=repo_root, log_path=preflight_log)
            (meta_dir / "preflight.json").write_text(
                json.dumps({"verify_file_matching_exit_code": code, "data_dir": str(data_dir), "label_dir": str(label_dir)}, indent=2) + "\n",
                encoding="utf-8",
            )

    if args.preflight_only:
        print(f"[INFO] Preflight done. Run dir: {run_dir}")
        return 0

    # Resume: if run dir exists and checkpoints exist, pick newest *_Latest.pth
    resume_from: Optional[str] = None
    if args.resume == "auto":
        ckpt_dir = output_root / "GNN_model" / "19classmodel_hole_zscore"
        if ckpt_dir.exists():
            latest = sorted(ckpt_dir.glob("*_Latest.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if latest:
                resume_from = str(latest[0])

    # Build training command
    passthrough = args.training_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    train_cmd: List[str] = [
        str(torchrun),
        "--standalone",
        f"--nproc_per_node={int(args.nproc_per_node)}",
        str(script_path),
        "--output_root",
        str(output_root),
        "--run_id",
        run_id,
    ]
    if resume_from is not None:
        train_cmd += ["--resume_from", resume_from]
    train_cmd += passthrough

    log_path = logs_dir / "train.log"
    (meta_dir / "train_cmd.txt").write_text(" ".join(train_cmd) + "\n", encoding="utf-8")

    if args.dry_run:
        print("[DRY RUN] train cmd:")
        print("  " + " ".join(train_cmd))
        print(f"[DRY RUN] run dir: {run_dir}")
        return 0

    start_time = _dt.datetime.now().isoformat()
    exit_code = _tee_process(train_cmd, cwd=workdir, log_path=log_path)
    end_time = _dt.datetime.now().isoformat()

    parsed = _parse_summary_from_log(log_path)
    metrics = _parse_metrics_from_log(log_path)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "run_id": run_id,
        "profile": args.profile,
        "exit_code": exit_code,
        "start_time": start_time,
        "end_time": end_time,
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        **parsed,
    }
    (meta_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n[INFO] Finished (exit_code={exit_code})")
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Summary: {meta_dir / 'summary.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

