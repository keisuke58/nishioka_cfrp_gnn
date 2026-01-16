from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(_read_text(path))


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _parse_dataset_info_txt(path: Path) -> Dict[str, Any]:
    # Example:
    # === Dataset Information ===
    # Dataset Type: Noise Defect-Free
    # Train pairs: 7000
    # Val pairs: 1500
    # Test pairs: 1500
    # Total pairs: 10000
    out: Dict[str, Any] = {"path": str(path)}
    if not path.exists():
        return out

    for raw in _read_text(path).splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        k, v = [s.strip() for s in line.split(":", 1)]
        key = k.lower().replace(" ", "_")
        # numeric fields
        if key.endswith("_pairs") or key in {"total_pairs", "number_of_classes", "nodes_per_sample"}:
            try:
                out[key] = int(v)
                continue
            except Exception:
                pass
        if key in {"best_macro_f1_score", "best_val_loss"}:
            try:
                out[key] = float(v)
                continue
            except Exception:
                pass
        if key in {"best_epoch"}:
            try:
                out[key] = int(v)
                continue
            except Exception:
                pass
        out[key] = v
    return out


def _safe_relpath(target: Path, base: Path) -> str:
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _pick_latest_dir(dirs: Iterable[Path]) -> Optional[Path]:
    best: Optional[Tuple[float, Path]] = None
    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        try:
            mtime = d.stat().st_mtime
        except Exception:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, d)
    return best[1] if best else None


def _find_predict_dir(run_dir: Path, meta_summary: Optional[Dict[str, Any]]) -> Optional[Path]:
    # Primary: meta/summary.json "predict_output_dir"
    if meta_summary:
        p = meta_summary.get("predict_output_dir")
        if isinstance(p, str) and p:
            cand = Path(p)
            if cand.exists():
                return cand
            # If the path in meta points to a different run_id, try to remap within this run_dir.
            # Typical layout: run_dir/outputs/Predict_data/<predict_dir_name>
            remap = run_dir / "outputs" / "Predict_data" / cand.name
            if remap.exists():
                return remap

    base = run_dir / "outputs" / "Predict_data"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("Predict")]
    return _pick_latest_dir(candidates)


def _find_loss_plots_dir(run_dir: Path) -> Optional[Path]:
    base = run_dir / "outputs" / "Predict_data"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("Loss_plots_")]
    return _pick_latest_dir(candidates)


def _find_final_model_path(run_dir: Path, meta_summary: Optional[Dict[str, Any]]) -> Optional[Path]:
    if meta_summary:
        p = meta_summary.get("final_model_path")
        if isinstance(p, str) and p:
            cand = Path(p)
            if cand.exists():
                return cand
            remap = run_dir / "outputs" / "GNN_model" / cand.name
            if remap.exists():
                return remap

    base = run_dir / "outputs"
    if not base.exists():
        return None
    # Keep it shallow-ish to avoid scanning gigantic trees.
    candidates: List[Path] = []
    for sub in ["GNN_model", "model", "models"]:
        d = base / sub
        if d.exists() and d.is_dir():
            candidates.extend(list(d.rglob("*.pth")))
    if not candidates:
        candidates = list(base.rglob("*.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return candidates[0]


def _read_csv_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            out.append({k: (v if v is not None else "") for k, v in row.items()})
            if max_rows is not None and i + 1 >= max_rows:
                break
    return out


def _try_float(s: Any) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _summarize_file_statistics(csv_path: Path, worst_k: int = 15) -> Dict[str, Any]:
    rows = _read_csv_rows(csv_path)
    if not rows:
        return {"path": str(csv_path), "count": 0}

    accs: List[Tuple[float, str]] = []
    ratios: List[Tuple[float, str]] = []
    for row in rows:
        fn = row.get("Filename", "")
        acc = _try_float(row.get("Accuracy"))
        if acc is not None:
            accs.append((acc, fn))
        ratio = _try_float(row.get("PredDefectRatio"))
        if ratio is not None:
            ratios.append((ratio, fn))

    accs_sorted = sorted(accs, key=lambda x: x[0])
    accs_sorted_desc = sorted(accs, key=lambda x: x[0], reverse=True)
    ratios_sorted = sorted(ratios, key=lambda x: x[0], reverse=True)

    mean_acc = sum(a for a, _ in accs) / max(1, len(accs))
    return {
        "path": str(csv_path),
        "count": len(rows),
        "mean_accuracy": mean_acc,
        "worst_accuracy": [{"filename": fn, "accuracy": a} for a, fn in accs_sorted[:worst_k]],
        "best_accuracy": [{"filename": fn, "accuracy": a} for a, fn in accs_sorted_desc[:worst_k]],
        "top_pred_defect_ratio": [{"filename": fn, "pred_defect_ratio": r} for r, fn in ratios_sorted[:worst_k]],
    }


def _summarize_threshold_decisions(csv_path: Path) -> Dict[str, Any]:
    """Summarize file_decisions_thresholds*.csv.

    Expected columns:
      - Filename, Accuracy, PredDefectRatio, PredDefect_t0_22, PredDefect_t0_23, ...
    """
    rows = _read_csv_rows(csv_path)
    if not rows:
        return {"path": str(csv_path), "count": 0, "threshold_counts": {}}

    # Sum any integer-ish PredDefect_t* columns
    threshold_cols = [k for k in rows[0].keys() if k.startswith("PredDefect_t")]
    counts: Dict[str, int] = {k: 0 for k in threshold_cols}
    ratio_vals: List[float] = []

    for row in rows:
        r = _try_float(row.get("PredDefectRatio"))
        if r is not None:
            ratio_vals.append(r)
        for k in threshold_cols:
            v = row.get(k)
            try:
                counts[k] += int(float(v)) if v not in (None, "") else 0
            except Exception:
                pass

    return {
        "path": str(csv_path),
        "count": len(rows),
        "mean_pred_defect_ratio": (sum(ratio_vals) / len(ratio_vals)) if ratio_vals else None,
        "threshold_counts": counts,
    }


def _find_first_png(run_dir: Path, patterns: List[str]) -> Optional[Path]:
    # outputs tree is usually manageable in run dirs; keep it scoped.
    base = run_dir / "outputs"
    if not base.exists():
        return None
    for pat in patterns:
        hits = sorted(base.rglob(pat))
        if hits:
            # newest
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
            return hits[0]
    return None


_RE_GPU_LINE = re.compile(r"^\s*GPU\s+\d+:\s+(.+?)\s*$")
_RE_USING_DEVICE = re.compile(r"^\s*Using device:\s+(.+?)\s*$")
_RE_TOTAL_SAMPLES = re.compile(r"^\s*Total samples:\s*([0-9]+)\s*$")
_RE_CLASS_DIST = re.compile(
    r"^\s*Class\s+(\d+):\s*([0-9]+)\s+samples\s*\(\s*([0-9.]+)%\)\s*,\s*weight:\s*([0-9eE\.\-]+)\s*$"
)


def _parse_train_log_summary(train_log: Path, max_classes: int = 64) -> Dict[str, Any]:
    """Extract GPU and class-imbalance summary from train.log (best-effort)."""
    out: Dict[str, Any] = {"path": str(train_log)}
    if not train_log.exists():
        return out

    gpu_names: List[str] = []
    using_device: List[str] = []
    total_samples: Optional[int] = None
    # Logs can print the class distribution block multiple times (multi-rank / multiple stages).
    # We'll collect blocks and keep the one with the largest total_samples (or sum of samples).
    blocks: List[Dict[str, Any]] = []
    capturing = False
    current_total: Optional[int] = None
    current_by_class: Dict[int, Dict[str, Any]] = {}

    in_class_block = False
    with train_log.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = _RE_GPU_LINE.match(line)
            if m:
                name = m.group(1).strip()
                if name and name not in gpu_names:
                    gpu_names.append(name)
                continue

            m = _RE_USING_DEVICE.match(line)
            if m:
                name = m.group(1).strip()
                if name and name not in using_device:
                    using_device.append(name)
                continue

            m = _RE_TOTAL_SAMPLES.match(line)
            if m:
                try:
                    current_total = int(m.group(1))
                    if total_samples is None:
                        total_samples = current_total
                except Exception:
                    pass

            if "Class distribution:" in line:
                # start a new capture
                capturing = True
                in_class_block = True
                current_by_class = {}
                continue

            if in_class_block:
                m = _RE_CLASS_DIST.match(line)
                if m:
                    try:
                        cid = int(m.group(1))
                        current_by_class[cid] = {
                            "class_id": cid,
                            "samples": int(m.group(2)),
                            "percent": float(m.group(3)),
                            "weight": float(m.group(4)),
                        }
                    except Exception:
                        pass
                    if len(current_by_class) >= max_classes:
                        in_class_block = False
                    continue
                # stop block on blank/next header
                if line.startswith("Class weights:") or line.startswith("==="):
                    in_class_block = False
                    if capturing and current_by_class:
                        # store this block
                        sample_sum = sum(int(v.get("samples", 0)) for v in current_by_class.values())
                        blocks.append(
                            {
                                "total_samples": current_total,
                                "sample_sum": sample_sum,
                                "by_class": current_by_class,
                            }
                        )
                    capturing = False

    out["gpu_names"] = gpu_names
    out["using_device"] = using_device
    out["total_samples"] = total_samples

    # pick best block
    best_block: Optional[Dict[str, Any]] = None
    for b in blocks:
        score = b.get("total_samples") or b.get("sample_sum") or 0
        if best_block is None:
            best_block = b
        else:
            best_score = best_block.get("total_samples") or best_block.get("sample_sum") or 0
            if score > best_score:
                best_block = b

    best_by_class = (best_block or {}).get("by_class") or {}
    class_rows = sorted(best_by_class.values(), key=lambda x: x.get("class_id", 0))
    out["class_distribution"] = class_rows

    # Derived imbalance summary
    if class_rows:
        majority = max(class_rows, key=lambda x: x.get("samples", 0))
        minority = min(class_rows, key=lambda x: x.get("samples", 10**18))
        out["imbalance_summary"] = {
            "majority_class": majority.get("class_id"),
            "majority_percent": majority.get("percent"),
            "minority_class": minority.get("class_id"),
            "minority_samples": minority.get("samples"),
            "minority_percent": minority.get("percent"),
        }
    return out

def _discover_predict_csv_dir(run_dir: Path) -> Optional[Path]:
    base = run_dir / "outputs" / "Predict_csv"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("Predict_csv")]
    return _pick_latest_dir(candidates)


def _discover_predict_csv_files(predict_csv_dir: Optional[Path]) -> Dict[str, Optional[Path]]:
    if not predict_csv_dir:
        return {"file_statistics": None, "file_decisions_thresholds": None}
    stats = sorted(predict_csv_dir.glob("file_statistics*.csv"))
    decs = sorted(predict_csv_dir.glob("file_decisions_thresholds*.csv"))
    return {
        "file_statistics": stats[-1] if stats else None,
        "file_decisions_thresholds": decs[-1] if decs else None,
    }


def _find_predict_truth_dir(run_dir: Path) -> Optional[Path]:
    base = run_dir / "outputs" / "Predict_truth"
    if not base.exists():
        return None
    # Typical: Predict_truth/<folder_with_timestamp>/
    dirs = [p for p in base.iterdir() if p.is_dir()]
    return _pick_latest_dir(dirs)


def _find_pred_data_dir(predict_dir: Optional[Path]) -> Optional[Path]:
    if not predict_dir:
        return None
    dirs = [p for p in predict_dir.iterdir() if p.is_dir() and p.name.startswith("pred_data_")]
    return _pick_latest_dir(dirs)


def _latest_glob(d: Optional[Path], pattern: str) -> Optional[Path]:
    if not d or not d.exists():
        return None
    hits = sorted(d.glob(pattern))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return hits[0]


def _select_spatial_visualizations(spatial_dir: Path, max_total: int = 12) -> List[Path]:
    if not spatial_dir.exists():
        return []
    # Prefer a mix: some Defect_* and some NoiseDefectFree_*
    defects = sorted([p for p in spatial_dir.glob("Defect*_spatial_visualization.png")])
    ndf = sorted([p for p in spatial_dir.glob("NoiseDefectFree*_spatial_visualization.png")])

    picked: List[Path] = []
    # 1/2 from each if possible
    half = max_total // 2
    picked.extend(defects[:half])
    picked.extend(ndf[: max_total - len(picked)])
    if len(picked) < max_total:
        # fill remaining from whichever has leftovers
        rest = defects[half:] + ndf[max_total - len(picked) :]
        for p in rest:
            if len(picked) >= max_total:
                break
            if p not in picked:
                picked.append(p)
    return picked


@dataclass(frozen=True)
class LoadedRun:
    run_dir: Path
    run_id: str
    generated_at: str
    meta: Dict[str, Any]
    metrics: Dict[str, Any]
    dataset: Dict[str, Any]
    artifacts: Dict[str, Any]


def load_run(run_dir: Path, max_spatial: int = 12) -> LoadedRun:
    run_dir = run_dir.expanduser().resolve()
    meta_dir = run_dir / "meta"
    logs_dir = run_dir / "logs"

    meta_run_config = _load_json_if_exists(meta_dir / "run_config.json") or {}
    meta_summary = _load_json_if_exists(meta_dir / "summary.json") or {}
    meta_preflight = _load_json_if_exists(meta_dir / "preflight.json") or {}

    metrics = _load_json_if_exists(run_dir / "metrics.json") or {}

    run_id = (
        str(meta_run_config.get("run_id") or "")
        or str(meta_summary.get("run_id") or "")
        or run_dir.name
    )

    # Artifacts discovery
    predict_dir = _find_predict_dir(run_dir, meta_summary)
    loss_dir = _find_loss_plots_dir(run_dir)
    final_model = _find_final_model_path(run_dir, meta_summary)

    spatial_dir = (predict_dir / "spatial_visualizations") if predict_dir else None
    spatial_examples = _select_spatial_visualizations(spatial_dir, max_total=max_spatial) if spatial_dir else []

    predict_truth_dir = _find_predict_truth_dir(run_dir)
    pred_data_dir = _find_pred_data_dir(predict_dir)

    predict_csv_dir = _discover_predict_csv_dir(run_dir)
    predict_csv_files = _discover_predict_csv_files(predict_csv_dir)

    # Dataset info txt (prefer loss_dir's dataset_info, fallback predict_dir's)
    dataset_info: Dict[str, Any] = {}
    loss_ds = _parse_dataset_info_txt(loss_dir / "dataset_info.txt") if loss_dir else {}
    pred_ds = _parse_dataset_info_txt(predict_dir / "dataset_info.txt") if predict_dir else {}

    # Merge: start from loss_ds (often contains best scores), fill missing with pred_ds (often contains nodes_per_sample)
    dataset_info.update(loss_ds or {})
    for k, v in (pred_ds or {}).items():
        if k not in dataset_info or dataset_info.get(k) in (None, "", 0):
            dataset_info[k] = v

    # Summaries from CSV (optional)
    file_stats_summary = (
        _summarize_file_statistics(predict_csv_files["file_statistics"]) if predict_csv_files["file_statistics"] else None
    )
    threshold_summary = (
        _summarize_threshold_decisions(predict_csv_files["file_decisions_thresholds"])
        if predict_csv_files["file_decisions_thresholds"]
        else None
    )

    def _latest_match(d: Optional[Path], pattern: str) -> Optional[str]:
        if not d or not d.exists():
            return None
        matches = sorted(d.glob(pattern))
        return str(matches[-1]) if matches else None

    # Optional: confusion matrix / confusion plot (best-effort)
    confusion_plot = _find_first_png(run_dir, patterns=["*confusion*.png", "*Confusion*.png"])
    # Prefer explicit Predict_truth plots if available
    confusion_truth_png = _latest_glob(predict_truth_dir, "confusion_matrix*.png")
    f1score_truth_png = _latest_glob(predict_truth_dir, "f1score_class*.png")
    detailed_metrics_truth_png = _latest_glob(predict_truth_dir, "detailed_class_metrics*.png")
    pred_vs_true_truth_png = _latest_glob(predict_truth_dir, "pred_vs_true_class_labels*.png")

    # NPY prediction aggregates (for npy->png plots)
    all_labels_npy = (pred_data_dir / "all_labels.npy") if pred_data_dir else None
    all_preds_npy = (pred_data_dir / "all_preds.npy") if pred_data_dir else None
    all_probs_npy = (pred_data_dir / "all_probs.npy") if pred_data_dir else None

    # Train.log summary (GPU + class imbalance)
    train_log_path = logs_dir / "train.log"
    train_log_summary = _parse_train_log_summary(train_log_path) if train_log_path.exists() else {"path": str(train_log_path)}

    # Key files to visualize (ensure "good/bad" are included)
    def _to_spatial_png(filename: str) -> Optional[str]:
        if not spatial_dir or not filename:
            return None
        stem = filename[:-4] if filename.endswith(".npy") else filename
        cand = spatial_dir / f"{stem}_spatial_visualization.png"
        return str(cand) if cand.exists() else None

    key_examples: Dict[str, Any] = {}
    # Prefer selecting examples that actually have spatial visualizations.
    stats_csv_path = predict_csv_files.get("file_statistics") if isinstance(predict_csv_files, dict) else None
    if spatial_dir and stats_csv_path and Path(stats_csv_path).exists():
        rows = _read_csv_rows(Path(stats_csv_path))

        def parse_acc(r: Dict[str, str]) -> Optional[float]:
            return _try_float(r.get("Accuracy"))

        # Build candidate lists
        with_acc = [(parse_acc(r), r.get("Filename", "")) for r in rows]
        with_acc = [(a, fn) for (a, fn) in with_acc if a is not None and fn]
        with_acc.sort(key=lambda x: float(x[0]))  # ascending (worst first)

        def pick_worst(k: int = 12) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for a, fn in with_acc:
                sp = _to_spatial_png(fn)
                if not sp:
                    continue
                out.append({"filename": fn, "accuracy": float(a), "spatial_png": sp})
                if len(out) >= k:
                    break
            return out

        def pick_best_defect(k: int = 12) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for a, fn in reversed(with_acc):
                if "Defect_" not in fn:
                    continue
                sp = _to_spatial_png(fn)
                if not sp:
                    continue
                out.append({"filename": fn, "accuracy": float(a), "spatial_png": sp})
                if len(out) >= k:
                    break
            return out

        key_examples["worst_accuracy"] = pick_worst(k=12)
        key_examples["best_accuracy_defect"] = pick_best_defect(k=12)
    elif isinstance(file_stats_summary, dict) and file_stats_summary.get("count", 0) > 0:
        # fallback to summary-only lists (may miss images)
        worst = file_stats_summary.get("worst_accuracy", []) or []
        best_acc = file_stats_summary.get("best_accuracy", []) or []
        best_defect = [x for x in best_acc if isinstance(x, dict) and "Defect_" in str(x.get("filename", ""))]
        key_examples["worst_accuracy"] = [{"filename": x.get("filename"), "accuracy": x.get("accuracy"), "spatial_png": _to_spatial_png(str(x.get("filename","")))} for x in worst if isinstance(x, dict)]
        key_examples["best_accuracy_defect"] = [{"filename": x.get("filename"), "accuracy": x.get("accuracy"), "spatial_png": _to_spatial_png(str(x.get("filename","")))} for x in best_defect if isinstance(x, dict)]

    artifacts: Dict[str, Any] = {
        "predict_dir": str(predict_dir) if predict_dir else None,
        "loss_plots_dir": str(loss_dir) if loss_dir else None,
        "final_model_path": str(final_model) if final_model else None,
        "train_log": str((logs_dir / "train.log")) if (logs_dir / "train.log").exists() else None,
        "predict_truth_dir": str(predict_truth_dir) if predict_truth_dir else None,
        "pred_data_dir": str(pred_data_dir) if pred_data_dir else None,
        "spatial_visualizations_dir": str(spatial_dir) if spatial_dir else None,
        "spatial_examples": [str(p) for p in spatial_examples],
        "pred_arrays": {
            "all_labels_npy": str(all_labels_npy) if (all_labels_npy and all_labels_npy.exists()) else None,
            "all_preds_npy": str(all_preds_npy) if (all_preds_npy and all_preds_npy.exists()) else None,
            "all_probs_npy": str(all_probs_npy) if (all_probs_npy and all_probs_npy.exists()) else None,
        },
        "plots": {
            "combined_plot": _latest_match(loss_dir, "combined_plot_*.png"),
            "loss_plot": _latest_match(loss_dir, "loss_plot_*.png"),
            "macro_f1_plot": _latest_match(loss_dir, "macro_f1_plot_*.png"),
            "confusion_plot": str(confusion_plot) if confusion_plot else None,
            "confusion_truth_png": str(confusion_truth_png) if confusion_truth_png else None,
            "f1score_truth_png": str(f1score_truth_png) if f1score_truth_png else None,
            "detailed_metrics_truth_png": str(detailed_metrics_truth_png) if detailed_metrics_truth_png else None,
            "pred_vs_true_truth_png": str(pred_vs_true_truth_png) if pred_vs_true_truth_png else None,
        },
        "predict_csv_dir": str(predict_csv_dir) if predict_csv_dir else None,
        "predict_csv_files": {k: str(v) if v else None for k, v in predict_csv_files.items()},
        "file_statistics_summary": file_stats_summary,
        "threshold_decisions_summary": threshold_summary,
        "train_log_summary": train_log_summary,
        "key_file_examples": key_examples,
    }

    meta: Dict[str, Any] = {
        "run_config": meta_run_config,
        "summary": meta_summary,
        "preflight": meta_preflight,
    }

    return LoadedRun(
        run_dir=run_dir,
        run_id=run_id,
        generated_at=datetime.now().isoformat(),
        meta=meta,
        metrics=metrics,
        dataset=dataset_info,
        artifacts=artifacts,
    )


def to_report_dict(loaded: LoadedRun, out_dir: Path) -> Dict[str, Any]:
    """Convert to a compact, stable dict for templates / LLM prompts.

    All paths are converted to either absolute or out_dir-relative strings later in rendering.
    """
    run_cfg = loaded.meta.get("run_config", {}) or {}
    summary = loaded.meta.get("summary", {}) or {}
    preflight = loaded.meta.get("preflight", {}) or {}
    metrics = loaded.metrics or {}

    best = metrics.get("best", {}) if isinstance(metrics.get("best"), dict) else {}
    test = metrics.get("test", {}) if isinstance(metrics.get("test"), dict) else {}

    # Key hyperparams (best-effort)
    args = run_cfg.get("training_args", []) if isinstance(run_cfg.get("training_args"), list) else []

    torchrun = run_cfg.get("torchrun")
    python_exec = None
    if isinstance(torchrun, str) and torchrun:
        tr = Path(torchrun)
        # torchrun is typically <env>/bin/torchrun
        cand = tr.parent / "python"
        if cand.exists():
            python_exec = str(cand)

    return {
        "generated_at": loaded.generated_at,
        "run_id": loaded.run_id,
        "run_dir": str(loaded.run_dir),
        "git_short_sha": run_cfg.get("git_short_sha"),
        "script": run_cfg.get("script"),
        "profile": run_cfg.get("profile") or summary.get("profile"),
        "env": {
            "torchrun": torchrun,
            "python": python_exec,
            "workdir": run_cfg.get("workdir"),
        },
        "training": {
            "nproc_per_node": run_cfg.get("nproc_per_node"),
            "args": args,
        },
        "dataset": loaded.dataset,
        "results": {
            "best": {
                "epoch": best.get("epoch"),
                "val_loss": best.get("val_loss"),
                "macro_f1": best.get("macro_f1") or summary.get("best_macro_f1"),
            },
            "test": test,
        },
        "preflight": preflight,
        "artifacts": loaded.artifacts,
        "out_dir": str(out_dir),
    }

