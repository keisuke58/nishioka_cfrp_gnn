#!/usr/bin/env python3
"""
M0-1: 現行モデルの性能を全OOD分割で記録

既存のベンチマーク実行結果およびruns内のmetrics.jsonを集約し、
baseline_metrics.json を生成する。

Usage:
    python tools/create_baseline_metrics.py
    python tools/create_baseline_metrics.py --output reports/baseline_metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


def parse_split_from_run_id(run_id: str) -> str:
    """run_id から split_type を推定 (e.g. bench_iid_s42, bench_quick_defect_size_s42 -> iid, defect_size)"""
    m = re.search(r"(?:bench_quick_|bench_)?(iid|defect_size|defect_ratio|layer)_s\d+", run_id)
    return m.group(1) if m else "iid"  # 既存メイン run は IID とみなす


def load_metrics_from_run(run_dir: Path) -> tuple[dict | None, str]:
    """run ディレクトリから metrics を読み込み。split_type は meta/run_config から優先。"""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None, "unknown"
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except Exception:
        return None, "unknown"
    split_type = parse_split_from_run_id(run_dir.name)
    run_config = run_dir / "meta" / "run_config.json"
    if run_config.exists():
        try:
            with open(run_config) as f:
                cfg = json.load(f)
            args = cfg.get("training_args", [])
            manifest = ""
            for i, a in enumerate(args):
                if a == "--split_manifest" and i + 1 < len(args):
                    manifest = args[i + 1]
                    break
            if "defect_size" in manifest:
                split_type = "defect_size"
            elif "defect_ratio" in manifest:
                split_type = "defect_ratio"
            elif "layer" in manifest:
                split_type = "layer"
        except Exception:
            pass
    return metrics, split_type


def collect_all_runs() -> list[dict]:
    """runs/ 以下から全 metrics を収集"""
    results = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        # ネストされた run (benchmark_xxx/bench_iid_s42_xxx/)
        for sub in run_dir.iterdir():
            if sub.is_dir() and (sub / "metrics.json").exists():
                m, split_type = load_metrics_from_run(sub)
                if m is not None:
                    results.append({
                        "run_id": sub.name,
                        "parent": run_dir.name,
                        "split_type": split_type,
                        "metrics": m,
                    })
        # 直下の run (20260116_xxx/)
        if (run_dir / "metrics.json").exists():
            m, split_type = load_metrics_from_run(run_dir)
            if m is not None:
                if split_type == "unknown":
                    split_type = "iid"  # 既存メイン run は通常 IID
                results.append({
                    "run_id": run_dir.name,
                    "parent": "",
                    "split_type": split_type,
                    "metrics": m,
                })
    return results


def _summary_for_split(rows: list[dict]) -> dict:
    f1s = [r["test_macro_f1"] or r["best_macro_f1"] for r in rows if (r["test_macro_f1"] or r["best_macro_f1"])]
    return {
        "count": len(rows),
        "macro_f1_mean": sum(f1s) / len(f1s) if f1s else None,
    }


def build_baseline_json(results: list[dict]) -> dict:
    """baseline_metrics.json 用の構造を構築"""
    by_split = {}
    for r in results:
        st = r["split_type"]
        if st not in by_split:
            by_split[st] = []
        m = r["metrics"]
        best = m.get("best", {})
        test = m.get("test", {})
        row = {
            "run_id": r["run_id"],
            "best_macro_f1": best.get("macro_f1"),
            "best_epoch": best.get("epoch"),
            "test_macro_f1": test.get("macro_f1"),
            "test_accuracy": test.get("accuracy"),
            "test_weighted_f1": test.get("weighted_f1"),
            "test_balanced_accuracy": test.get("balanced_accuracy"),
            "test_mcc": test.get("mcc"),
        }
        by_split[st].append(row)  # null でも記録（試行の記録として）
    return {
        "generated_at": datetime.now().isoformat(),
        "description": "M0-1: 現行モデル性能のベースライン記録（全OOD分割）",
        "by_split_type": by_split,
        "summary": {
            st: {
                "count": len(rows),
                "macro_f1_mean": (
                    (sum(f for r in rows if (f := (r["test_macro_f1"] or r["best_macro_f1"])) is not None) / n)
                    if (n := sum(1 for r in rows if (r["test_macro_f1"] or r["best_macro_f1"]))) else None
                ),
            }
            for st, rows in by_split.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="reports/baseline_metrics.json")
    args = parser.parse_args()
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = collect_all_runs()
    baseline = build_baseline_json(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] Wrote {len(results)} runs to {out_path}")
    for st, rows in baseline["by_split_type"].items():
        valid = [r for r in rows if (r.get("test_macro_f1") or r.get("best_macro_f1"))]
        if valid:
            f1s = [r.get("test_macro_f1") or r.get("best_macro_f1") for r in valid]
            print(f"  {st}: {len(valid)} runs, macro_f1 range: {min(f1s):.4f} - {max(f1s):.4f}")
        else:
            print(f"  {st}: {len(rows)} runs (no valid metrics yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
