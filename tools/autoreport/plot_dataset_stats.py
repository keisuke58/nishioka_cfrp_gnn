#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({k: (v if v is not None else "") for k, v in row.items()})
    return out


def _plot_bar(xs: List[str], ys: List[float], out: Path, title: str, ylabel: str, rotate: bool = True, logy: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(xs)), ys)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=45 if rotate else 0, ha="right" if rotate else "center", fontsize=7)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200)
    plt.close(fig)


def _plot_hist(vals: List[float], out: Path, title: str, xlabel: str, bins: int = 60, xlim: Optional[Tuple[float, float]] = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=bins, color="#4C78A8", edgecolor="black", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200)
    plt.close(fig)


_RE_DEF = re.compile(r"^Defect_L(\d+)_B(\d+)_el(\d+)_H(\d+)_W(\d+)\.npy$")
_RE_NDF = re.compile(r"^NoiseDefectFree_(\d+)\.npy$")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate dataset property plots from report_data.json + CSVs.")
    ap.add_argument("--report_data", required=True, help="Path to reports/<run>/report_data.json")
    ap.add_argument("--outdir", required=True, help="assets/ output directory")
    args = ap.parse_args()

    report = _load_json(Path(args.report_data).expanduser().resolve())
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    art = report.get("artifacts") or {}
    train_sum = art.get("train_log_summary") or {}
    class_dist = train_sum.get("class_distribution") or []

    # 1) Class distribution counts/weights
    if isinstance(class_dist, list) and class_dist:
        class_ids = [str(int(r.get("class_id"))) for r in class_dist if isinstance(r, dict) and "class_id" in r]
        counts = [float(r.get("samples", 0)) for r in class_dist if isinstance(r, dict)]
        weights = [float(r.get("weight", 0.0)) for r in class_dist if isinstance(r, dict)]
        if class_ids and counts:
            _plot_bar(class_ids, counts, outdir / "dataset_class_counts_log.png", "Class distribution (counts, log-scale)", "count", rotate=False, logy=True)
            _plot_bar(class_ids, counts, outdir / "dataset_class_counts.png", "Class distribution (counts)", "count", rotate=False, logy=False)
        if class_ids and weights:
            _plot_bar(class_ids, weights, outdir / "dataset_class_weights.png", "Class weights", "weight", rotate=False, logy=False)

    # 2) Dataset property from file_statistics.csv
    stats_path = ((art.get("predict_csv_files") or {}).get("file_statistics")) if isinstance(art.get("predict_csv_files"), dict) else None
    if isinstance(stats_path, str) and stats_path:
        rows = _read_csv_rows(Path(stats_path))
        # Defect vs NDF ratio
        n_def, n_ndf, n_other = 0, 0, 0
        layer_counts: Dict[str, int] = {}
        hw_counts: Dict[str, int] = {}
        ratios: List[float] = []
        for row in rows:
            fn = row.get("Filename", "")
            if _RE_NDF.match(fn):
                n_ndf += 1
            elif _RE_DEF.match(fn):
                n_def += 1
                m = _RE_DEF.match(fn)
                assert m
                layer = m.group(1)
                hw = f"H{m.group(4)}_W{m.group(5)}"
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
                hw_counts[hw] = hw_counts.get(hw, 0) + 1
            else:
                n_other += 1

            try:
                r = float(row.get("PredDefectRatio") or "nan")
                if r == r:
                    ratios.append(r)
            except Exception:
                pass

        _plot_bar(["Defect", "NoiseDefectFree", "Other"], [n_def, n_ndf, n_other], outdir / "dataset_defect_ndf_counts.png", "File type counts", "#files", rotate=False)

        if ratios:
            # In practice PredDefectRatio can be very small (e.g. < 0.02), so xlim=(0,1)
            # makes the histogram look blank. Auto-zoom to the observed range.
            rmax = max(ratios)
            if rmax <= 0.05:
                xhi = max(0.01, rmax * 1.10)
                xlim = (0.0, xhi)
                bins = 50
            else:
                xlim = (0.0, 1.0)
                bins = 60
            _plot_hist(ratios, outdir / "dataset_preddefectratio_hist.png", "PredDefectRatio distribution", "PredDefectRatio", bins=bins, xlim=xlim)

        if layer_counts:
            xs = sorted(layer_counts.keys(), key=lambda x: int(x))
            ys = [float(layer_counts[x]) for x in xs]
            _plot_bar(xs, ys, outdir / "dataset_layer_counts.png", "Defect layer distribution (L)", "#files", rotate=False)

        if hw_counts:
            xs = sorted(hw_counts.keys())
            ys = [float(hw_counts[x]) for x in xs]
            _plot_bar(xs, ys, outdir / "dataset_hw_counts.png", "Defect HxW distribution", "#files", rotate=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

