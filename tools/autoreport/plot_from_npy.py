#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple


def _load_npy(path: Path):
    import numpy as np  # type: ignore

    return np.load(str(path), allow_pickle=False)


def _confusion(labels, preds, num_classes: int):
    import numpy as np  # type: ignore

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # labels/preds can be 1D of class ids, or 2D one-hot
    if labels.ndim == 2:
        labels = labels.argmax(axis=1)
    if preds.ndim == 2:
        preds = preds.argmax(axis=1)
    labels = labels.astype(int).ravel()
    preds = preds.astype(int).ravel()
    for t, p in zip(labels, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _per_class_f1(cm) -> Tuple["list[float]", "list[float]", "list[float]"]:
    import numpy as np  # type: ignore

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision.tolist(), recall.tolist(), f1.tolist()


def _plot_heatmap(cm, out: Path, title: str, normalize: bool = False) -> None:
    import numpy as np  # type: ignore
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    data = cm.astype(float)
    if normalize:
        row = np.maximum(data.sum(axis=1, keepdims=True), 1.0)
        data = data / row

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(data.shape[0]))
    ax.set_yticks(range(data.shape[0]))
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200)
    plt.close(fig)


def _plot_bar(vals, out: Path, title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    xs = list(range(len(vals)))
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.bar(xs, vals)
    ax.set_title(title)
    ax.set_xlabel("class")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200)
    plt.close(fig)


def _plot_hist(vals, out: Path, title: str, xlabel: str, bins: int = 50, xlim: Optional[Tuple[float, float]] = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if xlim:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200)
    plt.close(fig)


def _write_top_confusions(cm, out_csv: Path, topk: int = 30) -> None:
    import csv
    import numpy as np  # type: ignore

    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    items = []
    for t in range(cm2.shape[0]):
        for p in range(cm2.shape[1]):
            c = int(cm2[t, p])
            if c > 0:
                row_sum = int(cm[t, :].sum())
                row_rate = (c / row_sum) if row_sum > 0 else 0.0
                items.append((c, row_rate, t, p))
    items.sort(key=lambda x: x[0], reverse=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "true_class", "pred_class", "count", "row_rate"])
        for i, (c, rr, t, p) in enumerate(items[:topk], start=1):
            w.writerow([i, t, p, c, f"{rr:.6f}"])


def _confidence_and_entropy(probs, num_classes: int):
    import numpy as np  # type: ignore

    p = probs.astype(float)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    p = p / s
    conf = p.max(axis=1)
    eps = 1e-12
    ent = -(p * np.log(p + eps)).sum(axis=1) / max(1.0, np.log(float(num_classes)))
    return conf.tolist(), ent.tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate confusion matrix / per-class plots from npy arrays.")
    ap.add_argument("--labels", required=True, help="Path to all_labels.npy (1D ids or 2D one-hot).")
    ap.add_argument("--preds", required=True, help="Path to all_preds.npy (1D ids or 2D one-hot).")
    ap.add_argument("--probs", default="", help="Optional path to all_probs.npy (N,C).")
    ap.add_argument("--outdir", required=True, help="Output directory for PNGs.")
    ap.add_argument("--num_classes", type=int, default=19)
    ap.add_argument("--prefix", default="auto", help="Filename prefix.")
    ap.add_argument("--topk", type=int, default=30, help="Top-K confusion pairs to export.")
    args = ap.parse_args()

    labels_p = Path(args.labels).expanduser().resolve()
    preds_p = Path(args.preds).expanduser().resolve()
    probs_p = Path(args.probs).expanduser().resolve() if args.probs else None
    outdir = Path(args.outdir).expanduser().resolve()
    ncls = int(args.num_classes)
    prefix = str(args.prefix)

    labels = _load_npy(labels_p)
    preds = _load_npy(preds_p)
    cm = _confusion(labels, preds, num_classes=ncls)
    prec, rec, f1 = _per_class_f1(cm)

    _plot_heatmap(cm, outdir / f"{prefix}_confusion_counts.png", "Confusion matrix (counts)", normalize=False)
    _plot_heatmap(cm, outdir / f"{prefix}_confusion_normalized.png", "Confusion matrix (row-normalized)", normalize=True)
    _plot_bar(f1, outdir / f"{prefix}_per_class_f1.png", "Per-class F1", "F1")
    _plot_bar(prec, outdir / f"{prefix}_per_class_precision.png", "Per-class Precision", "Precision")
    _plot_bar(rec, outdir / f"{prefix}_per_class_recall.png", "Per-class Recall", "Recall")
    _write_top_confusions(cm, outdir / f"{prefix}_top_confusions.csv", topk=int(args.topk))

    if probs_p and probs_p.exists():
        probs = _load_npy(probs_p)
        if getattr(probs, "ndim", 0) == 2 and probs.shape[1] == ncls:
            conf, ent = _confidence_and_entropy(probs, num_classes=ncls)
            _plot_hist(
                conf,
                outdir / f"{prefix}_confidence_hist.png",
                "Confidence distribution",
                "max probability",
                bins=60,
                xlim=(0.0, 1.0),
            )
            _plot_hist(
                ent,
                outdir / f"{prefix}_entropy_hist.png",
                "Entropy distribution (normalized)",
                "entropy",
                bins=60,
                xlim=(0.0, 1.0),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

