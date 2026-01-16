#!/usr/bin/env python3
"""
Re-render spatial visualization PNGs in a presentation-friendly layout.

This script is meant to post-process a run output folder WITHOUT re-training:
it regenerates each figure with larger readable titles and much less vertical whitespace.

Typical usage (for an existing run output):
  python3 tools/make_spatial_visualizations_presentation.py \
    --targets_from_png_dir "<RUN>/outputs/Predict_data/<PRED>/spatial_visualizations" \
    --predictions_npy_dir "<RUN>/outputs/Predict_data/<PRED>/predictions_npy" \
    --out_dir "<RUN>/outputs/Predict_data/<PRED>/spatial_visualizations_presentation"
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# Global font: Times New Roman (fallbacks for Linux)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


# ----------------------------
# Geometry (must match training/pred pipeline)
# ----------------------------
NUM_COLS = 57
NUM_ROWS = 125
VERTICES_PER_LAYER = 6971
TOTAL_ELEMENTS = NUM_COLS * NUM_ROWS

# Hole region definition (same as other scripts)
HOLE_COL_START = 26
HOLE_ROW_START1 = 51
HOLE_ROW_START2 = 77
HOLE_SIZE_COLS = 7
HOLE_SIZE_ROWS1 = 7
HOLE_SIZE_ROWS2 = 15


def _build_hole_set() -> set[int]:
    hole_elements: List[int] = []
    for r in range(HOLE_ROW_START1, HOLE_ROW_START1 + HOLE_SIZE_ROWS1):
        for c in range(HOLE_COL_START, HOLE_COL_START + HOLE_SIZE_COLS):
            hole_elements.append(r * NUM_COLS + c)
    for r in range(HOLE_ROW_START2, HOLE_ROW_START2 + HOLE_SIZE_ROWS2):
        for c in range(HOLE_COL_START, HOLE_COL_START + HOLE_SIZE_COLS):
            hole_elements.append(r * NUM_COLS + c)
    # Convert to 0-based indices (matching existing scripts)
    return set([i - 1 for i in hole_elements])


HOLE_SET = _build_hole_set()


def get_defect_cmap_with_white_zero() -> ListedColormap:
    """
    Custom colormap for class maps:
      - class 0 is black
      - others follow coolwarm
    """
    coolwarm = mpl.colormaps.get_cmap("coolwarm").resampled(19)
    colors = coolwarm(np.linspace(0, 1, 19))
    colors[0] = [0.0, 0.0, 0.0, 1.0]
    return ListedColormap(colors)


def _decode_if_logits(arr: np.ndarray) -> np.ndarray:
    """If arr is [N, C] logits/probs, decode to argmax; else keep [N]."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return np.argmax(a, axis=1)
    if a.ndim == 1:
        return a
    raise ValueError(f"Unexpected array shape: {a.shape}")


def _gridify_two_layers(vec_13942: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map a 13942-length vector (2*6971) into two (NUM_ROWS, NUM_COLS) grids
    with holes filled as NaN, then rotate/flip to match the notebook/viewer.
    """
    v = np.asarray(vec_13942)
    if v.shape[0] < 2 * VERTICES_PER_LAYER:
        raise ValueError(f"Expected >= {2 * VERTICES_PER_LAYER} values, got {v.shape[0]}")

    layer1 = v[:VERTICES_PER_LAYER]
    layer2 = v[VERTICES_PER_LAYER : 2 * VERTICES_PER_LAYER]

    layer1_full = np.full(TOTAL_ELEMENTS, np.nan, dtype=float)
    layer2_full = np.full(TOTAL_ELEMENTS, np.nan, dtype=float)

    idx = 0
    for i in range(TOTAL_ELEMENTS):
        if i in HOLE_SET:
            continue
        if idx >= VERTICES_PER_LAYER:
            break
        layer1_full[i] = float(layer1[idx])
        idx += 1

    idx = 0
    for i in range(TOTAL_ELEMENTS):
        if i in HOLE_SET:
            continue
        if idx >= VERTICES_PER_LAYER:
            break
        layer2_full[i] = float(layer2[idx])
        idx += 1

    g1 = layer1_full.reshape((NUM_ROWS, NUM_COLS))
    g2 = layer2_full.reshape((NUM_ROWS, NUM_COLS))
    # orientation (same as other scripts)
    return np.flipud(np.rot90(g1, k=1)), np.flipud(np.rot90(g2, k=1))


def _load_dpsss_for_base(
    base_name: str,
    defect_data_root: Path,
    ndf_data_root: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Returns (bottom, upper, data_type_label) or None if not found.
    """
    def _candidate_dirs(p: Path) -> List[Path]:
        # Accept either a split dir (.../train|val|test) or a dataset root containing them.
        if p.exists() and p.is_dir() and p.name in {"train", "val", "test"}:
            return [p]
        cands = [p / "test", p / "val", p / "train", p]
        return [d for d in cands if d.exists() and d.is_dir()]

    is_ndf = base_name.startswith("NoiseDefectFree_")
    data_type = "NoiseDefectFree" if is_ndf else "Defect"
    roots = _candidate_dirs(ndf_data_root if is_ndf else defect_data_root)

    fp: Optional[Path] = None
    for d in roots:
        cand = d / f"{base_name}.npy"
        if cand.exists():
            fp = cand
            break
    if fp is None:
        return None
    arr = np.load(fp)
    # DSPSS (z-score) is a float vector; use first 13942
    g1, g2 = _gridify_two_layers(np.asarray(arr[: 2 * VERTICES_PER_LAYER], dtype=float))
    return g1, g2, data_type


def _load_class_map_for_base(base_name: str, label_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    fp = label_dir / f"{base_name}_19label.npy"
    if not fp.exists():
        return None
    arr = _decode_if_logits(np.load(fp))
    return _gridify_two_layers(arr[: 2 * VERTICES_PER_LAYER])


def _load_pred_map_for_base(base_name: str, predictions_npy_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    fp = predictions_npy_dir / f"{base_name}_pred.npy"
    if not fp.exists():
        return None
    arr = _decode_if_logits(np.load(fp))
    return _gridify_two_layers(arr[: 2 * VERTICES_PER_LAYER])


_RE_BASE_FROM_PNG = re.compile(r"^(?P<base>.+)_spatial_visualization\.png$")
_RE_DEFECT = re.compile(
    r"^Defect_L(?P<L>\d+)_B(?P<B>\d+)_el(?P<el>\d+)_H(?P<H>\d+)_W(?P<W>\d+)$"
)
_RE_NDF = re.compile(r"^NoiseDefectFree_(?P<id>\d+)$")


def _pretty_title(base_name: str) -> str:
    """
    Turn a base_name into a presentation-friendly title without underscores.
    Examples:
      - Defect_L2_B124_el6170_H8_W8 -> "Defect (layer 2, block 124, element 6170, height 8, width 8)"
      - NoiseDefectFree_000002 -> "Noise defect-free (ID 000002)"
    """
    m = _RE_DEFECT.match(base_name)
    if m:
        L = int(m.group("L"))
        B = int(m.group("B"))
        el = int(m.group("el"))
        H = int(m.group("H"))
        W = int(m.group("W"))
        return f"Defect (layer {L}, block {B}, element {el}, height {H}, width {W})"
    m = _RE_NDF.match(base_name)
    if m:
        return f"Noise defect-free (ID {m.group('id')})"
    # Fallback: replace underscores with spaces
    return base_name.replace("_", " ")


def _bases_from_png_dir(png_dir: Path) -> List[str]:
    bases: List[str] = []
    for p in sorted(png_dir.glob("*_spatial_visualization.png")):
        m = _RE_BASE_FROM_PNG.match(p.name)
        if not m:
            continue
        bases.append(m.group("base"))
    return bases


def _bases_from_predictions_dir(predictions_npy_dir: Path) -> List[str]:
    bases: List[str] = []
    for p in sorted(predictions_npy_dir.glob("*_pred.npy")):
        bases.append(p.name.replace("_pred.npy", ""))
    return bases


def _set_clean_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_one(
    base_name: str,
    *,
    defect_data_root: Path,
    ndf_data_root: Path,
    label_dir: Path,
    predictions_npy_dir: Path,
    out_dir: Path,
    dpi: int = 200,
) -> bool:
    dpsss = _load_dpsss_for_base(base_name, defect_data_root, ndf_data_root)
    pred = _load_pred_map_for_base(base_name, predictions_npy_dir)
    if dpsss is None or pred is None:
        return False
    dpsss_b, dpsss_u, data_type = dpsss
    pred_b, pred_u = pred
    label = _load_class_map_for_base(base_name, label_dir)

    defect_cmap_custom = get_defect_cmap_with_white_zero()

    # Mask NaNs for nicer rendering (holes become background)
    dpsss_b_m = np.ma.masked_where(np.isnan(dpsss_b), dpsss_b)
    dpsss_u_m = np.ma.masked_where(np.isnan(dpsss_u), dpsss_u)

    pred_b_i = np.where(np.isnan(pred_b), 0, pred_b).astype(int)
    pred_u_i = np.where(np.isnan(pred_u), 0, pred_u).astype(int)
    pred_b_m = np.ma.masked_where(np.isnan(pred_b), pred_b_i)
    pred_u_m = np.ma.masked_where(np.isnan(pred_u), pred_u_i)

    if label is not None:
        lab_b, lab_u = label
        lab_b_i = np.where(np.isnan(lab_b), 0, lab_b).astype(int)
        lab_u_i = np.where(np.isnan(lab_u), 0, lab_u).astype(int)
        lab_b_m = np.ma.masked_where(np.isnan(lab_b), lab_b_i)
        lab_u_m = np.ma.masked_where(np.isnan(lab_u), lab_u_i)
    else:
        lab_b_m = lab_u_m = None

    # Figure: keep geometry un-stretched (aspect="equal") but avoid excessive vertical whitespace.
    # Use a shorter canvas and tighter hspace.
    fig, axs = plt.subplots(2, 3, figsize=(18.0, 8.5))
    # Reduce "top" to add more whitespace between header text and plots.
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.05, top=0.75, wspace=0.12, hspace=0.0035)

    fig.suptitle(_pretty_title(base_name), fontsize=24, fontweight="bold", y=0.825)
    # fig.text(
    #     0.5,
    #     0.88,
    #     f"{data_type}. Bottom layer expected labels: 0–9. Upper layer expected labels: 0 and 10–18.",
    #     ha="center",
    #     va="center",
    #     fontsize=18,
    # )

    # Column 1: DSPSS (colorbar per row)
    im_dpsss_0 = axs[0, 0].imshow(dpsss_b_m, cmap="jet", aspect="equal", interpolation="nearest")
    im_dpsss_1 = axs[1, 0].imshow(dpsss_u_m, cmap="jet", aspect="equal", interpolation="nearest")
    axs[0, 0].set_title("Bottom layer: DSPSS (z-score)", fontsize=18, pad=10)
    axs[1, 0].set_title("Upper layer: DSPSS (z-score)", fontsize=18, pad=10)

    # Column 2: Label (if available)
    if lab_b_m is not None and lab_u_m is not None:
        im_gt_0 = axs[0, 1].imshow(lab_b_m, cmap=defect_cmap_custom, vmin=0, vmax=18, aspect="equal", interpolation="nearest")
        im_gt_1 = axs[1, 1].imshow(lab_u_m, cmap=defect_cmap_custom, vmin=0, vmax=18, aspect="equal", interpolation="nearest")
        axs[0, 1].set_title("Bottom layer: Ground-truth label", fontsize=18, pad=10)
        axs[1, 1].set_title("Upper layer: Ground-truth label", fontsize=18, pad=10)
    else:
        # Create dummy mappables so we can still attach per-row colorbars consistently.
        im_gt_0 = axs[0, 1].imshow(
            np.ma.masked_array(np.zeros_like(pred_b_m, dtype=float), mask=np.ones_like(pred_b_m, dtype=bool)),
            cmap=defect_cmap_custom,
            vmin=0,
            vmax=18,
            aspect="equal",
            interpolation="nearest",
        )
        im_gt_1 = axs[1, 1].imshow(
            np.ma.masked_array(np.zeros_like(pred_u_m, dtype=float), mask=np.ones_like(pred_u_m, dtype=bool)),
            cmap=defect_cmap_custom,
            vmin=0,
            vmax=18,
            aspect="equal",
            interpolation="nearest",
        )
        axs[0, 1].text(0.5, 0.5, "No label file", ha="center", va="center", fontsize=18, transform=axs[0, 1].transAxes)
        axs[1, 1].text(0.5, 0.5, "No label file", ha="center", va="center", fontsize=18, transform=axs[1, 1].transAxes)
        axs[0, 1].set_title("Bottom layer: Ground-truth label", fontsize=18, pad=10)
        axs[1, 1].set_title("Upper layer: Ground-truth label", fontsize=18, pad=10)

    # Column 3: Prediction
    im_pr_0 = axs[0, 2].imshow(pred_b_m, cmap=defect_cmap_custom, vmin=0, vmax=18, aspect="equal", interpolation="nearest")
    im_pr_1 = axs[1, 2].imshow(pred_u_m, cmap=defect_cmap_custom, vmin=0, vmax=18, aspect="equal", interpolation="nearest")
    axs[0, 2].set_title("Bottom layer: Prediction", fontsize=18, pad=10)
    axs[1, 2].set_title("Upper layer: Prediction", fontsize=18, pad=10)

    # Colorbars:
    # - DSPSS: one per row (left column)
    # - Class: right two panels share one per row (GT + Prediction)
    # Make colorbars a bit shorter vertically (~3/4) for cleaner rows.
    cbar_d0 = fig.colorbar(im_dpsss_0, ax=axs[0, 0], shrink=0.85, pad=0.03)
    cbar_d0.set_label("DSPSS (z-score)", fontsize=18)
    cbar_d0.ax.tick_params(labelsize=11)

    cbar_d1 = fig.colorbar(im_dpsss_1, ax=axs[1, 0], shrink=0.85, pad=0.03)
    cbar_d1.set_label("DSPSS (z-score)", fontsize=18)
    cbar_d1.ax.tick_params(labelsize=11)

    cbar_c0 = fig.colorbar(im_gt_0, ax=[axs[0, 1], axs[0, 2]], shrink=0.85, pad=0.03)
    cbar_c0.set_label("Class", fontsize=18)
    cbar_c0.set_ticks(list(range(0, 19, 2)))
    cbar_c0.ax.tick_params(labelsize=11)

    cbar_c1 = fig.colorbar(im_gt_1, ax=[axs[1, 1], axs[1, 2]], shrink=0.85, pad=0.03)
    cbar_c1.set_label("Class", fontsize=18)
    cbar_c1.set_ticks(list(range(0, 19, 2)))
    cbar_c1.ax.tick_params(labelsize=11)

    for r in range(2):
        for c in range(3):
            _set_clean_axes(axs[r, c])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_name}_spatial_visualization.png"
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Make presentation-friendly spatial visualization PNGs.")
    ap.add_argument("--targets_from_png_dir", type=str, default="", help="If set, use *_spatial_visualization.png files here as targets.")
    ap.add_argument("--predictions_npy_dir", type=str, required=True, help="Directory containing *_pred.npy files.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for regenerated PNGs.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of number of files (0 = no limit).")
    ap.add_argument("--dpi", type=int, default=200, help="PNG dpi (default: 200).")

    ap.add_argument(
        "--defect_data_root",
        type=str,
        default="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore",
        help="Defect data root or split dir (contains *.npy, or has train/val/test subdirs).",
    )
    ap.add_argument(
        "--ndf_data_root",
        type=str,
        default="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise",
        help="NoiseDefectFree data root or split dir (contains *.npy, or has train/val/test subdirs).",
    )
    ap.add_argument(
        "--label_dir",
        type=str,
        default="/home/nishioka/GNN/GNN_hole_2026/all_19class_label",
        help="Label directory containing *_19label.npy.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    predictions_npy_dir = Path(args.predictions_npy_dir)
    out_dir = Path(args.out_dir)
    defect_data_root = Path(args.defect_data_root)
    ndf_data_root = Path(args.ndf_data_root)
    label_dir = Path(args.label_dir)

    print(f"predictions_npy_dir: {predictions_npy_dir}")
    print(f"exists: {predictions_npy_dir.exists()}")
    print(f"is_dir: {predictions_npy_dir.is_dir()}")
    files = list(predictions_npy_dir.glob("*_pred.npy"))
    print(f"files count: {len(files)}")
    print(f"first few files: {files[:5]}")

    if args.targets_from_png_dir:
        bases = _bases_from_png_dir(Path(args.targets_from_png_dir))
    else:
        bases = _bases_from_predictions_dir(predictions_npy_dir)

    print(f"bases: {bases}")

    if args.limit and args.limit > 0:
        bases = bases[: int(args.limit)]

    if not bases:
        print("[ERROR] No targets found.")
        return 2

    ok = 0
    skipped = 0
    for i, base in enumerate(bases, start=1):
        try:
            if render_one(
                base,
                defect_data_root=defect_data_root,
                ndf_data_root=ndf_data_root,
                label_dir=label_dir,
                predictions_npy_dir=predictions_npy_dir,
                out_dir=out_dir,
                dpi=int(args.dpi),
            ):
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            print(f"[WARN] failed: {base}: {e}")
        if i % 20 == 0:
            print(f"progress: {i}/{len(bases)} (ok={ok}, skipped={skipped})")

    print(f"done: ok={ok}, skipped={skipped}, out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

