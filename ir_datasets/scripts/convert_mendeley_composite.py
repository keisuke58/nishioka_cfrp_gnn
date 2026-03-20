"""
Dataset 2: Mendeley Composite Academic Samples → GNN-compatible IR features
Source: https://data.mendeley.com/datasets/v4knrwgj9y/2

Thermal imagery from composite material academic samples (Data in Brief, 2020)
- 3 CFRP + 3 GFRP plates (300x300x2mm)
- Geometries: planar, curved, trapezoidal
- 25 Teflon inserts per plate (delamination simulation)
- 12 thermal sequences (~2000 frames, 512x512, 120-145fps)
- Inspected from both sides (front/back)

Output: Per-sequence .npy files with 3D IR features [H, W, 3]
  - max_contrast, slope, peak_time
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


def find_sequence_files(raw_dir: str) -> list:
    """Find thermal sequence files (.mat, .npy, .raw, .seq)."""
    patterns = ["*.mat", "*.npy", "*.raw", "*.seq", "*.csv", "*.dat"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(raw_dir, "**", p), recursive=True))
    # Also look for directories containing frame-by-frame images
    frame_dirs = []
    for d in glob.glob(os.path.join(raw_dir, "**"), recursive=True):
        if os.path.isdir(d):
            frames = glob.glob(os.path.join(d, "*.png")) + \
                     glob.glob(os.path.join(d, "*.tif")) + \
                     glob.glob(os.path.join(d, "*.tiff"))
            if len(frames) > 50:  # Likely a frame sequence
                frame_dirs.append(d)
    return sorted(files), sorted(frame_dirs)


def load_sequence_from_frames(frame_dir: str, max_frames: int = 2000) -> np.ndarray:
    """Load a temporal sequence from individual frame images."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required")

    frames_paths = sorted(
        glob.glob(os.path.join(frame_dir, "*.png")) +
        glob.glob(os.path.join(frame_dir, "*.tif")) +
        glob.glob(os.path.join(frame_dir, "*.tiff"))
    )[:max_frames]

    if not frames_paths:
        return None

    first = np.array(Image.open(frames_paths[0]), dtype=np.float32)
    if first.ndim == 3:
        first = first[:, :, 0]
    H, W = first.shape
    sequence = np.zeros((len(frames_paths), H, W), dtype=np.float32)
    sequence[0] = first

    for i, fp in enumerate(frames_paths[1:], 1):
        img = np.array(Image.open(fp), dtype=np.float32)
        if img.ndim == 3:
            img = img[:, :, 0]
        sequence[i] = img

    return sequence


def load_sequence_from_mat(mat_path: str) -> Optional[np.ndarray]:
    """Load temporal sequence from .mat file."""
    if loadmat is None:
        raise ImportError("scipy required for .mat files")

    mat = loadmat(mat_path)
    # Look for 3D array (temporal sequence)
    for key, val in mat.items():
        if key.startswith("_"):
            continue
        if isinstance(val, np.ndarray) and val.ndim == 3:
            # Could be [T, H, W] or [H, W, T]
            if val.shape[0] < val.shape[-1]:
                # Likely [H, W, T] → transpose
                return val.transpose(2, 0, 1).astype(np.float32)
            return val.astype(np.float32)
    return None


def extract_temporal_features(frames: np.ndarray) -> np.ndarray:
    """
    Extract 3 temporal IR features from thermal sequence [T, H, W].

    Returns: [H, W, 3] (max_contrast, slope, peak_time)
    """
    T, H, W = frames.shape
    frames = frames.astype(np.float64)

    T0 = frames[0]
    delta_T = frames - T0[np.newaxis, :, :]

    # 1. Max contrast
    max_contrast = np.max(delta_T, axis=0)

    # 2. Peak time (normalized)
    peak_time = np.argmax(frames, axis=0).astype(np.float64)
    peak_time /= max(T - 1, 1)

    # 3. Cooling slope (log-decay in second half)
    slope = np.zeros((H, W), dtype=np.float64)
    t_start = max(T // 3, 2)

    if T - t_start >= 3:
        T_inf = frames[-1]
        cooling = frames[t_start:].copy()
        for t in range(cooling.shape[0]):
            diff = cooling[t] - T_inf
            cooling[t] = np.log(np.maximum(diff, 1e-10))

        t_range = np.arange(cooling.shape[0], dtype=np.float64)
        t_norm = t_range - t_range.mean()
        t_var = np.sum(t_norm ** 2) + 1e-10

        mean_log = cooling.mean(axis=0)
        slope_num = np.sum(
            t_norm[:, None, None] * (cooling - mean_log[None, :, :]),
            axis=0
        )
        slope = slope_num / t_var

    features = np.stack([max_contrast, slope, peak_time], axis=-1)
    return features


def zscore_normalize(features: np.ndarray) -> np.ndarray:
    """Z-score normalize each feature channel."""
    result = features.copy()
    for c in range(features.shape[-1]):
        ch = result[..., c]
        mean = ch.mean()
        std = ch.std() + 1e-8
        result[..., c] = (ch - mean) / std
    return result


def parse_sample_info(name: str) -> dict:
    """Try to extract plate/material/side info from filename."""
    info = {"material": "unknown", "geometry": "unknown", "side": "unknown"}
    name_lower = name.lower()
    if "cfrp" in name_lower or "carbon" in name_lower:
        info["material"] = "CFRP"
    elif "gfrp" in name_lower or "glass" in name_lower:
        info["material"] = "GFRP"
    if "planar" in name_lower or "flat" in name_lower:
        info["geometry"] = "planar"
    elif "curved" in name_lower:
        info["geometry"] = "curved"
    elif "trapez" in name_lower:
        info["geometry"] = "trapezoidal"
    if "front" in name_lower:
        info["side"] = "front"
    elif "back" in name_lower or "rear" in name_lower:
        info["side"] = "back"
    return info


def convert_mendeley_composite(
    raw_dir: str,
    output_dir: str,
    max_frames_per_seq: int = 2000,
) -> dict:
    """
    Convert Mendeley Composite Samples dataset to IR features.

    Args:
        raw_dir: Path to extracted dataset
        output_dir: Output path for processed .npy files

    Returns:
        Summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    mat_files, frame_dirs = find_sequence_files(raw_dir)
    print(f"Found {len(mat_files)} sequence files, {len(frame_dirs)} frame directories")

    stats = {"n_sequences": 0, "shapes": [], "materials": []}

    # Process .mat / .npy files
    for fpath in mat_files:
        basename = Path(fpath).stem
        info = parse_sample_info(basename)
        print(f"\nProcessing: {basename} ({info['material']})")

        try:
            if fpath.endswith(".mat"):
                seq = load_sequence_from_mat(fpath)
            elif fpath.endswith(".npy"):
                seq = np.load(fpath)
                if seq.ndim == 2:
                    # Single frame
                    from convert_mendeley_cfrp_seg import extract_features_from_single_image
                    features = extract_features_from_single_image(seq)
                    features = zscore_normalize(features)
                    out_path = os.path.join(output_dir, f"composite_{basename}_ir.npy")
                    np.save(out_path, features.astype(np.float32))
                    stats["n_sequences"] += 1
                    stats["shapes"].append(features.shape)
                    print(f"  Single image → {features.shape}")
                    continue
                seq = seq.astype(np.float32)
            else:
                print(f"  [SKIP] Unsupported format: {fpath}")
                continue

            if seq is None or seq.ndim != 3:
                print(f"  [SKIP] Could not load as 3D sequence")
                continue

            print(f"  Sequence shape: {seq.shape} (T={seq.shape[0]})")

            # Subsample if very long
            if seq.shape[0] > max_frames_per_seq:
                step = seq.shape[0] // max_frames_per_seq
                seq = seq[::step]
                print(f"  Subsampled to {seq.shape[0]} frames")

            features = extract_temporal_features(seq)
            features = zscore_normalize(features)

            out_path = os.path.join(output_dir, f"composite_{basename}_ir.npy")
            np.save(out_path, features.astype(np.float32))

            stats["n_sequences"] += 1
            stats["shapes"].append(features.shape)
            stats["materials"].append(info["material"])
            print(f"  → {features.shape}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # Process frame directories
    for fdir in frame_dirs:
        dirname = Path(fdir).name
        info = parse_sample_info(dirname)
        print(f"\nProcessing frames: {dirname} ({info['material']})")

        try:
            seq = load_sequence_from_frames(fdir, max_frames=max_frames_per_seq)
            if seq is None:
                print(f"  [SKIP] No frames found")
                continue

            print(f"  Loaded {seq.shape[0]} frames, resolution {seq.shape[1]}x{seq.shape[2]}")

            features = extract_temporal_features(seq)
            features = zscore_normalize(features)

            out_path = os.path.join(output_dir, f"composite_{dirname}_ir.npy")
            np.save(out_path, features.astype(np.float32))

            stats["n_sequences"] += 1
            stats["shapes"].append(features.shape)
            stats["materials"].append(info["material"])
            print(f"  → {features.shape}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # Summary
    summary_path = os.path.join(output_dir, "conversion_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Mendeley Composite Academic Samples Conversion Summary\n")
        f.write(f"Source: {raw_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Sequences converted: {stats['n_sequences']}\n")
        f.write(f"Materials: {dict(zip(*np.unique(stats['materials'], return_counts=True))) if stats['materials'] else 'N/A'}\n")
        f.write(f"Features: [max_contrast, slope, peak_time]\n")
        f.write(f"Normalization: z-score per channel\n")

    print(f"\nConversion complete: {stats['n_sequences']} sequences → {output_dir}")
    return stats


if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw", "mendeley_composite_samples")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed", "mendeley_composite_samples")

    convert_mendeley_composite(raw_dir, output_dir)
