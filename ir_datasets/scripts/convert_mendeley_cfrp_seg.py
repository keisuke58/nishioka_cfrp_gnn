"""
Dataset 1: Mendeley CFRP Defect Segmentation → GNN-compatible IR features
Source: https://data.mendeley.com/datasets/jrsb4b9yy5/1

Garcia Vargas & Fernandes (2025)
- 1034 thermal images (640x512px, MWIR, 55Hz pulsed thermography)
- CFRP (Carbon/PEEK), [02/902]6, 100x100mm
- Kapton tape inserts: 3 depths (0.13, 0.26, 0.39mm) x 3 sizes (2x2, 3x3, 4x4mm)
- Includes expert-annotated segmentation masks

Output: Per-sample .npy files with 3D IR features [H, W, 3]
  - max_contrast, slope, peak_time
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
except ImportError:
    Image = None
    print("WARNING: Pillow not installed. Install with: pip install Pillow")


def find_thermal_images(raw_dir: str) -> list:
    """Find thermal image files in the dataset directory."""
    extensions = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.bmp", "*.npy", "*.mat"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))
    # Exclude mask files
    files = [f for f in files if "mask" not in f.lower() and "label" not in f.lower()]
    return sorted(files)


def find_mask_files(raw_dir: str) -> list:
    """Find segmentation mask files."""
    extensions = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.bmp", "*.npy"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(raw_dir, "**/*mask*" + ext[1:]), recursive=True))
        files.extend(glob.glob(os.path.join(raw_dir, "**/*label*" + ext[1:]), recursive=True))
        files.extend(glob.glob(os.path.join(raw_dir, "**/*annotation*" + ext[1:]), recursive=True))
    return sorted(files)


def load_image(path: str) -> np.ndarray:
    """Load an image file as numpy array."""
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".mat"):
        from scipy.io import loadmat
        mat = loadmat(path)
        # Try common key names
        for key in ["data", "image", "thermal", "T", "img"]:
            if key in mat:
                return mat[key]
        # Return first non-meta key
        for key, val in mat.items():
            if not key.startswith("_") and isinstance(val, np.ndarray):
                return val
    else:
        if Image is None:
            raise ImportError("Pillow required for image loading")
        img = Image.open(path)
        return np.array(img, dtype=np.float32)


def extract_temporal_features_from_sequence(frames: np.ndarray) -> np.ndarray:
    """
    Extract 3 temporal features from a thermal image sequence.

    Args:
        frames: [T, H, W] thermal image sequence

    Returns:
        features: [H, W, 3] - (max_contrast, slope, peak_time)
    """
    T, H, W = frames.shape
    frames = frames.astype(np.float64)

    # Background: first frame (before flash)
    T0 = frames[0]

    # 1. max_contrast: max temperature rise
    delta_T = frames - T0[np.newaxis, :, :]
    max_contrast = np.max(delta_T, axis=0)  # [H, W]

    # 2. peak_time: frame index of maximum temperature
    peak_time = np.argmax(frames, axis=0).astype(np.float64)  # [H, W]
    peak_time = peak_time / max(T - 1, 1)  # Normalize to [0, 1]

    # 3. slope: cooling rate after peak
    # Compute log-decay slope using least squares on post-peak region
    slope = np.zeros((H, W), dtype=np.float64)
    # Use second half of sequence for slope estimation
    t_start = max(T // 4, 2)
    t_range = np.arange(t_start, T, dtype=np.float64)

    if len(t_range) >= 2:
        # Ambient temperature estimate (last frame)
        T_inf = frames[-1]
        for t_idx in range(t_start, T):
            diff = frames[t_idx] - T_inf
            diff = np.maximum(diff, 1e-10)
            frames[t_idx] = np.log(diff)

        log_frames = frames[t_start:]  # [T-t_start, H, W]
        t_norm = t_range - t_range.mean()
        t_var = np.sum(t_norm ** 2) + 1e-10

        # Vectorized slope computation
        mean_log = log_frames.mean(axis=0)
        slope_num = np.sum(t_norm[:, np.newaxis, np.newaxis] *
                          (log_frames - mean_log[np.newaxis, :, :]), axis=0)
        slope = slope_num / t_var  # [H, W]

    features = np.stack([max_contrast, slope, peak_time], axis=-1)  # [H, W, 3]
    return features


def extract_features_from_single_image(image: np.ndarray) -> np.ndarray:
    """
    Extract pseudo-features from a single thermal image (no temporal sequence).
    Uses spatial gradient and intensity as proxy features.

    Args:
        image: [H, W] single thermal image

    Returns:
        features: [H, W, 3] - (intensity, gradient_magnitude, laplacian)
    """
    image = image.astype(np.float64)

    # 1. Normalized intensity
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 1e-10:
        intensity = (image - img_min) / (img_max - img_min)
    else:
        intensity = np.zeros_like(image)

    # 2. Gradient magnitude (Sobel-like)
    gy = np.zeros_like(image)
    gx = np.zeros_like(image)
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
    grad_mag = np.sqrt(gx**2 + gy**2)
    gm_max = grad_mag.max()
    if gm_max > 1e-10:
        grad_mag = grad_mag / gm_max

    # 3. Laplacian (second derivative → defect edges)
    laplacian = np.zeros_like(image)
    laplacian[1:-1, 1:-1] = (
        image[2:, 1:-1] + image[:-2, 1:-1] +
        image[1:-1, 2:] + image[1:-1, :-2] - 4 * image[1:-1, 1:-1]
    )
    lap_abs = np.abs(laplacian)
    lap_max = lap_abs.max()
    if lap_max > 1e-10:
        laplacian = lap_abs / lap_max

    features = np.stack([intensity, grad_mag, laplacian], axis=-1)
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


def convert_mendeley_cfrp_seg(
    raw_dir: str,
    output_dir: str,
    max_samples: Optional[int] = None,
) -> dict:
    """
    Convert Mendeley CFRP segmentation dataset to GNN-compatible IR features.

    Args:
        raw_dir: Path to extracted Mendeley dataset
        output_dir: Path to save processed .npy files

    Returns:
        Summary dict with statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    thermal_files = find_thermal_images(raw_dir)
    mask_files = find_mask_files(raw_dir)

    if not thermal_files:
        print(f"ERROR: No thermal images found in {raw_dir}")
        print("Expected structure:")
        print(f"  {raw_dir}/")
        print(f"    images/  (thermal images)")
        print(f"    masks/   (segmentation masks)")
        return {"status": "error", "message": "No files found"}

    print(f"Found {len(thermal_files)} thermal images, {len(mask_files)} masks")

    if max_samples:
        thermal_files = thermal_files[:max_samples]

    stats = {"n_samples": 0, "shapes": [], "has_sequence": False}

    # Check if we have a temporal sequence (multiple frames per sample)
    # or individual images
    # Try loading first file to determine format
    first = load_image(thermal_files[0])
    if first.ndim == 3 and first.shape[0] > 3:
        # Temporal sequence [T, H, W]
        stats["has_sequence"] = True
        print(f"Detected temporal sequence format: {first.shape}")
    elif first.ndim == 2:
        print(f"Detected single image format: {first.shape}")
    elif first.ndim == 3 and first.shape[-1] <= 4:
        # Multi-channel image [H, W, C], take first channel
        print(f"Detected multi-channel image: {first.shape}")

    for idx, fpath in enumerate(thermal_files):
        try:
            img = load_image(fpath)

            if img.ndim == 3 and img.shape[0] > 3:
                # Temporal sequence
                features = extract_temporal_features_from_sequence(img)
            elif img.ndim == 2:
                features = extract_features_from_single_image(img)
            elif img.ndim == 3 and img.shape[-1] <= 4:
                features = extract_features_from_single_image(img[:, :, 0])
            else:
                print(f"  [SKIP] Unknown format: {img.shape} for {fpath}")
                continue

            # Z-score normalize
            features = zscore_normalize(features)

            # Save
            basename = Path(fpath).stem
            out_path = os.path.join(output_dir, f"mendeley_cfrp_{basename}_ir.npy")
            np.save(out_path, features.astype(np.float32))

            stats["n_samples"] += 1
            stats["shapes"].append(features.shape)

            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(thermal_files)}] {basename}: {features.shape}")

        except Exception as e:
            print(f"  [ERROR] {fpath}: {e}")
            continue

    # Save masks as labels
    mask_output_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_output_dir, exist_ok=True)
    for mpath in mask_files:
        try:
            mask = load_image(mpath)
            if mask.ndim == 3:
                mask = mask[:, :, 0]  # Take first channel
            basename = Path(mpath).stem
            np.save(os.path.join(mask_output_dir, f"{basename}.npy"), mask)
        except Exception as e:
            print(f"  [MASK ERROR] {mpath}: {e}")

    # Summary
    if stats["shapes"]:
        shapes_set = set(str(s) for s in stats["shapes"])
        stats["unique_shapes"] = list(shapes_set)

    summary_path = os.path.join(output_dir, "conversion_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Mendeley CFRP Segmentation Dataset Conversion Summary\n")
        f.write(f"Source: {raw_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Samples converted: {stats['n_samples']}\n")
        f.write(f"Unique shapes: {stats.get('unique_shapes', 'N/A')}\n")
        f.write(f"Features: [max_contrast/intensity, slope/gradient, peak_time/laplacian]\n")
        f.write(f"Normalization: z-score per channel\n")
        f.write(f"Masks saved: {len(mask_files)}\n")

    print(f"\nConversion complete: {stats['n_samples']} samples → {output_dir}")
    return stats


if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw", "mendeley_cfrp_seg")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed", "mendeley_cfrp_seg")

    convert_mendeley_cfrp_seg(raw_dir, output_dir)
