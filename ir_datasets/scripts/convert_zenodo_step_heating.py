"""
Dataset 3: Zenodo Step-Heating Thermography CFRP → GNN-compatible IR features
Source: https://zenodo.org/records/5426793

Pedrayes et al. (2022)
- 36 multi-band images (640x480, 30 channels each)
- Channels: PCT components, PPT, kurtosis, skewness, TSR coefficients
  for both heating (10s) and cooling (10s) phases
- CFRP laminate, subsurface delaminations
- Specimen rotated 10° between captures

Output: Per-sample .npy files with IR features [H, W, C]
  - Original 30 channels preserved OR reduced to 3 via PCA
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

try:
    import h5py
except ImportError:
    h5py = None


def find_multiband_files(raw_dir: str) -> list:
    """Find multi-band image files from Zenodo dataset.

    Excludes single-channel mask files (PNG) to prevent them from
    overwriting multi-channel TIFF outputs with the same basename.
    """
    patterns = ["*.npy", "*.mat", "*.hdf5", "*.h5", "*.tif", "*.tiff"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(raw_dir, "**", p), recursive=True))
    return sorted(files)


def load_multiband_image(path: str) -> Optional[np.ndarray]:
    """Load a multi-band image file.

    Returns: [H, W, C] or [C, H, W] array
    """
    ext = Path(path).suffix.lower()

    if ext == ".npy":
        return np.load(path)

    elif ext == ".mat":
        from scipy.io import loadmat
        mat = loadmat(path)
        for key, val in mat.items():
            if key.startswith("_"):
                continue
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                return val

    elif ext in (".hdf5", ".h5"):
        if h5py is None:
            raise ImportError("h5py required for HDF5 files")
        with h5py.File(path, "r") as f:
            # Try common dataset names
            for key in ["data", "image", "thermal", "features"]:
                if key in f:
                    return f[key][:]
            # Return first dataset
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    return f[key][:]

    elif ext in (".tif", ".tiff"):
        # Use tifffile for multi-band TIFFs (e.g., 30-channel)
        try:
            import tifffile
            img = tifffile.imread(path)
            return img.astype(np.float32)
        except ImportError:
            pass
        # Fallback to Pillow
        if Image is None:
            raise ImportError("tifffile or Pillow required")
        img = Image.open(path)
        frames = []
        try:
            while True:
                frames.append(np.array(img, dtype=np.float32))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        if len(frames) > 1:
            return np.stack(frames, axis=-1)  # [H, W, C]
        return np.array(frames[0], dtype=np.float32)

    elif ext == ".png":
        if Image is None:
            raise ImportError("Pillow required")
        return np.array(Image.open(path), dtype=np.float32)

    return None


def reduce_channels_pca(features: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Reduce multi-channel features to n_components using PCA.

    Args:
        features: [H, W, C] with C > n_components
    Returns:
        reduced: [H, W, n_components]
    """
    H, W, C = features.shape
    flat = features.reshape(-1, C)  # [H*W, C]

    # Standardize (center + scale) before PCA to prevent
    # high-variance channels from dominating all components
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    standardized = (flat - mean) / std

    # PCA via SVD
    try:
        U, S, Vt = np.linalg.svd(standardized, full_matrices=False)
        components = Vt[:n_components]  # [n_components, C]
        reduced_flat = standardized @ components.T  # [H*W, n_components]
    except np.linalg.LinAlgError:
        # Fallback: use first n_components channels
        print("  [WARN] SVD failed, using first channels as fallback")
        reduced_flat = standardized[:, :n_components]

    reduced = reduced_flat.reshape(H, W, n_components)
    return reduced


def zscore_normalize(features: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel."""
    result = features.copy()
    for c in range(features.shape[-1]):
        ch = result[..., c]
        mean = ch.mean()
        std = ch.std() + 1e-8
        result[..., c] = (ch - mean) / std
    return result


def convert_zenodo_step_heating(
    raw_dir: str,
    output_dir: str,
    reduce_to_3ch: bool = True,
    save_full_channels: bool = True,
) -> dict:
    """
    Convert Zenodo step-heating dataset to GNN-compatible IR features.

    Args:
        raw_dir: Path to extracted Zenodo dataset
        output_dir: Output directory for processed files
        reduce_to_3ch: If True, apply PCA to reduce to 3 channels
        save_full_channels: If True, also save full 30-channel version

    Returns:
        Summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    files = find_multiband_files(raw_dir)
    if not files:
        print(f"ERROR: No files found in {raw_dir}")
        print("Please download from: https://zenodo.org/records/5426793")
        return {"status": "error"}

    print(f"Found {len(files)} files in {raw_dir}")

    stats = {"n_samples": 0, "shapes": [], "original_channels": []}

    for fpath in files:
        basename = Path(fpath).stem
        print(f"\nProcessing: {basename}")

        try:
            data = load_multiband_image(fpath)
            if data is None:
                print(f"  [SKIP] Could not load")
                continue

            # Ensure [H, W, C] format
            if data.ndim == 2:
                data = data[:, :, np.newaxis]
            elif data.ndim == 3:
                # If [C, H, W], transpose
                if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                    data = data.transpose(1, 2, 0)

            H, W, C = data.shape
            print(f"  Shape: ({H}, {W}, {C})")
            stats["original_channels"].append(C)

            # Save full channels version
            if save_full_channels and C > 3:
                full_dir = os.path.join(output_dir, "full_channels")
                os.makedirs(full_dir, exist_ok=True)
                full_norm = zscore_normalize(data.astype(np.float64))
                np.save(
                    os.path.join(full_dir, f"zenodo_sh_{basename}_ir_full.npy"),
                    full_norm.astype(np.float32)
                )
                print(f"  Full channels saved: ({H}, {W}, {C})")

            # Reduce to 3 channels
            if reduce_to_3ch and C > 3:
                features = reduce_channels_pca(data.astype(np.float64), n_components=3)
                print(f"  PCA reduction: {C}ch → 3ch")
            elif C >= 3:
                features = data[:, :, :3].astype(np.float64)
            else:
                # Pad to 3 channels
                features = np.zeros((H, W, 3), dtype=np.float64)
                features[:, :, :C] = data

            features = zscore_normalize(features)

            out_path = os.path.join(output_dir, f"zenodo_sh_{basename}_ir.npy")
            np.save(out_path, features.astype(np.float32))

            stats["n_samples"] += 1
            stats["shapes"].append(features.shape)
            print(f"  → {features.shape}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # Summary
    summary_path = os.path.join(output_dir, "conversion_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Zenodo Step-Heating Thermography CFRP Conversion Summary\n")
        f.write(f"Source: {raw_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Samples converted: {stats['n_samples']}\n")
        f.write(f"Original channels: {stats['original_channels']}\n")
        f.write(f"Output features: 3ch (PCA reduced) + full channels\n")
        f.write(f"Normalization: z-score per channel\n")
        f.write(f"PCA applied: {reduce_to_3ch}\n")

    print(f"\nConversion complete: {stats['n_samples']} samples → {output_dir}")
    return stats


if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw", "zenodo_step_heating")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed", "zenodo_step_heating")

    convert_zenodo_step_heating(raw_dir, output_dir)
