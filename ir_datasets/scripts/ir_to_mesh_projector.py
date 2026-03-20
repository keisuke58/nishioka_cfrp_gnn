"""
IR Image → FEM Mesh Node Projection

Projects IR thermography features from image space [H, W, 3] onto
FEM mesh nodes [N, 3] for GNN input.

This is the critical bridge between public IR datasets and the existing
GNN pipeline (gnn_common/data_utils.py).

Flow:
  IR image [H, W, 3] → Affine/grid alignment → Bilinear interpolation
  → Node features [N, 3] (surface nodes only, interior = 0)
  → Concatenate with FEM features [N, 4] → [N, 7]
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass
class AffineCalibration:
    """2D affine transform parameters: FEM (x,y) → IR pixel (u,v)

    u = a00*x + a01*y + a02
    v = a10*x + a11*y + a12
    """
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    @classmethod
    def from_correspondences(cls, fem_points: np.ndarray, ir_points: np.ndarray):
        """Compute affine from ≥3 corresponding point pairs.

        Args:
            fem_points: [K, 2] FEM coordinates (x, y)
            ir_points: [K, 2] IR pixel coordinates (u, v)
        """
        K = fem_points.shape[0]
        assert K >= 3, "Need ≥3 point pairs"

        # Build system: [x, y, 1] * A = [u, v]
        src = np.column_stack([fem_points, np.ones(K)])  # [K, 3]
        # Solve for u and v separately
        Au, _, _, _ = np.linalg.lstsq(src, ir_points[:, 0], rcond=None)
        Av, _, _, _ = np.linalg.lstsq(src, ir_points[:, 1], rcond=None)

        M = np.eye(3, dtype=np.float64)
        M[0, :] = Au
        M[1, :] = Av
        return cls(matrix=M)

    @classmethod
    def from_bounds(cls, fem_bounds: Tuple, ir_shape: Tuple):
        """Auto-calibration: map FEM bounding box to IR image bounds.

        Args:
            fem_bounds: (x_min, x_max, y_min, y_max)
            ir_shape: (H, W) of IR image
        """
        x_min, x_max, y_min, y_max = fem_bounds
        H, W = ir_shape

        # Map FEM (x,y) → pixel (u,v) with margin
        margin = 0.05  # 5% border margin
        effective_W = W * (1 - 2 * margin)
        effective_H = H * (1 - 2 * margin)

        sx = effective_W / max(x_max - x_min, 1e-10)
        sy = effective_H / max(y_max - y_min, 1e-10)

        M = np.array([
            [sx, 0, W * margin - sx * x_min],
            [0, sy, H * margin - sy * y_min],
            [0, 0, 1]
        ], dtype=np.float64)

        return cls(matrix=M)

    def transform(self, fem_xy: np.ndarray) -> np.ndarray:
        """Transform FEM coordinates to IR pixel coordinates.

        Args:
            fem_xy: [N, 2] (x, y)
        Returns:
            ir_uv: [N, 2] (u, v) - column, row in image
        """
        N = fem_xy.shape[0]
        src = np.column_stack([fem_xy, np.ones(N)])  # [N, 3]
        uv = src @ self.matrix[:2, :].T  # [N, 2]
        return uv

    def save(self, path: str):
        np.save(path, self.matrix)

    @classmethod
    def load(cls, path: str):
        return cls(matrix=np.load(path))


def bilinear_interpolate(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on image at (u, v) coordinates.

    Args:
        image: [H, W, C] feature map
        u: [N] column coordinates (float)
        v: [N] row coordinates (float)

    Returns:
        values: [N, C]
    """
    H, W, C = image.shape
    N = len(u)

    # Clamp to valid range
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)

    du = u - u0
    dv = v - v0

    # Bilinear weights
    w00 = (1 - du) * (1 - dv)  # [N]
    w01 = (1 - du) * dv
    w10 = du * (1 - dv)
    w11 = du * dv

    # Interpolate
    values = (
        w00[:, None] * image[v0, u0] +
        w01[:, None] * image[v1, u0] +
        w10[:, None] * image[v0, u1] +
        w11[:, None] * image[v1, u1]
    )

    return values  # [N, C]


def project_ir_to_mesh(
    ir_features: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    surface_mask: Optional[np.ndarray] = None,
    calibration: Optional[AffineCalibration] = None,
    normalize: bool = False,
    n_surface_nodes: Optional[int] = None,
) -> np.ndarray:
    """
    Project IR image features onto FEM mesh nodes.

    Args:
        ir_features: [H, W, 3] IR feature map (from converter output,
                     already z-score normalized)
        x_coords: [N] mesh node x coordinates (should be normalized to [0,1])
        y_coords: [N] mesh node y coordinates (should be normalized to [0,1])
        z_coords: [N] mesh node z coordinates (should be normalized to [0,1])
        surface_mask: [N] bool, True for surface nodes. If None, auto-detect
                      using n_surface_nodes or the 2-layer convention
                      (first half = surface, second half = interior).
        calibration: Affine transform FEM→IR. Auto-computed if None.
        normalize: Z-score normalize output features. Default False because
                   the input IR features are typically already z-score normalized.
        n_surface_nodes: Number of surface nodes (first n_surface_nodes in the
                         array). Used for auto surface mask when surface_mask
                         is None.

    Returns:
        node_ir_features: [N, 3] IR features projected onto mesh nodes
    """
    N = len(x_coords)
    H, W = ir_features.shape[:2]
    C = ir_features.shape[2] if ir_features.ndim == 3 else 1

    if ir_features.ndim == 2:
        ir_features = ir_features[:, :, np.newaxis]

    # Auto-detect surface nodes if not provided
    # For 2-layer meshes: first half = surface (layer 1), second half = interior (layer 2)
    if surface_mask is None:
        if n_surface_nodes is not None:
            surface_mask = np.zeros(N, dtype=bool)
            surface_mask[:n_surface_nodes] = True
        elif N % 2 == 0:
            # 2-layer convention: first half is surface
            surface_mask = np.zeros(N, dtype=bool)
            surface_mask[:N // 2] = True
        else:
            # Fallback: all nodes are surface
            surface_mask = np.ones(N, dtype=bool)

    # Auto-calibration if not provided
    if calibration is None:
        # Use only surface node coordinates for bounding box
        sx = surface_mask
        fem_bounds = (
            x_coords[sx].min(), x_coords[sx].max(),
            y_coords[sx].min(), y_coords[sx].max(),
        )
        calibration = AffineCalibration.from_bounds(fem_bounds, (H, W))

    # Project surface nodes
    node_ir = np.zeros((N, C), dtype=np.float64)
    surface_idx = np.where(surface_mask)[0]

    if len(surface_idx) > 0:
        fem_xy = np.column_stack([x_coords[surface_idx], y_coords[surface_idx]])
        uv = calibration.transform(fem_xy)  # [n_surface, 2]

        # Bilinear interpolation for ALL channels simultaneously
        node_ir[surface_idx] = bilinear_interpolate(
            ir_features.astype(np.float64),
            uv[:, 0], uv[:, 1]
        )

    # Optional Z-score normalize using surface node statistics
    if normalize:
        for c in range(C):
            surface_vals = node_ir[surface_idx, c]
            if len(surface_vals) > 0:
                mean = surface_vals.mean()
                std = surface_vals.std() + 1e-8
                node_ir[:, c] = (node_ir[:, c] - mean) / std

    return node_ir.astype(np.float32)


def create_fused_node_features(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    dspss_values: np.ndarray,
    ir_features: np.ndarray,
    surface_mask: Optional[np.ndarray] = None,
    calibration: Optional[AffineCalibration] = None,
) -> np.ndarray:
    """
    Create fused FEM+IR node features for GNN input.

    Current pipeline: [x, y, z, DSPSS] = 4D
    This function: [x, y, z, DSPSS, ir1, ir2, ir3] = 7D

    Args:
        x_coords, y_coords, z_coords: [N] node coordinates
        dspss_values: [N] DSPSS stress values
        ir_features: [H, W, 3] IR feature map
        surface_mask: [N] bool
        calibration: Affine calibration

    Returns:
        fused_features: [N, 7]
    """
    # Original FEM features
    fem_features = np.vstack([x_coords, y_coords, z_coords, dspss_values]).T  # [N, 4]

    # Project IR onto mesh (input is already z-score normalized)
    ir_node_features = project_ir_to_mesh(
        ir_features, x_coords, y_coords, z_coords,
        surface_mask=surface_mask,
        calibration=calibration,
        normalize=False,
    )  # [N, 3]

    # Concatenate
    fused = np.hstack([fem_features, ir_node_features])  # [N, 7]
    return fused.astype(np.float32)


def batch_project_ir_dataset(
    ir_processed_dir: str,
    output_dir: str,
    coords_dir: str = "/home/nishioka/GNN/GNN_hole/GNN_hole_data",
) -> dict:
    """
    Batch-project all processed IR .npy files onto FEM mesh.

    Args:
        ir_processed_dir: Directory with *_ir.npy files [H, W, 3]
        output_dir: Directory to save *_ir_mesh.npy files [N, 3]
        coords_dir: Directory with FEM mesh coordinate files

    Returns:
        Summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load NORMALIZED FEM mesh coordinates (range [0, 1])
    # The normalized coordinates ensure proper mapping to IR image pixel space.
    # Raw coordinates (hole_x_2layer.npy) are in physical units (mm) and must
    # NOT be used — they produce pixel indices in the hundreds of thousands,
    # causing all nodes to clamp to the image boundary.
    coord_names_priority = [
        ("normalized_x_2layer.npy", "normalized_y_2layer.npy", "normalized_z_2layer.npy"),
        ("hole_x_2layer.npy", "hole_y_2layer.npy", "hole_z_2layer.npy"),
    ]

    x, y, z = None, None, None
    for xf, yf, zf in coord_names_priority:
        xp = os.path.join(coords_dir, xf)
        yp = os.path.join(coords_dir, yf)
        zp = os.path.join(coords_dir, zf)
        if os.path.exists(xp) and os.path.exists(yp) and os.path.exists(zp):
            x = np.load(xp)
            y = np.load(yp)
            z = np.load(zp)
            print(f"Loaded coordinates: {xf} (range x=[{x.min():.4f}, {x.max():.4f}], "
                  f"y=[{y.min():.4f}, {y.max():.4f}], z=[{z.min():.4f}, {z.max():.4f}])")
            break

    if x is None:
        raise FileNotFoundError(
            f"No coordinate files found in {coords_dir}. "
            f"Expected normalized_{{x,y,z}}_2layer.npy or hole_{{x,y,z}}_2layer.npy"
        )

    N = len(x)
    n_surface = N // 2  # 2-layer mesh: first half = surface
    print(f"FEM mesh: {N} nodes ({n_surface} surface + {N - n_surface} interior)")

    # Find IR feature files
    ir_files = sorted(Path(ir_processed_dir).glob("*_ir.npy"))
    print(f"Found {len(ir_files)} IR feature files")

    stats = {"n_projected": 0, "channel_stats": []}

    for ir_path in ir_files:
        try:
            ir_feat = np.load(str(ir_path))  # [H, W, 3]
            if ir_feat.ndim != 3:
                print(f"  [SKIP] {ir_path.name}: unexpected shape {ir_feat.shape}")
                continue

            node_ir = project_ir_to_mesh(
                ir_feat, x, y, z,
                normalize=False,  # Input is already z-score normalized
                n_surface_nodes=n_surface,
            )  # [N, 3]

            out_name = ir_path.stem.replace("_ir", "_ir_mesh") + ".npy"
            np.save(os.path.join(output_dir, out_name), node_ir)

            stats["n_projected"] += 1

            # Log channel statistics for verification
            ch_info = []
            for c in range(node_ir.shape[1]):
                surf_vals = node_ir[:n_surface, c]
                ch_info.append(f"ch{c}: mean={surf_vals.mean():.3f}, std={surf_vals.std():.3f}, "
                              f"range=[{surf_vals.min():.3f}, {surf_vals.max():.3f}]")

            if stats["n_projected"] <= 3 or stats["n_projected"] % 10 == 0:
                print(f"  [{stats['n_projected']}] {ir_path.name} → {out_name}")
                for info in ch_info:
                    print(f"       {info}")

        except Exception as e:
            print(f"  [ERROR] {ir_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nProjected {stats['n_projected']} IR files onto mesh → {output_dir}")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Project IR features onto FEM mesh")
    parser.add_argument("--ir-dir", default=None,
                       help="Directory with processed IR .npy files")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    parser.add_argument("--coords-dir", default="/home/nishioka/GNN/GNN_hole/GNN_hole_data",
                       help="FEM coordinate directory")
    args = parser.parse_args()

    # Default: process zenodo step heating dataset
    ir_dir = args.ir_dir
    if ir_dir is None:
        ir_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "processed", "zenodo_step_heating"
        )

    output = args.output
    if output is None:
        output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "processed", "mesh_projected", "zenodo"
        )

    print(f"IR input dir:  {ir_dir}")
    print(f"Output dir:    {output}")
    print(f"Coords dir:    {args.coords_dir}")
    print()

    batch_project_ir_dataset(ir_dir, output, args.coords_dir)
