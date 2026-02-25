#!/usr/bin/env python3
"""
Standalone data loader for the CFRP GNN sample dataset.

This module is self-contained — it has no dependency on the full GNN project.
It only requires numpy and torch_geometric (PyG).

Usage:
    from sample_data_loader import CFRPSampleDataset

    dataset = CFRPSampleDataset("sample_dataset")
    print(f"Number of samples: {len(dataset)}")
    data = dataset[0]
    print(f"Nodes: {data.num_nodes}, Features: {data.x.shape[1]}, Classes: {data.y.max()+1}")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
    from torch_geometric.data import Data, Dataset
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = None
    Dataset = object


class CFRPSampleDataset(Dataset if HAS_PYG else object):
    """PyG-compatible dataset loader for the CFRP GNN sample dataset.

    Args:
        root_dir: Path to the sample dataset directory (containing data/, labels/, metadata.json)
        transform: Optional PyG transform
        pre_transform: Optional PyG pre-transform
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        pre_transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / "data"
        self.label_dir = self.root_dir / "labels"
        self.metadata = self._load_metadata()

        # Build file list
        self._data_files = sorted(
            f.name for f in self.data_dir.iterdir()
            if f.suffix == ".npy" and not f.name.endswith("_19label.npy")
        )

        if HAS_PYG:
            super().__init__(str(self.root_dir), transform, pre_transform)

    def _load_metadata(self) -> dict:
        meta_path = self.root_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}

    def len(self) -> int:
        return len(self._data_files)

    def __len__(self) -> int:
        return self.len()

    def get(self, idx: int):
        return self._load_sample(idx)

    def __getitem__(self, idx: int):
        return self.get(idx)

    def _load_sample(self, idx: int):
        """Load a single sample as a PyG Data object."""
        if not HAS_PYG:
            raise ImportError(
                "torch and torch_geometric are required. "
                "Install with: pip install torch torch_geometric"
            )

        data_fname = self._data_files[idx]
        base = os.path.splitext(data_fname)[0]
        label_fname = f"{base}_19label.npy"

        # Load features
        features = np.load(self.data_dir / data_fname)
        num_nodes = self.metadata.get("num_nodes_per_sample")
        num_features = self.metadata.get("num_features", 4)

        if features.ndim == 1 and num_nodes:
            # Stacked 1D array → reshape to [N, F]
            features = features.reshape(num_features, num_nodes).T
        elif features.ndim == 1:
            # Try to infer shape
            if len(features) % 4 == 0:
                n = len(features) // 4
                features = features.reshape(4, n).T
            else:
                features = features.reshape(-1, 1)

        x = torch.tensor(features, dtype=torch.float)

        # Load labels
        label_path = self.label_dir / label_fname
        if label_path.exists():
            labels = np.load(label_path)
            if labels.ndim == 2:
                # One-hot → class indices
                y = torch.tensor(labels.argmax(axis=1), dtype=torch.long)
            else:
                y = torch.tensor(labels, dtype=torch.long)
        else:
            # Defect-free: all class 0
            y = torch.zeros(x.shape[0], dtype=torch.long)

        data = Data(x=x, y=y)
        data.filename = data_fname
        return data

    @staticmethod
    def load_all(root_dir: str) -> List:
        """Convenience method to load all samples as a list of PyG Data objects."""
        dataset = CFRPSampleDataset(root_dir)
        return [dataset[i] for i in range(len(dataset))]

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        lines = [
            f"CFRPSampleDataset: {self.root_dir}",
            f"  Samples: {len(self)}",
            f"  Metadata: {self.metadata.get('description', 'N/A')}",
        ]
        if self.metadata.get("num_nodes_per_sample"):
            lines.append(f"  Nodes/sample: {self.metadata['num_nodes_per_sample']}")
        if self.metadata.get("defect_size_distribution"):
            lines.append(f"  Size dist: {self.metadata['defect_size_distribution']}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "sample_dataset"
    ds = CFRPSampleDataset(root)
    print(ds.summary())
    if len(ds) > 0:
        sample = ds[0]
        print(f"\nSample 0: x={sample.x.shape}, y={sample.y.shape}, file={sample.filename}")
