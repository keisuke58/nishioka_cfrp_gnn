#!/usr/bin/env python3
"""
M3-5: 推論API（検出+位置+重症度+信頼度）

単一サンプル入力 → 構造化出力（検出、位置、サイズ、回帰、信頼度）

Usage:
    from tools.predict_api import PredictAPI
    api = PredictAPI(model_path="runs/.../GATModel_*_Best_Final.pth")
    out = api.predict(values_npy_path)
    # out = {"has_defect": bool, "location_pred": [N], "defect_nodes": int, "confidence": float, ...}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLE_DATA = REPO_ROOT / "GNN_hole/GNN_hole_data"
MAX_NODES = 13942
DEFAULT_MODEL = (
    REPO_ROOT
    / "runs/20260116_104929_nogit_dsNDF_ep2000_lr0p001_F10p730/outputs/GNN_model/19classmodel_hole_zscore/GATModel_20260116_104950_Best_Final.pth"
)


def _load_model(model_path: Path, device: torch.device):
    """GATModel (two-stage) をロード"""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "GNN_hole_2026/GNN_program"))
    from GNN_zscore_sub_noise_defect_free import GATModel

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    w = state.get("conv1.lin_src.weight", state.get("conv1.weight"))
    hidden = (w.shape[0] // 4) if w is not None else 16
    model = GATModel(hidden_channels=hidden, num_classes=19, dropout=0.0, edge_drop_prob=0.0)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


class PredictAPI:
    """単一サンプル推論API"""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path or DEFAULT_MODEL)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _load_model(self.model_path, self.device)
        self._x = np.load(HOLE_DATA / "normalized_x_2layer.npy")[:MAX_NODES]
        self._y = np.load(HOLE_DATA / "normalized_y_2layer.npy")[:MAX_NODES]
        self._z = np.load(HOLE_DATA / "normalized_z_2layer.npy")[:MAX_NODES]
        ei = np.load(HOLE_DATA / "hole_edges_2layer_best.npy")
        if ei.shape[0] != 2:
            ei = ei.T
        self._edge_index = torch.tensor(ei, dtype=torch.long, device=self.device)

    def predict(
        self,
        values: np.ndarray | str | Path,
    ) -> Dict[str, Any]:
        """
        単一サンプルを推論。

        Args:
            values: [N] float のノード値、または .npy パス

        Returns:
            {
                "has_defect": bool,
                "defect_nodes": int,
                "location_pred": np.ndarray [N],
                "pred_classes": list[int],
                "confidence": float,  # 予測の最大確率（欠陥ノードの平均）
                "error_code": 0,
            }
        """
        if isinstance(values, (str, Path)):
            values = np.load(values)[:MAX_NODES]
        values = np.asarray(values, dtype=np.float32).flatten()[:MAX_NODES]
        if len(values) != MAX_NODES:
            return {"error_code": 1, "message": "Input shape mismatch"}

        feat = np.vstack((self._x, self._y, self._z, values)).T
        x_t = torch.tensor(feat, dtype=torch.float, device=self.device)
        from torch_geometric.data import Data
        data = Data(x=x_t, edge_index=self._edge_index)

        with torch.no_grad():
            logits = self.model(data)
        if isinstance(logits, dict):
            logits = logits.get("location", logits.get("combined_logits", logits))
        if not isinstance(logits, torch.Tensor):
            return {"error_code": 2, "message": "Model output format error"}
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1)
        max_pred_prob = np.max(probs, axis=1)

        defect_mask = pred > 0
        n_defect = int(defect_mask.sum())
        has_defect = n_defect > 0
        confidence = float(np.mean(max_pred_prob[defect_mask])) if has_defect else float(np.mean(max_pred_prob))

        return {
            "has_defect": has_defect,
            "defect_nodes": n_defect,
            "location_pred": pred,
            "pred_classes": list(np.unique(pred)),
            "confidence": confidence,
            "error_code": 0,
        }


def main():
    parser = argparse.ArgumentParser(description="M3-5: Predict API CLI")
    parser.add_argument("input", type=str, help="Input .npy path")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    api = PredictAPI(model_path=args.model)
    out = api.predict(args.input)
    if out.get("error_code", 0) != 0:
        print(f"[ERROR] {out.get('message', 'Unknown error')}")
        return 1
    print(f"has_defect: {out['has_defect']}")
    print(f"defect_nodes: {out['defect_nodes']}")
    print(f"pred_classes: {out['pred_classes']}")
    print(f"confidence: {out['confidence']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
