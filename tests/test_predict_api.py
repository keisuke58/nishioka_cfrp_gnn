"""M3-5: predict_api の確認テスト"""

import unittest
import numpy as np
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPredictAPI(unittest.TestCase):
    """PredictAPI の動作確認（モデル読み込みをモックする軽量テスト）"""

    def test_import(self):
        from tools.predict_api import PredictAPI
        self.assertIsNotNone(PredictAPI)

    def test_predict_api_with_real_data(self):
        """実データで推論（モデルが存在する場合のみ）"""
        model_path = Path(__file__).parent.parent / "runs/20260116_104929_nogit_dsNDF_ep2000_lr0p001_F10p730/outputs/GNN_model/19classmodel_hole_zscore/GATModel_20260116_104950_Best_Final.pth"
        data_path = Path(__file__).parent.parent / "GNN_hole_2026/all_sub_hole_defect_zscore_noise/test/Defect_L10_B102_el1169_H2_W2.npy"
        if not model_path.exists() or not data_path.exists():
            self.skipTest("Model or test data not found")
        from tools.predict_api import PredictAPI
        api = PredictAPI(model_path=str(model_path))
        out = api.predict(str(data_path))
        self.assertEqual(out.get("error_code", 0), 0)
        self.assertIn("has_defect", out)
        self.assertIn("defect_nodes", out)
        self.assertIn("confidence", out)
        self.assertIn("location_pred", out)
        self.assertEqual(out["location_pred"].shape[0], 13942)


if __name__ == "__main__":
    unittest.main()
