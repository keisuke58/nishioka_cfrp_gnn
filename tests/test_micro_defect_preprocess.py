"""M2-2: micro_defect_preprocess の確認テスト"""

import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.micro_defect_preprocess import (
    halpin_tsai_E_ratio,
    effective_E_ratio,
    apply_defect_params,
    apply_defect_blur,
)


class TestHalpinTsai(unittest.TestCase):
    """Halpin-Tsai 有効弾性係数"""

    def test_phi_zero(self):
        self.assertAlmostEqual(halpin_tsai_E_ratio(0.0), 1.0, places=5)

    def test_phi_increases_reduces_E(self):
        e0 = halpin_tsai_E_ratio(0.0)
        e1 = halpin_tsai_E_ratio(0.1)
        e2 = halpin_tsai_E_ratio(0.2)
        self.assertLess(e2, e1)
        self.assertLess(e1, e0)

    def test_phi_clamped(self):
        e = halpin_tsai_E_ratio(0.5)
        self.assertGreaterEqual(e, 0.0)
        self.assertLessEqual(e, 1.0)


class TestApplyDefectParams(unittest.TestCase):
    """apply_defect_params の動作確認"""

    def test_basic_scale(self):
        rng = np.random.RandomState(42)
        v = rng.randn(100).astype(np.float32)
        mask = np.zeros(100, dtype=bool)
        mask[40:60] = True
        ei = np.array([[0, 1, 2], [1, 2, 3]])
        out = apply_defect_params(v, mask, ei, {"E_ratio": 0.5})
        self.assertEqual(out.shape, v.shape)
        # 欠陥ノードはスケールされる
        np.testing.assert_almost_equal(out[40:60], v[40:60] * 0.5)
        self.assertTrue(np.allclose(out[~mask], v[~mask]))

    def test_phi_overrides_E_ratio(self):
        v = np.ones(50, dtype=np.float32)
        mask = np.zeros(50, dtype=bool)
        mask[20:30] = True
        ei = np.array([[0], [1]])
        out_phi = apply_defect_params(v, mask, ei, {"phi": 0.1})
        out_E = apply_defect_params(v, mask, ei, {"E_ratio": halpin_tsai_E_ratio(0.1)})
        np.testing.assert_almost_equal(out_phi[20:30], out_E[20:30])


class TestApplyDefectBlur(unittest.TestCase):
    """境界ぼかしの動作確認"""

    def test_no_blur_preserves(self):
        v = np.arange(20, dtype=np.float32)
        mask = np.zeros(20, dtype=bool)
        mask[5:15] = True
        ei = np.array([[0, 1, 2], [1, 2, 3]])
        out = apply_defect_blur(v, mask, ei, blur_sigma=0.0, num_iter=1)
        np.testing.assert_almost_equal(out, v)

    def test_blur_smooth(self):
        v = np.zeros(30, dtype=np.float32)
        v[10:20] = 1.0
        mask = np.zeros(30, dtype=bool)
        mask[10:20] = True
        ei = np.array([[i, i + 1] for i in range(29)], dtype=np.int64).T
        out = apply_defect_blur(v, mask, ei, blur_sigma=1.0, num_iter=3)
        self.assertEqual(out.shape, v.shape)
        # 境界付近で値が変化する（厳密な値はグラフ構造依存）
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1))


if __name__ == "__main__":
    unittest.main()
