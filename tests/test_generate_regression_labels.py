"""M3-2: generate_regression_labels の確認テスト"""

import unittest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.generate_regression_labels import (
    extract_size_from_filename,
    compute_defect_centroid,
)


class TestExtractSize(unittest.TestCase):
    def test_defect_size_class(self):
        sc, size = extract_size_from_filename("Defect_L1_B1_el100_H2_W2.npy")
        self.assertEqual(sc, 0)
        self.assertAlmostEqual(size, 0.0, places=5)

    def test_medium(self):
        sc, size = extract_size_from_filename("Defect_L1_B1_el100_H4_W4.npy")
        self.assertEqual(sc, 1)
        self.assertAlmostEqual(size, 0.2, places=5)  # (16-4)/60

    def test_large(self):
        sc, size = extract_size_from_filename("Defect_L1_B1_el100_H8_W8.npy")
        self.assertEqual(sc, 2)
        self.assertAlmostEqual(size, 1.0, places=5)

    def test_ndf(self):
        sc, size = extract_size_from_filename("NoiseDefectFree_0.npy")
        self.assertEqual(sc, -1)
        self.assertEqual(size, 0.0)


class TestComputeDefectCentroid(unittest.TestCase):
    def test_no_defect(self):
        labels = np.zeros(100, dtype=np.int64)
        x, y, z = np.random.randn(100), np.random.randn(100), np.random.randn(100)
        cx, cy, cz = compute_defect_centroid(labels, x, y, z)
        self.assertEqual((cx, cy, cz), (0.0, 0.0, 0.0))

    def test_onehot_no_defect(self):
        labels = np.zeros((100, 19), dtype=np.int64)
        labels[:, 0] = 1
        x, y, z = np.random.randn(100), np.random.randn(100), np.random.randn(100)
        cx, cy, cz = compute_defect_centroid(labels, x, y, z)
        self.assertEqual((cx, cy, cz), (0.0, 0.0, 0.0))

    def test_defect_centroid(self):
        labels = np.zeros(10, dtype=np.int64)
        labels[3:7] = 3  # クラス3
        x = np.array([0, 0, 0, 1, 2, 3, 4, 0, 0, 0], dtype=np.float64)
        y = np.array([0, 0, 0, 1, 2, 3, 4, 0, 0, 0], dtype=np.float64)
        z = np.zeros(10, dtype=np.float64)
        cx, cy, cz = compute_defect_centroid(labels, x, y, z)
        self.assertAlmostEqual(cx, 2.5)  # (1+2+3+4)/4
        self.assertAlmostEqual(cy, 2.5)
        self.assertAlmostEqual(cz, 0.0)


if __name__ == "__main__":
    unittest.main()
