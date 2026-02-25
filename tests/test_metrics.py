"""メトリクス計算のテスト"""

import unittest
import numpy as np

# プロジェクトルートをパスに追加
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_common.metrics import metrics_from_confusion_matrix


class TestMetrics(unittest.TestCase):
    """メトリクス計算のテスト"""
    
    def test_metrics_from_confusion_matrix(self):
        """混同行列からのメトリクス計算テスト"""
        # 簡単な2x2混同行列
        cm = np.array([
            [10, 2],  # クラス0: 10正解, 2誤分類
            [3, 15]   # クラス1: 3誤分類, 15正解
        ])
        
        metrics = metrics_from_confusion_matrix(cm)
        
        # 基本的なメトリクスの存在確認
        self.assertIn('accuracy', metrics)
        self.assertIn('macro_f1', metrics)
        self.assertIn('weighted_f1', metrics)
        self.assertIn('precision_per_class', metrics)
        self.assertIn('recall_per_class', metrics)
        
        # 精度の計算確認
        total = cm.sum()
        correct = np.trace(cm)
        expected_accuracy = correct / total
        self.assertAlmostEqual(metrics['accuracy'], expected_accuracy, places=5)
    
    def test_metrics_empty_classes(self):
        """空のクラスを含む混同行列のテスト"""
        # 3x3混同行列（クラス2は出現しない）
        cm = np.array([
            [10, 2, 0],
            [3, 15, 0],
            [0, 0, 0]
        ])
        
        metrics = metrics_from_confusion_matrix(cm)
        
        # メトリクスが計算できることを確認
        self.assertIn('macro_f1', metrics)
        self.assertIn('macro_f1_sklearn_like', metrics)
        self.assertIn('macro_f1_support_only', metrics)
    
    def test_metrics_invalid_shape(self):
        """無効な形状の混同行列のテスト"""
        # 非正方行列
        cm = np.array([[10, 2, 3]])
        
        with self.assertRaises(ValueError):
            metrics_from_confusion_matrix(cm)


if __name__ == '__main__':
    unittest.main()
