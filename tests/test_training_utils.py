"""トレーニングユーティリティのテスト"""

import unittest
import torch

# プロジェクトルートをパスに追加
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_common.training_utils import set_seed, get_distributed_info


class TestTrainingUtils(unittest.TestCase):
    """トレーニングユーティリティのテスト"""
    
    def test_set_seed(self):
        """シード設定のテスト"""
        set_seed(42)
        
        # シードが設定されていることを確認（再現性のテスト）
        x1 = torch.rand(5)
        set_seed(42)
        x2 = torch.rand(5)
        
        # 同じシードで同じ乱数が生成されることを確認
        self.assertTrue(torch.allclose(x1, x2))
    
    def test_get_distributed_info(self):
        """分散学習情報取得のテスト"""
        rank, world_size, master_addr, master_port = get_distributed_info()
        
        # デフォルト値の確認
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)
        self.assertEqual(master_addr, '127.0.0.1')
        self.assertEqual(master_port, '12355')


if __name__ == '__main__':
    unittest.main()
