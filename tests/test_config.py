"""設定管理のテスト"""

import unittest
import os
import tempfile
from pathlib import Path
import yaml

# プロジェクトルートをパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.config_loader import Config, TrainingConfig, HardwareConfig


class TestConfig(unittest.TestCase):
    """設定管理のテスト"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        config = Config()
        self.assertEqual(config.training.learning_rate, 0.002)
        self.assertEqual(config.training.epochs, 2000)
        self.assertEqual(config.hardware.nproc_per_node, 4)
    
    def test_config_from_env(self):
        """環境変数からの設定読み込みテスト"""
        os.environ['LR'] = '0.001'
        os.environ['EPOCHS'] = '1000'
        os.environ['NPROC_PER_NODE'] = '2'
        
        try:
            config = Config.from_env()
            self.assertEqual(config.training.learning_rate, 0.001)
            self.assertEqual(config.training.epochs, 1000)
            self.assertEqual(config.hardware.nproc_per_node, 2)
        finally:
            # クリーンアップ
            os.environ.pop('LR', None)
            os.environ.pop('EPOCHS', None)
            os.environ.pop('NPROC_PER_NODE', None)
    
    def test_config_from_yaml(self):
        """YAMLファイルからの設定読み込みテスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_data = {
                'training': {
                    'learning_rate': 0.005,
                    'epochs': 500,
                },
                'hardware': {
                    'nproc_per_node': 8,
                }
            }
            yaml.dump(yaml_data, f)
            yaml_path = Path(f.name)
        
        try:
            config = Config.from_yaml(yaml_path)
            self.assertEqual(config.training.learning_rate, 0.005)
            self.assertEqual(config.training.epochs, 500)
            self.assertEqual(config.hardware.nproc_per_node, 8)
        finally:
            yaml_path.unlink()
    
    def test_config_to_dict(self):
        """設定の辞書変換テスト"""
        config = Config()
        config_dict = config.to_dict()
        
        self.assertIn('training', config_dict)
        self.assertIn('hardware', config_dict)
        self.assertEqual(config_dict['training']['learning_rate'], 0.002)


if __name__ == '__main__':
    unittest.main()
