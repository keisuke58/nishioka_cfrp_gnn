"""設定管理"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """データ設定"""
    standardized_data_folder: str = "/home/nishioka/GNN/Defect_4x4_Normalized1"
    label_data_folder: str = "/home/nishioka/GNN/Defect19Class_OneHot_test3"
    x_coords_path: str = "/home/nishioka/GNN/BasicdataforGNN/x_2layer_normalized.npy"
    y_coords_path: str = "/home/nishioka/GNN/BasicdataforGNN/y_2layer_normalized.npy"
    z_coords_path: str = "/home/nishioka/GNN/BasicdataforGNN/z_2layer_normalized.npy"
    edges_path: str = "/home/nishioka/GNN/BasicdataforGNN/edges_2layer.npy"
    max_nodes: int = 3654
    data_file_prefix: str = "Normalized1_Defect4x4_ELNOD"
    label_file_prefix: str = "Defect19Class_L"


@dataclass
class ModelConfig:
    """モデル設定"""
    model_type: str = "GCN"  # "GCN" or "GAT"
    hidden_channels: int = 128
    num_classes: int = 19
    dropout: float = 0.2
    num_heads: int = 4  # GAT用


@dataclass
class TrainingConfig:
    """トレーニング設定"""
    learning_rate: float = 0.005
    batch_size: int = 32
    epochs: int = 150
    weight_decay: float = 5e-4
    patience: int = 30
    k_folds: int = 5
    seed: int = 42
    model_save_dir: str = "/home/nishioka/GNN/GNNmodel/19classmodel"


@dataclass
class Config:
    """全体設定"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    
    @classmethod
    def default(cls):
        """デフォルト設定を返す"""
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig()
        )
    
    @classmethod
    def for_18_classes(cls):
        """18クラス用の設定"""
        return cls(
            data=DataConfig(
                label_data_folder="/home/nishioka/GNN/DefectClass_OneHot_test1",
                label_file_prefix="DefectClass_L"
            ),
            model=ModelConfig(num_classes=18),
            training=TrainingConfig(
                model_save_dir="/home/nishioka/GNN/GNNmodel/18classmodel"
            )
        )
    
    @classmethod
    def for_10_classes(cls):
        """10クラス用の設定"""
        return cls(
            model=ModelConfig(num_classes=10),
            training=TrainingConfig(
                model_save_dir="/home/nishioka/GNN/GNNmodel/10classmodel"
            )
        )