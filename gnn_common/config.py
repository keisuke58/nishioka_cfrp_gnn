"""設定管理"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


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
    model_type: str = "GCN"  # "GCN", "GAT", or "GATMultiTask"
    hidden_channels: int = 128
    num_classes: int = 19
    dropout: float = 0.2
    num_heads: int = 4  # GAT用
    input_channels: int = 4  # ノード特徴量の次元数 (4=FEM, 7=FEM+IR)
    multitask: bool = False
    num_size_classes: int = 4  # サイズクラス数 (micro/small/medium/large)
    size_weight: float = 1.0  # multi-task時のsize lossの重み
    logit_adjust_tau: float = 3.0  # LogitAdjust強度
    use_synthetic_ir: bool = False  # 合成IR特徴量を使用するか


@dataclass
class IRConfig:
    """IR thermography fusion settings (Issue #3)"""
    ir_data_dir: Optional[str] = None           # IR processed data directory
    ir_feature_dim: int = 3                     # Number of IR feature channels
    ir_normalize: bool = True                   # Z-score normalize IR features
    interpolation: str = "bilinear"             # Interpolation method
    fusion_mode: str = "concatenate"            # "concatenate" (→7D) or "add"
    calibration_file: Optional[str] = None      # Affine calibration .npy file
    use_synthetic_ir: bool = False              # Generate synthetic IR from DSPSS
    synthetic_noise_std: float = 0.05           # Noise for synthetic IR
    dataset_source: str = "none"               # "mendeley_cfrp", "mendeley_composite", "zenodo_sh", "synthetic"


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
    ir: IRConfig = None

    def __post_init__(self):
        if self.ir is None:
            self.ir = IRConfig()
    
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