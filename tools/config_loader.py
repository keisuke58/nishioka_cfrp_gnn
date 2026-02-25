#!/usr/bin/env python3
"""
設定ファイルの読み込みと管理ユーティリティ

環境変数とYAML設定ファイルの両方をサポート
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """トレーニング設定"""
    profile: str = "ndf_recommended"
    dataset_tag: str = "NDF"
    epochs: int = 2000
    patience: int = 300
    batch_size: int = 64
    hidden_channels: int = 16
    learning_rate: float = 0.002
    weight_decay: float = 5e-4
    dropout: float = 0.10
    edge_drop_prob: float = 0.01
    data_usage_ratio: float = 1.0


@dataclass
class HardwareConfig:
    """ハードウェア設定"""
    nproc_per_node: int = 4
    omp_num_threads: int = 4


@dataclass
class FeatureFlags:
    """機能フラグ"""
    use_amp: bool = True
    use_class_frequency_sampler: bool = True
    use_onecycle: bool = True


@dataclass
class LoggingConfig:
    """ロギング設定"""
    log_level: str = "INFO"
    verbose_print: bool = False


@dataclass
class SweepConfig:
    """スイープ設定"""
    max_retries: int = 0
    retry_delay: int = 60


@dataclass
class PathsConfig:
    """パス設定"""
    conda_py: str = "/home/nishioka/miniconda3/envs/gnn_final_env/bin/python"
    torchrun: str = "/home/nishioka/miniconda3/envs/gnn_final_env/bin/torchrun"
    script: str = "/home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free.py"
    workdir: str = "/home/nishioka/GNN/GNN_hole_2026/GNN_program"
    launcher: str = "/home/nishioka/GNN/tools/launch_run.py"
    output_base: str = "/home/nishioka/GNN/runs"


@dataclass
class Config:
    """全体設定"""
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    env: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> Config:
        """YAMLファイルから設定を読み込む"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
        return cls(
            paths=PathsConfig(**data.get('paths', {})),
            training=TrainingConfig(**data.get('training', {})),
            hardware=HardwareConfig(**data.get('hardware', {})),
            features=FeatureFlags(**data.get('features', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            sweep=SweepConfig(**data.get('sweep', {})),
            env=data.get('env', {}),
        )
    
    @classmethod
    def from_env(cls) -> Config:
        """環境変数から設定を読み込む"""
        def get_int(key: str, default: int) -> int:
            val = os.environ.get(key)
            return int(val) if val else default
        
        def get_float(key: str, default: float) -> float:
            val = os.environ.get(key)
            return float(val) if val else default
        
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, '').lower()
            if val in ('1', 'true', 'yes', 'on'):
                return True
            elif val in ('0', 'false', 'no', 'off'):
                return False
            return default
        
        return cls(
            paths=PathsConfig(
                conda_py=os.environ.get('CONDA_PY', PathsConfig.conda_py),
                torchrun=os.environ.get('TORCHRUN', PathsConfig.torchrun),
                script=os.environ.get('SCRIPT', PathsConfig.script),
                workdir=os.environ.get('WORKDIR', PathsConfig.workdir),
                launcher=os.environ.get('LAUNCHER', PathsConfig.launcher),
                output_base=os.environ.get('OUTPUT_BASE', PathsConfig.output_base),
            ),
            training=TrainingConfig(
                profile=os.environ.get('PROFILE', TrainingConfig.profile),
                dataset_tag=os.environ.get('DATASET_TAG', TrainingConfig.dataset_tag),
                epochs=get_int('EPOCHS', TrainingConfig.epochs),
                patience=get_int('PATIENCE', TrainingConfig.patience),
                batch_size=get_int('BATCH_SIZE', TrainingConfig.batch_size),
                hidden_channels=get_int('HIDDEN', TrainingConfig.hidden_channels),
                learning_rate=get_float('LR', TrainingConfig.learning_rate),
                weight_decay=get_float('WD', TrainingConfig.weight_decay),
                dropout=get_float('DROPOUT', TrainingConfig.dropout),
                edge_drop_prob=get_float('EDGE_DROP', TrainingConfig.edge_drop_prob),
                data_usage_ratio=get_float('DATA_USAGE_RATIO', TrainingConfig.data_usage_ratio),
            ),
            hardware=HardwareConfig(
                nproc_per_node=get_int('NPROC_PER_NODE', HardwareConfig.nproc_per_node),
                omp_num_threads=get_int('OMP_NUM_THREADS', HardwareConfig.omp_num_threads),
            ),
            features=FeatureFlags(
                use_amp=get_bool('USE_AMP', FeatureFlags.use_amp),
                use_class_frequency_sampler=get_bool('USE_CLASS_FREQ_SAMPLER', FeatureFlags.use_class_frequency_sampler),
                use_onecycle=FeatureFlags.use_onecycle,  # Always True if flag is present
            ),
            logging=LoggingConfig(
                log_level=os.environ.get('LOG_LEVEL', LoggingConfig.log_level),
                verbose_print=get_bool('VERBOSE_PRINT', LoggingConfig.verbose_print),
            ),
            sweep=SweepConfig(
                max_retries=get_int('MAX_RETRIES', SweepConfig.max_retries),
                retry_delay=get_int('RETRY_DELAY', SweepConfig.retry_delay),
            ),
            env={k: v for k, v in os.environ.items() if k.startswith(('TORCH_', 'NCCL_', 'CUDA_'))},
        )
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> Config:
        """設定を読み込む（YAML優先、なければ環境変数）"""
        if config_path and Path(config_path).exists():
            return cls.from_yaml(Path(config_path))
        return cls.from_env()
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書に変換"""
        return {
            'paths': self.paths.__dict__,
            'training': self.training.__dict__,
            'hardware': self.hardware.__dict__,
            'features': self.features.__dict__,
            'logging': self.logging.__dict__,
            'sweep': self.sweep.__dict__,
            'env': self.env,
        }


if __name__ == '__main__':
    # Test loading
    config = Config.load()
    print("Loaded configuration:")
    import json
    print(json.dumps(config.to_dict(), indent=2, default=str))
