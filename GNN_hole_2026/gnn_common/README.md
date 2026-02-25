# GNN共通モジュール

プロジェクト全体で使用できる共通モジュールです。

## 配置場所

- **プロジェクトルート**: `/home/nishioka/GNN/gnn_common/` - 全プロジェクトで使用可能
- **DDPフォルダ**: `/home/nishioka/GNN/DDP/common/` - DDPスクリプト用（同じ内容）

## 使用方法

### 1. DDPスクリプトから使用

```python
# DDP/ フォルダ内のスクリプトから
from common.models import GCNModel, GATModel
from common.data_utils import create_data_label_pairs, prepare_data
from common.training_utils import set_seed, setup, cleanup
```

### 2. GNN_hole_2026プロジェクトから使用

```python
# GNN_hole_2026/GNN_program/ フォルダ内のスクリプトから
import sys
sys.path.insert(0, '/home/nishioka/GNN')
from gnn_common.models import GCNModel, GATModel
from gnn_common.data_utils import create_data_label_pairs, prepare_data
from gnn_common.training_utils import set_seed, setup, cleanup
```

## モジュール構成

### `models.py`
- `GCNModel`: Graph Convolutional Network モデル
- `GATModel`: Graph Attention Network モデル（シンプル版）

**注意**: `GNN_zscore_sub_noise_defect_free.py`で使用されている2段階GATモデルとは異なります。

### `data_utils.py`
- `extract_layer_block()`: ファイル名から層とブロック番号を抽出
- `create_data_label_pairs()`: データファイルとラベルファイルのペアを作成
- `prepare_data()`: データを準備してDataオブジェクトのリストを返す
- `compute_class_weights()`: クラス重みを計算

### `training_utils.py`
- `set_seed()`: 再現性のためのシード設定
- `setup()`: プロセスグループの初期化
- `cleanup()`: プロセスグループのクリーンアップ
- `get_distributed_info()`: 分散学習の環境情報を取得

### `metrics.py`
- `calculate_metrics()`: 精度評価のメトリクス計算
- `evaluate_and_visualize()`: 精度評価と可視化

### `config.py`
- `DataConfig`: データ設定
- `ModelConfig`: モデル設定
- `TrainingConfig`: トレーニング設定
- `Config`: 全体設定（デフォルト、18クラス用、10クラス用のプリセットあり）

## 既存スクリプトとの統合

### `GNN_zscore_sub_noise_defect_free.py`の場合

このスクリプトは独自の実装を持っていますが、以下の関数は共通モジュールに置き換え可能です：

- `set_seed()` → `gnn_common.training_utils.set_seed()`
- `setup()` → `gnn_common.training_utils.setup()`
- `cleanup()` → `gnn_common.training_utils.cleanup()`
- `extract_layer_block()` → `gnn_common.data_utils.extract_layer_block()`（ただし、ファイル名形式が異なる可能性あり）
- `compute_class_weights()` → `gnn_common.data_utils.compute_class_weights()`

**注意**: モデル（`GATModel`）は2段階モデルのため、共通モジュールのものとは異なります。

## リファクタリングの利点

1. **コードの重複削減**: 共通機能を一箇所に集約
2. **保守性の向上**: バグ修正や機能追加が容易
3. **一貫性の確保**: すべてのスクリプトで同じ実装を使用
4. **テスト容易性**: 共通モジュールを個別にテスト可能
5. **設定管理**: 設定を一元管理

## 移行ガイド

既存のスクリプトをリファクタリングする場合：

1. 共通モジュールをインポート
2. 重複コードを削除
3. 共通モジュールの関数を使用
4. 設定を`Config`クラスで管理

詳細な例は `DDP/train_refactored_example.py` を参照してください。