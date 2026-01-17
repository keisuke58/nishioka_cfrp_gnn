# 共通モジュール

このディレクトリには、GNNトレーニングスクリプトで共通して使用されるモジュールが含まれています。

## モジュール構成

### `models.py`
- `GCNModel`: Graph Convolutional Network モデル
- `GATModel`: Graph Attention Network モデル

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

## 使用例

```python
from common.models import GCNModel, GATModel
from common.data_utils import create_data_label_pairs, prepare_data
from common.training_utils import set_seed, setup, cleanup
from common.metrics import calculate_metrics
from common.config import Config

# 設定の読み込み
config = Config.default()

# モデルの作成
model = GCNModel(
    hidden_channels=128,
    num_classes=19,
    dropout=0.2
)

# データの準備
pairs = create_data_label_pairs(data_files, label_files)
dataset = prepare_data(
    pairs,
    config.data.standardized_data_folder,
    config.data.label_data_folder,
    x_coords, y_coords, z_coords, edge_index
)
```

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

詳細な例は `train_refactored_example.py` を参照してください。