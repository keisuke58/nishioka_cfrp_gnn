# 統合ガイド

既存のトレーニングスクリプトに新機能を統合する方法

---

## 1. 動作確認

まず、新機能が正しく動作するか確認します：

```bash
# テストスクリプトを実行
python tools/test_new_features.py
```

すべてのテストがパスすることを確認してください。

---

## 2. OOD分割の統合

### 既存コードの変更箇所

通常、データ分割は以下のようなコードで行われています：

```python
# 既存コード（例）
pairs = create_data_label_pairs(data_files, label_files)
# ランダム分割
train_pairs, val_pairs, test_pairs = train_test_split(pairs, ...)
```

### 統合方法

```python
from gnn_common.integration_utils import apply_ood_split, load_config_for_splitting

# 設定を読み込む
split_config = load_config_for_splitting('config.yaml')

# OOD分割を適用
train_pairs, val_pairs, test_pairs = apply_ood_split(
    pairs,
    split_type=split_config['split_type'],
    label_data_folder=label_data_folder,
    test_ratio=split_config['test_ratio'],
    val_ratio=split_config['val_ratio'],
    seed=split_config['seed'],
    **split_config['ood_params']
)
```

### 設定ファイルの例

```yaml
# config.yaml
data_splitting:
  split_type: "defect_size"  # 'iid', 'defect_size', 'defect_ratio', 'layer'
  test_ratio: 0.2
  val_ratio: 0.1
  seed: 42
  ood_params:
    defect_size:
      size_threshold: 5
```

---

## 3. Cross-edgeの統合

### 既存コードの変更箇所

データ準備の部分：

```python
# 既存コード（例）
data_list, class_weights = prepare_data(
    pairs, normalized_data_folder, label_data_folder,
    x_coords, y_coords, z_coords, edge_index,
    max_nodes=13942, return_class_weights=True
)
```

### 統合方法

```python
from gnn_common.integration_utils import (
    prepare_data_with_cross_edges,
    load_config_for_cross_edges
)

# Cross-edge設定を読み込む
cross_edge_config = load_config_for_cross_edges('config.yaml')

# Cross-edge対応のデータ準備
data_list, class_weights = prepare_data_with_cross_edges(
    pairs, normalized_data_folder, label_data_folder,
    x_coords, y_coords, z_coords, edge_index,
    max_nodes=13942,
    return_class_weights=True,
    use_cross_edges=cross_edge_config['enabled'],
    cross_edge_k=cross_edge_config['k'],
    cross_edge_method=cross_edge_config['surface_method']
)
```

### モデルの選択

```python
from gnn_common.integration_utils import create_model_with_cross_edges

# Cross-edge対応モデルを作成
model = create_model_with_cross_edges(
    model_type='GAT',
    use_cross_edges=cross_edge_config['enabled'],
    hidden_channels=64,
    num_classes=19,
    num_heads=4,
    dropout=0.2
)
```

---

## 4. Localization指標の統合

### 既存コードの変更箇所

評価ループの部分：

```python
# 既存コード（例）
model.eval()
all_preds = []
all_labels = []
for batch in test_loader:
    out = model(batch)
    pred = out.argmax(dim=1)
    all_preds.extend(pred.cpu().numpy())
    all_labels.extend(batch.y.cpu().numpy())
# メトリクス計算
accuracy = (all_preds == all_labels).mean()
```

### 統合方法

```python
from gnn_common.integration_utils import evaluate_with_localization_metrics

# Localization指標を含む評価
results = evaluate_with_localization_metrics(
    model,
    test_loader,
    device,
    num_classes=19,
    calculate_localization=True,
    defect_class_ids=set(range(1, 19))  # クラス1-18が欠陥
)

# 結果の使用
print(f"Accuracy: {results['test_accuracy']:.4f}")
print(f"Macro F1: {results['macro_f1']:.4f}")

if results['localization']:
    loc = results['localization']
    print(f"Top-1 Accuracy: {loc['top_k_accuracy'][1]:.4f}")
    print(f"Top-5 Accuracy: {loc['top_k_accuracy'][5]:.4f}")
    print(f"Mean Distance Error: {loc['distance_error']['mean']:.4f}")
    print(f"AUPRC: {loc['auprc']:.4f}")
```

---

## 5. 完全な統合例

```python
import torch
from gnn_common.integration_utils import (
    apply_ood_split,
    prepare_data_with_cross_edges,
    evaluate_with_localization_metrics,
    create_model_with_cross_edges,
    load_config_for_splitting,
    load_config_for_cross_edges
)
from gnn_common.data_utils import create_data_label_pairs

# 1. 設定を読み込む
split_config = load_config_for_splitting('config.yaml')
cross_edge_config = load_config_for_cross_edges('config.yaml')

# 2. データペアを作成
pairs = create_data_label_pairs(data_files, label_files)

# 3. OOD分割
train_pairs, val_pairs, test_pairs = apply_ood_split(
    pairs,
    split_type=split_config['split_type'],
    label_data_folder=label_data_folder,
    **split_config['ood_params']
)

# 4. データ準備（Cross-edge対応）
train_data, class_weights = prepare_data_with_cross_edges(
    train_pairs, normalized_data_folder, label_data_folder,
    x_coords, y_coords, z_coords, edge_index,
    use_cross_edges=cross_edge_config['enabled'],
    cross_edge_k=cross_edge_config['k'],
    cross_edge_method=cross_edge_config['surface_method']
)

# 5. モデル作成
model = create_model_with_cross_edges(
    model_type='GAT',
    use_cross_edges=cross_edge_config['enabled'],
    hidden_channels=64,
    num_classes=19
)

# 6. 学習（通常通り）
# ... 学習ループ ...

# 7. 評価（Localization指標含む）
test_loader = DataLoader(test_data, batch_size=64)
results = evaluate_with_localization_metrics(
    model, test_loader, device,
    calculate_localization=True
)
```

---

## 6. アブレーション実験の実行

### 個別実験

```bash
# Cross-edge OFF
python tools/run_ablation_experiment.py \
    --experiment baseline \
    --split_type iid \
    --output_dir ./ablation_results

# Cross-edge ON
python tools/run_ablation_experiment.py \
    --experiment cross_edge_on \
    --split_type iid \
    --cross_edges \
    --cross_edge_k 1 \
    --output_dir ./ablation_results
```

### スイープ実行

```bash
bash tools/run_ablation_sweep.sh
```

---

## 7. トラブルシューティング

### OOD分割が動作しない

- `label_data_folder`が正しく設定されているか確認
- ラベルファイルが存在するか確認
- `max_nodes`が正しく設定されているか確認

### Cross-edgeが追加されない

- `use_cross_edges=True`が設定されているか確認
- Surfaceノードが正しく識別されているか確認（`verbose_print=True`で確認）
- 座標データが正しく読み込まれているか確認

### Localization指標がNaNになる

- 欠陥ノードが存在するか確認
- 座標データが正しく読み込まれているか確認
- `defect_class_ids`が正しく設定されているか確認

---

## 8. 次のステップ

1. **動作確認**: `python tools/test_new_features.py` を実行
2. **既存スクリプトへの統合**: 上記の例を参考に統合
3. **アブレーション実験**: 異なる設定で実験を実行
4. **結果の比較**: 実験結果を比較して最適な設定を決定
