# 実装機能の使用例

このドキュメントは、実装した機能（OOD分割、Localization指標、Cross-edge）の具体的な使用方法を示します。

---

## 1. OOD分割の使用

### 基本的な使用例

```python
from gnn_common.data_utils import create_ood_split, visualize_split_statistics

# データペアの準備（既存のコードから）
pairs = create_data_label_pairs(data_files, label_files)

# IID分割（ランダム）
train_pairs, val_pairs, test_pairs = create_ood_split(
    pairs,
    split_type='iid',
    test_ratio=0.2,
    val_ratio=0.1,
    seed=42
)

# OOD分割（欠陥サイズで分割）
train_pairs, val_pairs, test_pairs = create_ood_split(
    pairs,
    split_type='defect_size',
    label_data_folder='/path/to/labels',
    size_threshold=5,  # 5クラス以上の欠陥を持つサンプルをtestに
    test_ratio=0.2,
    val_ratio=0.1,
    seed=42
)

# OOD分割（層で分割）
train_pairs, val_pairs, test_pairs = create_ood_split(
    pairs,
    split_type='layer',
    label_data_folder='/path/to/labels',
    train_layers=[1, 2, 3],  # L1-L3をtrain、それ以外をtest
    test_ratio=0.2,
    val_ratio=0.1,
    seed=42
)

# 分割統計の可視化
visualize_split_statistics(
    train_pairs, val_pairs, test_pairs,
    label_data_folder='/path/to/labels',
    output_path='split_statistics.png'
)
```

### 設定ファイルからの使用

```yaml
# config.yaml
data_splitting:
  split_type: "defect_size"
  test_ratio: 0.2
  val_ratio: 0.1
  seed: 42
  ood_params:
    defect_size:
      size_threshold: 5
```

```python
import yaml
from gnn_common.data_utils import create_ood_split

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

split_config = config['data_splitting']
ood_params = split_config.get('ood_params', {}).get(split_config['split_type'], {})

train_pairs, val_pairs, test_pairs = create_ood_split(
    pairs,
    split_type=split_config['split_type'],
    test_ratio=split_config['test_ratio'],
    val_ratio=split_config['val_ratio'],
    seed=split_config['seed'],
    label_data_folder='/path/to/labels',
    **ood_params
)
```

---

## 2. Localization指標の計算

### 基本的な使用例

```python
import torch
import numpy as np
from gnn_common.metrics import calculate_localization_metrics

# モデルの評価ループ内で
model.eval()
all_predictions = []
all_labels = []
all_coordinates = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        
        # 予測確率を保存
        probs = torch.softmax(out, dim=1)
        all_predictions.append(probs.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        
        # 座標を取得（x, y, zはdata.xの最初の3次元）
        coords = batch.x[:, :3].cpu().numpy()
        all_coordinates.append(coords)

# 結合
predictions = np.vstack(all_predictions)  # [N, num_classes]
labels = np.concatenate(all_labels)       # [N]
coordinates = np.vstack(all_coordinates) # [N, 3]

# Localization指標を計算
metrics = calculate_localization_metrics(
    predictions, labels, coordinates,
    top_k_list=[1, 3, 5, 10]
)

print(f"Top-1 Accuracy: {metrics['top_k_accuracy'][1]:.4f}")
print(f"Top-3 Accuracy: {metrics['top_k_accuracy'][3]:.4f}")
print(f"Top-5 Accuracy: {metrics['top_k_accuracy'][5]:.4f}")
print(f"Mean Distance Error: {metrics['distance_error']['mean']:.4f}")
print(f"Median Distance Error: {metrics['distance_error']['median']:.4f}")
print(f"AUPRC: {metrics['auprc']:.4f}")
```

### 個別の指標を計算

```python
from gnn_common.metrics import (
    calculate_top_k_accuracy,
    calculate_distance_error
)

# Top-k Accuracyのみ
top_k_acc = calculate_top_k_accuracy(
    predictions, labels, top_k_list=[1, 3, 5, 10]
)

# 距離誤差のみ
dist_error = calculate_distance_error(
    predictions, labels, coordinates,
    defect_class_ids=set(range(1, 19))  # クラス1-18が欠陥
)
```

---

## 3. Cross-edge（Yehia方式）の使用

### データ準備時にCross-edgeを追加

```python
from gnn_common.data_utils import (
    prepare_data,
    identify_surface_nodes,
    add_cross_edges_to_data
)
import numpy as np

# 座標データをロード
x_coords = np.load('/path/to/x_coords.npy')
y_coords = np.load('/path/to/y_coords.npy')
z_coords = np.load('/path/to/z_coords.npy')
coordinates = np.stack([x_coords, y_coords, z_coords], axis=1)  # [N, 3]

# Surfaceノードを識別
surface_mask = identify_surface_nodes(
    coordinates, z_coords,
    method='outer_layer'  # または 'top_p'
)

# データを準備
data_list, class_weights = prepare_data(
    pairs, normalized_data_folder, label_data_folder,
    x_coords, y_coords, z_coords, edge_index,
    max_nodes=13942, return_class_weights=True
)

# 各データにCross-edgeを追加
for data in data_list:
    # 座標を取得（data.xの最初の3次元）
    coords = data.x[:, :3].numpy()
    z_coords_data = data.x[:, 2].numpy()
    
    # Surfaceノードを識別（このデータ用）
    surface_mask_data = identify_surface_nodes(
        coords, z_coords_data, method='outer_layer'
    )
    
    # Cross-edgeを追加
    data = add_cross_edges_to_data(
        data, coords, surface_mask_data, k=1
    )
```

### Cross-edge対応モデルの使用

```python
from gnn_common.models import GATModelWithCrossEdges

# モデルの作成
model = GATModelWithCrossEdges(
    hidden_channels=64,
    num_classes=19,
    num_heads=4,
    dropout=0.2,
    use_edge_type=False  # 現在は未使用
)

# 通常通り学習・推論
output = model(data)
```

### 設定ファイルからの使用

```yaml
# config.yaml
cross_edge:
  enabled: true
  k: 1
  surface_method: "outer_layer"
  top_p: 0.1
```

```python
import yaml
from gnn_common.data_utils import (
    identify_surface_nodes,
    add_cross_edges_to_data
)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

cross_edge_config = config.get('cross_edge', {})
if cross_edge_config.get('enabled', False):
    # Cross-edgeを有効化
    surface_mask = identify_surface_nodes(
        coordinates, z_coords,
        method=cross_edge_config.get('surface_method', 'outer_layer'),
        top_p=cross_edge_config.get('top_p', 0.1)
    )
    
    for data in data_list:
        coords = data.x[:, :3].numpy()
        z_coords_data = data.x[:, 2].numpy()
        surface_mask_data = identify_surface_nodes(
            coords, z_coords_data,
            method=cross_edge_config.get('surface_method', 'outer_layer')
        )
        data = add_cross_edges_to_data(
            data, coords, surface_mask_data,
            k=cross_edge_config.get('k', 1)
        )
```

---

## 4. 統合使用例（アブレーション実験）

```python
from gnn_common.data_utils import (
    create_data_label_pairs,
    create_ood_split,
    prepare_data,
    identify_surface_nodes,
    add_cross_edges_to_data
)
from gnn_common.models import GATModel, GATModelWithCrossEdges
from gnn_common.metrics import calculate_localization_metrics

# 1. データ準備
pairs = create_data_label_pairs(data_files, label_files)

# 2. OOD分割
train_pairs, val_pairs, test_pairs = create_ood_split(
    pairs,
    split_type='defect_size',
    label_data_folder='/path/to/labels',
    size_threshold=5,
    seed=42
)

# 3. データロード
train_data, class_weights = prepare_data(
    train_pairs, normalized_data_folder, label_data_folder,
    x_coords, y_coords, z_coords, edge_index,
    max_nodes=13942, return_class_weights=True
)

# 4. Cross-edgeの追加（オプション）
if use_cross_edges:
    for data in train_data:
        coords = data.x[:, :3].numpy()
        z_coords_data = data.x[:, 2].numpy()
        surface_mask = identify_surface_nodes(
            coords, z_coords_data, method='outer_layer'
        )
        data = add_cross_edges_to_data(data, coords, surface_mask, k=1)

# 5. モデルの選択
if use_cross_edges:
    model = GATModelWithCrossEdges(hidden_channels=64, num_classes=19)
else:
    model = GATModel(hidden_channels=64, num_classes=19)

# 6. 学習（通常通り）

# 7. 評価時にLocalization指標を計算
# （評価ループ内で）
metrics = calculate_localization_metrics(
    predictions, labels, coordinates,
    top_k_list=[1, 3, 5, 10]
)
```

---

## 5. トラブルシューティング

### OOD分割でエラーが出る場合

- `label_data_folder`が正しく設定されているか確認
- ラベルファイルが存在するか確認
- `max_nodes`が正しく設定されているか確認

### Localization指標がNaNになる場合

- 欠陥ノードが存在するか確認
- `defect_class_ids`が正しく設定されているか確認
- 座標データが正しく読み込まれているか確認

### Cross-edgeが追加されない場合

- `surface_node_mask`が正しく計算されているか確認（少なくとも1つのsurfaceノードが必要）
- `k`がsurfaceノード数以下か確認
- 座標データが正しく読み込まれているか確認

---

## 6. ファインチューニングの使用

### 基本的なファインチューニング

既存の学習済みモデルをロードして、新しいデータセットや異なるタスクでファインチューニングできます。

```bash
# 1. 重みのみをロードしてファインチューニング（optimizer/schedulerはリセット）
python GNN_zscore_sub_noise_defect_free.py \
  --fine_tune_from /path/to/pretrained_model.pth \
  --learning_rate 0.0001 \
  --epochs 500

# 2. バックボーンを凍結して分類ヘッドのみを学習
python GNN_zscore_sub_noise_defect_free.py \
  --fine_tune_from /path/to/pretrained_model.pth \
  --freeze_backbone \
  --learning_rate 0.001 \
  --epochs 300

# 3. 層別学習率を設定（バックボーンは低LR、分類ヘッドは高LR）
python GNN_zscore_sub_noise_defect_free.py \
  --fine_tune_from /path/to/pretrained_model.pth \
  --backbone_lr 0.0001 \
  --head_lr 0.001 \
  --epochs 500
```

### 学習の再開との違い

- `--resume_from`: 完全な再開（optimizer/schedulerの状態も復元、epochも継続）
- `--fine_tune_from`: ファインチューニング（重みのみロード、optimizer/schedulerはリセット、epochは0から開始）

```bash
# 学習の再開（中断した学習を続ける）
python GNN_zscore_sub_noise_defect_free.py \
  --resume_from /path/to/checkpoint.pth \
  --epochs 2000

# ファインチューニング（新しいデータセットで微調整）
python GNN_zscore_sub_noise_defect_free.py \
  --fine_tune_from /path/to/pretrained_model.pth \
  --learning_rate 0.0001 \
  --epochs 500
```

### 設定ファイルからの使用

```yaml
# config.yaml
training:
  fine_tune_from: "/path/to/pretrained_model.pth"
  freeze_backbone: false
  backbone_lr: 0.0001
  head_lr: 0.001
  learning_rate: 0.001  # backbone_lr/head_lrが未指定の場合のデフォルト
```

### 推奨されるファインチューニング戦略

1. **全層ファインチューニング**: 小さい学習率で全層を更新
   ```bash
   --fine_tune_from model.pth --learning_rate 0.0001
   ```

2. **分類ヘッドのみ学習**: バックボーンを凍結して分類ヘッドのみ更新
   ```bash
   --fine_tune_from model.pth --freeze_backbone --learning_rate 0.001
   ```

3. **層別学習率**: バックボーンは低LR、分類ヘッドは高LR
   ```bash
   --fine_tune_from model.pth --backbone_lr 0.0001 --head_lr 0.001
   ```

---

## 7. 次のステップ

1. **アブレーション実験**: Cross-edge ON/OFFで性能を比較
2. **OOD評価**: IIDとOODで性能差を確認
3. **Localization指標**: 位置推定の性能を定量的に評価
4. **ハイパーパラメータ調整**: Cross-edgeの`k`やsurface定義方法を調整
5. **ファインチューニング実験**: 異なるデータセットやタスクでの転移学習
