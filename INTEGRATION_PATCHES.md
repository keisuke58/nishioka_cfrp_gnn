# 統合パッチガイド

既存の `GNN_zscore_sub_noise_defect_free.py` に新機能を統合するための具体的なパッチ手順

---

## 統合箇所の特定

### 箇所1: データ分割部分（line 2996-3026）

**既存コード**:
```python
# train/val/testに分割
enforce_disjoint_groups = getattr(args, "enforce_disjoint_groups", True)
group_key = getattr(args, "group_key", "LBel")
if enforce_disjoint_groups:
    train_pairs, val_pairs, test_pairs, train_data_folder_map, val_data_folder_map, test_data_folder_map = group_disjoint_split(
        all_pairs,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        group_key=group_key,
        seed=42,
    )
else:
    # 単純な分割
    ...
```

**統合後**:
```python
# OOD分割の設定を読み込む
from gnn_common.integration_utils import apply_ood_split, load_config_for_splitting

split_config = load_config_for_splitting('config.yaml')

# OOD分割を使用する場合
if split_config['split_type'] != 'iid':
    # OOD分割を適用
    # 注意: all_pairsは(data_file, label_file, data_dir)のタプルなので、
    # まず(data_file, label_file)のペアに変換
    pairs_for_ood = [(p[0], p[1]) for p in all_pairs]
    
    train_pairs, val_pairs, test_pairs = apply_ood_split(
        pairs_for_ood,
        split_type=split_config['split_type'],
        label_data_folder=label_data_folder,
        test_ratio=0.15,
        val_ratio=0.15,
        seed=42,
        **split_config['ood_params']
    )
    
    # データフォルダマップを再構築
    all_pairs_dict = {p[0]: p[2] for p in all_pairs}
    train_data_folder_map = {p[0]: all_pairs_dict.get(p[0], noise_data_folder) for p in train_pairs}
    val_data_folder_map = {p[0]: all_pairs_dict.get(p[0], noise_data_folder) for p in val_pairs}
    test_data_folder_map = {p[0]: all_pairs_dict.get(p[0], noise_data_folder) for p in test_pairs}
    
    if rank == 0:
        print(f"[INFO] Using OOD split: {split_config['split_type']}")
else:
    # 既存の分割ロジックを使用
    enforce_disjoint_groups = getattr(args, "enforce_disjoint_groups", True)
    group_key = getattr(args, "group_key", "LBel")
    if enforce_disjoint_groups:
        train_pairs, val_pairs, test_pairs, train_data_folder_map, val_data_folder_map, test_data_folder_map = group_disjoint_split(
            all_pairs,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            group_key=group_key,
            seed=42,
        )
    else:
        # 単純な分割
        ...
```

---

### 箇所2: データ準備部分（line 3143-3149）

**既存コード**:
```python
train_dataset, class_weights = prepare_data(train_pairs, noise_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier, data_folder_map=train_data_folder_map_global)
val_dataset, _ = prepare_data(val_pairs, noise_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier, data_folder_map=val_data_folder_map_global)
test_dataset, _ = prepare_data(test_pairs, noise_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier, data_folder_map=test_data_folder_map_global)
```

**統合後**:
```python
from gnn_common.integration_utils import prepare_data_with_cross_edges, load_config_for_cross_edges

# Cross-edge設定を読み込む
cross_edge_config = load_config_for_cross_edges('config.yaml')

if rank == 0:
    if cross_edge_config['enabled']:
        print(f"[INFO] Cross-edges enabled: k={cross_edge_config['k']}, method={cross_edge_config['surface_method']}")
    else:
        print("[INFO] Cross-edges disabled")

# Cross-edge対応のデータ準備
train_dataset, class_weights = prepare_data_with_cross_edges(
    train_pairs, noise_data_folder, label_data_folder,
    x_coords, y_coords, z_coords, edge_index,
    class_weight_multiplier=class_weight_multiplier,
    data_folder_map=train_data_folder_map_global,
    use_cross_edges=cross_edge_config['enabled'],
    cross_edge_k=cross_edge_config['k'],
    cross_edge_method=cross_edge_config['surface_method'],
    max_nodes=13942,
    return_class_weights=True
)

val_dataset, _ = prepare_data_with_cross_edges(
    val_pairs, noise_data_folder, label_data_folder,
    x_coords, y_coords, z_coords, edge_index,
    class_weight_multiplier=class_weight_multiplier,
    data_folder_map=val_data_folder_map_global,
    use_cross_edges=cross_edge_config['enabled'],
    cross_edge_k=cross_edge_config['k'],
    cross_edge_method=cross_edge_config['surface_method'],
    max_nodes=13942,
    return_class_weights=False
)

test_dataset, _ = prepare_data_with_cross_edges(
    test_pairs, noise_data_folder, label_data_folder,
    x_coords, y_coords, z_coords, edge_index,
    class_weight_multiplier=class_weight_multiplier,
    data_folder_map=test_data_folder_map_global,
    use_cross_edges=cross_edge_config['enabled'],
    cross_edge_k=cross_edge_config['k'],
    cross_edge_method=cross_edge_config['surface_method'],
    max_nodes=13942,
    return_class_weights=False
)
```

---

### 箇所3: モデル作成部分（line 3124-3129）

**既存コード**:
```python
model = GATModel(
    hidden_channels=args.hidden_channels, 
    num_classes=19,
    dropout=dropout,
    edge_drop_prob=edge_drop_prob
).to(device)
```

**統合後**:
```python
from gnn_common.integration_utils import create_model_with_cross_edges, load_config_for_cross_edges

# Cross-edge設定を読み込む（データ準備と同じ設定を使用）
cross_edge_config = load_config_for_cross_edges('config.yaml')

# Cross-edge対応モデルを作成
model = create_model_with_cross_edges(
    model_type='GAT',
    use_cross_edges=cross_edge_config['enabled'],
    hidden_channels=args.hidden_channels,
    num_classes=19,
    num_heads=4,  # GATのデフォルト
    dropout=dropout
).to(device)

# 注意: GATModelWithCrossEdgesはedge_drop_probを直接サポートしていない場合があるため、
# 必要に応じてモデル定義を確認・修正
```

---

### 箇所4: 評価部分（line 4277-4355）

**既存コード**:
```python
test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader_eval):
        batch = batch.to(device)
        out = ddp_model(batch)
        y = batch.y
        loss = loss_fn(out, y)
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total_samples += y.size(0)
        test_preds.extend(pred.cpu().numpy())
        test_labels.extend(y.cpu().numpy())
        probs = torch.softmax(out, dim=1).cpu().numpy()
        test_probs.extend(probs)

test_loss = total_loss / len(test_loader_eval)
test_accuracy = correct / total_samples
```

**統合後**:
```python
from gnn_common.integration_utils import evaluate_with_localization_metrics

# Localization指標を含む評価
# 注意: 座標データが必要
results = evaluate_with_localization_metrics(
    ddp_model,
    test_loader_eval,
    device,
    num_classes=19,
    calculate_localization=True,
    defect_class_ids=set(range(1, 19))  # クラス1-18が欠陥
)

# 従来のメトリクス
test_loss = results['test_loss']
test_accuracy = results['test_accuracy']
macro_f1 = results['macro_f1']

# Localization指標の出力
if results['localization']:
    loc = results['localization']
    print(f"\n=== Localization Metrics ===")
    print(f"Top-1 Accuracy: {loc['top_k_accuracy'][1]:.4f}")
    print(f"Top-3 Accuracy: {loc['top_k_accuracy'][3]:.4f}")
    print(f"Top-5 Accuracy: {loc['top_k_accuracy'][5]:.4f}")
    print(f"Top-10 Accuracy: {loc['top_k_accuracy'][10]:.4f}")
    print(f"Mean Distance Error: {loc['distance_error']['mean']:.4f}")
    print(f"Median Distance Error: {loc['distance_error']['median']:.4f}")
    print(f"Max Distance Error: {loc['distance_error']['max']:.4f}")
    print(f"AUPRC: {loc['auprc']:.4f}")

# 既存の評価コードも残す（後方互換性のため）
test_preds = np.array([...])  # resultsから取得するか、既存のループも実行
test_labels = np.array([...])
test_probs = np.array([...])
```

---

## 完全な統合手順

### Step 1: 既存スクリプトをバックアップ

```bash
cd /home/nishioka/GNN/GNN_hole_2026/GNN_program
cp GNN_zscore_sub_noise_defect_free.py GNN_zscore_sub_noise_defect_free.py.backup
```

### Step 2: インポートを追加

既存スクリプトの先頭（import部分）に追加:

```python
# 新機能のインポート
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from gnn_common.integration_utils import (
    apply_ood_split,
    prepare_data_with_cross_edges,
    evaluate_with_localization_metrics,
    create_model_with_cross_edges,
    load_config_for_splitting,
    load_config_for_cross_edges
)
```

### Step 3: 各統合箇所をパッチ

上記の「統合後」コードで各箇所を置き換え

### Step 4: 設定ファイルの準備

```bash
cp /home/nishioka/GNN/config.yaml.example /home/nishioka/GNN/config.yaml
# 必要に応じて編集
```

---

## 注意事項

1. **既存のスクリプトはそのまま**: バックアップを取ってから作業
2. **段階的な統合**: まずOOD分割だけ統合して動作確認
3. **設定ファイル**: `config.yaml`が存在しない場合はデフォルト値が使用される
4. **後方互換性**: 既存の機能はすべて維持される

---

## テスト方法

統合後、以下で動作確認:

```bash
# 既存の方法で実行（IID分割、Cross-edgeなし）
python GNN_zscore_sub_noise_defect_free_with_new_features.py --epochs 10

# OOD分割を使用
# config.yamlで split_type: "defect_size" を設定

# Cross-edgeを使用
# config.yamlで cross_edge.enabled: true を設定
```
