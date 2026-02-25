# 実装ガイド（具体的なコード例）

このドキュメントは、NEXT_PLAN.mdで提示した機能を実装するための具体的なコード例を示します。

---

## 1. OOD分割の実装

### 1.1 `gnn_common/data_utils.py` に追加する関数

```python
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from sklearn.model_selection import train_test_split

def calculate_defect_statistics(
    pairs: List[Tuple[str, str]], 
    label_data_folder: str,
    max_nodes: int = 13942
) -> Dict[str, Dict]:
    """
    各データファイルの欠陥統計を計算
    
    Returns:
        {
            'filename': {
                'defect_ratio': float,  # 欠陥ノードの割合
                'defect_size': float,   # 欠陥クラス数（簡易版）
                'layer': int,          # 層番号
                'block': int,          # ブロック番号
                'defect_classes': set  # 出現する欠陥クラスID
            }
        }
    """
    stats = {}
    
    for data_file, label_file in pairs:
        try:
            label_path = os.path.join(label_data_folder, label_file)
            label = np.load(label_path)[:max_nodes]
            
            # ラベルをクラスIDに変換
            if len(label.shape) == 2 and label.shape[1] > 1:
                y = np.argmax(label, axis=1)
            else:
                y = label.flatten().astype(int)
            
            # 欠陥統計を計算
            defect_mask = y > 0  # クラス0以外が欠陥
            defect_ratio = defect_mask.sum() / len(y)
            defect_classes = set(y[defect_mask].tolist())
            
            # ファイル名から層・ブロック情報を抽出
            layer_block = extract_layer_block(data_file)
            layer = layer_block[0] if layer_block else -1
            block = layer_block[1] if layer_block else -1
            
            # 欠陥サイズ（簡易版：欠陥クラス数）
            defect_size = len(defect_classes)
            
            base_name = os.path.splitext(data_file)[0]
            stats[base_name] = {
                'defect_ratio': defect_ratio,
                'defect_size': defect_size,
                'layer': layer,
                'block': block,
                'defect_classes': defect_classes,
                'data_file': data_file,
                'label_file': label_file
            }
        except Exception as e:
            if VERBOSE_PRINT:
                print(f"Error calculating stats for {data_file}: {e}")
            continue
    
    return stats


def create_ood_split(
    pairs: List[Tuple[str, str]],
    split_type: str = 'iid',
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    label_data_folder: Optional[str] = None,
    max_nodes: int = 13942,
    seed: int = 42,
    **kwargs
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    OOD分割またはIID分割を作成
    
    Args:
        pairs: (data_file, label_file)のタプルのリスト
        split_type: 'iid', 'defect_size', 'defect_position', 'layer', 'bc'
        test_ratio: テストセットの割合
        val_ratio: バリデーションセットの割合（残りからの割合）
        label_data_folder: ラベルフォルダのパス（OOD分割に必要）
        max_nodes: 最大ノード数
        seed: 乱数シード
        **kwargs: 分割タイプ固有のパラメータ
    
    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    np.random.seed(seed)
    
    if split_type == 'iid':
        # IID分割（ランダム）
        train_pairs, temp_pairs = train_test_split(
            pairs, test_size=(test_ratio + val_ratio), random_state=seed
        )
        val_pairs, test_pairs = train_test_split(
            temp_pairs, test_size=test_ratio/(test_ratio + val_ratio), random_state=seed
        )
        return train_pairs, val_pairs, test_pairs
    
    # OOD分割の場合は統計情報が必要
    if label_data_folder is None:
        raise ValueError("label_data_folder is required for OOD splits")
    
    stats = calculate_defect_statistics(pairs, label_data_folder, max_nodes)
    
    if split_type == 'defect_size':
        # 欠陥サイズで分割
        # 小欠陥をtrain、大欠陥をtestに
        size_threshold = kwargs.get('size_threshold', 5)  # デフォルト: 5クラス以上
        
        train_pairs = []
        test_pairs = []
        
        for base_name, stat in stats.items():
            pair = (stat['data_file'], stat['label_file'])
            if stat['defect_size'] < size_threshold:
                train_pairs.append(pair)
            else:
                test_pairs.append(pair)
        
        # valはtrainから分離
        train_pairs, val_pairs = train_test_split(
            train_pairs, test_size=val_ratio/(1-test_ratio), random_state=seed
        )
        
    elif split_type == 'layer':
        # 層で分割（例: L1-L3をtrain、L4-L6をtest）
        train_layers = set(kwargs.get('train_layers', [1, 2, 3]))
        
        train_pairs = []
        test_pairs = []
        
        for base_name, stat in stats.items():
            pair = (stat['data_file'], stat['label_file'])
            if stat['layer'] in train_layers:
                train_pairs.append(pair)
            else:
                test_pairs.append(pair)
        
        # valはtrainから分離
        train_pairs, val_pairs = train_test_split(
            train_pairs, test_size=val_ratio/(1-test_ratio), random_state=seed
        )
        
    elif split_type == 'defect_ratio':
        # 欠陥比率で分割
        ratio_threshold = kwargs.get('ratio_threshold', 0.1)  # デフォルト: 10%
        
        train_pairs = []
        test_pairs = []
        
        for base_name, stat in stats.items():
            pair = (stat['data_file'], stat['label_file'])
            if stat['defect_ratio'] < ratio_threshold:
                train_pairs.append(pair)
            else:
                test_pairs.append(pair)
        
        # valはtrainから分離
        train_pairs, val_pairs = train_test_split(
            train_pairs, test_size=val_ratio/(1-test_ratio), random_state=seed
        )
        
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    
    return train_pairs, val_pairs, test_pairs


def visualize_split_statistics(
    train_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]],
    label_data_folder: str,
    output_path: Optional[str] = None,
    max_nodes: int = 13942
):
    """
    分割結果の統計を可視化
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    all_pairs = train_pairs + val_pairs + test_pairs
    stats = calculate_defect_statistics(all_pairs, label_data_folder, max_nodes)
    
    # 各分割の統計を集計
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 欠陥比率の分布
    ax = axes[0, 0]
    for split_name, pairs in splits.items():
        ratios = [
            stats[os.path.splitext(p[0])[0]]['defect_ratio']
            for p in pairs
            if os.path.splitext(p[0])[0] in stats
        ]
        ax.hist(ratios, alpha=0.5, label=split_name, bins=20)
    ax.set_xlabel('Defect Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Defect Ratio Distribution')
    ax.legend()
    
    # 2. 層の分布
    ax = axes[0, 1]
    for split_name, pairs in splits.items():
        layers = [
            stats[os.path.splitext(p[0])[0]]['layer']
            for p in pairs
            if os.path.splitext(p[0])[0] in stats
        ]
        ax.hist(layers, alpha=0.5, label=split_name, bins=range(1, 10))
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count')
    ax.set_title('Layer Distribution')
    ax.legend()
    
    # 3. 欠陥サイズの分布
    ax = axes[1, 0]
    for split_name, pairs in splits.items():
        sizes = [
            stats[os.path.splitext(p[0])[0]]['defect_size']
            for p in pairs
            if os.path.splitext(p[0])[0] in stats
        ]
        ax.hist(sizes, alpha=0.5, label=split_name, bins=range(1, 20))
    ax.set_xlabel('Defect Size (num classes)')
    ax.set_ylabel('Count')
    ax.set_title('Defect Size Distribution')
    ax.legend()
    
    # 4. サンプル数の比較
    ax = axes[1, 1]
    split_names = list(splits.keys())
    counts = [len(pairs) for pairs in splits.values()]
    ax.bar(split_names, counts)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Split Sizes')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Split statistics saved to {output_path}")
    else:
        plt.show()
```

---

## 2. Localization指標の実装

### 2.1 `gnn_common/metrics.py` に追加する関数

```python
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist

def calculate_top_k_accuracy(
    predictions: np.ndarray,  # [N, num_classes] 予測確率
    labels: np.ndarray,       # [N] 正解ラベル
    top_k_list: List[int] = [1, 3, 5, 10]
) -> Dict[int, float]:
    """
    Top-k Accuracyを計算
    
    Args:
        predictions: 予測確率 [N, num_classes]
        labels: 正解ラベル [N]
        top_k_list: 計算するkのリスト
    
    Returns:
        {k: accuracy} の辞書
    """
    num_samples = len(labels)
    top_k_acc = {}
    
    # 各サンプルについて、予測確率上位k個に正解が含まれるかチェック
    for k in top_k_list:
        top_k_preds = np.argsort(predictions, axis=1)[:, -k:]  # 上位k個のインデックス
        correct = np.sum([labels[i] in top_k_preds[i] for i in range(num_samples)])
        top_k_acc[k] = correct / num_samples
    
    return top_k_acc


def calculate_distance_error(
    predictions: np.ndarray,      # [N, num_classes] 予測確率
    labels: np.ndarray,           # [N] 正解ラベル
    coordinates: np.ndarray,      # [N, 3] ノード座標 (x, y, z)
    defect_class_ids: set = None  # 欠陥クラスIDの集合（Noneの場合は1以上）
) -> Dict[str, float]:
    """
    位置推定の距離誤差を計算
    
    Args:
        predictions: 予測確率 [N, num_classes]
        labels: 正解ラベル [N]
        coordinates: ノード座標 [N, 3]
        defect_class_ids: 欠陥クラスIDの集合
    
    Returns:
        {
            'mean_distance_error': float,
            'median_distance_error': float,
            'max_distance_error': float,
            'num_defects': int
        }
    """
    if defect_class_ids is None:
        defect_class_ids = set(range(1, predictions.shape[1]))
    
    # 欠陥ノードのみを対象
    defect_mask = np.array([label in defect_class_ids for label in labels])
    if not np.any(defect_mask):
        return {
            'mean_distance_error': np.nan,
            'median_distance_error': np.nan,
            'max_distance_error': np.nan,
            'num_defects': 0
        }
    
    defect_coords = coordinates[defect_mask]
    defect_labels = labels[defect_mask]
    defect_preds = predictions[defect_mask]
    
    # 各欠陥について、予測位置と正解位置の距離を計算
    distances = []
    
    for i, (true_label, pred_probs, coord) in enumerate(zip(defect_labels, defect_preds, defect_coords)):
        # 正解位置（同じクラスのノードの重心）
        same_class_mask = (labels == true_label) & defect_mask
        if np.sum(same_class_mask) > 0:
            true_center = np.mean(coordinates[same_class_mask], axis=0)
        else:
            true_center = coord  # フォールバック
        
        # 予測位置（予測確率が高いノードの重心）
        pred_class = np.argmax(pred_probs)
        if pred_class in defect_class_ids:
            pred_class_mask = (labels == pred_class) & defect_mask
            if np.sum(pred_class_mask) > 0:
                pred_center = np.mean(coordinates[pred_class_mask], axis=0)
            else:
                pred_center = coord  # フォールバック
        else:
            pred_center = coord  # 非欠陥と予測された場合
        
        # ユークリッド距離
        dist = np.linalg.norm(pred_center - true_center)
        distances.append(dist)
    
    distances = np.array(distances)
    
    return {
        'mean_distance_error': float(np.mean(distances)),
        'median_distance_error': float(np.median(distances)),
        'max_distance_error': float(np.max(distances)),
        'num_defects': len(distances)
    }


def calculate_localization_metrics(
    predictions: np.ndarray,      # [N, num_classes] 予測確率
    labels: np.ndarray,           # [N] 正解ラベル
    coordinates: np.ndarray,      # [N, 3] ノード座標
    top_k_list: List[int] = [1, 3, 5, 10],
    defect_class_ids: set = None
) -> Dict[str, any]:
    """
    Localization指標をまとめて計算
    
    Returns:
        {
            'top_k_accuracy': {1: 0.xx, 3: 0.xx, ...},
            'distance_error': {
                'mean': float,
                'median': float,
                'max': float,
                'num_defects': int
            },
            'auprc': float  # 欠陥検出のAUPRC
        }
    """
    # Top-k Accuracy
    top_k_acc = calculate_top_k_accuracy(predictions, labels, top_k_list)
    
    # Distance Error
    dist_error = calculate_distance_error(predictions, labels, coordinates, defect_class_ids)
    
    # AUPRC (欠陥検出の二値分類として)
    if defect_class_ids is None:
        defect_class_ids = set(range(1, predictions.shape[1]))
    
    binary_labels = np.array([1 if label in defect_class_ids else 0 for label in labels])
    # 欠陥クラスの予測確率の合計
    defect_probs = predictions[:, list(defect_class_ids)].sum(axis=1)
    
    if len(np.unique(binary_labels)) == 2:  # 両方のクラスが存在する場合のみ
        auprc = average_precision_score(binary_labels, defect_probs)
    else:
        auprc = np.nan
    
    return {
        'top_k_accuracy': top_k_acc,
        'distance_error': dist_error,
        'auprc': float(auprc)
    }
```

---

## 3. Cross-edge（Yehia方式）の実装

### 3.1 `gnn_common/data_utils.py` に追加する関数

```python
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def create_cross_edges(
    coordinates: np.ndarray,      # [N, 3] ノード座標
    surface_node_mask: np.ndarray, # [N] bool, Trueがsurfaceノード
    k: int = 1                    # 各ノードからsurfaceノードへのエッジ数
) -> torch.Tensor:
    """
    A-B cross-edgeを作成（Yehia方式）
    
    Args:
        coordinates: ノード座標 [N, 3]
        surface_node_mask: surfaceノードのマスク [N]
        k: 各ノードからsurfaceノードへのエッジ数
    
    Returns:
        cross_edge_index: [2, E] エッジインデックス
    """
    # Surfaceノード（B）とInteriorノード（A）を分離
    surface_indices = np.where(surface_node_mask)[0]
    interior_indices = np.where(~surface_node_mask)[0]
    
    if len(surface_indices) == 0:
        # Surfaceノードがない場合は空のエッジを返す
        return torch.empty((2, 0), dtype=torch.long)
    
    # Interiorノードの座標
    interior_coords = coordinates[interior_indices]
    surface_coords = coordinates[surface_indices]
    
    # kNNで最近傍のsurfaceノードを探す
    if k >= len(surface_indices):
        k = len(surface_indices)
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(surface_coords)
    distances, indices = nbrs.kneighbors(interior_coords)
    
    # エッジを作成: interior -> surface
    edge_list = []
    for i, interior_idx in enumerate(interior_indices):
        for j in range(k):
            surface_idx = surface_indices[indices[i, j]]
            edge_list.append([interior_idx, surface_idx])
    
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    cross_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return cross_edge_index


def add_cross_edges_to_data(
    data: Data,
    coordinates: np.ndarray,
    surface_node_mask: np.ndarray,
    k: int = 1
) -> Data:
    """
    Dataオブジェクトにcross-edgeを追加
    
    Args:
        data: PyG Dataオブジェクト
        coordinates: ノード座標 [N, 3]
        surface_node_mask: surfaceノードのマスク [N]
        k: 各ノードからsurfaceノードへのエッジ数
    
    Returns:
        更新されたDataオブジェクト
    """
    cross_edge_index = create_cross_edges(coordinates, surface_node_mask, k)
    
    # 既存のエッジと結合
    if cross_edge_index.size(1) > 0:
        if data.edge_index.size(1) > 0:
            # 既存エッジと結合
            data.edge_index = torch.cat([data.edge_index, cross_edge_index], dim=1)
        else:
            data.edge_index = cross_edge_index
    
    # エッジタイプを記録（オプション）
    if hasattr(data, 'edge_type'):
        # 既存のedge_typeがある場合
        original_types = data.edge_type
        cross_types = torch.ones(cross_edge_index.size(1), dtype=torch.long) * 2  # タイプ2 = cross-edge
        data.edge_type = torch.cat([original_types, cross_types])
    else:
        # 新規作成
        original_types = torch.zeros(data.edge_index.size(1) - cross_edge_index.size(1), dtype=torch.long)
        cross_types = torch.ones(cross_edge_index.size(1), dtype=torch.long) * 2
        data.edge_type = torch.cat([original_types, cross_types])
    
    return data


def identify_surface_nodes(
    coordinates: np.ndarray,  # [N, 3]
    z_coords: np.ndarray,     # [N] z座標（層方向）
    method: str = 'outer_layer',  # 'outer_layer' or 'top_p'
    top_p: float = 0.1        # method='top_p'の場合の上位p%
) -> np.ndarray:
    """
    Surfaceノードを識別
    
    Args:
        coordinates: ノード座標 [N, 3]
        z_coords: z座標（層方向） [N]
        method: 'outer_layer' (外層) または 'top_p' (上位p%)
        top_p: method='top_p'の場合の上位p%
    
    Returns:
        surface_node_mask: [N] bool, Trueがsurfaceノード
    """
    if method == 'outer_layer':
        # 外層（z座標が最大・最小の層）をsurfaceとする
        z_min, z_max = z_coords.min(), z_coords.max()
        z_threshold = (z_max - z_min) * 0.1  # 上下10%を外層とする
        surface_mask = (z_coords <= z_min + z_threshold) | (z_coords >= z_max - z_threshold)
        
    elif method == 'top_p':
        # z座標が上位p%のノードをsurfaceとする
        threshold = np.percentile(z_coords, 100 * (1 - top_p))
        surface_mask = z_coords >= threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return surface_mask
```

### 3.2 `gnn_common/models.py` に追加する関数

```python
class GATModelWithCrossEdges(torch.nn.Module):
    """Cross-edge対応のGATモデル"""
    def __init__(
        self, 
        hidden_channels=64, 
        num_classes=19, 
        num_heads=4, 
        dropout=0.2,
        use_edge_type=False  # エッジタイプを使うか
    ):
        super(GATModelWithCrossEdges, self).__init__()
        self.use_edge_type = use_edge_type
        
        self.conv1 = GATConv(4, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels * 2, heads=num_heads, concat=True)
        self.conv3 = GATConv(hidden_channels * 2 * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.fc = nn.Linear(hidden_channels * num_heads, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        
        # エッジタイプ埋め込み（オプション）
        if use_edge_type:
            self.edge_type_embedding = nn.Embedding(3, hidden_channels)  # タイプ0, 1, 2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # エッジタイプがある場合は特徴量に追加（簡易版）
        # より高度な実装では、GATConvのedge_attrとして渡す
        if self.use_edge_type and hasattr(data, 'edge_type'):
            # ここでは簡易版として、エッジタイプ情報をノード特徴に反映
            # 実際の実装では、GATConvのedge_attrパラメータを使う
            pass
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.fc(x)
        return x
```

---

## 4. 設定ファイルへの追加

### 4.1 `config.yaml.example` に追加

```yaml
# Data splitting configuration
data_splitting:
  split_type: "iid"  # 'iid', 'defect_size', 'defect_ratio', 'layer'
  test_ratio: 0.2
  val_ratio: 0.1
  seed: 42
  
  # OOD分割のパラメータ
  ood_params:
    defect_size:
      size_threshold: 5
    defect_ratio:
      ratio_threshold: 0.1
    layer:
      train_layers: [1, 2, 3]  # これらの層をtrainに

# Cross-edge configuration
cross_edge:
  enabled: false
  k: 1  # 各ノードからsurfaceノードへのエッジ数
  surface_method: "outer_layer"  # 'outer_layer' or 'top_p'
  top_p: 0.1  # surface_method='top_p'の場合

# Localization metrics
localization:
  calculate: true
  top_k_list: [1, 3, 5, 10]
  defect_class_ids: null  # nullの場合は1以上を欠陥とする
```

---

## 5. 使用方法の例

### 5.1 OOD分割の使用

```python
from gnn_common.data_utils import create_ood_split, visualize_split_statistics

# データペアの準備
pairs = create_data_label_pairs(data_files, label_files)

# OOD分割（欠陥サイズで分割）
train_pairs, val_pairs, test_pairs = create_ood_split(
    pairs,
    split_type='defect_size',
    label_data_folder='/path/to/labels',
    size_threshold=5,
    seed=42
)

# 分割統計の可視化
visualize_split_statistics(
    train_pairs, val_pairs, test_pairs,
    label_data_folder='/path/to/labels',
    output_path='split_statistics.png'
)
```

### 5.2 Localization指標の計算

```python
from gnn_common.metrics import calculate_localization_metrics
import numpy as np

# 予測と座標を取得
predictions = model_outputs  # [N, num_classes]
labels = true_labels  # [N]
coordinates = node_coords  # [N, 3]

# Localization指標を計算
metrics = calculate_localization_metrics(
    predictions, labels, coordinates,
    top_k_list=[1, 3, 5, 10]
)

print(f"Top-1 Accuracy: {metrics['top_k_accuracy'][1]:.4f}")
print(f"Top-5 Accuracy: {metrics['top_k_accuracy'][5]:.4f}")
print(f"Mean Distance Error: {metrics['distance_error']['mean']:.4f}")
print(f"AUPRC: {metrics['auprc']:.4f}")
```

### 5.3 Cross-edgeの使用

```python
from gnn_common.data_utils import (
    add_cross_edges_to_data, 
    identify_surface_nodes
)

# Surfaceノードを識別
z_coords = data.x[:, 2]  # z座標を取得
surface_mask = identify_surface_nodes(
    coordinates, z_coords, method='outer_layer'
)

# Cross-edgeを追加
data = add_cross_edges_to_data(
    data, coordinates, surface_mask, k=1
)

# モデルで使用
model = GATModelWithCrossEdges(use_edge_type=True)
output = model(data)
```

---

## 6. 実装の優先順位

1. **Phase 1**: OOD分割の実装（最優先）
   - `calculate_defect_statistics` の実装
   - `create_ood_split` の実装
   - 設定ファイルへの統合

2. **Phase 2**: Localization指標の実装
   - `calculate_localization_metrics` の実装
   - 評価ループへの統合

3. **Phase 3**: Cross-edgeの実装
   - `create_cross_edges` の実装
   - モデル側の対応
   - アブレーション実験の準備

---

## 7. テスト方法

各機能について、簡単なテストスクリプトを作成することを推奨します：

```python
# test_ood_split.py
from gnn_common.data_utils import create_ood_split

# テストデータで動作確認
pairs = [("file1.npy", "file1_19label.npy"), ...]
train, val, test = create_ood_split(pairs, split_type='defect_size', ...)
assert len(train) + len(val) + len(test) == len(pairs)
```
