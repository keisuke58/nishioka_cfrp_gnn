"""データ準備関連のユーティリティ関数（GNN_zscore_sub_noise_defect_free.pyを基に）"""
import os
import re
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# VERBOSE_PRINT のデフォルト値（環境変数から取得可能）
VERBOSE_PRINT = int(os.environ.get("VERBOSE_PRINT", "0") or "0")


def extract_layer_block(file_name):
    """ファイル名から層とブロック番号を抽出（すべてのH_Wサイズに対応）
    
    GNN_zscore_sub_noise_defect_free.pyの実装を基にしています。
    H2_W2, H4_W4, H8_W8, H5_W6, H7_W4, H4_W8, H8_W4など、すべてのサイズに対応。
    """
    try:
        # H2_W2, H4_W4, H8_W8, H5_W6, H7_W4, H4_W8, H8_W4など、すべてのサイズに対応
        match = re.search(r'L(\d+)_B(\d+)_el(\d+)_H(\d+)_W(\d+)', file_name)
        if match:
            layer = int(match.group(1))
            block = int(match.group(2))
            return (layer, block)
        else:
            # フォールバック: 古い形式 L(\d+)B(\d+) にも対応
            layer_block_str = re.search(r'L(\d+)B(\d+)', file_name)
            if layer_block_str:
                layer = int(layer_block_str.group(1))
                block = int(layer_block_str.group(2))
                return (layer, block)
            if VERBOSE_PRINT:
                print(f"Invalid file name format: {file_name}")
            return None
    except AttributeError:
        if VERBOSE_PRINT:
            print(f"Invalid file name format: {file_name}")
        return None


def create_data_label_pairs(data_files, label_files):
    """データファイルとラベルファイルのペアを作成
    
    GNN_zscore_sub_noise_defect_free.pyの実装を基にしています。
    ベース名（_19label.npyを除いた部分）でマッチングします。
    """
    data_label_pairs = {}
    unmatched_labels = set(label_files)  # ラベルファイルの集合
    no_label_counter = 0

    # データファイル名からベース名を抽出（_19label.npyを除いた部分）
    for data_file in data_files:
        # データファイル名からベース名を取得（.npyを除く）
        base_name = os.path.splitext(data_file)[0]
        data_label_pairs[base_name] = {"data": data_file}

    # ラベルファイルとマッチング
    for label_file in label_files:
        if label_file.endswith("_19label.npy"):
            # ラベルファイル名からベース名を取得（_19label.npyを除く）
            base_name = label_file.replace("_19label.npy", "")
            if base_name in data_label_pairs:
                data_label_pairs[base_name]["label"] = label_file
                unmatched_labels.discard(label_file)  # マッチしたラベルを削除
            else:
                # マッチしないラベルを記録（最初の3つだけ）
                if no_label_counter < 3 and VERBOSE_PRINT:
                    print(f"No matching data file for label: {label_file}")
                    no_label_counter += 1

    # ペアが作成されなかった場合を出力（3つに制限）
    no_data_counter = 0
    for base_name, v in data_label_pairs.items():
        if "label" not in v:
            if no_data_counter < 3 and VERBOSE_PRINT:  # 最大3つまで出力
                print(f"No label found for data file: {v['data']}")
                no_data_counter += 1

    # マッチしなかったラベルを出力（最初の3つだけ）
    unmatched_counter = 0
    for unmatched_label in unmatched_labels:
        if unmatched_counter < 3 and VERBOSE_PRINT:
            print(f"Unmatched label file: {unmatched_label}")
            unmatched_counter += 1

    valid_pairs = [(v["data"], v["label"]) for k, v in data_label_pairs.items() if "label" in v]
    
    # マッチしたペアの確認ログを追加
    if len(valid_pairs) > 0:
        print(f"✓ Successfully matched {len(valid_pairs)} pairs")
        if VERBOSE_PRINT:
            # 最初の3つのマッチしたペアの例を表示
            print("  Sample matched pairs:")
            for i, (data_file, label_file) in enumerate(valid_pairs[:3]):
                # ベース名が一致しているか確認
                data_base = os.path.splitext(data_file)[0]
                label_base = label_file.replace("_19label.npy", "")
                match_status = "✓" if data_base == label_base else "✗ MISMATCH"
                print(f"    [{i+1}] {match_status} Data: {data_file} <-> Label: {label_file}")
                if data_base != label_base:
                    print(f"         WARNING: Base names don't match! Data base: '{data_base}', Label base: '{label_base}'")
    else:
        print("⚠ WARNING: No valid pairs found!")
    
    return valid_pairs


def prepare_data(pairs, standardized_data_folder, label_data_folder, 
                 x_coords, y_coords, z_coords, edge_index, 
                 max_nodes=3654, return_class_weights=False):
    """
    データを準備してDataオブジェクトのリストを返す
    
    Args:
        pairs: (data_file, label_file)のタプルのリスト
        standardized_data_folder: 標準化されたデータフォルダのパス
        label_data_folder: ラベルデータフォルダのパス
        x_coords, y_coords, z_coords: 座標データ
        edge_index: エッジインデックス
        max_nodes: 最大ノード数（デフォルト: 3654）
        return_class_weights: クラス重みを返すかどうか
    
    Returns:
        data_list: Dataオブジェクトのリスト
        class_weights (optional): クラス重みのテンソル
    """
    data_list = []
    labels = []
    
    for data_file, label_file in pairs:
        data_file_path = os.path.join(standardized_data_folder, data_file)
        label_file_path = os.path.join(label_data_folder, label_file)

        values = np.load(data_file_path)[:max_nodes]
        label = np.load(label_file_path)[:max_nodes]

        node_features = np.vstack((x_coords, y_coords, z_coords, values)).T
        x = torch.tensor(node_features, dtype=torch.float)

        y = torch.argmax(torch.tensor(label, dtype=torch.float), dim=1).long()

        if return_class_weights:
            labels.extend(y.tolist())

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    if return_class_weights:
        class_weights = compute_class_weights(np.array(labels))
        return data_list, class_weights
    
    return data_list


def compute_class_weights(labels, multiplier=None, fix_class0_weight=True, class0_weight=1.0, num_classes=19):
    """
    クラス重みを計算（GNN_zscore_sub_noise_defect_free.pyの実装を基に）
    
    Args:
        labels: ラベル配列
        multiplier: 未使用（後方互換性のため）
        fix_class0_weight: Trueの場合、Class 0の重みを固定値にする（推奨）
        class0_weight: Class 0の固定重み値（fix_class0_weight=Trueの場合）
        num_classes: クラス数（デフォルト: 19）
    
    Returns:
        クラス重みテンソル
    """
    class_counts = np.bincount(labels, minlength=num_classes)  # Ensure all classes are counted
    
    if fix_class0_weight:
        # Class 0の重みを固定値にして、他のクラスだけ重み付け（崩壊防止に最も効果的）
        # これにより、Class 0が99.8%を占めていても、重みが極端に小さくなるのを防ぐ
        weights = np.ones(num_classes) * class0_weight  # Class 0を含む全てを初期化
        
        # Class 1以降に対してのみ逆頻度重みを計算
        for i in range(1, num_classes):
            if class_counts[i] > 0:
                weights[i] = class0_weight * (class_counts[0] / class_counts[i])
            else:
                weights[i] = class0_weight
        
        # 重みの上限を設定（極端に大きくなるのを防ぐ）
        max_weight = class0_weight * 100  # Class 0の100倍まで
        weights = np.clip(weights, class0_weight, max_weight)
    else:
        # 従来の方法（クリップ付き）
        weights = 1.0 / (class_counts + 1e-6)
        # クラス重みのクリップ: 極端に小さくなるのを防ぐ（崩壊防止）
        # 下限を0.1に設定（0.01では正規化後に再び小さくなるため）
        weights = np.clip(weights, 0.1, None)
    
    # 平均が1になるようにスケール（相対比は維持、損失寄与が弱くなるのを防ぐ）
    # 従来の weights.sum() による正規化は全ての重みを小さくしてしまうため、平均基準に変更
    class_weights = weights / np.mean(weights)
    
    return torch.tensor(class_weights, dtype=torch.float)


def prepare_data(pairs, normalized_data_folder, label_data_folder, 
                 x_coords, y_coords, z_coords, edge_index, 
                 class_weight_multiplier=None, data_folder_map=None,
                 max_nodes=13942, return_class_weights=True, verbose_print=None):
    """
    データを準備してDataオブジェクトのリストを返す
    （GNN_zscore_sub_noise_defect_free.pyの実装を基に）
    
    Args:
        pairs: (data_file, label_file)のタプルのリスト
        normalized_data_folder: デフォルトのデータフォルダ（data_folder_mapがNoneの場合に使用）
        label_data_folder: ラベルフォルダ
        x_coords, y_coords, z_coords: 座標データ
        edge_index: エッジインデックス
        class_weight_multiplier: クラス重みの乗数（未使用、後方互換性のため）
        data_folder_map: データファイル名からデータフォルダへのマッピング（オプション）
        max_nodes: 最大ノード数（デフォルト: 13942）
        return_class_weights: クラス重みを返すかどうか（デフォルト: True）
        verbose_print: 詳細出力フラグ（Noneの場合は環境変数VERBOSE_PRINTを使用）
    
    Returns:
        data_list: Dataオブジェクトのリスト
        class_weights: クラス重みのテンソル（return_class_weights=Trueの場合）
    """
    if verbose_print is None:
        verbose_print = VERBOSE_PRINT
    
    data_list = []
    labels = []
    pair_counter = 0  # 確認用カウンタ
    ndf_fallback_label_count = 0  # NoiseDefectFree_ のラベル自動生成回数（デバッグ用）
    missing_data_count = 0
    missing_label_count = 0
    load_error_count = 0
    feature_error_count = 0

    total_pairs = len(pairs)
    for pair_idx, (data_file, label_file) in enumerate(pairs):
        if pair_idx % 500 == 0 and pair_idx > 0:
            print(f"  Loading data: {pair_idx}/{total_pairs} pairs processed...")
        
        # データファイルのパスを決定（data_folder_mapがあればそれを使用）
        if data_folder_map is not None and data_file in data_folder_map:
            data_file_path = os.path.join(data_folder_map[data_file], data_file)
        else:
            data_file_path = os.path.join(normalized_data_folder, data_file)
        
        label_file_path = os.path.join(label_data_folder, label_file)
        label = None  # IMPORTANT: ループごとに初期化（前ループの値が残らないようにする）

        if not os.path.exists(data_file_path):
            missing_data_count += 1
            if verbose_print:
                print(f"データファイルが存在しません: {data_file_path}")
            continue
        if not os.path.exists(label_file_path):
            # 欠陥なし（NoiseDefectFree_）は単一ラベル（全ノード0）として扱えるようにする
            # ラベルファイルが無い場合でも学習/推論に含める
            if str(data_file).startswith("NoiseDefectFree_"):
                label = np.zeros((max_nodes,), dtype=np.int64)
                ndf_fallback_label_count += 1
            else:
                missing_label_count += 1
                if verbose_print:
                    print(f"ラベルファイルが存在しません: {label_file_path}")
                continue

        # データとラベルをロード
        try:
            values = np.load(data_file_path)[:max_nodes]
            # NoiseDefectFree_ でラベルが欠損していない場合は通常通りロード
            if label is None:
                label = np.load(label_file_path)[:max_nodes]
        except Exception as e:
            load_error_count += 1
            if verbose_print:
                print(f"データ読み込みエラー: {e}")
            continue

        # ノード特徴量を作成
        try:
            node_features = np.vstack((x_coords, y_coords, z_coords, values)).T
            x = torch.tensor(node_features, dtype=torch.float)
            
            # ラベルの形状を確認して適切に処理
            label_tensor = torch.tensor(label, dtype=torch.float)
            if len(label_tensor.shape) == 2 and label_tensor.shape[1] > 1:
                # One-hotエンコードされたラベルの場合
                y = torch.argmax(label_tensor, dim=1).long()
            elif len(label_tensor.shape) == 1 or (len(label_tensor.shape) == 2 and label_tensor.shape[1] == 1):
                # 直接クラスIDの場合
                y = label_tensor.long().squeeze()
            else:
                raise ValueError(f"Unexpected label shape: {label_tensor.shape}")
        except Exception as e:
            feature_error_count += 1
            if verbose_print:
                print(f"特徴量作成エラー: {e}, label shape: {label.shape if 'label' in locals() else 'unknown'}")
            continue

        # データリストとラベルを保存
        data = Data(x=x, edge_index=edge_index, y=y)
        # ファイル名を保存（推論時の対応付けのため）
        data.filename = data_file
        
        # ファイル単位のminority_ratioを計算して保存（sampler用）
        # minority_ratio = 1 - (count0 / N) で、高いほどminorityが多い
        unique_labels, counts = torch.unique(y, return_counts=True)
        total_nodes = len(y)
        count0 = counts[unique_labels == 0].item() if 0 in unique_labels else 0
        minority_ratio = 1.0 - (count0 / total_nodes) if total_nodes > 0 else 0.0
        data.minority_ratio = minority_ratio
        
        data_list.append(data)
        if return_class_weights:
            labels.extend(y.tolist())

        # デバッグ用にいくつかのペアを出力（ペアの整合性も確認）
        if verbose_print and pair_counter < 1:
            print(f"Pair {pair_counter + 1}:")
            print(f"Data file: {data_file_path}")
            print(f"Label file: {label_file_path}")
            # ペアの整合性を確認
            data_base = os.path.splitext(data_file)[0]
            label_base = label_file.replace("_19label.npy", "")
            if data_base == label_base:
                print(f"✓ Pair match verified: '{data_base}' <-> '{label_base}'")
            else:
                print(f"✗ Pair mismatch detected: '{data_base}' != '{label_base}'")
            print(f"x shape: {x.shape}")
            print(f"y shape: {y.shape}")
            # ラベルの詳細情報を出力
            print(f"Label file shape: {label.shape}")
            print(f"Label file dtype: {label.dtype}")
            print(f"Label file min/max: {label.min()}/{label.max()}")
            if len(label.shape) == 2:
                print(f"Label file is 2D (one-hot): shape={label.shape}")
            else:
                print(f"Label file is 1D (class IDs): shape={label.shape}")
            # 変換後のラベルの分布を確認
            unique_labels, counts = torch.unique(y, return_counts=True)
            print(f"Converted label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
            print(f"Converted label min/max: {y.min().item()}/{y.max().item()}")
            pair_counter += 1

    if ndf_fallback_label_count > 0:
        print(f"[INFO] NoiseDefectFree label fallback used: {ndf_fallback_label_count} files (generated all-zero labels)")

    # Avoid noisy per-sample warnings by default; summarize once.
    if (missing_data_count or missing_label_count or load_error_count or feature_error_count) and not verbose_print:
        print(
            "[INFO] prepare_data summary: "
            f"missing_data={missing_data_count}, missing_label={missing_label_count}, "
            f"load_errors={load_error_count}, feature_errors={feature_error_count}"
        )

    # クラス重みを計算
    if return_class_weights:
        if len(labels) > 0:
            labels_array = np.array(labels)
            class_weights = compute_class_weights(labels_array, multiplier=class_weight_multiplier)
            
            # クラス分布と重みを出力（デバッグ用）
            unique_labels, counts = np.unique(labels_array, return_counts=True)
            if verbose_print:
                print(f"\n=== Class Distribution and Weights ===")
                print(f"Total samples: {len(labels_array)}")
                print(f"Class distribution:")
                for cls, count in zip(unique_labels, counts):
                    weight = class_weights[cls].item()
                    percentage = count / len(labels_array) * 100
                    print(f"  Class {cls}: {count:8d} samples ({percentage:6.2f}%), weight: {weight:.6f}")
                print(f"Class weights: {class_weights.cpu().numpy()}")
                print(f"Class weights min/max: {class_weights.min().item():.6f}/{class_weights.max().item():.6f}")
            else:
                print(
                    f"[INFO] class_weights computed: total_labels={len(labels_array)}, "
                    f"min/max={class_weights.min().item():.6f}/{class_weights.max().item():.6f}"
                )
            return data_list, class_weights
        else:
            print("ラベルが存在しません。クラス重みの計算に失敗しました。")
            return None, None
    
    return data_list


def calculate_defect_statistics(
    pairs: List[Tuple[str, str]], 
    label_data_folder: str,
    max_nodes: int = 13942
) -> Dict[str, Dict]:
    """
    各データファイルの欠陥統計を計算
    
    Args:
        pairs: (data_file, label_file)のタプルのリスト
        label_data_folder: ラベルデータフォルダのパス
        max_nodes: 最大ノード数
    
    Returns:
        {
            'filename': {
                'defect_ratio': float,  # 欠陥ノードの割合
                'defect_size': float,   # 欠陥クラス数（簡易版）
                'layer': int,          # 層番号
                'block': int,          # ブロック番号
                'defect_classes': set,  # 出現する欠陥クラスID
                'data_file': str,
                'label_file': str
            }
        }
    """
    stats = {}
    
    for data_file, label_file in pairs:
        try:
            label_path = os.path.join(label_data_folder, label_file)
            
            # ラベルファイルが存在しない場合（NoiseDefectFree等）
            if not os.path.exists(label_path):
                if str(data_file).startswith("NoiseDefectFree_"):
                    # NoiseDefectFreeの場合は全ノードがクラス0
                    y = np.zeros(max_nodes, dtype=np.int64)
                else:
                    if VERBOSE_PRINT:
                        print(f"Label file not found: {label_path}, skipping...")
                    continue
            else:
                label = np.load(label_path)[:max_nodes]
                
                # ラベルをクラスIDに変換
                if len(label.shape) == 2 and label.shape[1] > 1:
                    y = np.argmax(label, axis=1)
                else:
                    y = label.flatten().astype(int)
            
            # 欠陥統計を計算
            defect_mask = y > 0  # クラス0以外が欠陥
            defect_ratio = defect_mask.sum() / len(y) if len(y) > 0 else 0.0
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
        split_type: 'iid', 'defect_size', 'defect_ratio', 'layer'
        test_ratio: テストセットの割合
        val_ratio: バリデーションセットの割合（残りからの割合）
        label_data_folder: ラベルフォルダのパス（OOD分割に必要）
        max_nodes: 最大ノード数
        seed: 乱数シード
        **kwargs: 分割タイプ固有のパラメータ
            - defect_size: size_threshold (int)
            - defect_ratio: ratio_threshold (float)
            - layer: train_layers (list of int)
    
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
        print(f"[INFO] IID split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
        return train_pairs, val_pairs, test_pairs
    
    # OOD分割の場合は統計情報が必要
    if label_data_folder is None:
        raise ValueError("label_data_folder is required for OOD splits")
    
    print(f"[INFO] Calculating defect statistics for OOD split (type={split_type})...")
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
        if len(train_pairs) > 0:
            train_pairs, val_pairs = train_test_split(
                train_pairs, test_size=val_ratio/(1-test_ratio), random_state=seed
            )
        else:
            val_pairs = []
            print("[WARN] No training pairs found for defect_size split!")
        
        print(f"[INFO] Defect size split (threshold={size_threshold}): "
              f"train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
        
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
        if len(train_pairs) > 0:
            train_pairs, val_pairs = train_test_split(
                train_pairs, test_size=val_ratio/(1-test_ratio), random_state=seed
            )
        else:
            val_pairs = []
            print("[WARN] No training pairs found for layer split!")
        
        print(f"[INFO] Layer split (train_layers={sorted(train_layers)}): "
              f"train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
        
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
        if len(train_pairs) > 0:
            train_pairs, val_pairs = train_test_split(
                train_pairs, test_size=val_ratio/(1-test_ratio), random_state=seed
            )
        else:
            val_pairs = []
            print("[WARN] No training pairs found for defect_ratio split!")
        
        print(f"[INFO] Defect ratio split (threshold={ratio_threshold}): "
              f"train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
        
    else:
        raise ValueError(f"Unknown split_type: {split_type}. "
                      f"Supported types: 'iid', 'defect_size', 'defect_ratio', 'layer'")
    
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
    
    Args:
        train_pairs: トレーニングペアのリスト
        val_pairs: バリデーションペアのリスト
        test_pairs: テストペアのリスト
        label_data_folder: ラベルデータフォルダのパス
        output_path: 出力パス（Noneの場合は表示）
        max_nodes: 最大ノード数
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn not available. Skipping visualization.")
        return
    
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
        if len(ratios) > 0:
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
            if os.path.splitext(p[0])[0] in stats and stats[os.path.splitext(p[0])[0]]['layer'] >= 0
        ]
        if len(layers) > 0:
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
        if len(sizes) > 0:
            ax.hist(sizes, alpha=0.5, label=split_name, bins=range(0, 20))
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
        print(f"[INFO] Split statistics saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


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
        raise ValueError(f"Unknown method: {method}. Supported: 'outer_layer', 'top_p'")
    
    return surface_mask


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
        original_size = data.edge_index.size(1) - cross_edge_index.size(1)
        if original_size > 0:
            original_types = torch.zeros(original_size, dtype=torch.long)
            cross_types = torch.ones(cross_edge_index.size(1), dtype=torch.long) * 2
            data.edge_type = torch.cat([original_types, cross_types])
        else:
            # 既存エッジがない場合
            data.edge_type = torch.ones(cross_edge_index.size(1), dtype=torch.long) * 2
    
    return data