"""データ準備関連のユーティリティ関数（GNN_zscore_sub_noise_defect_free.pyを基に）"""
import os
import re
import numpy as np
import torch
from torch_geometric.data import Data

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