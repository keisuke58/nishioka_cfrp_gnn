"""
既存のトレーニングスクリプトへの統合用ユーティリティ

OOD分割、Localization指標、Cross-edgeを既存のコードに統合するためのヘルパー関数
"""

import os
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from .data_utils import (
    create_ood_split,
    identify_surface_nodes,
    add_cross_edges_to_data,
    prepare_data as base_prepare_data
)
from .metrics import calculate_localization_metrics
from .models import GATModelWithCrossEdges


def apply_ood_split(
    pairs: List[Tuple[str, str]],
    split_type: str = 'iid',
    label_data_folder: Optional[str] = None,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
    **ood_params
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    OOD分割を適用（既存コードとの互換性を保つ）
    
    Args:
        pairs: (data_file, label_file)のタプルのリスト
        split_type: 'iid', 'defect_size', 'defect_ratio', 'layer'
        label_data_folder: ラベルフォルダのパス
        test_ratio: テストセットの割合
        val_ratio: バリデーションセットの割合
        seed: 乱数シード
        **ood_params: OOD分割のパラメータ
    
    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    return create_ood_split(
        pairs,
        split_type=split_type,
        label_data_folder=label_data_folder,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
        **ood_params
    )


def prepare_data_with_cross_edges(
    pairs: List[Tuple[str, str]],
    normalized_data_folder: str,
    label_data_folder: str,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    edge_index: torch.Tensor,
    max_nodes: int = 13942,
    return_class_weights: bool = True,
    use_cross_edges: bool = False,
    cross_edge_k: int = 1,
    cross_edge_method: str = 'outer_layer',
    verbose_print: Optional[bool] = None,
    **kwargs
):
    """
    Cross-edge対応のデータ準備関数
    
    Args:
        pairs: (data_file, label_file)のタプルのリスト
        normalized_data_folder: データフォルダ
        label_data_folder: ラベルフォルダ
        x_coords, y_coords, z_coords: 座標データ
        edge_index: 既存のエッジインデックス
        max_nodes: 最大ノード数
        return_class_weights: クラス重みを返すか
        use_cross_edges: Cross-edgeを使用するか
        cross_edge_k: Cross-edgeのk値
        cross_edge_method: Surfaceノード識別方法
        verbose_print: 詳細出力フラグ
        **kwargs: prepare_dataに渡すその他の引数
    
    Returns:
        data_list, class_weights (return_class_weights=Trueの場合)
        または data_list (return_class_weights=Falseの場合)
    """
    # 通常のデータ準備
    result = base_prepare_data(
        pairs,
        normalized_data_folder,
        label_data_folder,
        x_coords,
        y_coords,
        z_coords,
        edge_index,
        max_nodes=max_nodes,
        return_class_weights=return_class_weights,
        verbose_print=verbose_print,
        **kwargs
    )
    
    # 返り値の処理（base_prepare_dataがreturn_class_weightsに応じて適切に返すことを期待）
    # base_prepare_dataはreturn_class_weights=Trueの場合は(data_list, class_weights)を返し、
    # return_class_weights=Falseの場合はdata_listのみを返す
    if return_class_weights:
        # return_class_weights=Trueの場合、常にタプルを期待
        if isinstance(result, tuple) and len(result) == 2:
            data_list, class_weights = result
        elif isinstance(result, tuple) and len(result) == 1:
            # タプルだが要素が1つの場合（通常は発生しない）
            data_list = result[0]
            class_weights = None
        else:
            # 単一値の場合（通常は発生しない）
            data_list = result
            class_weights = None
    else:
        # return_class_weights=Falseの場合、resultはdata_listのみのはず
        if isinstance(result, tuple):
            # タプルの場合は最初の要素を取得（base_prepare_dataが誤ってタプルを返した場合のフォールバック）
            data_list = result[0]
            if len(result) > 1:
                # 警告を出力（デバッグ用）
                if verbose_print:
                    print(f"[WARN] prepare_data_with_cross_edges: base_prepare_data returned tuple when return_class_weights=False, using first element")
        else:
            data_list = result
        class_weights = None
    
    # Cross-edgeを追加
    if use_cross_edges:
        if verbose_print:
            print(f"[INFO] Adding cross-edges (k={cross_edge_k}, method={cross_edge_method})...")
        
        for i, data in enumerate(data_list):
            # 座標を取得（data.xの最初の3次元）
            coords = data.x[:, :3].numpy()
            z_coords_data = data.x[:, 2].numpy()
            
            # Surfaceノードを識別
            surface_mask = identify_surface_nodes(
                coords, z_coords_data,
                method=cross_edge_method
            )
            
            # Cross-edgeを追加（data_list内のオブジェクトを直接更新）
            updated_data = add_cross_edges_to_data(
                data, coords, surface_mask, k=cross_edge_k
            )
            # data_list内のオブジェクトを更新（in-place更新を確実にする）
            data_list[i] = updated_data
        
        if verbose_print:
            print(f"[INFO] Cross-edges added to {len(data_list)} data samples")
    
    # 返り値の処理（確実に正しい形式で返す）
    if return_class_weights:
        return data_list, class_weights
    else:
        return data_list


def evaluate_with_localization_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 19,
    calculate_localization: bool = True,
    defect_class_ids: Optional[set] = None
) -> Dict[str, Any]:
    """
    Localization指標を含む評価関数
    
    Args:
        model: モデル
        data_loader: データローダー
        device: デバイス
        num_classes: クラス数
        calculate_localization: Localization指標を計算するか
        defect_class_ids: 欠陥クラスIDの集合
    
    Returns:
        評価結果の辞書（従来のメトリクス + Localization指標）
    """
    import torch.nn.functional as F
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_coordinates = []
    total_loss = 0.0
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch)
            y = batch.y
            
            # 損失計算
            loss = loss_fn(out, y)
            total_loss += loss.item()
            
            # 予測
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # 座標を取得（data.xの最初の3次元）
            coords = batch.x[:, :3].cpu().numpy()
            all_coordinates.append(coords)
    
    # 結合
    all_probs = np.vstack(all_probs)
    all_coordinates = np.vstack(all_coordinates)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 従来のメトリクス
    accuracy = (all_preds == all_labels).mean()
    avg_loss = total_loss / len(data_loader)
    
    # 混同行列からメトリクスを計算（M1-3: 層別F1含む）
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    from .metrics import metrics_from_confusion_matrix
    cm_metrics = metrics_from_confusion_matrix(cm)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    results = {
        'test_loss': avg_loss,
        'test_accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        # M1-3: 層別F1
        'layer1_macro_f1': cm_metrics.get('layer1_macro_f1'),
        'layer2_macro_f1': cm_metrics.get('layer2_macro_f1'),
    }
    
    # Localization指標を追加
    if calculate_localization:
        try:
            loc_metrics = calculate_localization_metrics(
                all_probs, all_labels, all_coordinates,
                top_k_list=[1, 3, 5, 10],
                defect_class_ids=defect_class_ids
            )
            results['localization'] = loc_metrics
        except Exception as e:
            print(f"[WARN] Failed to calculate localization metrics: {e}")
            results['localization'] = None
    
    return results


def create_model_with_cross_edges(
    model_type: str = 'GAT',
    use_cross_edges: bool = False,
    **model_kwargs
) -> torch.nn.Module:
    """
    Cross-edge対応モデルを作成
    
    Args:
        model_type: 'GAT' または 'GCN'
        use_cross_edges: Cross-edgeを使用するか
        **model_kwargs: モデルのパラメータ
    
    Returns:
        モデルインスタンス
    """
    if model_type == 'GAT':
        if use_cross_edges:
            return GATModelWithCrossEdges(**model_kwargs)
        else:
            from .models import GATModel
            return GATModel(**model_kwargs)
    elif model_type == 'GCN':
        from .models import GCNModel
        return GCNModel(**model_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_config_for_splitting(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    設定ファイルから分割設定を読み込む
    
    Args:
        config_path: 設定ファイルのパス（Noneの場合はデフォルト）
    
    Returns:
        分割設定の辞書
    """
    import yaml
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        # デフォルト設定を返す
        return {
            'split_type': 'iid',
            'test_ratio': 0.2,
            'val_ratio': 0.1,
            'seed': 42,
            'ood_params': {}
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    split_config = config.get('data_splitting', {})
    split_type = split_config.get('split_type', 'iid')
    ood_params = split_config.get('ood_params', {}).get(split_type, {})
    
    return {
        'split_type': split_type,
        'test_ratio': split_config.get('test_ratio', 0.2),
        'val_ratio': split_config.get('val_ratio', 0.1),
        'seed': split_config.get('seed', 42),
        'ood_params': ood_params
    }


def load_config_for_cross_edges(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    設定ファイルからCross-edge設定を読み込む
    
    Args:
        config_path: 設定ファイルのパス（Noneの場合はデフォルト）
    
    Returns:
        Cross-edge設定の辞書
    """
    import yaml
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        # デフォルト設定を返す
        return {
            'enabled': False,
            'k': 1,
            'surface_method': 'outer_layer',
            'top_p': 0.1
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cross_edge_config = config.get('cross_edge', {})
    
    return {
        'enabled': cross_edge_config.get('enabled', False),
        'k': cross_edge_config.get('k', 1),
        'surface_method': cross_edge_config.get('surface_method', 'outer_layer'),
        'top_p': cross_edge_config.get('top_p', 0.1)
    }
