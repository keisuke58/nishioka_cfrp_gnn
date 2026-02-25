"""評価メトリクス関連の関数（GNN_zscore_sub_noise_defect_free.pyを基に）"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    classification_report, precision_recall_fscore_support,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    average_precision_score
)
from typing import List, Dict, Set, Optional
from scipy.spatial.distance import cdist


def metrics_from_confusion_matrix(cm: np.ndarray, eps: float = 1e-12):
    """
    Compute precision/recall/F1 directly from a confusion matrix.
    
    Notes:
    - Macro-F1 here matches sklearn's default label set behavior by averaging only over
      classes that appear in y_true OR y_pred (i.e., non-zero row or column in cm).
    - Weighted-* averages are weighted by true support (row sums), like sklearn.
    
    Args:
        cm: Confusion matrix (KxK numpy array)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Dictionary containing various metrics
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square (KxK). Got shape={cm.shape}")

    tp = np.diag(cm)
    support = cm.sum(axis=1)          # true counts per class
    pred_count = cm.sum(axis=0)       # predicted counts per class
    fp = pred_count - tp
    fn = support - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    total = support.sum()
    accuracy = tp.sum() / (total + eps) if total > 0 else 0.0

    # Macro definitions (make the "support=0" handling explicit):
    # - sklearn-like: labels in union(y_true, y_pred) => non-zero row OR column
    macro_mask_sklearn_like = (support + pred_count) > 0
    macro_f1_sklearn_like = float(f1[macro_mask_sklearn_like].mean()) if np.any(macro_mask_sklearn_like) else 0.0
    # - support-only: labels that appear in y_true (non-zero row). Recommended for "valに存在しないクラスでMacroが潰れる"回避。
    macro_mask_support_only = support > 0
    macro_f1_support_only = float(f1[macro_mask_support_only].mean()) if np.any(macro_mask_support_only) else 0.0
    # - all-classes: always average over all classes (includes support=0 as 0). Useful only for debugging.
    macro_f1_all = float(f1.mean()) if f1.size > 0 else 0.0

    weighted_precision = float((precision * support).sum() / (total + eps)) if total > 0 else 0.0
    weighted_recall = float((recall * support).sum() / (total + eps)) if total > 0 else 0.0
    weighted_f1 = float((f1 * support).sum() / (total + eps)) if total > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        # Backward compatible key (historically used by this script)
        "macro_f1": macro_f1_sklearn_like,
        # Explicit variants (Step0: macro定義の固定)
        "macro_f1_sklearn_like": macro_f1_sklearn_like,
        "macro_f1_support_only": macro_f1_support_only,
        "macro_f1_all": macro_f1_all,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def calculate_metrics(all_labels, all_preds, num_classes=19, show_plot=True):
    """
    精度評価のメトリクス計算関数
    
    Args:
        all_labels: 正解ラベルの配列
        all_preds: 予測ラベルの配列
        num_classes: クラス数
        show_plot: 混同行列を表示するかどうか
    
    Returns:
        precision, recall, f1: メトリクス値
    """
    # 各クラスごとの precision, recall, f1 を計算し、全体を平均
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 各クラスの指標を含むレポートを表示
    class_report = classification_report(all_labels, all_preds, zero_division=0)
    print("\nClassification Report:\n", class_report)
    
    # 混同行列を生成
    if show_plot:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=range(num_classes), 
                   yticklabels=range(num_classes))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    
    return precision, recall, f1


def evaluate_and_visualize(test_loader, model, device, num_classes=19):
    """
    精度評価と可視化のための関数
    
    Args:
        test_loader: テストデータローダー
        model: モデル
        device: デバイス
        num_classes: クラス数
    
    Returns:
        metrics_dict: メトリクスの辞書
    """
    import torch
    
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            y = batch.y

            # 損失計算
            loss = loss_fn(out, y)
            total_loss += loss.item()

            # 予測
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            correct += (pred == y).sum().item()
            total_samples += y.size(0)

    # 精度計算
    test_accuracy = correct / total_samples
    avg_test_loss = total_loss / len(test_loader)
    
    # Confusion Matrix と Classification Report の表示
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    class_report = classification_report(all_labels, all_preds, zero_division=0)

    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # ROC AUC スコア (multi-class) - 注意: 確率が必要
    try:
        # この実装は簡略化されています。実際には確率を保存する必要があります
        roc_auc = "ROC AUC calculation requires probabilities"
    except ValueError:
        roc_auc = "ROC AUC calculation failed due to missing classes in prediction."
    
    # 混同行列の可視化
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
               xticklabels=range(num_classes), 
               yticklabels=range(num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print(f"\nClassification Report:\n{class_report}")
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"ROC AUC Score: {roc_auc}")

    # 予測値 vs. 正解ラベルの散布図
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(all_labels)), all_labels, color="blue", 
               label="Actual Labels", alpha=0.6)
    plt.scatter(range(len(all_preds)), all_preds, color="red", marker="x", 
               label="Predicted Labels", alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("Predicted vs. Actual Class Labels")
    plt.legend()
    plt.show()
    
    return {
        'test_loss': avg_test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'balanced_accuracy': balanced_acc,
        'mcc': mcc
    }


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
        if k > predictions.shape[1]:
            k = predictions.shape[1]  # kがクラス数を超える場合は調整
        
        top_k_preds = np.argsort(predictions, axis=1)[:, -k:]  # 上位k個のインデックス
        correct = np.sum([labels[i] in top_k_preds[i] for i in range(num_samples)])
        top_k_acc[k] = correct / num_samples if num_samples > 0 else 0.0
    
    return top_k_acc


def calculate_distance_error(
    predictions: np.ndarray,      # [N, num_classes] 予測確率
    labels: np.ndarray,           # [N] 正解ラベル
    coordinates: np.ndarray,      # [N, 3] ノード座標 (x, y, z)
    defect_class_ids: Optional[Set[int]] = None  # 欠陥クラスIDの集合（Noneの場合は1以上）
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
    
    if len(distances) == 0:
        return {
            'mean_distance_error': np.nan,
            'median_distance_error': np.nan,
            'max_distance_error': np.nan,
            'num_defects': 0
        }
    
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
    defect_class_ids: Optional[Set[int]] = None
) -> Dict[str, any]:
    """
    Localization指標をまとめて計算
    
    Args:
        predictions: 予測確率 [N, num_classes]
        labels: 正解ラベル [N]
        coordinates: ノード座標 [N, 3]
        top_k_list: Top-k Accuracyを計算するkのリスト
        defect_class_ids: 欠陥クラスIDの集合（Noneの場合は1以上）
    
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
    dist_error_raw = calculate_distance_error(predictions, labels, coordinates, defect_class_ids)
    
    # Transform keys to match documented interface
    dist_error = {
        'mean': dist_error_raw.get('mean_distance_error', np.nan),
        'median': dist_error_raw.get('median_distance_error', np.nan),
        'max': dist_error_raw.get('max_distance_error', np.nan),
        'num_defects': dist_error_raw.get('num_defects', 0)
    }
    
    # AUPRC (欠陥検出の二値分類として)
    if defect_class_ids is None:
        defect_class_ids = set(range(1, predictions.shape[1]))
    
    binary_labels = np.array([1 if label in defect_class_ids else 0 for label in labels])
    # 欠陥クラスの予測確率の合計
    defect_probs = predictions[:, list(defect_class_ids)].sum(axis=1)
    
    if len(np.unique(binary_labels)) == 2:  # 両方のクラスが存在する場合のみ
        try:
            auprc = average_precision_score(binary_labels, defect_probs)
        except Exception as e:
            print(f"[WARN] Error calculating AUPRC: {e}")
            auprc = np.nan
    else:
        auprc = np.nan
    
    return {
        'top_k_accuracy': top_k_acc,
        'distance_error': dist_error,
        'auprc': float(auprc)
    }


def compute_benchmark_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    num_classes: int = 19,
    all_probs: Optional[np.ndarray] = None,
    coordinates: Optional[np.ndarray] = None,
    size_labels: Optional[np.ndarray] = None,
    size_preds: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    Compute the full standardized benchmark metric suite.

    Composes existing metric functions into a single dict suitable for
    benchmark comparison tables.

    Args:
        all_labels: Ground-truth class labels [N]
        all_preds: Predicted class labels [N]
        num_classes: Number of classes (default 19)
        all_probs: Predicted probabilities [N, num_classes] (optional, for top-k / AUPRC)
        coordinates: Node coordinates [N, 3] (optional, for distance error)
        size_labels: Ground-truth size class labels [G] (optional, multi-task)
        size_preds: Predicted size class labels [G] (optional, multi-task)

    Returns:
        Dictionary with keys:
            macro_f1, weighted_f1, balanced_accuracy, mcc, accuracy,
            top_k_accuracy (if all_probs), mean_distance_error (if coordinates),
            auprc (if all_probs), size_accuracy (if size_labels/preds)
    """
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)

    # --- Core classification metrics from confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm_metrics = metrics_from_confusion_matrix(cm)

    result: Dict[str, any] = {
        "accuracy": cm_metrics["accuracy"],
        "macro_f1": cm_metrics["macro_f1"],
        "macro_f1_support_only": cm_metrics["macro_f1_support_only"],
        "weighted_f1": cm_metrics["weighted_f1"],
        "weighted_precision": cm_metrics["weighted_precision"],
        "weighted_recall": cm_metrics["weighted_recall"],
        "balanced_accuracy": float(balanced_accuracy_score(all_labels, all_preds)),
        "mcc": float(matthews_corrcoef(all_labels, all_preds)),
    }

    # --- Localization metrics (require probabilities) ---
    if all_probs is not None:
        all_probs = np.asarray(all_probs)
        top_k = calculate_top_k_accuracy(all_probs, all_labels, top_k_list=[1, 3, 5])
        result["top_1_accuracy"] = top_k.get(1, np.nan)
        result["top_3_accuracy"] = top_k.get(3, np.nan)
        result["top_5_accuracy"] = top_k.get(5, np.nan)

        # AUPRC (binary: defect vs non-defect)
        defect_class_ids = set(range(1, num_classes))
        binary_labels = np.array([1 if l in defect_class_ids else 0 for l in all_labels])
        if len(np.unique(binary_labels)) == 2:
            defect_probs = all_probs[:, list(defect_class_ids)].sum(axis=1)
            try:
                result["auprc"] = float(average_precision_score(binary_labels, defect_probs))
            except Exception:
                result["auprc"] = np.nan
        else:
            result["auprc"] = np.nan

    # --- Distance error (require probabilities + coordinates) ---
    if all_probs is not None and coordinates is not None:
        coordinates = np.asarray(coordinates)
        dist = calculate_distance_error(all_probs, all_labels, coordinates)
        result["mean_distance_error"] = dist["mean_distance_error"]
        result["median_distance_error"] = dist["median_distance_error"]

    # --- Multi-task size accuracy ---
    if size_labels is not None and size_preds is not None:
        size_labels = np.asarray(size_labels)
        size_preds = np.asarray(size_preds)
        if len(size_labels) > 0:
            result["size_accuracy"] = float(np.mean(size_labels == size_preds))
        else:
            result["size_accuracy"] = np.nan

    return result