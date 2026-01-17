"""評価メトリクス関連の関数"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    classification_report, precision_recall_fscore_support,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score
)


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