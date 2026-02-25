import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import datetime
import time
from torch.utils.data import DistributedSampler, Sampler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score, roc_auc_score, roc_curve, auc
import optuna
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from Loss.focal_loss import FocalLoss

# Log-softmax version of Focal Loss for numerical stability
class FocalLossLogSoftmax(nn.Module):
    """Focal Loss using log_softmax for numerical stability"""
    def __init__(self, weights=None, gamma=2.0, reduction='mean'):
        super(FocalLossLogSoftmax, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weights is not None:
            self.register_buffer('weights', weights)
        else:
            self.weights = None
    
    def forward(self, logits, target):
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # Get log probability of the correct class
        target_one_hot = F.one_hot(target, num_classes=logits.size(1))
        log_probs_target = (log_probs * target_one_hot).sum(dim=1)
        probs_target = (probs * target_one_hot).sum(dim=1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - probs_target) ** self.gamma
        
        # Apply class weights if provided
        if self.weights is not None:
            class_weights = self.weights[target]
            loss = -class_weights * focal_weight * log_probs_target
        else:
            loss = -focal_weight * log_probs_target
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# モデル定義 (GAT)
# ----------------------------

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def edge_dropout(edge_index, drop_prob=0.0002):

    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > drop_prob  # 一定確率でTrue/Falseを生成
    edge_index_dropped = edge_index[:, mask]  # ドロップアウト後のエッジを保持
    
    return edge_index_dropped


class GATModel(torch.nn.Module):
    def __init__(self, hidden_channels=32, num_classes=19):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(4, hidden_channels, heads=4, concat=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels * 4)
        
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels * 2, heads=4, concat=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels * 8)
        
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=4, concat=True)
        self.batch_norm3 = nn.BatchNorm1d(hidden_channels * 4)
        
        # self.conv4 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        # self.batch_norm4 = nn.BatchNorm1d(hidden_channels * 4)
        
        self.fc = nn.Linear(hidden_channels * 4, num_classes)
        self.dropout = nn.Dropout(p=0.0002)

        # プロジェクションレイヤー
        self.proj1 = nn.Linear(4, hidden_channels * 4)
        self.proj2 = nn.Linear(hidden_channels * 4, hidden_channels * 8)
        self.proj3 = nn.Linear(hidden_channels * 8, hidden_channels * 4)

        self.apply(initialize_weights)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # エッジドロップアウトの適用（トレーニング時のみ）
        if self.training:
            edge_index = edge_dropout(edge_index, drop_prob=0.0002)

        # Layer 1 with residual connection
        x_residual = x
        x = F.relu(self.conv1(x, edge_index))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        if x_residual.size(1) != x.size(1):
            x_residual = self.proj1(x_residual)  # プロジェクションでサイズを合わせる
        x = x + x_residual

        # Layer 2 with residual connection
        x_residual = x
        x = F.relu(self.conv2(x, edge_index))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        if x_residual.size(1) != x.size(1):
            x_residual = self.proj2(x_residual)  # プロジェクションでサイズを合わせる
        x = x + x_residual

        # Layer 3 with residual connection
        x_residual = x
        x = F.relu(self.conv3(x, edge_index))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        if x_residual.size(1) != x.size(1):
            x_residual = self.proj3(x_residual)  # プロジェクションでサイズを合わせる
        x = x + x_residual

        # Fully connected layer (return logits, not probabilities)
        x = self.fc(x)
        # Note: softmax is applied in loss function or during inference
        return x

# Default gamma for focal loss (can be overridden by args)
default_gamma = 2.0
default_class_weight_multiplier = 5.0  # Increased from 1.0 to better handle imbalanced classes

def compute_class_weights(labels, multiplier=None):
    if multiplier is None:
        multiplier = default_class_weight_multiplier
    class_counts = np.bincount(labels, minlength=19)  # Ensure all 19 classes are counted
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1.0)
    
    # Use sklearn-style balanced weights: n_samples / (n_classes * np.bincount(y))
    n_samples = len(labels)
    n_classes = len(class_counts)
    class_weights = n_samples / (n_classes * class_counts)
    
    # Apply multiplier to minority classes (all except class 0)
    class_weights[1:] *= multiplier
    
    # Normalize to prevent extreme values
    class_weights = class_weights / class_weights.sum() * n_classes
    
    return torch.tensor(class_weights, dtype=torch.float)

def save_predictions_to_csv(test_pairs, all_preds, all_labels, all_probs, output_dir_csv, num_nodes_per_sample=13942):
    """
    予測結果をノード単位およびファイル単位でCSVに保存
    """
    os.makedirs(output_dir_csv, exist_ok=True)  # CSV保存先ディレクトリ

    # --- ファイル単位の統計情報を保存 ---
    file_results = []
    for i, filename_tuple in enumerate(test_pairs):
        filename = filename_tuple[0] if isinstance(filename_tuple, tuple) else filename_tuple
        start_idx = i * num_nodes_per_sample
        end_idx = start_idx + num_nodes_per_sample
    
        # 範囲チェック
        if end_idx > len(all_preds) or end_idx > len(all_labels):
            raise ValueError(f"Index out of range: start_idx={start_idx}, end_idx={end_idx}, total_samples={len(all_preds)}")
    
        # 各ファイルの予測情報を取得
        sample_preds = np.array(all_preds[start_idx:end_idx])  # 各ファイルの予測
        sample_labels = np.array(all_labels[start_idx:end_idx])  # 実際のラベル
    
        # クラス分布の計算（負の値や範囲外の値がないか確認）
        if np.any(sample_preds < 0) or np.any(sample_preds >= 20):
            raise ValueError(f"Invalid prediction values found in file: {filename}")
        class_counts = np.bincount(sample_preds.astype(int), minlength=19)  # 20クラス分の予測カウント
    
        # 精度の計算
        accuracy = (sample_preds == sample_labels).sum() / len(sample_labels)
    
        # 結果をリストに追加
        file_results.append({
            "Filename": filename,
            "Accuracy": accuracy,
            "Class Distribution": ",".join(map(str, class_counts))
        })
    
    # ファイル単位の結果をDataFrameに変換
    file_results_df = pd.DataFrame(file_results)
    
    # CSV保存
    file_results_csv_path = os.path.join(output_dir_csv, f"file_statistics_{timestamp}.csv")
    try:
        file_results_df.to_csv(file_results_csv_path, index=False)
        print(f"File-level statistics saved to {file_results_csv_path}")
    except Exception as e:
        print(f"Error saving file-level statistics: {e}")

def evaluate_and_visualize(final_test_loader, train_loader, ddp_model, device, class_weights, test_pairs, train_pairs, num_nodes_per_sample=13942, gamma=2.0):
    ddp_model.eval()# モデルを評価モードに切り替え
    test_preds = []
    test_labels = []
    test_probs = []
    total_loss = 0.0
    correct = 0
    total_samples = 0

    # Loss function for evaluation (use log_softmax version for consistency)
    loss_fn = FocalLossLogSoftmax(weights=class_weights, gamma=gamma, reduction='mean').to(device)
    
    with torch.no_grad():  # 評価時は勾配計算を無効化
        for batch in final_test_loader:
            batch = batch.to(device)
            out = ddp_model(batch)
            y = batch.y

            # Calculate loss
            loss = loss_fn(out, y)
            total_loss += loss.item()

            #softmax で確率を計算 (test_probs に格納)
            probs = F.softmax(out, dim=1).cpu().numpy()
            test_probs.extend(probs)

            # 実ラベル
            test_labels.extend(y.cpu().numpy())

            # 予測ラベル
            pred = out.argmax(dim=1).cpu().numpy()
            test_preds.extend(pred)

            correct += (pred == y.cpu().numpy()).sum()
            total_samples += y.size(0)

    test_accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_test_loss = total_loss / len(final_test_loader) if len(final_test_loader) > 0 else 0.0

    # --- 出力ディレクトリの作成 ---
    output_dir_test_predict = f"/home/nishioka/GNN/GNN_hole_2026/Predict_data/Predict19Class8x8_Test{timestamp}"
    os.makedirs(output_dir_test_predict, exist_ok=True)
    
    #ノード数スライスの整合性をチェック
    expected_test_total = len(test_pairs) * num_nodes_per_sample
    if len(test_preds) != expected_test_total:
        print("[WARNING] The total number of test_preds does not match test_pairs * num_nodes_per_sample.")
        print(f"          len(test_preds)={len(test_preds)}, expected={expected_test_total}")
        print("          Skipping file-by-file output for test data...")
    else:
        # 問題ない場合のみ、ファイルごとにスライスして保存
        for i, filename_tuple in enumerate(test_pairs):
            filename = filename_tuple[0] if isinstance(filename_tuple, tuple) else filename_tuple
            start_idx = i * num_nodes_per_sample
            end_idx = start_idx + num_nodes_per_sample

            # 1ファイルあたりの予測を取り出す
            sample_preds = np.array(test_preds[start_idx:end_idx])  
            base_filename = os.path.splitext(filename)[0]
            pred_filename = f"{base_filename}_pred.npy"
            np.save(os.path.join(output_dir_test_predict, pred_filename), sample_preds)

        print(f"Test data predictions saved for all test samples in: {output_dir_test_predict}") 
    
    all_preds_np = np.array(test_preds)    # テストデータ用
    all_labels_np = np.array(test_labels)  # テストデータ用
    all_probs_np = np.array(test_probs)    # テストデータ用

    print("Shape of all_preds:", all_preds_np.shape)
    print("Shape of all_labels:", all_labels_np.shape)
    print("Shape of all_probs:", all_probs_np.shape)

    output_dir1 = os.path.join(output_dir_test_predict, f"pred_data_{timestamp}")
    os.makedirs(output_dir1, exist_ok=True)

    np.save(os.path.join(output_dir1, "all_preds.npy"), all_preds_np)
    np.save(os.path.join(output_dir1, "all_labels.npy"), all_labels_np)
    np.save(os.path.join(output_dir1, "all_probs.npy"), all_probs_np)

    
    # --- メトリクス計算 (テストデータのみ)
    if len(all_preds_np) > 0 and len(all_labels_np) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_np, all_preds_np, average='weighted', zero_division=0
        )
        class_report = classification_report(all_labels_np, all_preds_np, zero_division=0)
        balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
        mcc = matthews_corrcoef(all_labels_np, all_preds_np)

        # ROC-AUC (multi-class)
        try:
            roc_auc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr')
        except ValueError:
            roc_auc = "ROC AUC calculation failed due to missing classes in prediction."

        cm = confusion_matrix(all_labels_np, all_preds_np)

        # ここで可視化 (混同行列や散布図など) を行う
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(20), yticklabels=range(20))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix (Test Data)")
        plt.show()

        # Calculate macro F1 for better evaluation of imbalanced classes
        macro_f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        macro_precision = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        macro_recall = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        
        print(f"\nClassification Report:\n{class_report}")
        print(f"\n=== Overall Metrics ===")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}, Weighted Recall: {recall:.4f}, Weighted F1-Score: {f1:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1-Score: {macro_f1:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
        print(f"ROC AUC Score: {roc_auc}")
        
        # Print class distribution for analysis
        unique_labels, label_counts = np.unique(all_labels_np, return_counts=True)
        unique_preds, pred_counts = np.unique(all_preds_np, return_counts=True)
        print(f"\n=== Class Distribution ===")
        print(f"Actual labels distribution: {dict(zip(unique_labels, label_counts))}")
        print(f"Predicted labels distribution: {dict(zip(unique_preds, pred_counts))}")

    else:
        print("[WARNING] No test predictions available for evaluation.")


    # Directory for saving images
    output_dir = f"/home/nishioka/GNN/GNN_hole_2026/Predict_truth/pred_vs_true_images_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
        
    # 1. 予測値 vs. 正解ラベルの散布図
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(all_labels_np)), all_labels_np, color="blue", label="Actual Labels", alpha=0.6)
    plt.scatter(range(len(all_preds_np)), all_preds_np, color="red", marker="x", label="Predicted Labels", alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("Predicted vs. Actual Class Labels")
    plt.legend()
    plt.savefig(f"{output_dir}/pred_vs_true_class_labels{timestamp}.png")  # Save as PNG
    plt.close()

    # 2. 混同行列のヒートマップ
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels_np, all_preds_np)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(20), yticklabels=range(20))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix{timestamp}.png")
    plt.close()

    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(all_labels_np, all_preds_np)
    # Apply log scale to avoid overwhelming contrast
    log_cm = np.log1p(cm)  # Use log1p to handle large values in a more balanced way
    # Set up the figure and heatmap for the PDF
    ax = sns.heatmap(log_cm, annot=cm, fmt="d", cmap="Blues", 
                     cbar_kws={'label': 'Counts (Log Scale)'}, vmin=0, square=True, 
                     annot_kws={"size": 8, "weight": "bold"}, linewidths=0.01, linecolor="gray")
    
    # Highlight the diagonal with red borders for correct classifications
    for i in range(cm.shape[0]):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    # Add labels and title
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix 2", fontsize=15)
    plt.savefig(f"{output_dir}/confusion_matrix2{timestamp}.png")

    # 4. クラスごとのF1スコア、Precision、Recallの棒グラフ
    class_report = classification_report(all_labels_np, all_preds_np, zero_division=0, output_dict=True)
    classes = list(range(19))
    f1_scores = [class_report[str(i)]['f1-score'] if str(i) in class_report else 0.0 for i in classes]
    precisions = [class_report[str(i)]['precision'] if str(i) in class_report else 0.0 for i in classes]
    recalls = [class_report[str(i)]['recall'] if str(i) in class_report else 0.0 for i in classes]
    supports = [class_report[str(i)]['support'] if str(i) in class_report else 0.0 for i in classes]
    
    # Create subplots for better visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1 Score
    axes[0, 0].bar(classes, f1_scores, color='skyblue', alpha=0.7)
    axes[0, 0].set_xlabel("Class", fontsize=12)
    axes[0, 0].set_ylabel("F1 Score", fontsize=12)
    axes[0, 0].set_title("F1 Score by Class", fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Precision
    axes[0, 1].bar(classes, precisions, color='lightgreen', alpha=0.7)
    axes[0, 1].set_xlabel("Class", fontsize=12)
    axes[0, 1].set_ylabel("Precision", fontsize=12)
    axes[0, 1].set_title("Precision by Class", fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Recall
    axes[1, 0].bar(classes, recalls, color='lightcoral', alpha=0.7)
    axes[1, 0].set_xlabel("Class", fontsize=12)
    axes[1, 0].set_ylabel("Recall", fontsize=12)
    axes[1, 0].set_title("Recall by Class", fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Support (class distribution)
    axes[1, 1].bar(classes, supports, color='plum', alpha=0.7)
    axes[1, 1].set_xlabel("Class", fontsize=12)
    axes[1, 1].set_ylabel("Support (Number of Samples)", fontsize=12)
    axes[1, 1].set_title("Class Distribution (Support)", fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')  # Log scale for better visualization of imbalanced classes
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_class_metrics{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the original F1 score plot for compatibility
    plt.figure(figsize=(12, 6))
    plt.bar(classes, f1_scores, color='skyblue', alpha=0.7)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("F1 Score by Class", fontsize=14, fontweight='bold')
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{output_dir}/f1score_class{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Images have been saved in {output_dir}")

# ----------------------------
# データペアリング関数
# ----------------------------
def extract_layer_block(file_name):
    """ファイル名から層とブロック番号を抽出"""
    try:
        layer_block_str = re.search(r'L(\d+)_B(\d+)_el(\d+)_H8_W8', file_name).groups()
        layer = int(layer_block_str[0])
        block = int(layer_block_str[1])
        return (layer, block)
    except AttributeError:
        print(f"Invalid file name format: {file_name}")
        return None

def create_data_label_pairs(data_files, label_files):
    """データファイルとラベルファイルのペアを作成"""
    data_label_pairs = {}
    unmatched_labels = set(label_files)  # ラベルファイルの集合
    no_label_counter = 0

    for data_file in data_files:
        layer_block = extract_layer_block(data_file)
        if layer_block:
            data_label_pairs[layer_block] = {"data": data_file}

    for label_file in label_files:
        layer_block = extract_layer_block(label_file)
        if layer_block and layer_block in data_label_pairs:
            if label_file.endswith("_19label.npy"):
                data_label_pairs[layer_block]["label"] = label_file
                unmatched_labels.discard(label_file)  # マッチしたラベルを削除

    # ペアが作成されなかった場合を出力（3つに制限）
    for k, v in data_label_pairs.items():
        if "label" not in v:
            if no_label_counter < 3:  # 最大3つまで出力
                print(f"No label found for data file: {v['data']}")
                no_label_counter += 1

    # マッチしなかったラベルを出力
    for unmatched_label in unmatched_labels:
        print(f"Unmatched label file: {unmatched_label}")

    valid_pairs = [(v["data"], v["label"]) for k, v in data_label_pairs.items() if "label" in v]
    return valid_pairs


# ----------------------------
# Class-Balanced Sampler
# ----------------------------
class ClassBalancedSampler(Sampler):
    """Class-balanced sampler that ensures each batch contains samples from all classes"""
    def __init__(self, dataset, num_classes=19, samples_per_class=None, shuffle=True):
        self.dataset = dataset
        self.num_classes = num_classes
        self.shuffle = shuffle
        
        # Calculate dominant class for each graph
        self.class_indices = {i: [] for i in range(num_classes)}
        for idx, data in enumerate(dataset):
            # Get the most frequent class in this graph
            classes, counts = torch.unique(data.y, return_counts=True)
            dominant_class = classes[counts.argmax()].item()
            self.class_indices[dominant_class].append(idx)
        
        # Determine samples per class
        if samples_per_class is None:
            # Use the minimum class size to ensure all classes are represented
            min_class_size = min(len(indices) for indices in self.class_indices.values() if len(indices) > 0)
            self.samples_per_class = min_class_size
        else:
            self.samples_per_class = samples_per_class
        
        # Create balanced indices
        self.indices = []
        for class_idx in range(num_classes):
            if len(self.class_indices[class_idx]) > 0:
                class_samples = self.class_indices[class_idx]
                if self.shuffle:
                    np.random.shuffle(class_samples)
                # Repeat if needed to reach samples_per_class
                if len(class_samples) < self.samples_per_class:
                    class_samples = (class_samples * ((self.samples_per_class // len(class_samples)) + 1))[:self.samples_per_class]
                else:
                    class_samples = class_samples[:self.samples_per_class]
                self.indices.extend(class_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        if self.shuffle:
            # Reshuffle class indices
            for class_idx in range(self.num_classes):
                if len(self.class_indices[class_idx]) > 0:
                    np.random.shuffle(self.class_indices[class_idx])
            # Recreate balanced indices
            indices = []
            for class_idx in range(self.num_classes):
                if len(self.class_indices[class_idx]) > 0:
                    class_samples = self.class_indices[class_idx]
                    if len(class_samples) < self.samples_per_class:
                        class_samples = (class_samples * ((self.samples_per_class // len(class_samples)) + 1))[:self.samples_per_class]
                    else:
                        class_samples = class_samples[:self.samples_per_class]
                    indices.extend(class_samples)
            np.random.shuffle(indices)
            return iter(indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class DistributedClassBalancedSampler(DistributedSampler):
    """Distributed version of ClassBalancedSampler"""
    def __init__(self, dataset, num_classes=19, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.num_classes = num_classes
        
        # Calculate dominant class for each graph
        self.class_indices = {i: [] for i in range(num_classes)}
        for idx, data in enumerate(dataset):
            classes, counts = torch.unique(data.y, return_counts=True)
            dominant_class = classes[counts.argmax()].item()
            self.class_indices[dominant_class].append(idx)
        
        # Create balanced indices per rank
        self._generate_indices()
    
    def _generate_indices(self):
        # Determine samples per class (minimum across all classes)
        min_class_size = min(len(indices) for indices in self.class_indices.values() if len(indices) > 0)
        samples_per_class = min_class_size // self.num_replicas
        if samples_per_class == 0:
            samples_per_class = 1
        
        # Create balanced indices for this rank
        indices = []
        for class_idx in range(self.num_classes):
            if len(self.class_indices[class_idx]) > 0:
                class_samples = self.class_indices[class_idx]
                if self.shuffle:
                    np.random.shuffle(class_samples)
                # Distribute across replicas
                rank_samples = class_samples[self.rank::self.num_replicas]
                if len(rank_samples) < samples_per_class:
                    rank_samples = (rank_samples * ((samples_per_class // len(rank_samples)) + 1))[:samples_per_class]
                else:
                    rank_samples = rank_samples[:samples_per_class]
                indices.extend(rank_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        self.indices = indices
    
    def __iter__(self):
        if self.shuffle:
            self._generate_indices()
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


# ----------------------------
# データ準備関数
# ----------------------------
def prepare_data(pairs, normalized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=None):
    data_list = []
    labels = []
    pair_counter = 0  # 確認用カウンタ

    for data_file, label_file in pairs:
        data_file_path = os.path.join(normalized_data_folder, data_file)
        label_file_path = os.path.join(label_data_folder, label_file)

        if not os.path.exists(data_file_path):
            print(f"データファイルが存在しません: {data_file_path}")
            continue
        if not os.path.exists(label_file_path):
            print(f"ラベルファイルが存在しません: {label_file_path}")
            continue

        # データとラベルをロード
        try:
            values = np.load(data_file_path)[:13942]
            label = np.load(label_file_path)[:13942]
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            continue

        # ノード特徴量を作成
        try:
            node_features = np.vstack((x_coords, y_coords, z_coords, values)).T
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.argmax(torch.tensor(label, dtype=torch.float), dim=1).long()
        except Exception as e:
            print(f"特徴量作成エラー: {e}")
            continue

        # データリストとラベルを保存
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
        labels.extend(y.tolist())

        # デバッグ用にいくつかのペアを出力
        if pair_counter < 1:
            print(f"Pair {pair_counter + 1}:")
            print(f"Data file: {data_file_path}")
            print(f"Label file: {label_file_path}")
            print(f"x shape: {x.shape}")
            print(f"y shape: {y.shape}")
            pair_counter += 1

    # クラス重みを計算
    if len(labels) > 0:
        class_weights = compute_class_weights(np.array(labels), multiplier=class_weight_multiplier)
    else:
        print("ラベルが存在しません。クラス重みの計算に失敗しました。")
        return None, None

    return data_list, class_weights

# ----------------------------
# シード設定関数
# ----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# プロセスグループの初期化
# ----------------------------
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

# ----------------------------
# プロセスグループのクリーンアップ
# ----------------------------
def cleanup():
    dist.destroy_process_group()

# ----------------------------
# メイン関数
# ----------------------------
def main(args):
    set_seed(42)

    # Initialize process group
    dist.init_process_group(
        backend='nccl',         # 'nccl' for GPUs, 'gloo' for CPUs
        init_method='env://',   # initialization method
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE'])
    )

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')

    if rank == 0:
        print(f"Rank: {rank}, World Size: {world_size}, Master Addr: {master_addr}, Master Port: {master_port}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"Rank {rank} using device: {device}")

    # Data preparation and model definition
    normalized_data_folder = "/home/nishioka/GNN/GNN_hole_2026/DSPSS_8x8/8x8_norm_clip_final"
    label_data_folder = "/home/nishioka/GNN/GNN_hole/GNN_19class/Def88_19class_label_except1"

    x_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
    y_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
    z_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")

    edges = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy")
    edge_index = torch.tensor(edges.T, dtype=torch.long).to(device)

    data_files = [f for f in os.listdir(normalized_data_folder) if f.startswith("Defect_L")]
    label_files = [
        f for f in os.listdir(label_data_folder)
        if f.startswith("Defect_L") and f.endswith("_19label.npy")
    ]

    pairs = create_data_label_pairs(data_files, label_files)
    if rank == 0:
        print(f"Total valid pairs: {len(pairs)}")

    model = GATModel(hidden_channels=args.hidden_channels, num_classes=19).to(device)
    
    # Create the DDP model after initializing the process group
    ddp_model = DDP(model, device_ids=[rank])

    # KFold Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold = 1
    all_fold_metrics = []
    
    for train_index, val_index in kf.split(pairs):
        if rank == 0:
            print(f"Starting Fold {fold}")

        train_pairs = [pairs[i] for i in train_index]
        val_pairs = [pairs[i] for i in val_index]

        if rank == 0:
            print(f"Train Pairs (Fold {fold}): Total {len(train_pairs)}")
            for i, pair in enumerate(train_pairs[:5]):  # 最初の5ペアを表示
                print(f"Train Pair {i + 1}: Data file: {pair[0]}, Label file: {pair[1]}")
        
        if rank == 0:
            print(f"Validation Pairs (Fold {fold}): Total {len(val_pairs)}")
            for i, pair in enumerate(val_pairs[:5]):  # 最初の5ペアを表示
                print(f"Validation Pair {i + 1}: Data file: {pair[0]}, Label file: {pair[1]}")
        
        if rank == 0:
            print(f"Proceeding with training for Fold {fold}...")

        # 修正ポイント：class_weightsを受け取る
        class_weight_multiplier = getattr(args, 'class_weight_multiplier', default_class_weight_multiplier)
        train_dataset, class_weights = prepare_data(train_pairs, normalized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier)
        val_dataset, _ = prepare_data(val_pairs, normalized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier)

        if train_dataset is not None:
            if rank == 0:
                print(f"Train Dataset Size (Fold {fold}): {len(train_dataset)}")
        if val_dataset is not None:
            if rank == 0:
                print(f"Validation Dataset Size (Fold {fold}): {len(val_dataset)}")
            
        # Use class-balanced sampler for training
        train_sampler = DistributedClassBalancedSampler(
            train_dataset, 
            num_classes=19, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
        
        if rank == 0:
            print(f"Using Class-Balanced Sampler for training (Fold {fold})")

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Learning rate scheduler: CosineAnnealingLR or OneCycleLR
        use_onecycle = getattr(args, 'use_onecycle', False)
        if use_onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=args.learning_rate,
                epochs=args.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                anneal_strategy='cos'
            )
            if rank == 0:
                print(f"Learning rate scheduler initialized: OneCycleLR (max_lr={args.learning_rate})")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.epochs,
                eta_min=1e-6
            )
            if rank == 0:
                print(f"Learning rate scheduler initialized: CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)")

        if rank == 0:
            precision_per_epoch = []
            recall_per_epoch = []
            f1_score_per_epoch = []
            macro_f1_per_epoch = []
            val_accuracy_per_epoch = []
            train_loss_per_epoch = []
            val_loss_per_epoch = []

        gamma = getattr(args, 'gamma', default_gamma)
        # Use log_softmax version for numerical stability
        use_log_softmax = getattr(args, 'use_log_softmax', True)
        if use_log_softmax:
            loss_fn = FocalLossLogSoftmax(weights=class_weights, gamma=gamma, reduction='mean').to(device)
            if rank == 0:
                print(f"Using FocalLossLogSoftmax (gamma={gamma}) for numerical stability")
        else:
            loss_fn = FocalLoss(weights=class_weights, gamma=gamma, reduction='mean').to(device)
            if rank == 0:
                print(f"Using FocalLoss (gamma={gamma})")

        # Time tracking for estimated completion time
        fold_start_time = time.time()
        epoch_times = []  # Store time per epoch for estimation
        
        best_val_loss = float('inf')
        best_macro_f1 = 0.0  # Track best macro F1 for model selection (only for early stopping)
        counter = 0
        macro_f1_counter = 0  # Counter for macro F1 based early stopping
        # AMP (Mixed Precision Training)
        use_amp = getattr(args, 'use_amp', True)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if rank == 0 and use_amp:
            print(f"Mixed Precision Training (AMP) enabled")

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            ddp_model.train()
            train_sampler.set_epoch(epoch)
            total_loss = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = ddp_model(batch)
                        y = batch.y
                        loss = loss_fn(out, y)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = ddp_model(batch)
                    y = batch.y
                    loss = loss_fn(out, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                
                # Update OneCycleLR scheduler per step
                if use_onecycle:
                    scheduler.step()

            avg_train_loss = torch.tensor(total_loss / len(train_loader), dtype=torch.float).to(device)
            dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
            avg_train_loss /= world_size

            ddp_model.eval()
            val_loss = 0.0
            correct = 0
            total_samples = 0
            all_preds = []
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = ddp_model(batch)
                    y = batch.y

                    loss = loss_fn(out, y)
                    val_loss += loss.item()

                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total_samples += y.size(0)

                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(y.cpu().numpy())

            avg_val_loss = torch.tensor(val_loss / len(val_loader), dtype=torch.float).to(device)
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
            avg_val_loss /= world_size

            total_correct = torch.tensor(correct, dtype=torch.float, device=device)
            total_samples_t = torch.tensor(total_samples, dtype=torch.float, device=device)
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_t, op=dist.ReduceOp.SUM)
            
            val_accuracy = total_correct / total_samples_t

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            if rank == 0:
                epoch_times.append(epoch_time)
                # Keep only last 10 epoch times for more accurate estimation
                if len(epoch_times) > 10:
                    epoch_times.pop(0)

            # ここに100エポックごとの詳細出力を追加
            if epoch % 100 == 0 and rank == 0:
                print(f"\n--- Epoch {epoch} ---")
                print(f"Class Weights: {class_weights.cpu().numpy()}")
                
                for param_group in optimizer.param_groups:
                    print(f"Learning Rate: {param_group['lr']}")
        
                print(f"Train Loss: {avg_train_loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}")
        
                if all_preds and all_labels:
                    all_preds_np = np.concatenate(all_preds)
                    all_labels_np = np.concatenate(all_labels)
                    precision, recall, f1, support = precision_recall_fscore_support(all_labels_np, all_preds_np, zero_division=0)
                    print(f"Per-Class Precision: {precision}")
                    print(f"Per-Class Recall: {recall}")
                    print(f"Per-Class F1-Score: {f1}")
                    print(f"Support per Class: {support}")
        
                    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                        all_labels_np, all_preds_np, average='weighted', zero_division=0)
                    print(f"Weighted Precision: {weighted_precision:.4f}, Weighted Recall: {weighted_recall:.4f}, Weighted F1-Score: {weighted_f1:.4f}")

            if rank == 0:
                if all_preds and all_labels:
                    all_preds_np = np.concatenate(all_preds)
                    all_labels_np = np.concatenate(all_labels)
                    cm = confusion_matrix(all_labels_np, all_preds_np)
                    # plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])  # この関数が定義されていない場合はコメントアウト

                    precision = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
                    recall = recall_score(all_labels_np, all_preds_np, average='weighted')
                    f1 = f1_score(all_labels_np, all_preds_np, average='weighted')
                    macro_f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
                    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Weighted F1-Score: {f1:.4f}, Macro F1-Score: {macro_f1:.4f}')
                else:
                    # デフォルト値を設定（通常は発生しないが、念のため）
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                    macro_f1 = 0.0

            if rank == 0:
                precision_per_epoch.append(precision)
                recall_per_epoch.append(recall)
                f1_score_per_epoch.append(f1)
                macro_f1_per_epoch.append(macro_f1)
                val_accuracy_per_epoch.append(val_accuracy.item())
                train_loss_per_epoch.append(avg_train_loss.item())
                val_loss_per_epoch.append(avg_val_loss.item())

            # Update learning rate scheduler (for CosineAnnealingLR)
            if not use_onecycle:
                scheduler.step()
            
            # Track best metrics for early stopping only (no model saving)
            if rank == 0:
                improved_loss = avg_val_loss.item() < best_val_loss
                improved_macro_f1 = macro_f1 > best_macro_f1
                
                if improved_loss:
                    best_val_loss = avg_val_loss.item()
                    counter = 0
                else:
                    counter += 1
                
                if improved_macro_f1:
                    best_macro_f1 = macro_f1
                    macro_f1_counter = 0
                else:
                    macro_f1_counter += 1
                
                # Early stopping: stop if both val_loss and macro_f1 don't improve
                if counter >= args.patience and macro_f1_counter >= args.patience:
                    if rank == 0:
                        print(f"Early stopping triggered for Fold {fold} at epoch {epoch} (val_loss: {counter}, macro_f1: {macro_f1_counter})")
                    break

            # 100エポックごとの出力と予想時間の表示
            if rank == 0 and epoch % 100 == 0:
                # Calculate estimated completion time
                if len(epoch_times) > 0:
                    avg_epoch_time = np.mean(epoch_times)
                    remaining_epochs = args.epochs - epoch
                    estimated_remaining_time = avg_epoch_time * remaining_epochs
                    
                    # Format time
                    hours = int(estimated_remaining_time // 3600)
                    minutes = int((estimated_remaining_time % 3600) // 60)
                    seconds = int(estimated_remaining_time % 60)
                    
                    elapsed_time = time.time() - fold_start_time
                    elapsed_hours = int(elapsed_time // 3600)
                    elapsed_minutes = int((elapsed_time % 3600) // 60)
                    elapsed_seconds = int(elapsed_time % 60)
                    
                    print(f'\n=== Fold {fold}, Epoch {epoch}/{args.epochs} ===')
                    print(f'Train Loss: {avg_train_loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}, Macro F1: {macro_f1:.4f}')
                    print(f'Elapsed Time: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}')
                    print(f'Estimated Remaining Time: {hours:02d}:{minutes:02d}:{seconds:02d} (Avg {avg_epoch_time:.2f}s/epoch)')
                    print('=' * 50)
                # epochs = range(1, args.epochs + 1)

                # plt.figure(figsize=(10, 6))
            
                # plt.plot(epochs, [t.cpu().numpy() if torch.is_tensor(t) else t for t in train_loss_per_epoch], label="Training Loss", color="blue", linewidth=2, marker='o')
                # plt.plot(epochs, [v.cpu().numpy() if torch.is_tensor(v) else v for v in val_loss_per_epoch], label="Validation Loss", color="red", linewidth=2, marker='s')
            
                # plt.xlabel("Epochs", fontsize=14)
                # plt.ylabel("Loss", fontsize=14)
                # plt.title("Training and Validation Loss over Epochs", fontsize=16)
                # plt.legend(fontsize=12)
                # plt.grid(True, linestyle='--', alpha=0.7)
                # plt.yscale('log')  # ログスケール
            
                # # 保存        
                # loss_plot_path = f"/home/nishioka/GNN/GNN_hole_2026/Predict_truth/pred_vs_true_images_{timestamp}/Loss_Line_Plot_{timestamp}.png"
                # save_dir = os.path.dirname(loss_plot_path)
                
                # # ディレクトリが存在しない場合は作成
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                
                # plt.savefig(loss_plot_path, bbox_inches="tight", dpi=300)
                # plt.show()

        if rank == 0:
            fold_elapsed_time = time.time() - fold_start_time
            fold_hours = int(fold_elapsed_time // 3600)
            fold_minutes = int((fold_elapsed_time % 3600) // 60)
            fold_seconds = int(fold_elapsed_time % 60)
            
            all_fold_metrics.append({
                "fold": fold,
                "best_val_loss": best_val_loss,
                "best_macro_f1": best_macro_f1,
                "val_accuracy": val_accuracy
            })
            print(f"\n{'='*60}")
            print(f"Completed Fold {fold}")
            print(f"Best Val Loss: {best_val_loss:.4f}, Best Macro F1: {best_macro_f1:.4f}")
            print(f"Fold Total Time: {fold_hours:02d}:{fold_minutes:02d}:{fold_seconds:02d}")
            print(f"{'='*60}\n")
        fold += 1

    dist.barrier()

    if rank == 0:
        print("Starting final training on all data...")

        train_val_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
        class_weight_multiplier = getattr(args, 'class_weight_multiplier', default_class_weight_multiplier)
        train_val_dataset, class_weights = prepare_data(train_val_pairs, normalized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier)
        test_dataset, _ = prepare_data(test_pairs, normalized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier)
        if train_val_dataset is None or class_weights is None:
            raise ValueError("Failed to prepare train_val_dataset or class_weights")

        test_dataset, _ = prepare_data(
            test_pairs,
            normalized_data_folder,
            label_data_folder,
            x_coords, y_coords, z_coords, edge_index
        )
        if test_dataset is None:
            raise ValueError("Failed to prepare test_dataset")
            
    # --------------------------------------------------------
    # 最終的な推論・保存・可視化の流れを修正した例
    # --------------------------------------------------------
        final_test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Use the last fold's model (no best model selection)
        ddp_model.eval()
        
        if rank == 0:
            print("Using the last fold's model for inference.")
        print("Evaluating on Test Data...")
        
        # ---- 1) テストデータ推論で all_preds / all_labels / all_probs を正しく埋める ----
        test_preds = []
        test_labels = []
        test_probs = []  # 確率を格納するリスト
        
        gamma = getattr(args, 'gamma', default_gamma)
        # Use log_softmax version for numerical stability
        use_log_softmax = getattr(args, 'use_log_softmax', True)
        if use_log_softmax:
            loss_fn = FocalLossLogSoftmax(weights=class_weights, gamma=gamma, reduction='mean').to(device)
        else:
            loss_fn = FocalLoss(weights=class_weights, gamma=gamma, reduction='mean').to(device)
        
        correct = 0
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(final_test_loader):
                batch = batch.to(device)
                out = ddp_model(batch)
                y = batch.y
        
                loss = loss_fn(out, y)
                total_loss += loss.item()
        
                # 予測ラベル
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_samples += y.size(0)
        
                # 予測ラベル・実ラベル・確率を格納
                test_preds.extend(pred.cpu().numpy())
                test_labels.extend(y.cpu().numpy())
                
                # 確率を格納 (softmax の結果が必要な場合)
                probs = F.softmax(out, dim=1).cpu().numpy()
                test_probs.extend(probs)
        
                # --- デバッグ用 ---
                if batch_idx < 1:  # 最初の1バッチのみ様子を見る
                    print(f"[DEBUG] Batch {batch_idx + 1}:")
                    print(f" - Loss: {loss.item():.4f}")
                    print(f" - Predictions (first 10): {pred[:10]}")
                    print(f" - Labels (first 10): {y[:10]}")
                    print(f" - Probabilities shape: {probs.shape}")  # (batch_size, num_classes)
        
        test_loss = total_loss / len(final_test_loader)
        test_accuracy = correct / total_samples
        
        # numpy 配列に変換
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        test_probs = np.array(test_probs)
        
        print(f"\n[INFO] Test set result:")
        print(f" - Test Loss: {test_loss:.4f}")
        print(f" - Test Accuracy: {test_accuracy:.4f}")
        precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
        recall = recall_score(test_labels, test_preds, average='weighted')
        f1 = f1_score(test_labels, test_preds, average='weighted')
        macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        macro_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
        macro_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(test_labels, test_preds)
        mcc = matthews_corrcoef(test_labels, test_preds)
        
        print(f" - Weighted Precision: {precision:.4f}, Weighted Recall: {recall:.4f}, Weighted F1-Score: {f1:.4f}")
        print(f" - Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1-Score: {macro_f1:.4f}")
        print(f" - Balanced Accuracy: {balanced_acc:.4f}")
        print(f" - Matthews Correlation Coefficient: {mcc:.4f}")
        print(f" - test_preds.shape={test_preds.shape}, test_labels.shape={test_labels.shape}, test_probs.shape={test_probs.shape}")
        
        # Print per-class performance summary
        class_report_dict = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)
        print(f"\n=== Per-Class Performance Summary ===")
        print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
        print("-" * 60)
        for i in range(19):
            if str(i) in class_report_dict:
                metrics = class_report_dict[str(i)]
                print(f"{i:<8} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} {int(metrics['support']):<12}")
        
        # ---- 2) 予測結果を CSV に保存 ----
        output_dir_csv = f"/home/nishioka/GNN/GNN_hole/Predict_csv/Predict_csv_{timestamp}"
        os.makedirs(output_dir_csv, exist_ok=True)
        
        # ここで「テスト用 all_preds / all_labels / all_probs」を渡す
        # → sample_preds の形で各ファイルごとに切り分ける場合、test_pairs の数とノード数を意識
        save_predictions_to_csv(
            test_pairs,
            test_preds,
            test_labels,
            test_probs,
            output_dir_csv,
            num_nodes_per_sample=13942
        )
        
        print("[INFO] CSV save completed.")
        
        # ---- 3) 可視化や追加の評価 ----
        # evaluate_and_visualize 内でさらにテストデータ・トレインデータを推論する場合は、
        # そちらと重複しないように注意。（すでに test_preds / test_labels / test_probs があるなら活用）
        
        gamma = getattr(args, 'gamma', default_gamma)
        evaluate_and_visualize(
            final_test_loader,
            train_loader,
            ddp_model,
            device,
            class_weights,
            test_pairs,
            train_pairs,
            num_nodes_per_sample=13942,
            gamma=gamma
        )
        
        # ---- 4) 最後に使用したモデルを保存 ----
        save_dir = "/home/nishioka/GNN/GNN_hole_2026/GNN_model/19classmodel_hole"
        os.makedirs(save_dir, exist_ok=True)
        final_model_path = f"{save_dir}/{type(model).__name__}_{timestamp}_Final.pth"
        torch.save(ddp_model.module.state_dict(), final_model_path)
        print(f"[FINAL_INFO] Final model saved to {final_model_path}")
    
    cleanup()

# ----------------------------
# エントリーポイント
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training Script')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Number of hidden channels')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=300, help='Early stopping patience')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma parameter (default: 2.0)')
    parser.add_argument('--class_weight_multiplier', type=float, default=5.0, help='Class weight multiplier for minority classes (default: 5.0, increased for better handling of imbalanced classes)')
    parser.add_argument('--use_log_softmax', action='store_true', default=True, help='Use log_softmax version of FocalLoss for numerical stability (default: True)')
    parser.add_argument('--no_log_softmax', action='store_false', dest='use_log_softmax', help='Disable log_softmax version of FocalLoss')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision training (AMP) (default: True)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp', help='Disable mixed precision training')
    parser.add_argument('--use_onecycle', action='store_true', default=False, help='Use OneCycleLR scheduler instead of CosineAnnealingLR (default: False)')

    args = parser.parse_args()

    main(args)