import os
import sys
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

# Layer-aware loss wrapper: 層ごとのラベル制約を追加
class LayerAwareLoss(nn.Module):
    """
    層ごとのラベル制約を追加する損失関数ラッパー
    Layer 1 (前半): 0, 1~9 のみ
    Layer 2 (後半): 0, 10~18 のみ
    
    データリークではない理由: これはデータの構造的制約であり、物理的な意味に基づく
    """
    def __init__(self, base_loss_fn, vertices_per_layer=6971, constraint_weight=1.0):
        """
        Args:
            base_loss_fn: ベースとなる損失関数（LogitAdjustLoss, FocalLossLogSoftmax, CrossEntropyLossなど）
            vertices_per_layer: 1層あたりのノード数（デフォルト: 6971）
            constraint_weight: 制約違反のペナルティ重み（デフォルト: 1.0）
        """
        super(LayerAwareLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.vertices_per_layer = vertices_per_layer
        self.constraint_weight = constraint_weight
    
    def forward(self, logits, target, batch=None):
        """
        Args:
            logits: [N, num_classes] のロジット
            target: [N] のクラスインデックス
            batch: [N] のバッチインデックス（各グラフのノードがどのグラフに属するか）
                   バッチサイズが1の場合はNoneでも可
        """
        # ベース損失を計算
        base_loss = self.base_loss_fn(logits, target)
        
        # 層ごとの制約違反ペナルティを計算
        constraint_loss = self._compute_constraint_penalty(logits, batch)
        
        return base_loss + self.constraint_weight * constraint_loss
    
    def _compute_constraint_penalty(self, logits, batch):
        """
        層ごとのラベル制約違反に対するペナルティを計算
        """
        # バッチが指定されていない場合、単一グラフと仮定
        if batch is None:
            # 単一グラフの場合、最初のvertices_per_layerがLayer 1、残りがLayer 2
            N = logits.size(0)
            if N <= self.vertices_per_layer:
                # ノード数が少ない場合は制約を適用しない
                return torch.tensor(0.0, device=logits.device)
            
            layer1_mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
            layer1_mask[:self.vertices_per_layer] = True
            layer2_mask = ~layer1_mask
        else:
            # 複数グラフの場合、各グラフ内でLayer 1とLayer 2を分離
            # batch内の各グラフについて、最初のvertices_per_layerがLayer 1
            unique_batches = torch.unique(batch)
            layer1_mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
            layer2_mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
            
            for b in unique_batches:
                batch_mask = (batch == b)
                batch_indices = torch.where(batch_mask)[0]
                if len(batch_indices) > self.vertices_per_layer:
                    layer1_indices = batch_indices[:self.vertices_per_layer]
                    layer2_indices = batch_indices[self.vertices_per_layer:]
                    layer1_mask[layer1_indices] = True
                    layer2_mask[layer2_indices] = True
        
        # 予測確率を計算
        probs = F.softmax(logits, dim=1)
        
        # Layer 1の制約違反: 10以上のクラスの予測確率にペナルティ
        layer1_logits = logits[layer1_mask]
        layer1_probs = probs[layer1_mask]
        if layer1_logits.size(0) > 0:
            # クラス10~18の予測確率の合計にペナルティ
            invalid_classes_layer1 = torch.arange(10, 19, device=logits.device)
            layer1_penalty = layer1_probs[:, invalid_classes_layer1].sum(dim=1).mean()
        else:
            layer1_penalty = torch.tensor(0.0, device=logits.device)
        
        # Layer 2の制約違反: 1~9のクラスの予測確率にペナルティ
        layer2_logits = logits[layer2_mask]
        layer2_probs = probs[layer2_mask]
        if layer2_logits.size(0) > 0:
            # クラス1~9の予測確率の合計にペナルティ
            invalid_classes_layer2 = torch.arange(1, 10, device=logits.device)
            layer2_penalty = layer2_probs[:, invalid_classes_layer2].sum(dim=1).mean()
        else:
            layer2_penalty = torch.tensor(0.0, device=logits.device)
        
        return layer1_penalty + layer2_penalty


# LogitAdjust Loss (優先度1の施策: Focal卒業候補)
class LogitAdjustLoss(nn.Module):
    """
    LogitAdjust Loss: logitsにtau*log(pi)を加算してクラス不均衡を補正
    Focal Lossより「全部0」崩壊を防ぎつつminorityを押し上げる効果が高い
    
    Reference: "Long-tail learning via logit adjustment" (Menon et al., 2020)
    
    Note: class_priorは必須（バッチ推定はDDP+不均衡で揺れて危険なため禁止）
    """
    def __init__(self, class_prior: torch.Tensor, tau=1.0, reduction='mean'):
        """
        Args:
            class_prior: 各クラスの事前確率（必須、train全体から計算した固定値）
            tau: 調整強度（デフォルト: 1.0, 推奨範囲: 1.0~2.0）
            reduction: 'mean' or 'sum'
        """
        super(LogitAdjustLoss, self).__init__()
        self.tau = float(tau)
        self.reduction = reduction
        # 固定priorを登録（DDP安全）
        self.register_buffer('log_pi', torch.log(class_prior + 1e-8))
    
    def forward(self, logits, target):
        """
        Args:
            logits: [N, num_classes] のロジット
            target: [N] のクラスインデックス
        """
        # Logit adjustment: logits + tau * log(pi)
        adjusted_logits = logits + self.tau * self.log_pi.unsqueeze(0)
        
        # Cross-entropy loss
        return F.cross_entropy(adjusted_logits, target, reduction=self.reduction)


# Log-softmax version of Focal Loss for numerical stability
class FocalLossLogSoftmax(nn.Module):
    """Focal Loss using log_softmax for numerical stability"""
    def __init__(self, weights=None, gamma=3.0, reduction='mean'):
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
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
import matplotlib.cm as cm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# フォントの設定
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

# カラーマップの設定
dpsss_cmap = 'jet'         # DSPSSデータ用：カラフルな 'jet'
defect_cmap = 'coolwarm'   # Defect Label & Predict Data用：赤青の 'coolwarm'

# グリッド設定
num_cols = 57
num_rows = 125
vertices_per_layer = 6971
total_elements = num_cols * num_rows

# 穴領域の設定
hole_col_start = 26
hole_row_start1 = 51
hole_row_start2 = 77
hole_size_cols = 7
hole_size_rows1 = 7
hole_size_rows2 = 15

# 穴の位置計算
hole_elements = []
for r in range(hole_row_start1, hole_row_start1 + hole_size_rows1):
    for c in range(hole_col_start, hole_col_start + hole_size_cols):
        elem = r * num_cols + c
        hole_elements.append(elem)

for r in range(hole_row_start2, hole_row_start2 + hole_size_rows2):
    for c in range(hole_col_start, hole_col_start + hole_size_cols):
        elem = r * num_cols + c
        hole_elements.append(elem)

hole_elements = [node - 1 for node in hole_elements]
# set化して高速化（線形探索→O(1)検索）
hole_set = set(hole_elements)

# クラス0を黒で表示するカスタムカラーマップ
def get_defect_cmap_with_white_zero():
    """クラス0を黒で表示するカスタムカラーマップ"""
    # coolwarmカラーマップを取得
    coolwarm = cm.get_cmap('coolwarm', 19)
    colors = coolwarm(np.linspace(0, 1, 19))
    # クラス0を黒に設定
    colors[0] = [0.0, 0.0, 0.0, 1.0]  # 黒
    return ListedColormap(colors)

def process_file(file_path):
    """ファイルを読み込んで2層に分けて処理"""
    data = np.load(file_path)
    if len(data.shape) == 2:
        decoded_data = np.argmax(data, axis=1)
    elif len(data.shape) == 1:
        decoded_data = data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    layer1_data = decoded_data[:vertices_per_layer]
    layer2_data = decoded_data[vertices_per_layer:]

    layer1_full = np.empty(total_elements)
    layer1_full[:] = np.nan
    layer2_full = np.empty(total_elements)
    layer2_full[:] = np.nan

    data_index = 0
    for i in range(total_elements):
        if i not in hole_set and data_index < vertices_per_layer:
            layer1_full[i] = layer1_data[data_index]
            data_index += 1

    data_index = 0
    for i in range(total_elements):
        if i not in hole_set and data_index < vertices_per_layer:
            layer2_full[i] = layer2_data[data_index]
            data_index += 1

    layer1_reshaped = layer1_full.reshape((num_rows, num_cols))
    layer2_reshaped = layer2_full.reshape((num_rows, num_cols))

    return np.flipud(np.rot90(layer1_reshaped, k=1)), np.flipud(np.rot90(layer2_reshaped, k=1))

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
    """Edge dropout with device-aware mask generation (DDP安全)"""
    num_edges = edge_index.size(1)
    # device mismatch修正: edge_indexと同じdeviceでmaskを生成
    mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
    edge_index_dropped = edge_index[:, mask]  # ドロップアウト後のエッジを保持
    
    return edge_index_dropped


class GATModel(torch.nn.Module):
    def __init__(self, hidden_channels=8, num_classes=19, dropout=0.1, edge_drop_prob=0.01):
        """
        Args:
            hidden_channels: 隠れ層のチャネル数
            num_classes: クラス数
            dropout: Dropout確率（デフォルト: 0.1, 推奨範囲: 0.0~0.3）
            edge_drop_prob: Edge dropout確率（デフォルト: 0.01, 推奨範囲: 0.0~0.05）
        """
        super(GATModel, self).__init__()
        self.edge_drop_prob = edge_drop_prob
        
        self.conv1 = GATConv(4, hidden_channels, heads=4, concat=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels * 4)
        
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels * 2, heads=4, concat=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels * 8)
        
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=4, concat=True)
        self.batch_norm3 = nn.BatchNorm1d(hidden_channels * 4)
        
        # self.conv4 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        # self.batch_norm4 = nn.BatchNorm1d(hidden_channels * 4)
        
        self.fc = nn.Linear(hidden_channels * 4, num_classes)
        self.dropout = nn.Dropout(p=dropout)

        # プロジェクションレイヤー
        self.proj1 = nn.Linear(4, hidden_channels * 4)
        self.proj2 = nn.Linear(hidden_channels * 4, hidden_channels * 8)
        self.proj3 = nn.Linear(hidden_channels * 8, hidden_channels * 4)

        self.apply(initialize_weights)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # エッジドロップアウトの適用（トレーニング時のみ）
        if self.training:
            edge_index = edge_dropout(edge_index, drop_prob=self.edge_drop_prob)

        # Layer 1 with residual connection (成功版と同じ方法)
        x_res = self.proj1(x)
        x = F.relu(self.conv1(x, edge_index)) + x_res
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        # Layer 2 with residual connection
        x_res = self.proj2(x)
        x = F.relu(self.conv2(x, edge_index)) + x_res
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        # Layer 3 with residual connection
        x_res = self.proj3(x)
        x = F.relu(self.conv3(x, edge_index)) + x_res
        x = self.batch_norm3(x)
        x = self.dropout(x)
        
        # Fully connected layer
        x = self.fc(x)
        
        # Apply layer constraint mask to logits
        # Layer1: only classes {0..9} allowed, Layer2: only classes {0,10..18} allowed
        vertices_per_layer = 6971
        batch = getattr(data, 'batch', None)
        
        if batch is None:
            # Single graph: first vertices_per_layer nodes are Layer1, rest are Layer2
            N = x.size(0)
            if N > vertices_per_layer:
                # Layer1: mask classes 10-18
                layer1_mask = torch.zeros(N, dtype=torch.bool, device=x.device)
                layer1_mask[:vertices_per_layer] = True
                invalid_classes_layer1 = torch.arange(10, 19, device=x.device)
                x[layer1_mask][:, invalid_classes_layer1] = -1e9
                
                # Layer2: mask classes 1-9
                layer2_mask = ~layer1_mask
                invalid_classes_layer2 = torch.arange(1, 10, device=x.device)
                x[layer2_mask][:, invalid_classes_layer2] = -1e9
        else:
            # Multiple graphs: for each graph, first vertices_per_layer nodes are Layer1
            unique_batches = torch.unique(batch)
            for b in unique_batches:
                batch_mask = (batch == b)
                batch_indices = torch.where(batch_mask)[0]
                if len(batch_indices) > vertices_per_layer:
                    layer1_indices = batch_indices[:vertices_per_layer]
                    layer2_indices = batch_indices[vertices_per_layer:]
                    
                    # Layer1: mask classes 10-18
                    invalid_classes_layer1 = torch.arange(10, 19, device=x.device)
                    x[layer1_indices][:, invalid_classes_layer1] = -1e9
                    
                    # Layer2: mask classes 1-9
                    invalid_classes_layer2 = torch.arange(1, 10, device=x.device)
                    x[layer2_indices][:, invalid_classes_layer2] = -1e9
        
        # Return logits (not softmax) for numerical stability and proper loss computation
        # Softmax will be applied in loss function if needed, or only when computing probabilities
        return x

# Default gamma for focal loss (can be overridden by args)
# クラス不均衡が極端な場合、gammaを上げることで難易度の高いサンプルに焦点を当てる
default_gamma = 3.0  # Increased from 2.0 to better handle extreme class imbalance
default_class_weight_multiplier = 5.0  # Increased from 1.0 to better handle imbalanced classes

def compute_class_weights(labels, multiplier=None, fix_class0_weight=True, class0_weight=1.0):
    """
    クラス重みを計算
    
    Args:
        labels: ラベル配列
        multiplier: 未使用（後方互換性のため）
        fix_class0_weight: Trueの場合、Class 0の重みを固定値にする（推奨）
        class0_weight: Class 0の固定重み値（fix_class0_weight=Trueの場合）
    
    Returns:
        クラス重みテンソル
    """
    class_counts = np.bincount(labels, minlength=19)  # Ensure all 19 classes are counted
    
    if fix_class0_weight:
        # Class 0の重みを固定値にして、他のクラスだけ重み付け（崩壊防止に最も効果的）
        # これにより、Class 0が99.8%を占めていても、重みが極端に小さくなるのを防ぐ
        weights = np.ones(19) * class0_weight  # Class 0を含む全てを初期化
        
        # Class 1-18に対してのみ逆頻度重みを計算
        for i in range(1, 19):
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

def save_predictions_to_csv(test_pairs, all_preds, all_labels, all_probs, output_dir_csv, num_nodes_per_sample=13942):
    """
    予測結果をノード単位およびファイル単位でCSVに保存
    分散学習環境では、実際に処理されたデータ数に基づいて処理する
    """
    os.makedirs(output_dir_csv, exist_ok=True)  # CSV保存先ディレクトリ

    # 実際に処理されたファイル数を計算（分散学習では全ファイルが処理されない可能性がある）
    actual_num_samples = len(all_preds)
    expected_num_files = len(test_pairs)
    actual_num_files = actual_num_samples // num_nodes_per_sample
    
    # データ数の整合性チェック
    if actual_num_samples % num_nodes_per_sample != 0:
        print(f"[WARNING] Total predictions ({actual_num_samples}) is not divisible by num_nodes_per_sample ({num_nodes_per_sample})")
        print(f"         This may indicate incomplete batches. Processing {actual_num_files} complete files.")
    
    # 実際に処理されたファイル数に基づいて処理
    num_files_to_process = min(actual_num_files, expected_num_files)
    
    if num_files_to_process < expected_num_files:
        print(f"[INFO] Processing {num_files_to_process} files out of {expected_num_files} total test files")
        print(f"       (This is normal in distributed training where each rank processes a subset)")
    
    # --- ファイル単位の統計情報を保存 ---
    file_results = []
    for i in range(num_files_to_process):
        filename_tuple = test_pairs[i] if i < len(test_pairs) else f"unknown_file_{i}"
        filename = filename_tuple[0] if isinstance(filename_tuple, tuple) else filename_tuple
        start_idx = i * num_nodes_per_sample
        end_idx = start_idx + num_nodes_per_sample
    
        # 範囲チェック
        if end_idx > len(all_preds) or end_idx > len(all_labels):
            print(f"[WARNING] Skipping file {i}: Index out of range (start_idx={start_idx}, end_idx={end_idx}, total_samples={len(all_preds)})")
            continue
    
        # 各ファイルの予測情報を取得
        sample_preds = np.array(all_preds[start_idx:end_idx])  # 各ファイルの予測
        sample_labels = np.array(all_labels[start_idx:end_idx])  # 実際のラベル
    
        # クラス分布の計算（負の値や範囲外の値がないか確認）
        if np.any(sample_preds < 0) or np.any(sample_preds >= 19):
            print(f"[WARNING] Invalid prediction values found in file: {filename}, skipping...")
            continue
        class_counts = np.bincount(sample_preds.astype(int), minlength=19)  # 19クラス分の予測カウント
    
        # 精度の計算
        accuracy = (sample_preds == sample_labels).sum() / len(sample_labels)
    
        # 結果をリストに追加
        file_results.append({
            "Filename": filename,
            "Accuracy": accuracy,
            "Class Distribution": ",".join(map(str, class_counts))
        })
    
    # ファイル単位の結果をDataFrameに変換
    if len(file_results) > 0:
        file_results_df = pd.DataFrame(file_results)
        
        # CSV保存
        file_results_csv_path = os.path.join(output_dir_csv, f"file_statistics_{timestamp}.csv")
        try:
            file_results_df.to_csv(file_results_csv_path, index=False)
            print(f"File-level statistics saved to {file_results_csv_path} ({len(file_results)} files)")
        except Exception as e:
            print(f"Error saving file-level statistics: {e}")
    else:
        print(f"[WARNING] No file results to save. This may indicate a data mismatch issue.")

def evaluate_and_visualize(final_test_loader, train_loader, model, device, class_weights, test_pairs, train_pairs, num_nodes_per_sample=13942, gamma=2.0, rank=0, world_size=1, test_dataset=None, x_coords=None, y_coords=None, z_coords=None):
    """
    推論と可視化を実行
    注意: 分散学習の場合、rank 0のみで実行して順序を保証する
    model: 通常のモデル（ddp_model.module）を渡すこと（DDP collectivesを避けるため）
    
    Args:
        x_coords, y_coords, z_coords: 座標データ（可視化の軸ラベル用）
    """
    model.eval()# モデルを評価モードに切り替え
    
    # rank 0のみで実行（順序を保証するため）
    if rank != 0:
        print(f"[INFO] Rank {rank}: Skipping evaluation (only rank 0 evaluates)")
        return
    
    # rank 0でtest_loaderを再作成（順序を保証するため、samplerなしで全データを順番に処理）
    if test_dataset is not None:
        print(f"[INFO] Recreating test_loader on rank 0 without DistributedSampler to ensure correct file-prediction mapping")
        from torch_geometric.loader import DataLoader
        final_test_loader = DataLoader(test_dataset, batch_size=final_test_loader.batch_size, shuffle=False)
    
    test_preds = []
    test_labels = []
    test_probs = []
    total_loss = 0.0
    correct = 0
    total_samples = 0

    # Loss function for evaluation - model outputs logits, use same approach as training
    # Default to log_softmax version for numerical stability
    use_log_softmax = True  # Default to True for numerical stability
    if use_log_softmax:
        loss_fn = FocalLossLogSoftmax(weights=class_weights.to(device), gamma=gamma).to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
    
    print(f"[INFO] Starting evaluation on rank {rank} (ensuring correct file-prediction mapping)")
    
    with torch.no_grad():  # 評価時は勾配計算を無効化
        for batch_idx, batch in enumerate(final_test_loader):
            batch = batch.to(device)
            out = model(batch)  # Model outputs logits (model is already ddp_model.module)
            y = batch.y

            # Calculate loss (out is logits, loss function handles it)
            loss = loss_fn(out, y)
            total_loss += loss.item()

            # Model outputs logits, apply softmax to get probabilities
            probs = torch.softmax(out, dim=1).cpu().numpy()
            test_probs.extend(probs)

            # 実ラベル
            test_labels.extend(y.cpu().numpy())

            # 予測ラベル
            pred = out.argmax(dim=1).cpu().numpy()
            test_preds.extend(pred)

            correct += (pred == y.cpu().numpy()).sum()
            total_samples += y.size(0)
            
            # バッチ内の各サンプルのファイル名を追跡
            # batchは複数のグラフを結合したものなので、各グラフのファイル名を取得
            if hasattr(batch, 'filename'):
                # batch.filenameがリストの場合
                if isinstance(batch.filename, list):
                    batch_filenames = batch.filename
                else:
                    # 単一のファイル名の場合、バッチサイズ分繰り返す
                    batch_filenames = [batch.filename] * (y.size(0) // num_nodes_per_sample)
            else:
                # ファイル名が取得できない場合は、batch内の各Dataオブジェクトから取得を試みる
                # これは複雑なので、別の方法を検討
                batch_filenames = None
            
            # バッチを個別のグラフに分割してファイル名を追跡
            # 注意: DataLoaderがバッチを結合しているため、個別のファイル名を取得するのは困難
            # 代わりに、予測結果を保存する際に、実際に処理された順序で保存する

    test_accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_test_loss = total_loss / len(final_test_loader) if len(final_test_loader) > 0 else 0.0

    # --- 出力ディレクトリの作成 ---
    output_dir_test_predict = f"/home/nishioka/GNN/GNN_hole_2026/Predict_data/Predict19ClassZscore_Test{timestamp}"
    os.makedirs(output_dir_test_predict, exist_ok=True)
    
    # 予測結果のnpyファイルを保存する専用フォルダを作成
    predictions_npy_dir = os.path.join(output_dir_test_predict, "predictions_npy")
    os.makedirs(predictions_npy_dir, exist_ok=True)
    
    #ノード数スライスの整合性をチェック
    expected_test_total = len(test_pairs) * num_nodes_per_sample
    actual_num_samples = len(test_preds)
    actual_num_files = actual_num_samples // num_nodes_per_sample
    
    if actual_num_samples != expected_test_total:
        print("[INFO] The total number of test_preds does not match test_pairs * num_nodes_per_sample.")
        print(f"       len(test_preds)={actual_num_samples}, expected={expected_test_total}")
        print(f"       This is normal in distributed training. Processing {actual_num_files} files out of {len(test_pairs)} total.")
    
    # 分散学習の場合、各ランクが異なるサブセットを処理するため、
    # 実際に処理されたファイルの順序で予測結果を保存する必要がある
    # しかし、DataLoaderから個別のファイル名を取得するのは困難なため、
    # 別のアプローチを取る：test_pairsの順序ではなく、実際の処理順序で保存
    
    # 実際に処理されたファイル数に基づいて保存
    num_files_to_process = min(actual_num_files, len(test_pairs))
    
    if num_files_to_process > 0:
        # 重要: 分散学習では、test_pairsの順序とtest_predsの順序が一致しない
        # そのため、test_pairsの順序で保存するのではなく、
        # 実際に処理された順序（test_predsの順序）で保存する必要がある
        
        # しかし、どのファイルがどの予測に対応するかを知る必要がある
        # 暫定的な解決策：test_pairsの順序で保存するが、警告を出す
        
        print(f"[WARNING] In distributed training, the order of test_pairs and test_preds may not match.")
        print(f"         Saving predictions in the order they were processed, not in test_pairs order.")
        print(f"         This may cause file name mismatches. Please verify the predictions manually.")
        
        # ファイルごとにスライスして保存
        saved_count = 0
        invalid_count = 0
        
        # 実際に処理されたファイルの順序で保存
        # 注意: これは暫定的な解決策で、完全に正確ではない可能性がある
        for i in range(num_files_to_process):
            start_idx = i * num_nodes_per_sample
            end_idx = start_idx + num_nodes_per_sample

            # 範囲チェック
            if end_idx > len(test_preds):
                print(f"[WARNING] Skipping file index {i}: Index out of range")
                continue

            # 1ファイルあたりの予測を取り出す
            sample_preds = np.array(test_preds[start_idx:end_idx])
            
            # 予測結果の検証
            if len(sample_preds) != num_nodes_per_sample:
                print(f"[WARNING] File index {i}: Expected {num_nodes_per_sample} predictions, got {len(sample_preds)}")
                invalid_count += 1
                continue
            
            # 予測値の範囲チェック（0-18の範囲内か）
            if np.any(sample_preds < 0) or np.any(sample_preds >= 19):
                print(f"[WARNING] File index {i}: Invalid prediction values detected (min={sample_preds.min()}, max={sample_preds.max()})")
                invalid_count += 1
                # 範囲外の値を修正（0-18にクリップ）
                sample_preds = np.clip(sample_preds, 0, 18)
            
            # test_pairsから対応するファイル名を取得（順序が一致していることを前提）
            if i < len(test_pairs):
                filename_tuple = test_pairs[i]
                filename = filename_tuple[0] if isinstance(filename_tuple, tuple) else filename_tuple
            else:
                # test_pairsの範囲外の場合、インデックスベースのファイル名を使用
                filename = f"unknown_file_{i}.npy"
                print(f"[WARNING] File index {i} is out of test_pairs range. Using default filename: {filename}")
            
            # 予測結果の統計情報を出力（最初の数ファイルのみ）
            if saved_count < 3:
                unique_preds, counts = np.unique(sample_preds, return_counts=True)
                print(f"[DEBUG] File {i} ({filename}): Prediction distribution: {dict(zip(unique_preds.tolist(), counts.tolist()))}")
            
            base_filename = os.path.splitext(filename)[0]
            pred_filename = f"{base_filename}_pred.npy"
            np.save(os.path.join(predictions_npy_dir, pred_filename), sample_preds)
            saved_count += 1

        print(f"Test data predictions saved for {saved_count} test samples in: {predictions_npy_dir}")
        if invalid_count > 0:
            print(f"[WARNING] {invalid_count} files had invalid predictions and were skipped or corrected")
        
        # 予測結果の全体統計を出力
        print(f"\n=== Prediction Statistics ===")
        all_preds_for_stats = np.array(test_preds)
        unique_preds, counts = np.unique(all_preds_for_stats, return_counts=True)
        print(f"Total predictions: {len(all_preds_for_stats)}")
        print(f"Prediction class distribution:")
        for cls, count in zip(unique_preds, counts):
            percentage = count / len(all_preds_for_stats) * 100
            print(f"  Class {cls}: {count:8d} predictions ({percentage:6.2f}%)")
        print(f"Prediction value range: {all_preds_for_stats.min()} - {all_preds_for_stats.max()}")
    else:
        print("[WARNING] No files to save. This may indicate a data mismatch issue.")
    
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
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(19), yticklabels=range(19))
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
    output_dir = f"/home/nishioka/GNN/GNN_hole_2026/Predict_truth/pred_vs_true_images_zscore_{timestamp}"
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
    # [FIX S-3] Fix confusion matrix label range from 20 to 19 classes
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(19), yticklabels=range(19))
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
    
    # --- 予測結果の空間的可視化（DSPSS、Label、Predictionの比較） ---
    print("\n予測結果の空間的可視化を開始...")
    visualization_output_dir = os.path.join(output_dir_test_predict, "spatial_visualizations")
    os.makedirs(visualization_output_dir, exist_ok=True)
    
    # Z-score標準化データのディレクトリ
    zscore_data_base = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore"
    test_data_folder = os.path.join(zscore_data_base, "test")
    label_data_folder = "/home/nishioka/GNN/GNN_hole_2026/all_19class_label"
    
    # カスタムカラーマップ（クラス0を黒）
    defect_cmap_custom = get_defect_cmap_with_white_zero()
    
    # 座標データの読み込み（軸ラベル用）
    if x_coords is None or y_coords is None or z_coords is None:
        try:
            x_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
            y_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
            z_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")
        except Exception as e:
            print(f"警告: 座標データの読み込みに失敗しました: {e}")
            x_coords = y_coords = z_coords = None
    
    # 座標データを2層に分割してリシェイプ（可視化用）
    if x_coords is not None:
        x_layer1 = x_coords[:vertices_per_layer]
        x_layer2 = x_coords[vertices_per_layer:]
        y_layer1 = y_coords[:vertices_per_layer]
        y_layer2 = y_coords[vertices_per_layer:]
        z_layer1 = z_coords[:vertices_per_layer]
        z_layer2 = z_coords[vertices_per_layer:]
        
        # 座標をグリッドにマッピング
        x_layer1_grid = np.empty(total_elements)
        x_layer1_grid[:] = np.nan
        x_layer2_grid = np.empty(total_elements)
        x_layer2_grid[:] = np.nan
        y_layer1_grid = np.empty(total_elements)
        y_layer1_grid[:] = np.nan
        y_layer2_grid = np.empty(total_elements)
        y_layer2_grid[:] = np.nan
        z_layer1_grid = np.empty(total_elements)
        z_layer1_grid[:] = np.nan
        z_layer2_grid = np.empty(total_elements)
        z_layer2_grid[:] = np.nan
        
        data_index = 0
        for j in range(total_elements):
            if j not in hole_set and data_index < vertices_per_layer:
                x_layer1_grid[j] = x_layer1[data_index]
                y_layer1_grid[j] = y_layer1[data_index]
                z_layer1_grid[j] = z_layer1[data_index]
                data_index += 1
        
        data_index = 0
        for j in range(total_elements):
            if j not in hole_set and data_index < vertices_per_layer:
                x_layer2_grid[j] = x_layer2[data_index]
                y_layer2_grid[j] = y_layer2[data_index]
                z_layer2_grid[j] = z_layer2[data_index]
                data_index += 1
        
        # リシェイプと回転（データと同じ変換を適用）
        x_layer1_reshaped = np.flipud(np.rot90(x_layer1_grid.reshape((num_rows, num_cols)), k=1))
        x_layer2_reshaped = np.flipud(np.rot90(x_layer2_grid.reshape((num_rows, num_cols)), k=1))
        y_layer1_reshaped = np.flipud(np.rot90(y_layer1_grid.reshape((num_rows, num_cols)), k=1))
        y_layer2_reshaped = np.flipud(np.rot90(y_layer2_grid.reshape((num_rows, num_cols)), k=1))
        z_layer1_reshaped = np.flipud(np.rot90(z_layer1_grid.reshape((num_rows, num_cols)), k=1))
        z_layer2_reshaped = np.flipud(np.rot90(z_layer2_grid.reshape((num_rows, num_cols)), k=1))
        
        # 軸の範囲を計算（ラベル表示用）
        x_min_layer1, x_max_layer1 = np.nanmin(x_layer1_reshaped), np.nanmax(x_layer1_reshaped)
        x_min_layer2, x_max_layer2 = np.nanmin(x_layer2_reshaped), np.nanmax(x_layer2_reshaped)
        y_min_layer1, y_max_layer1 = np.nanmin(y_layer1_reshaped), np.nanmax(y_layer1_reshaped)
        y_min_layer2, y_max_layer2 = np.nanmin(y_layer2_reshaped), np.nanmax(y_layer2_reshaped)
        z_min_layer1, z_max_layer1 = np.nanmin(z_layer1_reshaped), np.nanmax(z_layer1_reshaped)
        z_min_layer2, z_max_layer2 = np.nanmin(z_layer2_reshaped), np.nanmax(z_layer2_reshaped)
    else:
        x_layer1_reshaped = x_layer2_reshaped = None
        y_layer1_reshaped = y_layer2_reshaped = None
        z_layer1_reshaped = z_layer2_reshaped = None
        x_min_layer1 = x_max_layer1 = x_min_layer2 = x_max_layer2 = None
        y_min_layer1 = y_max_layer1 = y_min_layer2 = y_max_layer2 = None
        z_min_layer1 = z_max_layer1 = z_min_layer2 = z_max_layer2 = None
    
    # テストデータの可視化（最大50ファイル）
    max_visualize = min(50, len(test_pairs))
    visualized_count = 0  # 実際に可視化されたファイル数をカウント
    for i, (data_file, label_file) in enumerate(test_pairs[:max_visualize]):
        try:
            # ファイルパス
            data_file_path = os.path.join(test_data_folder, data_file)
            label_file_path = os.path.join(label_data_folder, label_file)
            # 予測ファイルは predictions_npy フォルダに保存されている
            predictions_npy_dir = os.path.join(output_dir_test_predict, "predictions_npy")
            pred_file_path = os.path.join(predictions_npy_dir, f"{os.path.splitext(data_file)[0]}_pred.npy")
            
            # ファイル存在確認
            if not os.path.exists(data_file_path):
                continue
            if not os.path.exists(pred_file_path):
                continue
            
            # データを読み込み・処理
            # DSPSSデータ（Z-score標準化データ）
            dpsss_data = np.load(data_file_path)[:13942]
            dpsss_layer1_full = np.empty(total_elements)
            dpsss_layer1_full[:] = np.nan
            dpsss_layer2_full = np.empty(total_elements)
            dpsss_layer2_full[:] = np.nan
            
            data_index = 0
            for j in range(total_elements):
                if j not in hole_set and data_index < vertices_per_layer:
                    dpsss_layer1_full[j] = dpsss_data[data_index]
                    data_index += 1
            
            data_index = 0
            for j in range(total_elements):
                if j not in hole_set and data_index < vertices_per_layer:
                    dpsss_layer2_full[j] = dpsss_data[vertices_per_layer + data_index]
                    data_index += 1
            
            dpsss_layer1 = np.flipud(np.rot90(dpsss_layer1_full.reshape((num_rows, num_cols)), k=1))
            dpsss_layer2 = np.flipud(np.rot90(dpsss_layer2_full.reshape((num_rows, num_cols)), k=1))
            
            # 予測データ
            pred_layer1, pred_layer2 = process_file(pred_file_path)
            
            # ラベルデータ（存在する場合）
            label_layer1 = None
            label_layer2 = None
            if os.path.exists(label_file_path):
                label_layer1, label_layer2 = process_file(label_file_path)
            
            # 可視化
            num_cols_vis = 3  # DSPSS, Defect Label, Prediction
            num_rows_vis = 2  # Layer 1, Layer 2
            
            plt.figure(figsize=(18, 12))
            
            # Layer 1 (Bottom Layer - 下の層)
            # DSPSS Layer 1
            ax1 = plt.subplot(num_rows_vis, num_cols_vis, 1)
            im1 = plt.imshow(dpsss_layer1, cmap=dpsss_cmap, aspect="equal", interpolation='none')
            plt.title(f"Bottom Layer - DSPSS (Z-score)\n{os.path.basename(data_file)}", fontsize=12, pad=50)
            plt.colorbar(im1, label="Z-score Normalized DSPSS", fraction=0.025, pad=0.04)
            
            # Defect Label Layer 1
            # 注意: データ構造では Layer 1（前半）には 0, 1~9 のみが存在
            ax2 = plt.subplot(num_rows_vis, num_cols_vis, 2)
            if label_layer1 is not None:
                label_layer1_int = np.where(np.isnan(label_layer1), 0, label_layer1).astype(int)
                label_layer1_int_masked = np.ma.masked_where(np.isnan(label_layer1), label_layer1_int)
                im = plt.imshow(label_layer1_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                              interpolation='none', vmin=0, vmax=18)
                plt.title(f"Bottom Layer - Defect Label\n(ラベル: 0, 1~9)\n{os.path.basename(label_file)}", fontsize=12, pad=50)
                cbar = plt.colorbar(im, label="Defect Label", fraction=0.025, pad=0.04)
                cbar.set_ticks(range(0, 19))
            else:
                plt.text(0.5, 0.5, "No Label", ha='center', va='center', fontsize=14)
                plt.axis('off')
            
            # Prediction Layer 1
            # 注意: データ構造では Layer 1（前半）には 0, 1~9 のみが存在すべき
            ax3 = plt.subplot(num_rows_vis, num_cols_vis, 3)
            pred_layer1_int = np.where(np.isnan(pred_layer1), 0, pred_layer1).astype(int)
            pred_layer1_int_masked = np.ma.masked_where(np.isnan(pred_layer1), pred_layer1_int)
            pred_max = int(np.max(pred_layer1_int[~np.isnan(pred_layer1)])) if np.any(~np.isnan(pred_layer1)) else 18
            im = plt.imshow(pred_layer1_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                          interpolation='none', vmin=0, vmax=max(pred_max, 18))
            plt.title(f"Bottom Layer - Prediction\n(期待されるラベル: 0, 1~9)\n{os.path.basename(pred_file_path)}", fontsize=12, pad=50)
            cbar = plt.colorbar(im, label="Prediction", fraction=0.025, pad=0.04)
            cbar.set_ticks(range(0, max(pred_max, 18) + 1))
            
            # Layer 2 (Upper Layer - 上の層)
            # DSPSS Layer 2
            ax4 = plt.subplot(num_rows_vis, num_cols_vis, 4)
            im4 = plt.imshow(dpsss_layer2, cmap=dpsss_cmap, aspect="equal", interpolation='none')
            plt.title(f"Upper Layer - DSPSS (Z-score)\n{os.path.basename(data_file)}", fontsize=12, pad=50)
            plt.colorbar(im4, label="Z-score Normalized DSPSS", fraction=0.025, pad=0.04)
            
            # Defect Label Layer 2
            # 注意: データ構造では Layer 2（後半）には 0, 10~18 のみが存在
            ax5 = plt.subplot(num_rows_vis, num_cols_vis, 5)
            if label_layer2 is not None:
                label_layer2_int = np.where(np.isnan(label_layer2), 0, label_layer2).astype(int)
                label_layer2_int_masked = np.ma.masked_where(np.isnan(label_layer2), label_layer2_int)
                im = plt.imshow(label_layer2_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                              interpolation='none', vmin=0, vmax=18)
                plt.title(f"Upper Layer - Defect Label\n(ラベル: 0, 10~18)\n{os.path.basename(label_file)}", fontsize=12, pad=50)
                cbar = plt.colorbar(im, label="Defect Label", fraction=0.025, pad=0.04)
                cbar.set_ticks(range(0, 19))
            else:
                plt.text(0.5, 0.5, "No Label", ha='center', va='center', fontsize=14)
                plt.axis('off')
            
            # Prediction Layer 2
            # 注意: データ構造では Layer 2（後半）には 0, 10~18 のみが存在すべき
            ax6 = plt.subplot(num_rows_vis, num_cols_vis, 6)
            pred_layer2_int = np.where(np.isnan(pred_layer2), 0, pred_layer2).astype(int)
            pred_layer2_int_masked = np.ma.masked_where(np.isnan(pred_layer2), pred_layer2_int)
            pred_max = int(np.max(pred_layer2_int[~np.isnan(pred_layer2)])) if np.any(~np.isnan(pred_layer2)) else 18
            im = plt.imshow(pred_layer2_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                          interpolation='none', vmin=0, vmax=max(pred_max, 18))
            plt.title(f"Upper Layer - Prediction\n(期待されるラベル: 0, 10~18)\n{os.path.basename(pred_file_path)}", fontsize=12, pad=50)
            cbar = plt.colorbar(im, label="Prediction", fraction=0.025, pad=0.04)
            cbar.set_ticks(range(0, max(pred_max, 18) + 1))
            
            plt.tight_layout()
            
            # PNGファイルとして保存
            base_filename = os.path.splitext(data_file)[0]
            output_path = os.path.join(visualization_output_dir, f"{base_filename}_spatial_visualization.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            visualized_count += 1  # 実際に可視化されたファイル数をカウント
            if visualized_count % 10 == 0:
                print(f"  可視化処理中: {visualized_count} ファイル")
                
        except Exception as e:
            print(f"  警告: {data_file} の可視化中にエラーが発生しました: {e}")
            continue
    
    print(f"空間的可視化完了! {visualized_count} ファイルのPNG画像を {visualization_output_dir} に保存しました")

# ----------------------------
# データペアリング関数
# ----------------------------
def extract_layer_block(file_name):
    """ファイル名から層とブロック番号を抽出（すべてのH_Wサイズに対応）"""
    try:
        # H2_W2, H4_W4, H8_W8, H5_W6, H7_W4, H4_W8, H8_W4など、すべてのサイズに対応
        match = re.search(r'L(\d+)_B(\d+)_el(\d+)_H(\d+)_W(\d+)', file_name)
        if match:
            layer = int(match.group(1))
            block = int(match.group(2))
            return (layer, block)
        else:
            print(f"Invalid file name format: {file_name}")
            return None
    except AttributeError:
        print(f"Invalid file name format: {file_name}")
        return None

def create_data_label_pairs(data_files, label_files):
    """データファイルとラベルファイルのペアを作成"""
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
                if no_label_counter < 3:
                    print(f"No matching data file for label: {label_file}")
                    no_label_counter += 1

    # ペアが作成されなかった場合を出力（3つに制限）
    no_data_counter = 0
    for base_name, v in data_label_pairs.items():
        if "label" not in v:
            if no_data_counter < 3:  # 最大3つまで出力
                print(f"No label found for data file: {v['data']}")
                no_data_counter += 1

    # マッチしなかったラベルを出力（最初の3つだけ）
    unmatched_counter = 0
    for unmatched_label in unmatched_labels:
        if unmatched_counter < 3:
            print(f"Unmatched label file: {unmatched_label}")
            unmatched_counter += 1

    valid_pairs = [(v["data"], v["label"]) for k, v in data_label_pairs.items() if "label" in v]
    
    # マッチしたペアの確認ログを追加
    if len(valid_pairs) > 0:
        print(f"✓ Successfully matched {len(valid_pairs)} pairs")
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
        
        # Calculate dominant class for each graph (with progress indication)
        if rank == 0:
            print(f"Initializing ClassBalancedSampler: processing {len(dataset)} samples...")
        self.class_indices = {i: [] for i in range(num_classes)}
        for idx, data in enumerate(dataset):
            if idx % 1000 == 0 and rank == 0:
                print(f"  Processing sample {idx}/{len(dataset)}...")
            classes, counts = torch.unique(data.y, return_counts=True)
            dominant_class = classes[counts.argmax()].item()
            self.class_indices[dominant_class].append(idx)
        
        if rank == 0:
            print(f"ClassBalancedSampler initialization complete. Class distribution:")
            for cls_idx in range(num_classes):
                if len(self.class_indices[cls_idx]) > 0:
                    print(f"  Class {cls_idx}: {len(self.class_indices[cls_idx])} samples")
        
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


class DistributedMinorityRatioSampler(DistributedSampler):
    """
    ファイル単位のminority_ratio基準で重み付けサンプリングを行うDistributedSampler（DDP安全版）
    minority_ratioが高いファイル（minority classが多いファイル）を多くサンプルする
    
    DDP安全: torch.multinomialでtotal_sizeを固定し、全rankのバッチ数が揃う
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, 
                 weight_power=1.0, min_weight=0.1, seed=42):
        """
        Args:
            dataset: PyG Dataオブジェクトのリスト（各Dataにminority_ratio属性が必要）
            num_replicas: DDPの総プロセス数
            rank: 現在のプロセスrank
            shuffle: シャッフルするか
            weight_power: minority_ratioの重み付けの強さ（1.0=線形、2.0=2乗）
            min_weight: 最小重み（0に近いminority_ratioでもサンプルされるように）
            seed: 乱数シード（DDP安全のため）
        """
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.weight_power = float(weight_power)
        self.min_weight = float(min_weight)
        self._build_weights()
    
    def _build_weights(self):
        """重みを構築（全rankで同じ重みを使用）"""
        if self.rank == 0:
            print(f"Initializing DistributedMinorityRatioSampler: processing {len(self.dataset)} samples...")
        
        ratios = []
        for idx, data in enumerate(self.dataset):
            if idx % 1000 == 0 and self.rank == 0:
                print(f"  Processing sample {idx}/{len(self.dataset)}...")
            # minority_ratioが設定されていない場合は計算
            if not hasattr(data, 'minority_ratio'):
                unique_labels, counts = torch.unique(data.y, return_counts=True)
                total_nodes = len(data.y)
                count0 = counts[unique_labels == 0].item() if 0 in unique_labels else 0
                r = 1.0 - (count0 / total_nodes) if total_nodes > 0 else 0.0
            else:
                r = float(data.minority_ratio)
            ratios.append(r)
        
        # Store ratios for dynamic weight_power updates
        self.ratios = np.array(ratios)
        self._update_weights()
        
        if self.rank == 0:
            print(f"MinorityRatioSampler initialization complete.")
            print(f"  Minority ratio range: {min(ratios):.4f} - {max(ratios):.4f}")
            print(f"  Weight range: {self.weights.min().item():.4f} - {self.weights.max().item():.4f}")
    
    def _update_weights(self):
        """重みを更新（weight_powerが変更された場合に呼び出す）"""
        # 重みを計算: minority_ratio^weight_power + min_weight
        w = (self.ratios ** self.weight_power) + self.min_weight
        # 正規化（確率分布にする）
        w = w / (w.sum() + 1e-12)
        self.weights = torch.tensor(w, dtype=torch.double)
    
    def set_weight_power(self, new_power):
        """weight_powerを更新して重みを再計算"""
        self.weight_power = float(new_power)
        self._update_weights()
    
    def __iter__(self):
        """DDP安全な重み付きサンプリング"""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # total_size個を「重み付きで復元抽出」（全rankで同じシード→同じ順序）
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g).tolist()
        
        # rankごとに等分（ここがDDP安全：全rankのバッチ数が揃う）
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


# ----------------------------
# データ準備関数
# ----------------------------
def prepare_data(pairs, normalized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=None):
    data_list = []
    labels = []
    pair_counter = 0  # 確認用カウンタ

    total_pairs = len(pairs)
    for pair_idx, (data_file, label_file) in enumerate(pairs):
        if pair_idx % 500 == 0 and pair_idx > 0:
            print(f"  Loading data: {pair_idx}/{total_pairs} pairs processed...")
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
        labels.extend(y.tolist())

        # デバッグ用にいくつかのペアを出力（ペアの整合性も確認）
        if pair_counter < 1:
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

    # クラス重みを計算
    if len(labels) > 0:
        labels_array = np.array(labels)
        class_weights = compute_class_weights(labels_array, multiplier=class_weight_multiplier)
        
        # クラス分布と重みを出力（デバッグ用）
        unique_labels, counts = np.unique(labels_array, return_counts=True)
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
def cleanup(device=None, rank=0):
    """Safely cleanup distributed process group with error handling and timeout protection"""
    try:
        # Check if process group is initialized
        if not dist.is_initialized():
            return
        
        # Get device if not provided (use default)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # First, ensure all CUDA operations are complete
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                if rank == 0:
                    print(f"[WARNING] CUDA sync failed during cleanup: {e}")
        
        # Try to synchronize all ranks before destroying
        # Use a lightweight barrier without timeout (timeout not supported in this PyTorch version)
        # Skip barrier if communicator is already aborted (common after early stopping)
        try:
            # Check if communicator is still valid before attempting barrier
            # If it's aborted, skip barrier and go straight to destroy
            if dist.is_initialized():
                try:
                    # Barrier is more reliable than all_reduce for final sync
                    # But skip if communicator might be aborted
                    dist.barrier()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except (RuntimeError, Exception) as barrier_error:
                    # If barrier fails (e.g., communicator aborted), that's OK
                    # Model is already saved, just proceed to destroy
                    if rank == 0 and "aborted" not in str(barrier_error).lower():
                        print(f"[WARNING] Final barrier before cleanup failed (continuing anyway): {barrier_error}")
        except Exception as sync_error:
            # If sync fails, log but continue - model is already saved
            if rank == 0:
                print(f"[WARNING] Final barrier before cleanup failed (continuing anyway): {sync_error}")
        
        # Now destroy the process group
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as destroy_error:
            # If destroy fails, that's OK - process will exit anyway
            if rank == 0:
                print(f"[WARNING] Process group destroy failed (this is OK): {destroy_error}")
        
    except Exception as e:
        # Log error but don't raise - allow process to exit gracefully
        if rank == 0:
            print(f"[WARNING] Error during cleanup: {e}")
        # Force destroy if normal destroy fails
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass  # Ignore errors during forced cleanup

# ----------------------------
# メイン関数
# ----------------------------
def main(args):
    set_seed(42)

    # DDP Debug Environment Variables (optional, for troubleshooting):
    # Set these before running to get detailed NCCL/PyTorch distributed debug info:
    #   export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch distributed debug
    #   export NCCL_DEBUG=INFO                 # NCCL debug info
    #   export NCCL_DEBUG_SUBSYS=ALL           # All NCCL subsystems
    # These are helpful for diagnosing collective timeout issues but not required for normal operation.

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
        # Record execution command for logging
        command_str = ' '.join(sys.argv)
        print(f"\n{'='*60}")
        print(f"Execution Command:")
        print(f"  {command_str}")
        print(f"{'='*60}\n")
        
        print(f"Rank: {rank}, World Size: {world_size}, Master Addr: {master_addr}, Master Port: {master_port}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # デバイス設定（利用可能なGPU数を確認）
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if rank >= num_gpus:
            print(f"Warning: Rank {rank} exceeds available GPU count ({num_gpus}). Using GPU {rank % num_gpus} instead.")
            local_rank = rank % num_gpus
        else:
            local_rank = rank
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        local_rank = None
    
    if rank == 0:
        print(f"Rank {rank} using device: {device}")

    # Data preparation and model definition
    # Z-score標準化データのディレクトリ
    zscore_data_base = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore"
    train_data_folder = os.path.join(zscore_data_base, "train")
    val_data_folder = os.path.join(zscore_data_base, "val")
    test_data_folder = os.path.join(zscore_data_base, "test")
    
    # クラスラベルのディレクトリ
    label_data_folder = "/home/nishioka/GNN/GNN_hole_2026/all_19class_label"

    x_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
    y_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
    z_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")

    edges = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy")
    # edge_indexはCPUで保持（DataLoader/num_workers/fork周りで安全）
    # 学習時は batch.to(device) でGPUへ移動される
    edge_index = torch.tensor(edges.T, dtype=torch.long)  # CPUのまま

    # train/val/testからデータファイルを読み込む
    train_data_files = [f for f in os.listdir(train_data_folder) if f.startswith("Defect_L") and f.endswith(".npy")]
    val_data_files = [f for f in os.listdir(val_data_folder) if f.startswith("Defect_L") and f.endswith(".npy")]
    test_data_files = [f for f in os.listdir(test_data_folder) if f.startswith("Defect_L") and f.endswith(".npy")]
    
    # ラベルファイルを読み込む
    label_files = [
        f for f in os.listdir(label_data_folder)
        if f.startswith("Defect_L") and f.endswith("_19label.npy")
    ]

    if rank == 0:
        print(f"Train data files: {len(train_data_files)}")
        print(f"Val data files: {len(val_data_files)}")
        print(f"Test data files: {len(test_data_files)}")
        print(f"Label files: {len(label_files)}")

    # データとラベルのペアを作成
    if rank == 0:
        print("\n=== Creating data-label pairs ===")
    train_pairs = create_data_label_pairs(train_data_files, label_files)
    val_pairs = create_data_label_pairs(val_data_files, label_files)
    test_pairs = create_data_label_pairs(test_data_files, label_files)

    if rank == 0:
        print(f"\n=== Pair Summary ===")
        print(f"Train pairs: {len(train_pairs)}")
        print(f"Val pairs: {len(val_pairs)}")
        print(f"Test pairs: {len(test_pairs)}")
        print(f"Total pairs: {len(train_pairs) + len(val_pairs) + len(test_pairs)}")
        
        # ペアの整合性を確認（最初のペアの例を表示）
        if len(train_pairs) > 0:
            print(f"\n=== Verifying pair integrity ===")
            sample_pair = train_pairs[0]
            data_file, label_file = sample_pair
            data_base = os.path.splitext(data_file)[0]
            label_base = label_file.replace("_19label.npy", "")
            if data_base == label_base:
                print(f"✓ Pair integrity check passed: '{data_base}' matches '{label_base}'")
            else:
                print(f"✗ Pair integrity check FAILED: '{data_base}' != '{label_base}'")
    
    # データ使用率の適用（10%刻み）
    data_usage_ratio = getattr(args, 'data_usage_ratio', 1.0)
    if data_usage_ratio < 1.0:
        if rank == 0:
            print(f"\nUsing {data_usage_ratio*100:.0f}% of data (data_usage_ratio={data_usage_ratio})")
        
        # ランダムシードを固定して再現性を確保
        np.random.seed(42)
        random.seed(42)
        
        # 各データセットから指定された割合をサンプリング
        num_train = max(1, int(len(train_pairs) * data_usage_ratio))
        num_val = max(1, int(len(val_pairs) * data_usage_ratio))
        num_test = max(1, int(len(test_pairs) * data_usage_ratio))
        
        train_indices = np.random.choice(len(train_pairs), num_train, replace=False)
        val_indices = np.random.choice(len(val_pairs), num_val, replace=False)
        test_indices = np.random.choice(len(test_pairs), num_test, replace=False)
        
        train_pairs = [train_pairs[i] for i in sorted(train_indices)]
        val_pairs = [val_pairs[i] for i in sorted(val_indices)]
        test_pairs = [test_pairs[i] for i in sorted(test_indices)]
        
        if rank == 0:
            print(f"After sampling: Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}, Test pairs: {len(test_pairs)}")

    # Dropout/EdgeDropのパラメータを取得（デフォルト値を推奨値に変更）
    dropout = getattr(args, 'dropout', 0.1)  # デフォルト: 0.1（推奨範囲: 0.0~0.3）
    edge_drop_prob = getattr(args, 'edge_drop_prob', 0.01)  # デフォルト: 0.01（推奨範囲: 0.0~0.05）
    model = GATModel(
        hidden_channels=args.hidden_channels, 
        num_classes=19,
        dropout=dropout,
        edge_drop_prob=edge_drop_prob
    ).to(device)
    
    # Create the DDP model after initializing the process group
    # device_idsはlocal_rankを使用（利用可能なGPU数に合わせる）
    if torch.cuda.is_available() and local_rank is not None:
        ddp_device_ids = [local_rank]
    else:
        ddp_device_ids = None
    ddp_model = DDP(model, device_ids=ddp_device_ids)

    # データセットの準備
    class_weight_multiplier = getattr(args, 'class_weight_multiplier', default_class_weight_multiplier)
    if rank == 0:
        print(f"\nPreparing train dataset ({len(train_pairs)} pairs)...")
    train_dataset, class_weights = prepare_data(train_pairs, train_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier)
    if rank == 0:
        print(f"Preparing val dataset ({len(val_pairs)} pairs)...")
    val_dataset, _ = prepare_data(val_pairs, val_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier)
    if rank == 0:
        print(f"Preparing test dataset ({len(test_pairs)} pairs)...")
    test_dataset, _ = prepare_data(test_pairs, test_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier)

    if train_dataset is None or class_weights is None:
        if rank == 0:
            raise ValueError("Failed to prepare train_dataset or class_weights")
        return

    if val_dataset is None:
        if rank == 0:
            raise ValueError("Failed to prepare val_dataset")
        return

    if test_dataset is None:
        if rank == 0:
            raise ValueError("Failed to prepare test_dataset")
        return

    if rank == 0:
        print(f"Train Dataset Size: {len(train_dataset)}")
        print(f"Val Dataset Size: {len(val_dataset)}")
        print(f"Test Dataset Size: {len(test_dataset)}")
    
    if rank == 0:
        print("\nInitializing samplers...")
    
    # Sampler選択: minority_ratio基準のsamplerを使うかどうか
    use_minority_sampler = getattr(args, 'use_minority_sampler', False)
    minority_weight_power = getattr(args, 'minority_weight_power', 1.0)
    
    if use_minority_sampler:
        if rank == 0:
            print("Creating train sampler with minority_ratio weighting...")
        train_sampler = DistributedMinorityRatioSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            weight_power=minority_weight_power
        )
    else:
        if rank == 0:
            print("Creating train sampler (standard DistributedSampler)...")
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    if rank == 0:
        print("Creating val sampler...")
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    if rank == 0:
        print("Creating test sampler...")
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    if rank == 0:
        print("Creating DataLoaders...")
    # Use drop_last=True for training to ensure all ranks have same number of batches (prevents NCCL timeout)
    # For validation, use drop_last=False to avoid empty batches when validation data is small
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, drop_last=False)
    
    # DDP Safety: Assert all ranks have non-empty loaders
    assert len(train_loader) > 0, f"Rank {rank}: train_loader is empty!"
    assert len(val_loader) > 0, f"Rank {rank}: val_loader is empty!"
    
    # DDP Safety: Log per-rank step counts at epoch 1 to verify identical batch counts
    if rank == 0:
        sampler_type = "DistributedMinorityRatioSampler" if use_minority_sampler else "DistributedSampler"
        print(f"Using {sampler_type} for training (ensures identical batch counts across ranks)")
        print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}, Test loader batches: {len(test_loader)}")
        print(f"[DDP AUDIT] Rank 0: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    
    # All ranks print their batch counts (will be identical with drop_last=True and DistributedSampler)
    print(f"[DDP AUDIT] Rank {rank}: train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler: CosineAnnealingLR or OneCycleLR
    # OneCycleLRをデフォルト推奨（GATはLRに敏感で、収束とminority学習が改善しやすい）
    use_onecycle = getattr(args, 'use_onecycle', False)  # argparseの値を使用
    onecycle_max_lr = getattr(args, 'onecycle_max_lr', None)  # max_lrを独立パラメータ化
    if onecycle_max_lr is None:
        # デフォルトはlearning_rateの1-3倍（GATは高いLRが効きやすい）
        onecycle_max_lr = args.learning_rate * 2.0 if args.learning_rate < 1e-3 else args.learning_rate
    
    if use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=onecycle_max_lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        if rank == 0:
            print(f"Learning rate scheduler initialized: OneCycleLR (max_lr={onecycle_max_lr}, base_lr={args.learning_rate})")
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

    # Loss関数の選択（優先度1の施策: LogitAdjust推奨）
    use_logit_adjust = getattr(args, 'use_logit_adjust', False)
    logit_adjust_tau = getattr(args, 'logit_adjust_tau', 1.0)
    use_layer_constraint = getattr(args, 'use_layer_constraint', True)  # デフォルトで有効
    layer_constraint_weight = getattr(args, 'layer_constraint_weight', 1.0)  # 制約違反のペナルティ重み
    
    # Initialize variables for per-node loss computation
    gamma = None
    class_prior = None
    use_log_softmax = getattr(args, 'use_log_softmax', False)
    
    # ベース損失関数を作成
    if use_logit_adjust:
        # LogitAdjust Loss（Focal卒業候補、全部0崩壊を防ぎつつminorityを押し上げ）
        # クラス事前確率を計算
        labels_array = np.concatenate([data.y.cpu().numpy() for data in train_dataset])
        class_counts = np.bincount(labels_array, minlength=19)
        class_prior = torch.tensor(class_counts / (class_counts.sum() + 1e-8), dtype=torch.float)
        base_loss_fn = LogitAdjustLoss(class_prior=class_prior.to(device), tau=logit_adjust_tau).to(device)
        if rank == 0:
            print(f"Using LogitAdjustLoss (tau={logit_adjust_tau}) - recommended for imbalanced datasets")
    else:
        gamma = getattr(args, 'gamma', default_gamma)
        # Use log_softmax version if requested, otherwise use standard CrossEntropyLoss for logits
        if use_log_softmax:
            base_loss_fn = FocalLossLogSoftmax(weights=class_weights.to(device), gamma=gamma).to(device)
            if rank == 0:
                print(f"Using FocalLossLogSoftmax (gamma={gamma}) - logits input with log_softmax for numerical stability")
        else:
            # For logits, use standard CrossEntropyLoss (equivalent to log_softmax + nll_loss)
            base_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
            if rank == 0:
                print(f"Using CrossEntropyLoss (logits input) - standard loss for logits")
    
    # 層ごとのラベル制約を追加（デフォルトで有効）
    # 注意: hard0 miningではper-node lossを使用するため、層制約は最終的な損失計算時に適用
    if use_layer_constraint:
        loss_fn = LayerAwareLoss(
            base_loss_fn=base_loss_fn,
            vertices_per_layer=vertices_per_layer,
            constraint_weight=layer_constraint_weight
        ).to(device)
        if rank == 0:
            print(f"Layer-aware constraint enabled (weight={layer_constraint_weight})")
            print(f"  Layer 1: labels 0, 1~9 only")
            print(f"  Layer 2: labels 0, 10~18 only")
    else:
        loss_fn = base_loss_fn
        if rank == 0:
            print(f"Layer-aware constraint disabled")
    if rank == 0:
        print(f"Class weights for training: {class_weights.cpu().numpy()}")
        print(f"Class weights range: min={class_weights.min().item():.6f}, max={class_weights.max().item():.6f}")

    # Helper function to compute per-node loss for hard0 mining
    def compute_per_node_loss(logits, target, loss_type, class_weights=None, gamma=None, class_prior=None, tau=None, device=None):
        """
        Compute per-node loss with reduction='none' for hard0 mining.
        
        Args:
            logits: [N, num_classes] logits
            target: [N] class indices
            loss_type: 'ce', 'focal_logsoftmax', or 'logit_adjust'
            class_weights: Class weights tensor (for CE and Focal)
            gamma: Gamma parameter for Focal loss
            class_prior: Class prior for LogitAdjust
            tau: Tau parameter for LogitAdjust
            device: Device for tensors
        """
        if loss_type == 'ce':
            # Standard CrossEntropyLoss with class weights
            if class_weights is not None:
                return F.cross_entropy(logits, target, weight=class_weights, reduction='none')
            else:
                return F.cross_entropy(logits, target, reduction='none')
        elif loss_type == 'focal_logsoftmax':
            # FocalLossLogSoftmax per-node computation
            log_probs = F.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)
            target_one_hot = F.one_hot(target, num_classes=logits.size(1))
            log_probs_target = (log_probs * target_one_hot).sum(dim=1)
            probs_target = (probs * target_one_hot).sum(dim=1)
            focal_weight = (1 - probs_target) ** gamma
            if class_weights is not None:
                class_weights_selected = class_weights[target]
                loss = -class_weights_selected * focal_weight * log_probs_target
            else:
                loss = -focal_weight * log_probs_target
            return loss
        elif loss_type == 'logit_adjust':
            # LogitAdjust per-node computation
            log_pi = torch.log(class_prior + 1e-8)
            adjusted_logits = logits + tau * log_pi.unsqueeze(0)
            return F.cross_entropy(adjusted_logits, target, reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    # Time tracking for estimated completion time
    train_start_time = time.time()
    epoch_times = []  # Store time per epoch for estimation
    
    best_val_loss = float('inf')
    best_macro_f1 = 0.0  # Track best macro F1 for model selection (only for early stopping)
    counter = 0
    macro_f1_counter = 0  # Counter for macro F1 based early stopping
    best_model_state = None  # Store best model state dict
    best_epoch = 0  # Track epoch with best model
    early_stopped = False  # Flag to track if early stopping occurred
    # Shared early stopping flag using tensor (more reliable than file-based)
    early_stop_tensor = torch.tensor(0, dtype=torch.int32, device=device)  # 0 = continue, 1 = stop
    # AMP (Mixed Precision Training)
    use_amp = getattr(args, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if rank == 0 and use_amp:
        print(f"Mixed Precision Training (AMP) enabled")
    
    # Checkpoint saving directory
    checkpoint_dir = "/home/nishioka/GNN/GNN_hole_2026/GNN_model/19classmodel_hole_zscore"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Don't check for early stopping at the start - check at the end after synchronization
        # This prevents race conditions with pending NCCL operations
        
        # DDP Safety: Verify DataLoader lengths on epoch 1 (all ranks should have identical counts)
        if epoch == 1:
            train_len = len(train_loader)
            val_len = len(val_loader)
            print(f"[DDP AUDIT Epoch 1] Rank {rank}: train_batches={train_len}, val_batches={val_len}")
            # All ranks should have the same lengths due to drop_last=True and DistributedSampler
            # This will be verified by comparing logs across ranks
        
        # 2-stage schedule: determine active parameters for this epoch
        if epoch < args.stage2_epoch:
            active_hard0_keep_ratio = args.stage1_hard0_keep_ratio
            active_minority_weight_power = args.stage1_minority_weight_power
        else:
            active_hard0_keep_ratio = args.stage2_hard0_keep_ratio
            active_minority_weight_power = args.stage2_minority_weight_power
        
        # Update sampler weight_power if using minority sampler
        if use_minority_sampler and hasattr(train_sampler, 'set_weight_power'):
            train_sampler.set_weight_power(active_minority_weight_power)
        
        epoch_start_time = time.time()
        ddp_model.train()
        # sampler.set_epochのガード（custom samplerの場合に存在しない可能性がある）
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)
        total_loss = 0

        # Determine loss type for per-node computation
        if use_logit_adjust:
            loss_type_for_per_node = 'logit_adjust'
        elif use_log_softmax:
            loss_type_for_per_node = 'focal_logsoftmax'
        else:
            loss_type_for_per_node = 'ce'

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = ddp_model(batch)
                    y = batch.y
                    
                    # Hard0 mining for training (only if active_hard0_keep_ratio < 1.0)
                    if active_hard0_keep_ratio < 1.0:
                        # Compute per-node loss
                        per_node_loss = compute_per_node_loss(
                            out, y, loss_type_for_per_node,
                            class_weights=class_weights.to(device) if not use_logit_adjust and class_weights is not None else None,
                            gamma=gamma if use_log_softmax else None,
                            class_prior=class_prior.to(device) if use_logit_adjust and class_prior is not None else None,
                            tau=logit_adjust_tau if use_logit_adjust else None,
                            device=device
                        )
                        
                        # Separate class 0 and non-0 nodes
                        class0_mask = (y == 0)
                        non0_mask = ~class0_mask
                        
                        # Keep all non-0 nodes
                        non0_losses = per_node_loss[non0_mask]
                        
                        # For class 0, keep top-k by loss (hardest samples)
                        topk_indices = None
                        if class0_mask.sum() > 0:
                            class0_losses = per_node_loss[class0_mask]
                            num_class0 = class0_losses.size(0)
                            k = max(1, int(num_class0 * active_hard0_keep_ratio))
                            _, topk_indices = torch.topk(class0_losses, k, largest=True)
                            topk0_losses = class0_losses[topk_indices]
                            
                            # Concatenate selected losses
                            selected_losses = torch.cat([non0_losses, topk0_losses]) if non0_losses.size(0) > 0 else topk0_losses
                        else:
                            selected_losses = non0_losses
                        
                        # Apply class-mean loss if enabled
                        if args.use_class_mean_loss and selected_losses.size(0) > 0:
                            # Get labels for selected losses
                            if class0_mask.sum() > 0 and topk_indices is not None:
                                selected_y = torch.cat([y[non0_mask], y[class0_mask][topk_indices]]) if non0_losses.size(0) > 0 else y[class0_mask][topk_indices]
                            else:
                                selected_y = y[non0_mask]
                            
                            # Compute per-class mean loss
                            unique_classes = torch.unique(selected_y)
                            per_class_means = []
                            for cls in unique_classes:
                                cls_mask = (selected_y == cls)
                                if cls_mask.sum() > 0:
                                    cls_mean = selected_losses[cls_mask].mean()
                                    per_class_means.append(cls_mean)
                            
                            if len(per_class_means) > 0:
                                loss = torch.stack(per_class_means).mean()
                            else:
                                loss = selected_losses.mean()
                        else:
                            if selected_losses.numel() > 0:
                                loss = selected_losses.mean()
                            else:
                                # Fallback to standard loss if no samples selected
                                # 層制約を適用する場合、batch情報を渡す
                                if use_layer_constraint:
                                    loss = loss_fn(out, y, batch=batch.batch)
                                else:
                                    loss = loss_fn(out, y)
                    else:
                        # No hard0 mining: use standard loss
                        # 層制約を適用する場合、batch情報を渡す
                        if use_layer_constraint:
                            loss = loss_fn(out, y, batch=batch.batch)
                        else:
                            loss = loss_fn(out, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = ddp_model(batch)
                y = batch.y
                
                # Hard0 mining for training (only if active_hard0_keep_ratio < 1.0)
                if active_hard0_keep_ratio < 1.0:
                    # Compute per-node loss
                    per_node_loss = compute_per_node_loss(
                        out, y, loss_type_for_per_node,
                        class_weights=class_weights.to(device) if not use_logit_adjust and class_weights is not None else None,
                        gamma=gamma if use_log_softmax else None,
                        class_prior=class_prior.to(device) if use_logit_adjust and class_prior is not None else None,
                        tau=logit_adjust_tau if use_logit_adjust else None,
                        device=device
                    )
                    
                    # Separate class 0 and non-0 nodes
                    class0_mask = (y == 0)
                    non0_mask = ~class0_mask
                    
                    # Keep all non-0 nodes
                    non0_losses = per_node_loss[non0_mask]
                    
                    # For class 0, keep top-k by loss (hardest samples)
                    topk_indices = None
                    if class0_mask.sum() > 0:
                        class0_losses = per_node_loss[class0_mask]
                        num_class0 = class0_losses.size(0)
                        k = max(1, int(num_class0 * active_hard0_keep_ratio))
                        _, topk_indices = torch.topk(class0_losses, k, largest=True)
                        topk0_losses = class0_losses[topk_indices]
                        
                        # Concatenate selected losses
                        selected_losses = torch.cat([non0_losses, topk0_losses]) if non0_losses.size(0) > 0 else topk0_losses
                    else:
                        selected_losses = non0_losses
                    
                    # Apply class-mean loss if enabled
                    if args.use_class_mean_loss and selected_losses.size(0) > 0:
                        # Get labels for selected losses
                        if class0_mask.sum() > 0 and topk_indices is not None:
                            selected_y = torch.cat([y[non0_mask], y[class0_mask][topk_indices]]) if non0_losses.size(0) > 0 else y[class0_mask][topk_indices]
                        else:
                            selected_y = y[non0_mask]
                        
                        # Compute per-class mean loss
                        unique_classes = torch.unique(selected_y)
                        per_class_means = []
                        for cls in unique_classes:
                            cls_mask = (selected_y == cls)
                            if cls_mask.sum() > 0:
                                cls_mean = selected_losses[cls_mask].mean()
                                per_class_means.append(cls_mean)
                        
                        if len(per_class_means) > 0:
                            loss = torch.stack(per_class_means).mean()
                        else:
                            loss = selected_losses.mean()
                    else:
                        if selected_losses.numel() > 0:
                            loss = selected_losses.mean()
                        else:
                            # Fallback to standard loss if no samples selected
                            # 層制約を適用する場合、batch情報を渡す
                            if use_layer_constraint:
                                loss = loss_fn(out, y, batch=batch.batch)
                            else:
                                loss = loss_fn(out, y)
                else:
                    # No hard0 mining: use standard loss
                    # 層制約を適用する場合、batch情報を渡す
                    if use_layer_constraint:
                        loss = loss_fn(out, y, batch=batch.batch)
                    else:
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
        
        # Ensure NCCL operation completes before proceeding
        torch.cuda.synchronize()

        ddp_model.eval()
        val_loss = 0.0
        correct = 0
        total_samples = 0
        # Build local confusion matrix (19x19) on each rank - much lighter than all_gather_object
        num_classes = 19
        cm_local = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
        
        with torch.no_grad():
            val_batch_count = 0
            local_samples = 0
            for batch in val_loader:
                batch = batch.to(device)
                out = ddp_model(batch)
                y = batch.y

                # 層制約を適用する場合、batch情報を渡す（評価時も制約を確認）
                if use_layer_constraint:
                    loss = loss_fn(out, y, batch=batch.batch)
                else:
                    loss = loss_fn(out, y)
                val_loss += loss.item()

                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                batch_size = y.size(0)
                total_samples += batch_size
                local_samples += batch_size
                
                # Update local confusion matrix (高速化: torch.bincountで一括更新)
                # k = true_label * num_classes + pred_label で1次元インデックス化
                y_flat = y.view(-1).long()
                pred_flat = pred.view(-1).long()
                k = y_flat * num_classes + pred_flat
                # bincountで各組み合わせの出現回数をカウント
                cm_update = torch.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
                cm_local += cm_update
                
                val_batch_count += 1
            
            # Debug: Log validation processing info
            local_cm_sum = cm_local.sum().item()
            if epoch <= 10 or epoch % 100 == 0:  # Log first 10 epochs and every 100 epochs
                print(f"[DEBUG] Rank {rank} Epoch {epoch}: val_batches={val_batch_count}, local_samples={local_samples}, local_cm_sum={local_cm_sum}")
            
            # Debug: Log if no batches were processed (should not happen with drop_last=False)
            if val_batch_count == 0:
                print(f"[WARNING] Rank {rank}: No validation batches processed at epoch {epoch}")

        avg_val_loss = torch.tensor(val_loss / len(val_loader), dtype=torch.float).to(device)
        dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
        avg_val_loss /= world_size

        total_correct = torch.tensor(correct, dtype=torch.float, device=device)
        total_samples_t = torch.tensor(total_samples, dtype=torch.float, device=device)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_t, op=dist.ReduceOp.SUM)
        
        # Reduce confusion matrix across all ranks (lightweight operation)
        dist.all_reduce(cm_local, op=dist.ReduceOp.SUM)
        
        # Ensure NCCL operations complete before proceeding
        torch.cuda.synchronize()
        
        val_accuracy = total_correct / total_samples_t
        
        # Convert confusion matrix to numpy on rank 0 for metric computation
        if rank == 0:
            cm_np = cm_local.cpu().numpy()
            # Debug: Check confusion matrix sum
            cm_sum = cm_np.sum()
            total_samples_val = total_samples_t.item()
            
            # Always log debug info for first 10 epochs or when there's a problem
            cm_nonzero_count = np.count_nonzero(cm_np)
            if epoch <= 10 or epoch % 100 == 0 or cm_sum == 0:
                print(f"[DEBUG] Rank 0 Epoch {epoch}: total_samples_t={total_samples_val}, cm_sum={cm_sum}, cm_nonzero={cm_nonzero_count}")
                # Debug: Show confusion matrix row/column sums to understand distribution
                row_sums = cm_np.sum(axis=1)  # True labels distribution
                col_sums = cm_np.sum(axis=0)  # Predicted labels distribution
                print(f"[DEBUG] Epoch {epoch} - True labels (row sums): {row_sums}")
                print(f"[DEBUG] Epoch {epoch} - Predicted labels (col sums): {col_sums}")
                # Show which classes have non-zero entries
                nonzero_indices = np.nonzero(cm_np)
                if len(nonzero_indices[0]) > 0:
                    print(f"[DEBUG] Epoch {epoch} - Non-zero entries: {len(nonzero_indices[0])} positions")
                    # Show first 20 non-zero entries
                    for idx in range(min(20, len(nonzero_indices[0]))):
                        i, j = nonzero_indices[0][idx], nonzero_indices[1][idx]
                        print(f"[DEBUG]   CM[{i},{j}] = {cm_np[i,j]}")
            
            if cm_sum == 0:
                print(f"[WARNING] Empty confusion matrix at epoch {epoch} - total_samples_t={total_samples_val}, cm_sum={cm_sum}")
                # Additional debug: check if cm_local was properly reduced
                print(f"[DEBUG] cm_local before reduction check: sum={cm_local.sum().item()}, device={cm_local.device}")
            
            # Reconstruct predictions and labels from confusion matrix for compatibility
            # (only needed for backward compatibility with existing metric computation code)
            all_preds_np = []
            all_labels_np = []
            for i in range(num_classes):
                for j in range(num_classes):
                    count = int(cm_np[i, j])
                    all_labels_np.extend([i] * count)
                    all_preds_np.extend([j] * count)
            # Check if we have any data (confusion matrix is not empty)
            if len(all_preds_np) > 0 and len(all_labels_np) > 0:
                all_preds_np = np.array(all_preds_np)
                all_labels_np = np.array(all_labels_np)
                # Debug: Check reconstructed arrays
                if epoch <= 10 or epoch % 100 == 0:
                    unique_labels = np.unique(all_labels_np)
                    unique_preds = np.unique(all_preds_np)
                    print(f"[DEBUG] Epoch {epoch} - Reconstructed: {len(all_labels_np)} labels, {len(all_preds_np)} preds")
                    print(f"[DEBUG] Epoch {epoch} - Unique labels: {unique_labels}, Unique preds: {unique_preds}")
            else:
                # Empty confusion matrix - no validation data processed
                all_preds_np = None
                all_labels_np = None
                if epoch % 10 == 0:  # Log warning every 10 epochs to avoid spam
                    print(f"[WARNING] Empty confusion matrix at epoch {epoch} - no validation data processed (cm_sum={cm_sum}, total_samples_t={total_samples_t.item()})")
        else:
            all_preds_np = None
            all_labels_np = None
            cm_np = None
        
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
    
            if all_preds_np is not None and all_labels_np is not None:
                # Compute metrics from reconstructed predictions/labels
                precision, recall, f1, support = precision_recall_fscore_support(
                    all_labels_np, all_preds_np, zero_division=0)
                print(f"Per-Class Precision: {precision}")
                print(f"Per-Class Recall: {recall}")
                print(f"Per-Class F1-Score: {f1}")
                print(f"Support per Class: {support}")
    
                weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                    all_labels_np, all_preds_np, average='weighted', zero_division=0)
                print(f"Weighted Precision: {weighted_precision:.4f}, Weighted Recall: {weighted_recall:.4f}, Weighted F1-Score: {weighted_f1:.4f}")

        if rank == 0:
            if cm_np is not None and all_preds_np is not None and all_labels_np is not None and len(all_preds_np) > 0:
                # Use confusion matrix directly (already computed from all_reduce)
                cm = cm_np

                precision = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
                recall = recall_score(all_labels_np, all_preds_np, average='weighted')
                f1 = f1_score(all_labels_np, all_preds_np, average='weighted')
                macro_f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
                if epoch % 50 == 0:  # Print every 50 epochs to reduce output
                    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Weighted F1-Score: {f1:.4f}, Macro F1-Score: {macro_f1:.4f}')
            else:
                # デフォルト値を設定（混同行列が空の場合）
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                macro_f1 = 0.0
                if epoch % 50 == 0:  # Log warning every 50 epochs to avoid spam
                    print(f'[WARNING] Cannot compute metrics - empty validation data at epoch {epoch}')
                    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Weighted F1-Score: {f1:.4f}, Macro F1-Score: {macro_f1:.4f}')

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
        
        # CRITICAL: Ensure all NCCL operations from this epoch are complete
        # All collectives (all_reduce) are already synchronous, so we just need CUDA sync
        # Removed unnecessary barrier here - all_reduce operations are already synchronized
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # ========================================================================
        # DDP SAFETY AUDIT: Collective Call Order (ALL ranks execute in SAME order)
        # ========================================================================
        # Every epoch, ALL ranks execute these collectives in EXACTLY this order:
        # 1. dist.all_reduce(avg_train_loss, op=SUM)      [Line ~1556]
        # 2. dist.all_reduce(avg_val_loss, op=SUM)        [Line ~1588]
        # 3. dist.all_reduce(total_correct, op=SUM)        [Line ~1593]
        # 4. dist.all_reduce(total_samples_t, op=SUM)      [Line ~1594]
        # 5. dist.all_reduce(cm_local, op=SUM)             [Line ~1597]
        # 6. dist.broadcast(stop_tensor, src=0)            [Line ~1756] - ALL ranks call this!
        #
        # IMPORTANT:
        # - NO collectives inside "if rank == 0:" conditionals
        # - NO early returns that skip collectives on some ranks
        # - All collectives are synchronous (no async operations)
        # - drop_last=True ensures identical batch counts across ranks
        # ========================================================================
        barrier_success = True
        
        # Track best metrics and save checkpoints
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
            
            # Update best model state when model improves (for early stopping)
            if improved_loss or improved_macro_f1:
                import copy
                best_model_state = copy.deepcopy(ddp_model.module.state_dict())
                best_epoch = epoch
            
            # Save checkpoint every 100 epochs (and delete old ones, keeping only the latest)
            if epoch % 100 == 0:
                try:
                    # Delete old checkpoint files with the same pattern (keep only latest)
                    import glob
                    pattern = f"{checkpoint_dir}/{type(model).__name__}_{timestamp}_Best_Epoch*.pth"
                    old_checkpoints = glob.glob(pattern)
                    for old_checkpoint in old_checkpoints:
                        try:
                            os.remove(old_checkpoint)
                            print(f"[CHECKPOINT] Deleted old checkpoint: {old_checkpoint}")
                        except Exception as e:
                            print(f"[WARNING] Failed to delete old checkpoint {old_checkpoint}: {e}")
                    
                    # Save new checkpoint
                    checkpoint_path = f"{checkpoint_dir}/{type(model).__name__}_{timestamp}_Best_Epoch{epoch}_ValLoss{best_val_loss:.4f}_MacroF1{best_macro_f1:.4f}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ddp_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_val_loss': best_val_loss,
                        'best_macro_f1': best_macro_f1,
                        'timestamp': timestamp
                    }, checkpoint_path)
                    print(f"[CHECKPOINT] Model saved at epoch {epoch}: {checkpoint_path}")
                    
                    # Also save as "best" checkpoint (overwrites previous best)
                    best_checkpoint_path = f"{checkpoint_dir}/{type(model).__name__}_{timestamp}_Best.pth"
                    torch.save(ddp_model.module.state_dict(), best_checkpoint_path)
                except Exception as e:
                    print(f"[WARNING] Failed to save checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Early stopping: stop if both val_loss and macro_f1 don't improve
            # IMPORTANT: Only trigger early stopping if barrier succeeded
            # If barrier failed, skip early stopping this epoch to avoid hanging
            # barrier_success is available to all ranks (defined before this block)
            should_stop = (counter >= args.patience and macro_f1_counter >= args.patience and barrier_success)
        else:
            # Non-rank-0: initialize should_stop (will be overwritten by broadcast)
            should_stop = False
        
        # CRITICAL DDP FIX: ALL ranks must call dist.broadcast(), not just rank 0
        # Rank 0 computes should_stop, but ALL ranks participate in the collective
        stop_tensor = torch.tensor(1 if should_stop else 0, dtype=torch.int32, device=device)
        try:
            # ALL ranks call broadcast - rank 0 sends, all ranks receive
            dist.broadcast(stop_tensor, src=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure broadcast completes
            
            # All ranks read the broadcast value
            should_stop_all = stop_tensor.item() == 1
        except Exception as e:
            # If broadcast fails, log and use fallback
            if rank == 0:
                print(f"[WARNING] Broadcast of early stop flag failed: {e}")
                should_stop_all = should_stop
            else:
                print(f"[WARNING] Rank {rank}: Broadcast of early stop flag failed: {e}")
                should_stop_all = False
        
        # [FIX S-2] Early stopping - ensure ALL ranks break at the SAME indentation level
        if should_stop_all:
            early_stopped = True
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"  - Val Loss hasn't improved for {counter} epochs")
                print(f"  - Macro F1 hasn't improved for {macro_f1_counter} epochs")
                print(f"  - Best Val Loss: {best_val_loss:.4f}")
                print(f"  - Best Macro F1: {best_macro_f1:.4f}")
                print(f"  - Best model was at epoch {best_epoch}")
                print(f"{'='*60}\n")
                
                # Load and save best model
                if best_model_state is not None:
                    try:
                        print(f"[INFO] Loading best model from epoch {best_epoch}...")
                        ddp_model.module.load_state_dict(best_model_state)
                        print(f"[INFO] Best model loaded successfully")
                        
                        # Save best model immediately
                        print("[INFO] Saving best model after early stopping...")
                        save_dir = "/home/nishioka/GNN/GNN_hole_2026/GNN_model/19classmodel_hole_zscore"
                        os.makedirs(save_dir, exist_ok=True)
                        best_final_path = f"{save_dir}/{type(model).__name__}_{timestamp}_Best_EarlyStop.pth"
                        torch.save(best_model_state, best_final_path)
                        print(f"[FINAL_INFO] Best model (epoch {best_epoch}) saved to {best_final_path}")
                        print(f"[INFO] Skipping post-training evaluation to prevent NCCL timeout")
                        print(f"[INFO] Model is saved and ready for use")
                    except Exception as e:
                        print(f"[WARNING] Failed to save best model after early stopping: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Break immediately - all ranks are synchronized, no pending operations
            # This is safe because:
            # 1. Barrier completed successfully (barrier_success == True)
            # 2. All NCCL operations from this epoch are complete
            # 3. Broadcast completed, so all ranks know to break
            # 4. All ranks will break at the same point
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
                
                elapsed_time = time.time() - train_start_time
                elapsed_hours = int(elapsed_time // 3600)
                elapsed_minutes = int((elapsed_time % 3600) // 60)
                elapsed_seconds = int(elapsed_time % 60)
                
                print(f'\n=== Epoch {epoch}/{args.epochs} ===')
                print(f'Train Loss: {avg_train_loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}, Macro F1: {macro_f1:.4f}')
                print(f'Elapsed Time: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}')
                print(f'Estimated Remaining Time: {hours:02d}:{minutes:02d}:{seconds:02d} (Avg {avg_epoch_time:.2f}s/epoch)')
                print('=' * 50)

    # Early stopping is now handled by breaking from the loop directly
    # No need to check tensor or file - early_stopped flag is set when breaking
    # This is simpler and more reliable
    
    # CRITICAL: Only synchronize if early stopping did NOT occur
    # If early stopping occurred, we already synchronized before setting the flag
    # Additional barriers here can cause hangs if communicator is aborted
    if not early_stopped:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Lightweight barrier to ensure all ranks complete training loop
            # Note: timeout parameter not supported in this PyTorch version
            dist.barrier()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] Post-training barrier failed: {e}")
    
    # Synchronize all ranks after training loop ONLY if early stopping didn't occur
    # Note: If early stopping occurred, best model was already saved
    # But we still want to run evaluation/visualization on rank 0
    if early_stopped:
        # Early stopping occurred - model already saved
        if rank == 0:
            print(f"[INFO] Early stopping occurred - will still run post-training evaluation/visualization")
            print(f"[INFO] Best model saved at: {checkpoint_dir}/{type(model).__name__}_{timestamp}_Best_EarlyStop.pth")
        
        # Clean up flag file (if it exists from fallback)
        try:
            early_stop_flag_file = f"/home/nishioka/GNN/GNN_hole_2026/GNN_model/19classmodel_hole_zscore/.early_stop_{timestamp}.flag"
            if rank == 0 and os.path.exists(early_stop_flag_file):
                os.remove(early_stop_flag_file)
        except:
            pass
        
        # Early stopping occurred - model already saved
        # Skip additional synchronization to avoid hanging on aborted communicators
        # The synchronization already happened before setting the early stop flag
        # But we still want to run evaluation/visualization on rank 0
        # Other ranks should exit here to avoid NCCL timeout
        if rank != 0:
            cleanup(device=device, rank=rank)
            return
        # Rank 0 continues to post-training evaluation/visualization section below
    else:
        # Normal completion - synchronize before post-training processing
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Note: timeout parameter not supported in this PyTorch version
            dist.barrier()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] Barrier failed after training loop: {e}")

    if rank == 0:
        train_elapsed_time = time.time() - train_start_time
        train_hours = int(train_elapsed_time // 3600)
        train_minutes = int((train_elapsed_time % 3600) // 60)
        train_seconds = int(train_elapsed_time % 60)
        
        # Reconstruct command for final output
        command_str = ' '.join(sys.argv)
        
        print(f"\n{'='*60}")
        print(f"Training Completed")
        print(f"Best Val Loss: {best_val_loss:.4f}, Best Macro F1: {best_macro_f1:.4f}")
        print(f"Best model was at epoch {best_epoch}")
        print(f"Total Training Time: {train_hours:02d}:{train_minutes:02d}:{train_seconds:02d}")
        print(f"{'='*60}")
        print(f"\nExecution Command (for reference):")
        print(f"  {command_str}")
        print(f"{'='*60}\n")
        
        # Ensure best model is loaded (if not already loaded during early stopping)
        if best_model_state is not None:
            try:
                # Only load if not already loaded (early stopping already loaded it)
                current_state = ddp_model.module.state_dict()
                # Simple check: if current state matches best state, skip loading
                if not all(torch.equal(current_state[k], best_model_state[k]) for k in current_state.keys()):
                    print(f"[INFO] Loading best model from epoch {best_epoch}...")
                    ddp_model.module.load_state_dict(best_model_state)
                    print(f"[INFO] Best model loaded successfully")
                else:
                    print(f"[INFO] Best model already loaded")
            except Exception as e:
                print(f"[WARNING] Failed to load best model: {e}")
                import traceback
                traceback.print_exc()
        
        # Save final model immediately (before other time-consuming operations)
        try:
            print("[INFO] Saving final model...")
            save_dir = "/home/nishioka/GNN/GNN_hole_2026/GNN_model/19classmodel_hole_zscore"
            os.makedirs(save_dir, exist_ok=True)
            final_model_path = f"{save_dir}/{type(model).__name__}_{timestamp}_Final.pth"
            torch.save(ddp_model.module.state_dict(), final_model_path)
            print(f"[FINAL_INFO] Final model saved to {final_model_path}")
            
            # Also save best model as final if available (if not already saved during early stopping)
            if best_model_state is not None:
                best_final_path = f"{save_dir}/{type(model).__name__}_{timestamp}_Best_Final.pth"
                # Check if early stop file exists, if not save it
                early_stop_path = f"{save_dir}/{type(model).__name__}_{timestamp}_Best_EarlyStop.pth"
                if not os.path.exists(early_stop_path):
                    torch.save(best_model_state, best_final_path)
                    print(f"[FINAL_INFO] Best model (epoch {best_epoch}) saved to {best_final_path}")
                else:
                    print(f"[INFO] Best model already saved during early stopping")
        except Exception as e:
            print(f"[ERROR] Failed to save final model: {e}")
            import traceback
            traceback.print_exc()
    
    # CRITICAL: Do NOT synchronize after model saving
    # Rank 0 will do time-consuming operations (plotting, evaluation, visualization)
    # Other ranks should exit immediately to avoid NCCL timeout
    # Model is already saved, so these operations are optional
    
    if rank == 0:
        # Loss plotを保存（アーリーストップ時も含む）
        try:
            if len(train_loss_per_epoch) > 0 and len(val_loss_per_epoch) > 0:
                print("Saving loss plots...")
                output_plot_dir = f"/home/nishioka/GNN/GNN_hole_2026/Predict_data/Loss_plots_{timestamp}"
                os.makedirs(output_plot_dir, exist_ok=True)
                
                epochs = range(1, len(train_loss_per_epoch) + 1)
                
                # Loss plot
                try:
                    plt.figure(figsize=(12, 6))
                    plt.plot(epochs, train_loss_per_epoch, 'b-', label='Train Loss', linewidth=2)
                    plt.plot(epochs, val_loss_per_epoch, 'r-', label='Val Loss', linewidth=2)
                    plt.xlabel('Epoch', fontsize=12)
                    plt.ylabel('Loss', fontsize=12)
                    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
                    plt.legend(fontsize=11)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_plot_dir, f"loss_plot_{timestamp}.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Loss plot saved to: {output_plot_dir}/loss_plot_{timestamp}.png")
                except Exception as e:
                    print(f"[WARNING] Failed to save loss plot: {e}")
                
                # Macro F1 plot
                if len(macro_f1_per_epoch) > 0:
                    try:
                        plt.figure(figsize=(12, 6))
                        plt.plot(epochs, macro_f1_per_epoch, 'g-', label='Macro F1-Score', linewidth=2)
                        plt.xlabel('Epoch', fontsize=12)
                        plt.ylabel('Macro F1-Score', fontsize=12)
                        plt.title('Macro F1-Score per Epoch', fontsize=14, fontweight='bold')
                        plt.legend(fontsize=11)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_plot_dir, f"macro_f1_plot_{timestamp}.png"), dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"✓ Macro F1 plot saved to: {output_plot_dir}/macro_f1_plot_{timestamp}.png")
                    except Exception as e:
                        print(f"[WARNING] Failed to save macro F1 plot: {e}")
                
                # Combined plot (Loss + Macro F1)
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Loss plot
                    ax1.plot(epochs, train_loss_per_epoch, 'b-', label='Train Loss', linewidth=2)
                    ax1.plot(epochs, val_loss_per_epoch, 'r-', label='Val Loss', linewidth=2)
                    ax1.set_xlabel('Epoch', fontsize=12)
                    ax1.set_ylabel('Loss', fontsize=12)
                    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
                    ax1.legend(fontsize=11)
                    ax1.grid(True, alpha=0.3)
                    
                    # Macro F1 plot
                    if len(macro_f1_per_epoch) > 0:
                        ax2.plot(epochs, macro_f1_per_epoch, 'g-', label='Macro F1-Score', linewidth=2)
                        ax2.set_xlabel('Epoch', fontsize=12)
                        ax2.set_ylabel('Macro F1-Score', fontsize=12)
                        ax2.set_title('Macro F1-Score per Epoch', fontsize=14, fontweight='bold')
                        ax2.legend(fontsize=11)
                        ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_plot_dir, f"combined_plot_{timestamp}.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Combined plot saved to: {output_plot_dir}/combined_plot_{timestamp}.png")
                except Exception as e:
                    print(f"[WARNING] Failed to save combined plot: {e}")
            else:
                print("[WARNING] No loss data available for plotting")
        except Exception as e:
            print(f"[ERROR] Failed to save loss plots: {e}")
            import traceback
            traceback.print_exc()

        # Test evaluation and visualization run on rank 0 only
        # Other ranks can skip to avoid NCCL timeout
        # Model is already saved, so these operations are optional
        if rank == 0:
            print("Evaluating on Test Data...")
            
            # [FIX S-1] Create a new test DataLoader without DistributedSampler for rank-0 evaluation
            # This ensures full test dataset evaluation with correct order matching test_pairs
            test_loader_eval = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False
            )
            
            # テストデータの評価
            ddp_model.eval()
            
            test_preds = []
            test_labels = []
            test_probs = []  # 確率を格納するリスト
            
            # --- loss function for test must match training choice ---
            use_logit_adjust = getattr(args, 'use_logit_adjust', False)
            logit_adjust_tau = getattr(args, 'logit_adjust_tau', 1.0)
            use_log_softmax = getattr(args, 'use_log_softmax', False)
            gamma = getattr(args, 'gamma', default_gamma)
            use_layer_constraint = getattr(args, 'use_layer_constraint', True)
            layer_constraint_weight = getattr(args, 'layer_constraint_weight', 1.0)
            
            # ベース損失関数を作成（学習時と同じ）
            if use_logit_adjust:
                # trainと同じpriorを作る（train_datasetから固定）
                labels_array = np.concatenate([d.y.cpu().numpy() for d in train_dataset])
                class_counts = np.bincount(labels_array, minlength=19)
                class_prior = torch.tensor(class_counts / (class_counts.sum() + 1e-8), dtype=torch.float, device=device)
                base_loss_fn = LogitAdjustLoss(class_prior=class_prior, tau=logit_adjust_tau).to(device)
            else:
                if use_log_softmax:
                    base_loss_fn = FocalLossLogSoftmax(weights=class_weights.to(device), gamma=gamma).to(device)
                else:
                    base_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
            
            # 層制約を適用（学習時と同じ設定）
            if use_layer_constraint:
                loss_fn = LayerAwareLoss(
                    base_loss_fn=base_loss_fn,
                    vertices_per_layer=vertices_per_layer,
                    constraint_weight=layer_constraint_weight
                ).to(device)
            else:
                loss_fn = base_loss_fn
            
            correct = 0
            total_loss = 0.0
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader_eval):
                    batch = batch.to(device)
                    out = ddp_model.module(batch)  # Model outputs logits (use .module to avoid DDP collectives)
                    y = batch.y
            
                    # 層制約を適用する場合、batch情報を渡す
                    if use_layer_constraint:
                        loss = loss_fn(out, y, batch=batch.batch)
                    else:
                        loss = loss_fn(out, y)
                    total_loss += loss.item()
            
                    # 予測ラベル
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total_samples += y.size(0)
            
                    # 予測ラベル・実ラベル・確率を格納
                    test_preds.extend(pred.cpu().numpy())
                    test_labels.extend(y.cpu().numpy())
                    
                    # Model outputs logits, apply softmax to get probabilities
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    test_probs.extend(probs)
            
            test_loss = total_loss / len(test_loader_eval)  # 修正: test_loader_evalを使う
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
            
            # ---- 予測結果を CSV に保存 ----
            output_dir_csv = f"/home/nishioka/GNN/GNN_hole_2026/Predict_csv/Predict_csv_zscore_{timestamp}"
            os.makedirs(output_dir_csv, exist_ok=True)
            
            save_predictions_to_csv(
                test_pairs,
                test_preds,
                test_labels,
                test_probs,
                output_dir_csv,
                num_nodes_per_sample=13942
            )
            
            print("[INFO] CSV save completed.")
            
            # ---- 可視化や追加の評価 ----
            # CRITICAL: Visualization runs only on rank 0
            # Other ranks have already exited, so no synchronization needed
            try:
                gamma = getattr(args, 'gamma', default_gamma)
                evaluate_and_visualize(
                    test_loader,
                    train_loader,
                    ddp_model.module,  # Use .module to avoid DDP collectives in rank0-only evaluation
                    device,
                    class_weights,
                    test_pairs,
                    train_pairs,
                    num_nodes_per_sample=13942,
                    gamma=gamma,
                    rank=rank,
                    world_size=world_size,
                    test_dataset=test_dataset,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    z_coords=z_coords
                )
            except Exception as e:
                print(f"[ERROR] Failed during evaluation/visualization: {e}")
                import traceback
                traceback.print_exc()
                # Continue even if visualization fails - model is already saved
        else:
            # Other ranks skip evaluation and visualization, exit immediately
            print(f"[INFO] Rank {rank}: Skipping test evaluation and visualization (rank 0 only)")
    
    cleanup(device=device, rank=rank)

# ----------------------------
# エントリーポイント
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training Script for Z-score Normalized Data')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Number of hidden channels')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate (default: 0.002, reduced from 0.01 to prevent collapse)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=300, help='Early stopping patience')
    parser.add_argument('--gamma', type=float, default=3.0, help='Focal loss gamma parameter (default: 3.0, increased for extreme class imbalance)')
    parser.add_argument('--class_weight_multiplier', type=float, default=5.0, help='Class weight multiplier for minority classes (default: 5.0, increased for better handling of imbalanced classes)')
    parser.add_argument('--use_logit_adjust', action='store_true', default=False, help='Use LogitAdjustLoss instead of FocalLoss (default: False, recommended for imbalanced datasets - priority 1)')
    parser.add_argument('--logit_adjust_tau', type=float, default=1.0, help='Tau parameter for LogitAdjustLoss (default: 1.0, recommended range: 1.0~2.0)')
    # Boolean引数の修正: store_true + default=Trueは常にTrueになる問題を修正
    # default=Falseにして、--no_xxxでOFFにする方式（互換性重視）
    parser.add_argument('--use_log_softmax', action='store_true', default=False, help='Use log_softmax version of FocalLoss for numerical stability (default: False)')
    parser.add_argument('--no_log_softmax', action='store_false', dest='use_log_softmax', help='Disable log_softmax version of FocalLoss')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use mixed precision training (AMP) (default: False)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp', help='Disable mixed precision training')
    parser.add_argument('--use_onecycle', action='store_true', default=False, help='Use OneCycleLR scheduler instead of CosineAnnealingLR (default: False)')
    parser.add_argument('--no_onecycle', dest='use_onecycle', action='store_false', help='Disable OneCycleLR scheduler (use CosineAnnealingLR)')
    parser.add_argument('--onecycle_max_lr', type=float, default=None, help='Max learning rate for OneCycleLR (default: learning_rate * 2.0, recommended: 1e-3 ~ 3e-3 for GAT)')
    parser.add_argument('--use_minority_sampler', action='store_true', default=False, help='Use DistributedMinorityRatioSampler for training (default: False, recommended for imbalanced datasets)')
    parser.add_argument('--minority_weight_power', type=float, default=1.0, help='Power for minority_ratio weighting in sampler (default: 1.0, higher=more emphasis on minority-rich files)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability (default: 0.1, recommended range: 0.0~0.3)')
    parser.add_argument('--edge_drop_prob', type=float, default=0.01, help='Edge dropout probability (default: 0.01, recommended range: 0.0~0.05)')
    parser.add_argument('--data_usage_ratio', type=float, default=0.5, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       help='Ratio of data to use (10%% increments: 0.1, 0.2, ..., 1.0). Default: 1.0 (100%%)')
    # Layer constraint: デフォルトで有効（--no_layer_constraintで無効化可能）
    parser.add_argument('--no_layer_constraint', dest='use_layer_constraint', action='store_false', default=True, help='Disable layer-aware label constraints (default: enabled, Layer 1: 0,1~9 only, Layer 2: 0,10~18 only)')
    parser.add_argument('--layer_constraint_weight', type=float, default=1.0, help='Weight for layer constraint penalty (default: 1.0, higher=stronger constraint)')
    parser.add_argument('--hard0_keep_ratio', type=float, default=1.0, help='Keep ratio of class-0 nodes by hardest loss. 1.0 disables.')
    parser.add_argument('--hard0_strategy', type=str, default='topk', choices=['topk'], help='Hard0 mining strategy (only topk implemented)')
    parser.add_argument('--use_class_mean_loss', action='store_true', default=False, help='Compute loss as mean over per-class means (only for classes present in batch)')
    parser.add_argument('--stage2_epoch', type=int, default=401, help='Epoch to start stage2')
    parser.add_argument('--stage1_minority_weight_power', type=float, default=1.5, help='Minority weight power for stage1')
    parser.add_argument('--stage2_minority_weight_power', type=float, default=1.0, help='Minority weight power for stage2')
    parser.add_argument('--stage1_hard0_keep_ratio', type=float, default=1.0, help='Hard0 keep ratio for stage1')
    parser.add_argument('--stage2_hard0_keep_ratio', type=float, default=1.0, help='Hard0 keep ratio for stage2')

    args = parser.parse_args()

    main(args)
