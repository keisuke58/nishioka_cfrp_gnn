#!/usr/bin/env python3
"""
評価専用スクリプト
保存済みの最良モデルを使用してテストデータの評価のみを実行します。
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    balanced_accuracy_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

# 既存のスクリプトから必要な関数とクラスをインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'gnn_common'))

from GNN_zscore_sub_noise_defect_free_integrated import (
    GATModel,
    group_id_from_filename,
    group_disjoint_split,
    default_class_weight_multiplier,
    default_gamma,
    FocalLossLogSoftmax,
    LogitAdjustLoss,
    LayerAwareLoss,
    vertices_per_layer
)

from gnn_common.integration_utils import (
    prepare_data_with_cross_edges,
    load_config_for_cross_edges,
    load_config_for_splitting,
    apply_ood_split
)

# Localization metrics
from gnn_common.metrics import calculate_localization_metrics


def load_model(model_path, hidden_channels=16, dropout=0.1, edge_drop_prob=0.01, device='cuda:0'):
    """保存済みモデルをロード"""
    print(f"[INFO] Loading model from: {model_path}")
    model = GATModel(
        hidden_channels=hidden_channels,
        num_classes=19,
        dropout=dropout,
        edge_drop_prob=edge_drop_prob
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # チェックポイントの形式を確認
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # state_dictが直接保存されている場合
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"[INFO] Model loaded successfully")
    return model


def prepare_test_dataset(args, rank=0):
    """テストデータセットを準備"""
    # データパスの設定
    noise_data_base = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise"
    noise_data_folder = os.path.join(noise_data_base, "train")
    defect_data_base = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise"
    defect_train_folder = os.path.join(defect_data_base, "train")
    defect_val_folder = os.path.join(defect_data_base, "val")
    defect_test_folder = os.path.join(defect_data_base, "test")
    label_data_folder = "/home/nishioka/GNN/GNN_hole_2026/all_19class_label"
    
    # 座標データの読み込み
    x_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
    y_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
    z_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")
    
    edges = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy")
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    # データファイルの読み込み
    noise_data_files = sorted([f for f in os.listdir(noise_data_folder) 
                               if f.startswith("NoiseDefectFree_") and f.endswith(".npy")])
    
    all_defect_files = []
    if os.path.exists(defect_train_folder):
        all_defect_files.extend([(f, defect_train_folder) for f in os.listdir(defect_train_folder) 
                                  if f.startswith("Defect_L") and f.endswith(".npy")])
    if os.path.exists(defect_val_folder):
        all_defect_files.extend([(f, defect_val_folder) for f in os.listdir(defect_val_folder) 
                                  if f.startswith("Defect_L") and f.endswith(".npy")])
    if os.path.exists(defect_test_folder):
        all_defect_files.extend([(f, defect_test_folder) for f in os.listdir(defect_test_folder) 
                                  if f.startswith("Defect_L") and f.endswith(".npy")])
    
    # ペアの作成
    noise_pairs = []
    for data_file in noise_data_files:
        label_file = data_file.replace("NoiseDefectFree_", "").replace(".npy", "_19label.npy")
        label_path = os.path.join(label_data_folder, label_file)
        if os.path.exists(label_path) or data_file.startswith("NoiseDefectFree_"):
            noise_pairs.append((data_file, label_file))
    
    defect_pairs = []
    for defect_file, defect_dir in all_defect_files:
        label_file = defect_file.replace(".npy", "_19label.npy")
        label_path = os.path.join(label_data_folder, label_file)
        if os.path.exists(label_path):
            defect_pairs.append((defect_file, label_file, defect_dir))
    
    # 全データを統合
    all_pairs = []
    all_pairs.extend([(pair[0], pair[1], noise_data_folder) for pair in noise_pairs])
    all_pairs.extend([(pair[0], pair[1], pair[2]) for pair in defect_pairs])
    
    np.random.seed(42)
    random.seed(42)
    np.random.shuffle(all_pairs)
    
    # 分割設定の読み込み
    split_config = load_config_for_splitting('config.yaml')
    group_key = "LBel"
    
    if split_config['split_type'] != 'iid':
        pairs_for_ood = [(p[0], p[1]) for p in all_pairs]
        train_pairs, val_pairs, test_pairs = apply_ood_split(
            pairs_for_ood,
            split_type=split_config['split_type'],
            label_data_folder=label_data_folder,
            test_ratio=0.15,
            val_ratio=0.15,
            seed=42,
            **split_config.get('ood_params', {})
        )
        all_pairs_dict = {p[0]: p[2] for p in all_pairs}
        test_data_folder_map = {p[0]: all_pairs_dict.get(p[0], noise_data_folder) for p in test_pairs}
    else:
        # IID分割
        total_samples = len(all_pairs)
        num_train = int(total_samples * 0.7)
        num_val = int(total_samples * 0.15)
        num_test = total_samples - num_train - num_val
        test_pairs = [(pair[0], pair[1]) for pair in all_pairs[num_train+num_val:]]
        test_data_folder_map = {pair[0]: pair[2] for pair in all_pairs[num_train+num_val:]}
    
    # Cross-edge設定
    cross_edge_config = load_config_for_cross_edges('config.yaml')
    
    # テストデータセットの準備
    print(f"[INFO] Preparing test dataset ({len(test_pairs)} pairs)...")
    test_dataset = prepare_data_with_cross_edges(
        test_pairs, noise_data_folder, label_data_folder,
        x_coords, y_coords, z_coords, edge_index,
        class_weight_multiplier=default_class_weight_multiplier,
        data_folder_map=test_data_folder_map,
        use_cross_edges=cross_edge_config['enabled'],
        cross_edge_k=cross_edge_config['k'],
        cross_edge_method=cross_edge_config['surface_method'],
        max_nodes=13942,
        return_class_weights=False
    )
    
    print(f"[INFO] Test Dataset Size: {len(test_dataset)}")
    
    return test_dataset, test_pairs


def evaluate_model(model, test_dataset, args, device='cuda:0'):
    """モデルを評価"""
    print("[INFO] Evaluating on Test Data...")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    model.eval()
    
    test_preds = []
    test_labels = []
    test_probs = []
    
    # 損失関数の設定（学習時と同じ）
    use_logit_adjust = getattr(args, 'use_logit_adjust', False)
    logit_adjust_tau = getattr(args, 'logit_adjust_tau', 1.0)
    use_log_softmax = getattr(args, 'use_log_softmax', False)
    gamma = getattr(args, 'gamma', default_gamma)
    use_layer_constraint = getattr(args, 'use_layer_constraint', True)
    layer_constraint_weight = getattr(args, 'layer_constraint_weight', 1.0)
    
    # クラス重みの計算（簡易版）
    class_weights = torch.ones(19, dtype=torch.float32)
    
    if use_logit_adjust:
        # LogitAdjustLoss用のpriorを計算（簡易版）
        labels_array = np.concatenate([d.y.cpu().numpy() for d in test_dataset])
        class_counts = np.bincount(labels_array, minlength=19)
        class_prior = torch.tensor(class_counts / (class_counts.sum() + 1e-8), dtype=torch.float, device=device)
        base_loss_fn = LogitAdjustLoss(class_prior=class_prior, tau=logit_adjust_tau).to(device)
    else:
        if use_log_softmax:
            base_loss_fn = FocalLossLogSoftmax(weights=class_weights.to(device), gamma=gamma).to(device)
        else:
            base_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
    
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
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            out = model(batch)
            y = batch.y
            
            if use_layer_constraint:
                loss = loss_fn(out, y, batch=batch.batch)
            else:
                loss = loss_fn(out, y)
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total_samples += y.size(0)
            
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(y.cpu().numpy())
            probs = torch.softmax(out, dim=1).cpu().numpy()
            test_probs.extend(probs)
    
    test_loss = total_loss / len(test_loader)
    test_accuracy = correct / total_samples
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    
    # 基本メトリクス
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
    
    # Localization Metrics
    try:
        print(f"\n[INFO] Calculating localization metrics...")
        all_coordinates = []
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            coords = batch.x[:, :3].cpu().numpy()
            all_coordinates.append(coords)
        all_coordinates = np.vstack(all_coordinates)
        
        loc_metrics = calculate_localization_metrics(
            test_probs, test_labels, all_coordinates,
            top_k_list=[1, 3, 5, 10],
            defect_class_ids=set(range(1, 19))
        )
        
        print(f"\n=== Localization Metrics ===")
        print(f"Top-1 Accuracy: {loc_metrics['top_k_accuracy'][1]:.4f}")
        print(f"Top-3 Accuracy: {loc_metrics['top_k_accuracy'][3]:.4f}")
        print(f"Top-5 Accuracy: {loc_metrics['top_k_accuracy'][5]:.4f}")
        print(f"Top-10 Accuracy: {loc_metrics['top_k_accuracy'][10]:.4f}")
        print(f"Mean Distance Error: {loc_metrics['distance_error']['mean']:.4f}")
        print(f"Median Distance Error: {loc_metrics['distance_error']['median']:.4f}")
        print(f"Max Distance Error: {loc_metrics['distance_error']['max']:.4f}")
        print(f"AUPRC: {loc_metrics['auprc']:.4f}")
    except Exception as e:
        print(f"[WARN] Failed to calculate localization metrics: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'balanced_acc': balanced_acc,
        'mcc': mcc
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluation-only script for saved GNN model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Number of hidden channels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--edge_drop_prob', type=float, default=0.01, help='Edge dropout probability')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--use_logit_adjust', action='store_true', default=False, help='Use LogitAdjustLoss')
    parser.add_argument('--logit_adjust_tau', type=float, default=1.0, help='Tau parameter for LogitAdjustLoss')
    parser.add_argument('--use_log_softmax', action='store_true', default=False, help='Use log_softmax version of FocalLoss')
    parser.add_argument('--gamma', type=float, default=3.0, help='Focal loss gamma parameter')
    parser.add_argument('--use_layer_constraint', action='store_true', default=True, help='Use layer constraint')
    parser.add_argument('--layer_constraint_weight', type=float, default=1.0, help='Layer constraint weight')
    
    args = parser.parse_args()
    
    # デバイスの設定
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"[WARNING] CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"[INFO] Using device: {device}")
    
    # モデルのロード
    model = load_model(
        args.model_path,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        edge_drop_prob=args.edge_drop_prob,
        device=device
    )
    
    # テストデータセットの準備
    test_dataset, test_pairs = prepare_test_dataset(args)
    
    # 評価の実行
    results = evaluate_model(model, test_dataset, args, device=device)
    
    print(f"\n[INFO] Evaluation completed successfully!")
    print(f"[INFO] Macro F1 Score: {results['macro_f1']:.4f}")


if __name__ == '__main__':
    import random
    import torch
    main()
