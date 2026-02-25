#!/usr/bin/env python3
"""
新機能の動作確認スクリプト

OOD分割、Localization指標、Cross-edgeの動作を確認します。
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# プロジェクトルートをパスに追加
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from gnn_common.data_utils import (
    create_ood_split,
    calculate_defect_statistics,
    visualize_split_statistics,
    identify_surface_nodes,
    create_cross_edges,
    add_cross_edges_to_data
)
from gnn_common.metrics import (
    calculate_localization_metrics,
    calculate_top_k_accuracy,
    calculate_distance_error
)
from gnn_common.models import GATModelWithCrossEdges
from torch_geometric.data import Data


def test_ood_split():
    """OOD分割のテスト"""
    print("\n" + "="*80)
    print("Test 1: OOD Split")
    print("="*80)
    
    # ダミーデータでテスト
    print("\n[INFO] Creating dummy data pairs...")
    pairs = [
        (f"Defect_L{i}_B{j}_el{k}_H4_W4.npy", f"Defect_L{i}_B{j}_el{k}_H4_W4_19label.npy")
        for i in range(1, 5)  # L1-L4
        for j in range(1, 3)  # B1-B2
        for k in [100, 200]   # el100, el200
    ]
    
    print(f"[INFO] Created {len(pairs)} dummy pairs")
    
    # 実際のデータがある場合はそれを使用
    label_data_folder = os.environ.get(
        "LABEL_DATA_FOLDER",
        "/home/nishioka/GNN/Defect19Class_OneHot_test3"
    )
    
    if not os.path.exists(label_data_folder):
        print(f"[WARN] Label data folder not found: {label_data_folder}")
        print("[INFO] Skipping OOD split test (requires actual label files)")
        return True
    
    # 実際に存在するペアのみをフィルタ
    existing_pairs = []
    for data_file, label_file in pairs:
        label_path = os.path.join(label_data_folder, label_file)
        if os.path.exists(label_path):
            existing_pairs.append((data_file, label_file))
    
    if len(existing_pairs) == 0:
        print("[WARN] No existing pairs found. Using first few pairs for testing...")
        existing_pairs = pairs[:10]  # 最初の10個を使用
    
    print(f"[INFO] Testing with {len(existing_pairs)} pairs")
    
    try:
        # IID分割のテスト
        print("\n[INFO] Testing IID split...")
        train_pairs, val_pairs, test_pairs = create_ood_split(
            existing_pairs,
            split_type='iid',
            test_ratio=0.2,
            val_ratio=0.1,
            seed=42
        )
        print(f"[OK] IID split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
        
        # OOD分割のテスト（欠陥サイズ）
        print("\n[INFO] Testing OOD split (defect_size)...")
        train_pairs_ood, val_pairs_ood, test_pairs_ood = create_ood_split(
            existing_pairs,
            split_type='defect_size',
            label_data_folder=label_data_folder,
            size_threshold=3,
            test_ratio=0.2,
            val_ratio=0.1,
            seed=42
        )
        print(f"[OK] OOD split (defect_size): train={len(train_pairs_ood)}, val={len(val_pairs_ood)}, test={len(test_pairs_ood)}")
        
        # 統計情報の計算
        print("\n[INFO] Calculating defect statistics...")
        stats = calculate_defect_statistics(existing_pairs, label_data_folder)
        print(f"[OK] Calculated statistics for {len(stats)} files")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] OOD split test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_localization_metrics():
    """Localization指標のテスト"""
    print("\n" + "="*80)
    print("Test 2: Localization Metrics")
    print("="*80)
    
    try:
        # ダミーデータを作成
        print("\n[INFO] Creating dummy data...")
        num_samples = 100
        num_classes = 19
        
        # 予測確率（ランダム）
        predictions = np.random.rand(num_samples, num_classes)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)  # 正規化
        
        # 正解ラベル（0-18のクラス）
        labels = np.random.randint(0, num_classes, num_samples)
        
        # 座標（3D）
        coordinates = np.random.rand(num_samples, 3) * 10  # 0-10の範囲
        
        # Top-k Accuracyのテスト
        print("\n[INFO] Testing Top-k Accuracy...")
        top_k_acc = calculate_top_k_accuracy(
            predictions, labels, top_k_list=[1, 3, 5, 10]
        )
        print(f"[OK] Top-k Accuracy: {top_k_acc}")
        
        # Distance Errorのテスト
        print("\n[INFO] Testing Distance Error...")
        dist_error = calculate_distance_error(
            predictions, labels, coordinates,
            defect_class_ids=set(range(1, num_classes))
        )
        print(f"[OK] Distance Error: {dist_error}")
        
        # 統合指標のテスト
        print("\n[INFO] Testing integrated localization metrics...")
        metrics = calculate_localization_metrics(
            predictions, labels, coordinates,
            top_k_list=[1, 3, 5, 10]
        )
        print(f"[OK] Localization metrics:")
        print(f"  Top-k Accuracy: {metrics['top_k_accuracy']}")
        print(f"  Distance Error: {metrics['distance_error']}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Localization metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_edges():
    """Cross-edgeのテスト"""
    print("\n" + "="*80)
    print("Test 3: Cross-edges")
    print("="*80)
    
    try:
        # ダミーデータを作成
        print("\n[INFO] Creating dummy graph data...")
        num_nodes = 100
        
        # 座標（3D）
        coordinates = np.random.rand(num_nodes, 3) * 10
        z_coords = coordinates[:, 2]
        
        # Surfaceノードの識別
        print("\n[INFO] Identifying surface nodes...")
        surface_mask = identify_surface_nodes(
            coordinates, z_coords,
            method='outer_layer'
        )
        num_surface = surface_mask.sum()
        print(f"[OK] Identified {num_surface} surface nodes out of {num_nodes}")
        
        if num_surface == 0:
            print("[WARN] No surface nodes found. Adjusting z_coords...")
            # z座標を調整してsurfaceノードを確保
            z_coords = np.linspace(0, 10, num_nodes)
            coordinates[:, 2] = z_coords
            surface_mask = identify_surface_nodes(
                coordinates, z_coords,
                method='outer_layer'
            )
            num_surface = surface_mask.sum()
            print(f"[OK] After adjustment: {num_surface} surface nodes")
        
        # Cross-edgeの作成
        print("\n[INFO] Creating cross-edges...")
        cross_edge_index = create_cross_edges(
            coordinates, surface_mask, k=1
        )
        print(f"[OK] Created {cross_edge_index.size(1)} cross-edges")
        
        # Dataオブジェクトへの追加
        print("\n[INFO] Adding cross-edges to Data object...")
        # ダミーのノード特徴量とエッジ
        x = torch.randn(num_nodes, 4)  # [N, 4] (x, y, z, feature)
        edge_index = torch.randint(0, num_nodes, (2, 50))  # 既存のエッジ
        
        data = Data(x=x, edge_index=edge_index)
        data = add_cross_edges_to_data(
            data, coordinates, surface_mask, k=1
        )
        
        print(f"[OK] Data object updated:")
        print(f"  Original edges: {edge_index.size(1)}")
        print(f"  Total edges (with cross-edges): {data.edge_index.size(1)}")
        print(f"  Cross-edges added: {data.edge_index.size(1) - edge_index.size(1)}")
        
        # モデルのテスト
        print("\n[INFO] Testing GATModelWithCrossEdges...")
        model = GATModelWithCrossEdges(
            hidden_channels=16,
            num_classes=19,
            num_heads=2,
            dropout=0.1
        )
        
        # フォワードパス
        output = model(data)
        print(f"[OK] Model forward pass successful: output shape = {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Cross-edges test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("="*80)
    print("New Features Test Suite")
    print("="*80)
    
    results = []
    
    # テスト実行
    results.append(("OOD Split", test_ood_split()))
    results.append(("Localization Metrics", test_localization_metrics()))
    results.append(("Cross-edges", test_cross_edges()))
    
    # 結果サマリー
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
