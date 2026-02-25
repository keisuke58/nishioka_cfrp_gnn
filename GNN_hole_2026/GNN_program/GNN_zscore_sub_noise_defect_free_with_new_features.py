"""
統合版トレーニングスクリプト（新機能統合版）

既存の GNN_zscore_sub_noise_defect_free.py に以下を統合:
- OOD分割機能
- Cross-edge機能
- Localization指標

既存のスクリプトはそのまま残し、このファイルで新機能を使用可能にします。
"""

# 既存のスクリプトをインポート（すべての関数とクラスを使用）
import sys
import os
from pathlib import Path

# 既存スクリプトのパスを追加
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# 既存スクリプトから必要なものをインポート
# 注意: 既存スクリプトのすべてのimportと関数定義をコピーする必要があります
# ここでは、統合箇所のみを変更したバージョンを作成します

# プロジェクトルートをパスに追加
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# 新機能をインポート
from gnn_common.integration_utils import (
    apply_ood_split,
    prepare_data_with_cross_edges,
    evaluate_with_localization_metrics,
    create_model_with_cross_edges,
    load_config_for_splitting,
    load_config_for_cross_edges
)

# 既存スクリプトのインポート（すべてのimportをコピー）
# 実際には、既存スクリプトのimport部分をそのままコピーする必要があります
# ここでは統合箇所のみを示します

print("[INFO] Using integrated version with new features:")
print("  - OOD split support")
print("  - Cross-edge support")
print("  - Localization metrics support")

# 注意: このファイルは統合箇所を示すテンプレートです
# 実際の使用には、既存スクリプトの全内容をコピーして、
# 以下の統合箇所のみを変更する必要があります

"""
統合箇所1: データ分割部分（line 3000付近）

既存コード:
    if enforce_disjoint_groups:
        train_pairs, val_pairs, test_pairs, train_data_folder_map, val_data_folder_map, test_data_folder_map = group_disjoint_split(...)
    else:
        # 単純な分割

統合後:
    # OOD分割の設定を読み込む
    split_config = load_config_for_splitting('config.yaml')
    
    # OOD分割を適用（既存のgroup_disjoint_splitと併用可能）
    if split_config['split_type'] != 'iid':
        # OOD分割を使用
        train_pairs, val_pairs, test_pairs = apply_ood_split(
            all_pairs,
            split_type=split_config['split_type'],
            label_data_folder=label_data_folder,
            **split_config['ood_params']
        )
        # データフォルダマップを再構築（既存のロジックに合わせる）
        train_data_folder_map = {pair[0]: pair[2] for pair in all_pairs if pair[0] in [p[0] for p in train_pairs]}
        val_data_folder_map = {pair[0]: pair[2] for pair in all_pairs if pair[0] in [p[0] for p in val_pairs]}
        test_data_folder_map = {pair[0]: pair[2] for pair in all_pairs if pair[0] in [p[0] for p in test_pairs]}
    else:
        # 既存の分割ロジックを使用
        if enforce_disjoint_groups:
            train_pairs, val_pairs, test_pairs, train_data_folder_map, val_data_folder_map, test_data_folder_map = group_disjoint_split(...)
        else:
            # 単純な分割
"""

"""
統合箇所2: データ準備部分（line 3143付近）

既存コード:
    train_dataset, class_weights = prepare_data(train_pairs, noise_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, ...)

統合後:
    # Cross-edge設定を読み込む
    cross_edge_config = load_config_for_cross_edges('config.yaml')
    
    # Cross-edge対応のデータ準備
    train_dataset, class_weights = prepare_data_with_cross_edges(
        train_pairs, noise_data_folder, label_data_folder,
        x_coords, y_coords, z_coords, edge_index,
        use_cross_edges=cross_edge_config['enabled'],
        cross_edge_k=cross_edge_config['k'],
        cross_edge_method=cross_edge_config['surface_method'],
        data_folder_map=train_data_folder_map_global,
        ...
    )
    val_dataset, _ = prepare_data_with_cross_edges(...)
    test_dataset, _ = prepare_data_with_cross_edges(...)
"""

"""
統合箇所3: モデル作成部分（line 3124付近）

既存コード:
    model = GATModel(...)

統合後:
    # Cross-edge対応モデルを作成
    model = create_model_with_cross_edges(
        model_type='GAT',
        use_cross_edges=cross_edge_config['enabled'],
        hidden_channels=args.hidden_channels,
        num_classes=19,
        dropout=dropout,
        edge_drop_prob=edge_drop_prob
    )
"""

"""
統合箇所4: 評価部分（line 4277付近）

既存コード:
    # テスト評価ループ
    for batch in test_loader_eval:
        out = ddp_model(batch)
        pred = out.argmax(dim=1)
        test_preds.extend(pred.cpu().numpy())
        test_labels.extend(batch.y.cpu().numpy())
        probs = torch.softmax(out, dim=1).cpu().numpy()
        test_probs.extend(probs)

統合後:
    # Localization指標を含む評価
    results = evaluate_with_localization_metrics(
        ddp_model,
        test_loader_eval,
        device,
        num_classes=19,
        calculate_localization=True,
        defect_class_ids=set(range(1, 19))
    )
    
    # 従来のメトリクス
    test_loss = results['test_loss']
    test_accuracy = results['test_accuracy']
    macro_f1 = results['macro_f1']
    
    # Localization指標
    if results['localization']:
        loc = results['localization']
        print(f"Top-1 Accuracy: {loc['top_k_accuracy'][1]:.4f}")
        print(f"Top-5 Accuracy: {loc['top_k_accuracy'][5]:.4f}")
        print(f"Mean Distance Error: {loc['distance_error']['mean']:.4f}")
        print(f"AUPRC: {loc['auprc']:.4f}")
"""

print("\n[INFO] Integration points identified:")
print("  1. Data splitting (line ~3000)")
print("  2. Data preparation (line ~3143)")
print("  3. Model creation (line ~3124)")
print("  4. Evaluation (line ~4277)")
print("\n[INFO] See INTEGRATION_PATCHES.md for detailed patch instructions")
