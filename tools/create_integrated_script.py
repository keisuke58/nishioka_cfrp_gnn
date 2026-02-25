#!/usr/bin/env python3
"""
既存のトレーニングスクリプトに新機能を統合する自動パッチスクリプト

既存の GNN_zscore_sub_noise_defect_free.py を読み込んで、
新機能（OOD分割、Cross-edge、Localization指標）を統合した
新しいファイルを作成します。
"""

import sys
import re
from pathlib import Path

# プロジェクトルート
repo_root = Path(__file__).resolve().parents[1]
script_dir = repo_root / "GNN_hole_2026" / "GNN_program"
original_script = script_dir / "GNN_zscore_sub_noise_defect_free.py"
output_script = script_dir / "GNN_zscore_sub_noise_defect_free_integrated.py"


def create_integrated_script():
    """統合版スクリプトを作成"""
    print(f"[INFO] Reading original script: {original_script}")
    
    if not original_script.exists():
        print(f"[ERROR] Original script not found: {original_script}")
        return 1
    
    # 既存スクリプトを読み込む
    with open(original_script, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"[INFO] Read {len(lines)} lines from original script")
    
    # バックアップを作成（念のため）
    backup_script = script_dir / f"{original_script.stem}_backup_{Path(__file__).stem}.py"
    with open(backup_script, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"[INFO] Created backup: {backup_script}")
    
    # 新しいファイルに書き込む
    new_lines = []
    
    # 1. インポート部分に新機能を追加
    # 最後のimport文の後を探す
    last_import_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')):
            last_import_line = i
    
    # 空行をスキップして、適切な位置を返す
    import_insertion_point = last_import_line + 1
    for i in range(last_import_line + 1, min(last_import_line + 10, len(lines))):
        if lines[i].strip() == '' or lines[i].strip().startswith('#'):
            import_insertion_point = i + 1
        else:
            break
    
    print(f"[INFO] Found import insertion point at line {import_insertion_point + 1}")
    
    # 既存の行をコピー（インポート部分まで）
    new_lines.extend(lines[:import_insertion_point])
    
    # 新機能のインポートを追加
    new_lines.append("\n")
    new_lines.append("# ============================================================================\n")
    new_lines.append("# New Features Integration (OOD Split, Cross-edge, Localization Metrics)\n")
    new_lines.append("# ============================================================================\n")
    new_lines.append("import sys\n")
    new_lines.append("from pathlib import Path\n")
    new_lines.append("repo_root = Path(__file__).resolve().parents[2]\n")
    new_lines.append("sys.path.insert(0, str(repo_root))\n")
    new_lines.append("\n")
    new_lines.append("from gnn_common.integration_utils import (\n")
    new_lines.append("    apply_ood_split,\n")
    new_lines.append("    prepare_data_with_cross_edges,\n")
    new_lines.append("    evaluate_with_localization_metrics,\n")
    new_lines.append("    create_model_with_cross_edges,\n")
    new_lines.append("    load_config_for_splitting,\n")
    new_lines.append("    load_config_for_cross_edges\n")
    new_lines.append(")\n")
    new_lines.append("\n")
    new_lines.append("# ============================================================================\n")
    new_lines.append("\n")
    
    # 残りの行をコピー
    new_lines.extend(lines[import_insertion_point:])
    
    # 2. データ分割部分を置き換え（line ~2996付近）
    # "train/val/testに分割" のコメントを探す
    split_start = None
    split_end = None
    
    for i, line in enumerate(new_lines):
        if 'train/val/testに分割' in line:
            split_start = i
            # 次の主要セクション（グローバル変数の保存）まで
            for j in range(i + 1, min(i + 50, len(new_lines))):
                if 'train_data_folder_map_global' in new_lines[j] and '=' in new_lines[j]:
                    split_end = j + 3  # 3行後まで
                    break
            break
    
    if split_start is not None and split_end is not None:
        print(f"[INFO] Found data split section: lines {split_start + 1}-{split_end + 1}")
        
        # 置き換えコードを準備
        replacement_code = """    # ============================================================================
    # OOD Split Integration
    # ============================================================================
    # OOD分割の設定を読み込む
    split_config = load_config_for_splitting('config.yaml')
    
    # OOD分割を使用する場合
    if split_config['split_type'] != 'iid':
        if rank == 0:
            print(f"[INFO] Using OOD split: {split_config['split_type']}")
        
        # all_pairsは(data_file, label_file, data_dir)のタプルなので、
        # まず(data_file, label_file)のペアに変換
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
        
        # データフォルダマップを再構築
        all_pairs_dict = {p[0]: p[2] for p in all_pairs}
        train_data_folder_map = {p[0]: all_pairs_dict.get(p[0], noise_data_folder) for p in train_pairs}
        val_data_folder_map = {p[0]: all_pairs_dict.get(p[0], noise_data_folder) for p in val_pairs}
        test_data_folder_map = {p[0]: all_pairs_dict.get(p[0], noise_data_folder) for p in test_pairs}
    else:
        # 既存の分割ロジックを使用（IID分割）
        # train/val/testに分割
        # 重要: group単位で分割して leakage (val∩test など) をゼロにする
        enforce_disjoint_groups = getattr(args, "enforce_disjoint_groups", True)
        group_key = getattr(args, "group_key", "LBel")
        if enforce_disjoint_groups:
            train_pairs, val_pairs, test_pairs, train_data_folder_map, val_data_folder_map, test_data_folder_map = group_disjoint_split(
                all_pairs,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                group_key=group_key,
                seed=42,
            )
        else:
            total_samples = len(all_pairs)
            num_train = int(total_samples * 0.7)
            num_val = int(total_samples * 0.15)
            num_test = total_samples - num_train - num_val
            train_pairs = [(pair[0], pair[1]) for pair in all_pairs[:num_train]]
            val_pairs = [(pair[0], pair[1]) for pair in all_pairs[num_train:num_train+num_val]]
            test_pairs = [(pair[0], pair[1]) for pair in all_pairs[num_train+num_val:]]
            # データフォルダのマッピングを保存（prepare_dataで使用するため）
            train_data_folder_map = {pair[0]: pair[2] for pair in all_pairs[:num_train]}
            val_data_folder_map = {pair[0]: pair[2] for pair in all_pairs[num_train:num_train+num_val]}
            test_data_folder_map = {pair[0]: pair[2] for pair in all_pairs[num_train+num_val:]}
    
"""
        
        # 置き換え
        new_lines = new_lines[:split_start] + [replacement_code] + new_lines[split_end:]
        print(f"[INFO] Replaced data split section")
    else:
        print(f"[WARN] Could not find data split section, skipping...")
    
    # 3. データ準備部分を置き換え（line ~3143付近）
    prep_start = None
    prep_end = None
    
    for i, line in enumerate(new_lines):
        if 'Preparing train dataset' in line and 'print' in line:
            prep_start = i - 1  # 少し前から
            # 3つのprepare_data呼び出しを見つける
            prepare_count = 0
            for j in range(i, min(i + 20, len(new_lines))):
                if 'prepare_data(' in new_lines[j]:
                    prepare_count += 1
                    if prepare_count == 3:
                        prep_end = j + 1
                        break
            break
    
    if prep_start is not None and prep_end is not None:
        print(f"[INFO] Found data preparation section: lines {prep_start + 1}-{prep_end + 1}")
        
        # Cross-edge設定を読み込むコードを追加
        prep_code = """    # ============================================================================
    # Cross-edge Integration
    # ============================================================================
    # Cross-edge設定を読み込む
    cross_edge_config = load_config_for_cross_edges('config.yaml')
    
    if rank == 0:
        if cross_edge_config['enabled']:
            print(f"[INFO] Cross-edges enabled: k={cross_edge_config['k']}, method={cross_edge_config['surface_method']}")
        else:
            print("[INFO] Cross-edges disabled")
    
    # データセットの準備（Cross-edge対応）
    class_weight_multiplier = getattr(args, 'class_weight_multiplier', default_class_weight_multiplier)
    if rank == 0:
        print(f"\\nPreparing train dataset ({len(train_pairs)} pairs)...")
    train_dataset, class_weights = prepare_data_with_cross_edges(
        train_pairs, noise_data_folder, label_data_folder,
        x_coords, y_coords, z_coords, edge_index,
        class_weight_multiplier=class_weight_multiplier,
        data_folder_map=train_data_folder_map_global,
        use_cross_edges=cross_edge_config['enabled'],
        cross_edge_k=cross_edge_config['k'],
        cross_edge_method=cross_edge_config['surface_method'],
        max_nodes=13942,
        return_class_weights=True
    )
    if rank == 0:
        print(f"Preparing val dataset ({len(val_pairs)} pairs)...")
    val_dataset, _ = prepare_data_with_cross_edges(
        val_pairs, noise_data_folder, label_data_folder,
        x_coords, y_coords, z_coords, edge_index,
        class_weight_multiplier=class_weight_multiplier,
        data_folder_map=val_data_folder_map_global,
        use_cross_edges=cross_edge_config['enabled'],
        cross_edge_k=cross_edge_config['k'],
        cross_edge_method=cross_edge_config['surface_method'],
        max_nodes=13942,
        return_class_weights=False
    )
    if rank == 0:
        print(f"Preparing test dataset ({len(test_pairs)} pairs)...")
    test_dataset, _ = prepare_data_with_cross_edges(
        test_pairs, noise_data_folder, label_data_folder,
        x_coords, y_coords, z_coords, edge_index,
        class_weight_multiplier=class_weight_multiplier,
        data_folder_map=test_data_folder_map_global,
        use_cross_edges=cross_edge_config['enabled'],
        cross_edge_k=cross_edge_config['k'],
        cross_edge_method=cross_edge_config['surface_method'],
        max_nodes=13942,
        return_class_weights=False
    )
"""
        
        # 置き換え
        new_lines = new_lines[:prep_start] + [prep_code] + new_lines[prep_end:]
        print(f"[INFO] Replaced data preparation section")
    else:
        print(f"[WARN] Could not find data preparation section, skipping...")
    
    # 4. 評価部分にLocalization指標を追加（line ~4277付近）
    eval_start = None
    
    for i, line in enumerate(new_lines):
        if 'Evaluating on Test Data' in line or ('test_loader_eval' in line and 'DataLoader' in line):
            eval_start = i
            break
    
    if eval_start is not None:
        print(f"[INFO] Found evaluation section: line {eval_start + 1}")
        
        # test_probs = np.array の後にLocalization指標を追加
        for i in range(eval_start, min(eval_start + 150, len(new_lines))):
            if 'test_probs = np.array' in new_lines[i]:
                # その後の数行の後に追加
                insertion_point = i + 5
                
                # Localization指標を追加
                loc_metrics_code = """            # ============================================================================
            # Localization Metrics Integration
            # ============================================================================
            # Localization指標を計算
            try:
                from gnn_common.metrics import calculate_localization_metrics
                
                # 座標データを取得（Localization指標用）
                all_coordinates = []
                for batch_idx, batch in enumerate(test_loader_eval):
                    batch = batch.to(device)
                    coords = batch.x[:, :3].cpu().numpy()
                    all_coordinates.append(coords)
                all_coordinates = np.vstack(all_coordinates)
                
                loc_metrics = calculate_localization_metrics(
                    test_probs, test_labels, all_coordinates,
                    top_k_list=[1, 3, 5, 10],
                    defect_class_ids=set(range(1, 19))
                )
                
                print(f"\\n=== Localization Metrics ===")
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
            
"""
                
                new_lines.insert(insertion_point, loc_metrics_code)
                print(f"[INFO] Added localization metrics to evaluation section at line {insertion_point + 1}")
                break
    
    # 新しいファイルに書き込む
    print(f"[INFO] Writing integrated script to: {output_script}")
    with open(output_script, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"[INFO] ✓ Integrated script created successfully!")
    print(f"[INFO] Original script: {original_script}")
    print(f"[INFO] Integrated script: {output_script}")
    print(f"[INFO] Backup: {backup_script}")
    print(f"\\n[INFO] Next steps:")
    print(f"  1. Review the integrated script: {output_script}")
    print(f"  2. Test with a small run (--epochs 10)")
    print(f"  3. Configure config.yaml for OOD split / Cross-edge if needed")
    
    return 0


if __name__ == "__main__":
    sys.exit(create_integrated_script())
