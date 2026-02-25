#!/usr/bin/env python3
"""
ノイズありの欠陥なしデータセットのラベルを作成するスクリプト

方針:
1. 既存のデータファイル（all_sub_hole_defect_zscore/train, val, test）を取得
2. 1つのラベルファイルを作成（全て0、欠陥なし）
3. ラベルは全て0（欠陥なし）で、形状は (num_nodes, 19) のone-hot形式
4. 各データファイルに対応するラベルファイル名でシンボリックリンクまたはコピーを作成
5. 既存のラベルファイルと同じディレクトリ（all_19class_label）に保存
"""

import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil

def get_data_files(data_base_dir):
    """データファイルのリストを取得"""
    train_dir = os.path.join(data_base_dir, "train")
    val_dir = os.path.join(data_base_dir, "val")
    test_dir = os.path.join(data_base_dir, "test")
    
    train_files = []
    val_files = []
    test_files = []
    
    if os.path.exists(train_dir):
        train_files = [f for f in os.listdir(train_dir) 
                      if f.startswith("Defect_L") and f.endswith(".npy")]
    
    if os.path.exists(val_dir):
        val_files = [f for f in os.listdir(val_dir) 
                    if f.startswith("Defect_L") and f.endswith(".npy")]
    
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) 
                     if f.startswith("Defect_L") and f.endswith(".npy")]
    
    return train_files, val_files, test_files

def get_num_nodes(data_file_path):
    """データファイルからノード数を取得"""
    try:
        data = np.load(data_file_path)
        # データファイルの形状を確認
        if len(data.shape) == 1:
            # 1次元配列の場合
            num_nodes = len(data)
        elif len(data.shape) == 2:
            # 2次元配列の場合（行数がノード数）
            num_nodes = data.shape[0]
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        # 13942で制限（既存のコードと同じ）
        num_nodes = min(num_nodes, 13942)
        return num_nodes
    except Exception as e:
        print(f"Error loading data file {data_file_path}: {e}")
        return None

def create_zero_label(num_nodes, num_classes=19):
    """
    全て0のラベル（欠陥なし）を作成
    
    Args:
        num_nodes: ノード数
        num_classes: クラス数（デフォルト: 19）
    
    Returns:
        label: (num_nodes, num_classes) のone-hot形式のラベル配列
               全てのノードがクラス0（欠陥なし）を表す
    """
    # one-hot形式: クラス0は [1, 0, 0, ..., 0]
    label = np.zeros((num_nodes, num_classes), dtype=np.int64)
    label[:, 0] = 1  # 最初の列（クラス0）を1に設定
    return label

def create_labels_for_dataset(data_files, data_dir, label_output_dir, master_label_path, split_name="train", use_symlink=True):
    """
    データセット（train/val/test）のラベルを作成
    1つのマスターラベルファイルから、各データファイルに対応するラベルファイルを作成
    
    Args:
        data_files: データファイル名のリスト
        data_dir: データファイルが存在するディレクトリ
        label_output_dir: ラベルファイルを保存するディレクトリ
        master_label_path: マスターラベルファイルのパス
        split_name: データセットの名前（train/val/test）
        use_symlink: シンボリックリンクを使用するか（Falseの場合はコピー）
    """
    print(f"\n=== Processing {split_name} dataset ===")
    print(f"Found {len(data_files)} data files")
    
    created_count = 0
    skipped_count = 0
    error_count = 0
    
    for data_file in tqdm(data_files, desc=f"Creating labels for {split_name}"):
        # ベース名を取得（.npyを除く）
        base_name = os.path.splitext(data_file)[0]
        
        # ラベルファイル名を作成
        label_file_name = f"{base_name}_19label.npy"
        label_file_path = os.path.join(label_output_dir, label_file_name)
        
        # 既にラベルファイルが存在する場合はスキップ
        if os.path.exists(label_file_path):
            skipped_count += 1
            continue
        
        # マスターラベルファイルからシンボリックリンクまたはコピーを作成
        try:
            if use_symlink:
                # シンボリックリンクを作成
                os.symlink(os.path.abspath(master_label_path), label_file_path)
            else:
                # コピーを作成
                shutil.copy2(master_label_path, label_file_path)
            created_count += 1
        except Exception as e:
            print(f"Error creating label file {label_file_path}: {e}")
            error_count += 1
    
    print(f"  Created: {created_count} labels")
    print(f"  Skipped (already exists): {skipped_count} labels")
    print(f"  Errors: {error_count} labels")
    
    return created_count, skipped_count, error_count

def main():
    parser = argparse.ArgumentParser(description="Create defect-free labels for noise dataset")
    parser.add_argument("--data_base_dir", type=str, 
                       default="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore",
                       help="Base directory containing train/val/test subdirectories")
    parser.add_argument("--label_output_dir", type=str,
                       default="/home/nishioka/GNN/GNN_hole_2026/all_19class_label",
                       help="Directory to save label files")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"],
                       default="all", help="Which split to process (default: all)")
    parser.add_argument("--num_nodes", type=int, default=13942,
                       help="Number of nodes in the label (default: 13942)")
    parser.add_argument("--use_symlink", action="store_true", default=False,
                       help="Use symbolic links instead of copying files")
    
    args = parser.parse_args()
    
    # ディレクトリの存在確認
    if not os.path.exists(args.data_base_dir):
        print(f"Error: Data base directory does not exist: {args.data_base_dir}")
        return
    
    # ラベル出力ディレクトリを作成（存在しない場合）
    os.makedirs(args.label_output_dir, exist_ok=True)
    print(f"Label output directory: {args.label_output_dir}")
    
    # マスターラベルファイルを作成（1つだけ）
    master_label_name = "noise_defect_free_19label.npy"
    master_label_path = os.path.join(args.label_output_dir, master_label_name)
    
    if not os.path.exists(master_label_path):
        print(f"\n=== Creating master label file ===")
        print(f"Master label file: {master_label_path}")
        print(f"Number of nodes: {args.num_nodes}")
        
        # ラベルを作成（全て0、欠陥なし）
        label = create_zero_label(args.num_nodes, num_classes=19)
        
        # マスターラベルファイルを保存
        np.save(master_label_path, label)
        print(f"✓ Master label file created")
        print(f"  Shape: {label.shape}")
        print(f"  Dtype: {label.dtype}")
        print(f"  All zeros (class 0): {np.all(label[:, 0] == 1) and np.all(label[:, 1:] == 0)}")
    else:
        print(f"\n=== Master label file already exists ===")
        print(f"Master label file: {master_label_path}")
        label = np.load(master_label_path)
        print(f"  Shape: {label.shape}")
        print(f"  Dtype: {label.dtype}")
    
    # データファイルを取得
    train_files, val_files, test_files = get_data_files(args.data_base_dir)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    total_created = 0
    total_skipped = 0
    total_errors = 0
    
    # 各データセットのラベルを作成（マスターラベルファイルから）
    if args.split in ["train", "all"]:
        train_dir = os.path.join(args.data_base_dir, "train")
        created, skipped, errors = create_labels_for_dataset(
            train_files, train_dir, args.label_output_dir, master_label_path, "train", args.use_symlink
        )
        total_created += created
        total_skipped += skipped
        total_errors += errors
    
    if args.split in ["val", "all"]:
        val_dir = os.path.join(args.data_base_dir, "val")
        created, skipped, errors = create_labels_for_dataset(
            val_files, val_dir, args.label_output_dir, master_label_path, "val", args.use_symlink
        )
        total_created += created
        total_skipped += skipped
        total_errors += errors
    
    if args.split in ["test", "all"]:
        test_dir = os.path.join(args.data_base_dir, "test")
        created, skipped, errors = create_labels_for_dataset(
            test_files, test_dir, args.label_output_dir, master_label_path, "test", args.use_symlink
        )
        total_created += created
        total_skipped += skipped
        total_errors += errors
    
    print(f"\n=== Summary ===")
    print(f"Total created: {total_created} labels")
    print(f"Total skipped: {total_skipped} labels")
    print(f"Total errors: {total_errors} labels")
    
    # サンプルラベルファイルを確認
    if total_created > 0:
        print(f"\n=== Verifying created labels ===")
        # 最初に作成されたラベルファイルを確認
        sample_files = []
        if args.split in ["train", "all"] and len(train_files) > 0:
            base_name = os.path.splitext(train_files[0])[0]
            sample_files.append(os.path.join(args.label_output_dir, f"{base_name}_19label.npy"))
        elif args.split in ["val", "all"] and len(val_files) > 0:
            base_name = os.path.splitext(val_files[0])[0]
            sample_files.append(os.path.join(args.label_output_dir, f"{base_name}_19label.npy"))
        elif args.split in ["test", "all"] and len(test_files) > 0:
            base_name = os.path.splitext(test_files[0])[0]
            sample_files.append(os.path.join(args.label_output_dir, f"{base_name}_19label.npy"))
        
        for sample_file in sample_files:
            if os.path.exists(sample_file):
                label = np.load(sample_file)
                print(f"Sample label file: {os.path.basename(sample_file)}")
                print(f"  Shape: {label.shape}")
                print(f"  Dtype: {label.dtype}")
                print(f"  Min: {label.min()}, Max: {label.max()}")
                print(f"  First row: {label[0]}")
                print(f"  All zeros (class 0): {np.all(label[:, 0] == 1) and np.all(label[:, 1:] == 0)}")

if __name__ == "__main__":
    main()
