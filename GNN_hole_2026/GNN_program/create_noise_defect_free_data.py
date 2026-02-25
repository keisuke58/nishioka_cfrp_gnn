#!/usr/bin/env python3
"""
ノイズありの欠陥なしデータセットを作成するスクリプト

方針:
1. 既存のデータファイル（all_sub_hole_defect_zscore/train, val, test）を取得
2. 各データファイルにGaussian noiseを追加して、欠陥なしデータを作成
3. ノイズ強度は調整可能（データの標準偏差に対する相対値、または絶対値）
4. 1つのラベルファイル（noise_defect_free_19label.npy）を作成（全てクラス0）
5. 新しいデータファイルとラベルファイルを保存
"""

import os
import numpy as np
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

def analyze_data_statistics(data_files, data_dir, num_samples=10):
    """
    データファイルの統計情報を分析して、適切なノイズ強度を提案
    
    Args:
        data_files: データファイル名のリスト
        data_dir: データファイルが存在するディレクトリ
        num_samples: 分析に使用するサンプル数
    
    Returns:
        stats: 統計情報の辞書（mean, std, min, max）
    """
    print(f"\n=== Analyzing data statistics (using {min(num_samples, len(data_files))} samples) ===")
    
    all_data = []
    sample_files = data_files[:min(num_samples, len(data_files))]
    
    for data_file in sample_files:
        data_file_path = os.path.join(data_dir, data_file)
        try:
            data = np.load(data_file_path)
            if len(data.shape) == 1:
                all_data.append(data)
            elif len(data.shape) == 2:
                all_data.append(data.flatten())
        except Exception as e:
            print(f"Warning: Could not load {data_file}: {e}")
            continue
    
    if len(all_data) == 0:
        return None
    
    all_data = np.concatenate(all_data)
    
    stats = {
        'mean': np.mean(all_data),
        'std': np.std(all_data),
        'min': np.min(all_data),
        'max': np.max(all_data),
        'median': np.median(all_data),
        'q25': np.percentile(all_data, 25),
        'q75': np.percentile(all_data, 75)
    }
    
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  Min: {stats['min']:.6f}")
    print(f"  Max: {stats['max']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  Q25: {stats['q25']:.6f}, Q75: {stats['q75']:.6f}")
    
    # ノイズ強度の提案
    print(f"\n=== Suggested noise intensities ===")
    print(f"  Relative to std (noise_std_ratio):")
    print(f"    - Light noise: 0.05-0.1 * std = {0.05*stats['std']:.6f} - {0.1*stats['std']:.6f}")
    print(f"    - Medium noise: 0.1-0.2 * std = {0.1*stats['std']:.6f} - {0.2*stats['std']:.6f}")
    print(f"    - Heavy noise: 0.2-0.5 * std = {0.2*stats['std']:.6f} - {0.5*stats['std']:.6f}")
    
    return stats

def add_gaussian_noise(data, noise_std, noise_mean=0.0, seed=None):
    """
    Gaussian noiseをデータに追加
    
    Args:
        data: 入力データ（1次元または2次元配列）
        noise_std: ノイズの標準偏差
        noise_mean: ノイズの平均（デフォルト: 0.0）
        seed: 乱数シード（デフォルト: None）
    
    Returns:
        noisy_data: ノイズが追加されたデータ
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(noise_mean, noise_std, size=data.shape)
    noisy_data = data + noise
    
    return noisy_data

def create_noise_data_from_zero(noise_std, noise_mean=0.0, num_nodes=13942, seed=None):
    """
    ゼロベクトル（全て0）にノイズを追加した欠陥なしデータを作成
    
    Args:
        noise_std: ノイズの標準偏差
        noise_mean: ノイズの平均（デフォルト: 0.0）
        num_nodes: ノード数（デフォルト: 13942）
        seed: 乱数シード（デフォルト: None）
    
    Returns:
        noisy_data: ノイズが追加されたデータ（全て0のベースにノイズを追加）
    """
    # ゼロベクトルを作成
    zero_data = np.zeros(num_nodes, dtype=np.float32)
    
    # ノイズを追加
    noisy_data = add_gaussian_noise(zero_data, noise_std, noise_mean, seed)
    
    return noisy_data

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

def create_noise_dataset_from_zero(output_dir, noise_std, noise_mean=0.0, 
                                   num_nodes=13942, num_samples=6000, seed=None):
    """
    ゼロベクトル（全て0）にノイズを追加した欠陥なしデータセットを作成
    
    Args:
        output_dir: 出力ディレクトリ
        noise_std: ノイズの標準偏差
        noise_mean: ノイズの平均（デフォルト: 0.0）
        num_nodes: ノード数（デフォルト: 13942）
        num_samples: 作成するサンプル数（デフォルト: 6000）
        seed: 乱数シード（デフォルト: None）
    
    Returns:
        created_count: 作成されたファイル数
        error_count: エラー数
        created_files: 作成されたファイル名のリスト
    """
    print(f"\n=== Creating defect-free noise dataset ===")
    print(f"Number of samples: {num_samples}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Noise std: {noise_std:.6f}, Noise mean: {noise_mean:.6f}")
    
    created_count = 0
    error_count = 0
    created_files = []
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in tqdm(range(num_samples), desc="Creating defect-free samples"):
        # 新しいファイル名を生成（NoiseDefectFree_000001.npy形式）
        new_filename = f"NoiseDefectFree_{idx+1:06d}.npy"
        output_file_path = os.path.join(output_dir, new_filename)
        
        # 既に存在する場合はスキップ
        if os.path.exists(output_file_path):
            created_files.append(new_filename)
            continue
        
        # ゼロベクトルにノイズを追加したデータを作成
        # 各ファイルごとに異なるシードを使用（再現性のため）
        file_seed = seed + idx if seed is not None else None
        noisy_data = create_noise_data_from_zero(noise_std, noise_mean, num_nodes, file_seed)
        
        if noisy_data is None:
            error_count += 1
            continue
        
        # データを保存
        try:
            np.save(output_file_path, noisy_data)
            created_files.append(new_filename)
            created_count += 1
        except Exception as e:
            print(f"Error saving {output_file_path}: {e}")
            error_count += 1
    
    print(f"  Created: {created_count} files")
    print(f"  Errors: {error_count} files")
    
    return created_count, error_count, created_files

def main():
    parser = argparse.ArgumentParser(description="Create defect-free dataset with Gaussian noise")
    parser.add_argument("--data_base_dir", type=str, 
                       default="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore",
                       help="Base directory containing train/val/test subdirectories")
    parser.add_argument("--output_base_dir", type=str,
                       default="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise",
                       help="Base directory for output noise data")
    parser.add_argument("--label_output_dir", type=str,
                       default="/home/nishioka/GNN/GNN_hole_2026/all_19class_label",
                       help="Directory to save label files")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"],
                       default="all", help="Which split to process (default: all)")
    
    # ノイズ強度の設定（2つの方法から選択）
    parser.add_argument("--noise_std_ratio", type=float, default=None,
                       help="Noise std as ratio of data std (e.g., 0.1 means 10%% of data std)")
    parser.add_argument("--noise_std_absolute", type=float, default=None,
                       help="Absolute noise std value")
    parser.add_argument("--noise_mean", type=float, default=0.0,
                       help="Noise mean (default: 0.0)")
    parser.add_argument("--max_nodes", type=int, default=13942,
                       help="Maximum number of nodes (default: 13942)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze data statistics without creating dataset")
    parser.add_argument("--num_samples", type=int, default=6000,
                       help="Number of defect-free samples to create from zero (default: 6000)")
    
    args = parser.parse_args()
    
    # ディレクトリの存在確認
    if not os.path.exists(args.data_base_dir):
        print(f"Error: Data base directory does not exist: {args.data_base_dir}")
        return
    
    # データファイルを取得
    train_files, val_files, test_files = get_data_files(args.data_base_dir)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    # データ統計を分析
    if len(train_files) > 0:
        train_dir = os.path.join(args.data_base_dir, "train")
        stats = analyze_data_statistics(train_files, train_dir, num_samples=20)
    else:
        print("Warning: No train files found for statistics analysis")
        stats = None
    
    if args.analyze_only:
        print("\n=== Analysis only mode - exiting ===")
        return
    
    # ノイズ強度を決定
    if args.noise_std_ratio is not None and args.noise_std_absolute is not None:
        print("Error: Cannot specify both noise_std_ratio and noise_std_absolute")
        return
    
    if args.noise_std_ratio is not None:
        if stats is None:
            print("Error: Cannot use noise_std_ratio without data statistics")
            return
        noise_std = args.noise_std_ratio * stats['std']
        print(f"\n=== Noise intensity ===")
        print(f"Using noise_std_ratio: {args.noise_std_ratio}")
        print(f"Data std: {stats['std']:.6f}")
        print(f"Calculated noise std: {noise_std:.6f}")
    elif args.noise_std_absolute is not None:
        noise_std = args.noise_std_absolute
        print(f"\n=== Noise intensity ===")
        print(f"Using absolute noise std: {noise_std:.6f}")
    else:
        # デフォルト: データの標準偏差の10%
        if stats is None:
            print("Error: Cannot determine default noise std without data statistics")
            print("Please specify --noise_std_ratio or --noise_std_absolute")
            return
        noise_std = 0.1 * stats['std']
        print(f"\n=== Noise intensity (default: 10%% of data std) ===")
        print(f"Data std: {stats['std']:.6f}")
        print(f"Default noise std: {noise_std:.6f}")
        print("(Use --noise_std_ratio or --noise_std_absolute to override)")
    
    # ラベル出力ディレクトリを作成
    os.makedirs(args.label_output_dir, exist_ok=True)
    
    # マスターラベルファイルを作成（1つだけ）
    master_label_name = "noise_defect_free_19label.npy"
    master_label_path = os.path.join(args.label_output_dir, master_label_name)
    
    if not os.path.exists(master_label_path):
        print(f"\n=== Creating master label file ===")
        print(f"Master label file: {master_label_path}")
        print(f"Number of nodes: {args.max_nodes}")
        
        # ラベルを作成（全て0、欠陥なし）
        label = create_zero_label(args.max_nodes, num_classes=19)
        
        # マスターラベルファイルを保存
        np.save(master_label_path, label)
        print(f"✓ Master label file created")
        print(f"  Shape: {label.shape}")
        print(f"  Dtype: {label.dtype}")
        print(f"  All zeros (class 0): {np.all(label[:, 0] == 1) and np.all(label[:, 1:] == 0)}")
    else:
        print(f"\n=== Master label file already exists ===")
        print(f"Master label file: {master_label_path}")
    
    total_created = 0
    total_errors = 0
    all_created_files = []
    
    # num_samplesが指定されている場合、ゼロベクトルから欠陥なしデータを作成
    if args.num_samples is not None:
        # trainディレクトリに全て保存
        train_output_dir = os.path.join(args.output_base_dir, "train")
        
        created_count, error_count, created_files = create_noise_dataset_from_zero(
            train_output_dir, noise_std, args.noise_mean, args.max_nodes, 
            args.num_samples, args.seed
        )
        
        total_created = created_count
        total_errors = error_count
        all_created_files = [os.path.splitext(f)[0] for f in created_files]
    else:
        # 従来の動作（各splitごとに処理）- 既存のデータファイルから作成
        if args.split in ["train", "all"] and len(train_files) > 0:
            train_dir = os.path.join(args.data_base_dir, "train")
            train_output_dir = os.path.join(args.output_base_dir, "train")
            # この場合は既存の関数を使う（必要に応じて実装）
            print("Warning: num_samples not specified, using existing data files")
            print("To create defect-free data from zero, please specify --num_samples")
    
    # 各データファイルに対応するラベルファイルを作成（マスターラベルファイルから）
    print(f"\n=== Creating label files for noise dataset ===")
    label_created = 0
    label_skipped = 0
    
    # 作成されたデータファイルのリストを使用
    if len(all_created_files) == 0:
        # 従来の動作（既存のファイル名を使用）
        all_data_files = []
        if args.split in ["train", "all"] and len(train_files) > 0:
            all_data_files.extend(train_files)
        if args.split in ["val", "all"] and len(val_files) > 0:
            all_data_files.extend(val_files)
        if args.split in ["test", "all"] and len(test_files) > 0:
            all_data_files.extend(test_files)
        all_created_files = [os.path.splitext(f)[0] for f in all_data_files]
    
    for base_name in tqdm(all_created_files, desc="Creating label files"):
        label_file_name = f"{base_name}_19label.npy"
        label_file_path = os.path.join(args.label_output_dir, label_file_name)
        
        # 既に存在する場合はスキップ
        if os.path.exists(label_file_path):
            label_skipped += 1
            continue
        
        # マスターラベルファイルからシンボリックリンクまたはコピーを作成
        try:
            # シンボリックリンクを作成（相対パスで）
            rel_path = os.path.relpath(master_label_path, args.label_output_dir)
            os.symlink(rel_path, label_file_path)
            label_created += 1
        except Exception as e:
            # シンボリックリンクが失敗した場合はコピー
            try:
                shutil.copy2(master_label_path, label_file_path)
                label_created += 1
            except Exception as e2:
                print(f"Error creating label file {label_file_path}: {e2}")
    
    print(f"  Created: {label_created} label files")
    print(f"  Skipped: {label_skipped} label files")
    
    print(f"\n=== Summary ===")
    print(f"Total created: {total_created} data files")
    print(f"Total errors: {total_errors} data files")
    print(f"Label files created: {label_created}")
    print(f"Master label file: {master_label_path}")
    print(f"\nOutput directory: {args.output_base_dir}")
    print(f"Label directory: {args.label_output_dir}")

if __name__ == "__main__":
    main()
