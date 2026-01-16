import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob

# パスの設定
source_dir = "/home/nishioka/GNN/GNN_hole_2026/Defect_hole_4x4_Region1_21_npy"
output_dir = "/home/nishioka/GNN/GNN_hole_2026/Defect_hole_4x4_Region1_21_npy_normalized"

# 出力ディレクトリの作成
os.makedirs(output_dir, exist_ok=True)

# 全ファイルの取得
npy_files = sorted(glob.glob(os.path.join(source_dir, "*.npy")))
print(f"見つかったファイル数: {len(npy_files)}")

# 全データを読み込んで統計を計算
print("\nデータを読み込んで統計を計算中...")
all_values = []
file_stats = []

for i, file_path in enumerate(npy_files):
    data = np.load(file_path)
    all_values.append(data.flatten())
    
    file_stats.append({
        'filename': os.path.basename(file_path),
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'shape': data.shape
    })
    
    if (i + 1) % 50 == 0:
        print(f"  処理中: {i + 1}/{len(npy_files)} ファイル")

# 全データを結合
all_data = np.concatenate(all_values)
print(f"\n全データの統計:")
print(f"  データポイント数: {len(all_data):,}")
print(f"  最小値: {np.min(all_data):.6f}")
print(f"  最大値: {np.max(all_data):.6f}")
print(f"  平均値: {np.mean(all_data):.6f}")
print(f"  標準偏差: {np.std(all_data):.6f}")
print(f"  中央値: {np.median(all_data):.6f}")

# 正規化用のパラメータ（Min-Max正規化）
global_min = np.min(all_data)
global_max = np.max(all_data)
print(f"\n正規化パラメータ:")
print(f"  Min: {global_min:.6f}")
print(f"  Max: {global_max:.6f}")
print(f"  範囲: {global_max - global_min:.6f}")

# 可視化
print("\nグラフを作成中...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 全データのヒストグラム
axes[0, 0].hist(all_data, bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('値')
axes[0, 0].set_ylabel('頻度')
axes[0, 0].set_title('全データの分布（ヒストグラム）')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(global_min, color='r', linestyle='--', label=f'Min: {global_min:.2f}')
axes[0, 0].axvline(global_max, color='r', linestyle='--', label=f'Max: {global_max:.2f}')
axes[0, 0].legend()

# 2. 各ファイルの最大値の分布
file_maxes = [stat['max'] for stat in file_stats]
axes[0, 1].hist(file_maxes, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('最大値')
axes[0, 1].set_ylabel('ファイル数')
axes[0, 1].set_title('各ファイルの最大値の分布')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(np.max(file_maxes), color='r', linestyle='--', 
                   label=f'全体最大: {np.max(file_maxes):.2f}')
axes[0, 1].legend()

# 3. 各ファイルの最小値の分布
file_mins = [stat['min'] for stat in file_stats]
axes[1, 0].hist(file_mins, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_xlabel('最小値')
axes[1, 0].set_ylabel('ファイル数')
axes[1, 0].set_title('各ファイルの最小値の分布')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(np.min(file_mins), color='r', linestyle='--', 
                   label=f'全体最小: {np.min(file_mins):.2f}')
axes[1, 0].legend()

# 4. サンプルデータの可視化（最初の5ファイル）
axes[1, 1].set_title('サンプルデータ（最初の5ファイル）')
for i in range(min(5, len(npy_files))):
    sample_data = np.load(npy_files[i])
    axes[1, 1].plot(sample_data[:1000], alpha=0.6, label=f'File {i+1}')
axes[1, 1].set_xlabel('インデックス')
axes[1, 1].set_ylabel('値')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(output_dir, "data_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  グラフを保存: {plot_path}")
plt.close()

# 正規化と保存
print("\nデータを正規化して保存中...")
normalized_stats = []

for i, (file_path, stat) in enumerate(zip(npy_files, file_stats)):
    data = np.load(file_path)
    
    # Min-Max正規化: (x - min) / (max - min) -> [0, 1]
    normalized_data = (data - global_min) / (global_max - global_min)
    
    # 統計を記録
    normalized_stats.append({
        'filename': stat['filename'],
        'original_min': stat['min'],
        'original_max': stat['max'],
        'normalized_min': np.min(normalized_data),
        'normalized_max': np.max(normalized_data),
        'normalized_mean': np.mean(normalized_data),
        'normalized_std': np.std(normalized_data)
    })
    
    # 保存
    output_path = os.path.join(output_dir, stat['filename'])
    np.save(output_path, normalized_data)
    
    if (i + 1) % 50 == 0:
        print(f"  処理中: {i + 1}/{len(npy_files)} ファイル")

print(f"\n正規化完了！")
print(f"  保存先: {output_dir}")
print(f"  保存ファイル数: {len(npy_files)}")

# 正規化後の統計を確認
all_normalized = []
for file_path in npy_files:
    normalized_data = np.load(os.path.join(output_dir, os.path.basename(file_path)))
    all_normalized.append(normalized_data.flatten())

all_normalized_data = np.concatenate(all_normalized)
print(f"\n正規化後の統計:")
print(f"  最小値: {np.min(all_normalized_data):.6f}")
print(f"  最大値: {np.max(all_normalized_data):.6f}")
print(f"  平均値: {np.mean(all_normalized_data):.6f}")
print(f"  標準偏差: {np.std(all_normalized_data):.6f}")

# 統計情報をCSVに保存
import pandas as pd
df_stats = pd.DataFrame(normalized_stats)
csv_path = os.path.join(output_dir, "normalization_statistics.csv")
df_stats.to_csv(csv_path, index=False)
print(f"\n統計情報を保存: {csv_path}")

# 正規化後のデータの可視化
print("\n正規化後のグラフを作成中...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 正規化後の全データのヒストグラム
axes[0, 0].hist(all_normalized_data, bins=100, edgecolor='black', alpha=0.7, color='purple')
axes[0, 0].set_xlabel('正規化後の値')
axes[0, 0].set_ylabel('頻度')
axes[0, 0].set_title('正規化後の全データの分布')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(0, color='r', linestyle='--', label='Min: 0')
axes[0, 0].axvline(1, color='r', linestyle='--', label='Max: 1')
axes[0, 0].legend()

# 2. 正規化前後の比較（サンプル）
sample_idx = 0
original = np.load(npy_files[sample_idx])
normalized = np.load(os.path.join(output_dir, os.path.basename(npy_files[sample_idx])))
axes[0, 1].plot(original[:1000], alpha=0.7, label='正規化前', color='blue')
axes[0, 1].set_xlabel('インデックス')
axes[0, 1].set_ylabel('値')
axes[0, 1].set_title(f'正規化前（サンプル: {os.path.basename(npy_files[sample_idx])}）')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(normalized[:1000], alpha=0.7, label='正規化後', color='red')
axes[1, 0].set_xlabel('インデックス')
axes[1, 0].set_ylabel('正規化後の値')
axes[1, 0].set_title(f'正規化後（サンプル: {os.path.basename(npy_files[sample_idx])}）')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 3. 正規化前後の分布比較
axes[1, 1].hist(all_data, bins=100, alpha=0.5, label='正規化前', color='blue', density=True)
axes[1, 1].hist(all_normalized_data, bins=100, alpha=0.5, label='正規化後', color='red', density=True)
axes[1, 1].set_xlabel('値')
axes[1, 1].set_ylabel('密度')
axes[1, 1].set_title('正規化前後の分布比較（正規化）')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
normalized_plot_path = os.path.join(output_dir, "normalized_data_analysis.png")
plt.savefig(normalized_plot_path, dpi=150, bbox_inches='tight')
print(f"  グラフを保存: {normalized_plot_path}")
plt.close()

print("\n" + "="*60)
print("処理完了！")
print("="*60)
