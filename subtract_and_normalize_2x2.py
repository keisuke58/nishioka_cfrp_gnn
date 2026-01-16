import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

# パスの設定
source_dir = "/home/nishioka/GNN/GNN_hole_2026/Defect_hole_2x2_Region1_21_npy"
reference_file = "/home/nishioka/GNN/GNN_hole_2026/DSPSS_8x8/hole_no_defect_original.npy"
subtracted_dir = "/home/nishioka/GNN/GNN_hole_2026/Defect_hole_2x2_Region1_21_npy_subtracted"
normalized_dir = "/home/nishioka/GNN/GNN_hole_2026/Defect_hole_2x2_Region1_21_npy_subtracted_normalized"

# 出力ディレクトリの作成
os.makedirs(subtracted_dir, exist_ok=True)
os.makedirs(normalized_dir, exist_ok=True)

# 基準データの読み込み
print("="*60)
print("ステップ1: 引き算処理")
print("="*60)
print(f"基準データを読み込み中: {reference_file}")
reference_data = np.load(reference_file)
print(f"  形状: {reference_data.shape}")
print(f"  範囲: [{np.min(reference_data):.6f}, {np.max(reference_data):.6f}]")

# 全ファイルの取得
npy_files = sorted(glob.glob(os.path.join(source_dir, "*.npy")))
print(f"\n見つかったファイル数: {len(npy_files)}")

# 引き算処理
print("\n引き算処理中...")
subtracted_stats = []
all_subtracted_values = []

for i, file_path in enumerate(npy_files):
    data = np.load(file_path)
    
    # 形状チェック
    if data.shape != reference_data.shape:
        print(f"警告: {os.path.basename(file_path)} の形状が異なります")
        print(f"  データ形状: {data.shape}, 基準形状: {reference_data.shape}")
        continue
    
    # 引き算
    subtracted_data = data - reference_data
    
    # 統計を記録
    subtracted_stats.append({
        'filename': os.path.basename(file_path),
        'original_min': np.min(data),
        'original_max': np.max(data),
        'subtracted_min': np.min(subtracted_data),
        'subtracted_max': np.max(subtracted_data),
        'subtracted_mean': np.mean(subtracted_data),
        'subtracted_std': np.std(subtracted_data)
    })
    
    all_subtracted_values.append(subtracted_data.flatten())
    
    # 保存
    output_path = os.path.join(subtracted_dir, os.path.basename(file_path))
    np.save(output_path, subtracted_data)
    
    if (i + 1) % 100 == 0:
        print(f"  処理中: {i + 1}/{len(npy_files)} ファイル")

print(f"\n引き算完了！")
print(f"  保存先: {subtracted_dir}")
print(f"  保存ファイル数: {len(subtracted_stats)}")

# 全引き算データの統計
all_subtracted_data = np.concatenate(all_subtracted_values)
print(f"\n引き算後の全データ統計:")
print(f"  データポイント数: {len(all_subtracted_data):,}")
print(f"  最小値: {np.min(all_subtracted_data):.6f}")
print(f"  最大値: {np.max(all_subtracted_data):.6f}")
print(f"  平均値: {np.mean(all_subtracted_data):.6f}")
print(f"  標準偏差: {np.std(all_subtracted_data):.6f}")
print(f"  中央値: {np.median(all_subtracted_data):.6f}")

# 統計情報をCSVに保存
df_subtracted = pd.DataFrame(subtracted_stats)
csv_path = os.path.join(subtracted_dir, "subtraction_statistics.csv")
df_subtracted.to_csv(csv_path, index=False)
print(f"\n引き算統計情報を保存: {csv_path}")

# 可視化（引き算後）
print("\n引き算後のグラフを作成中...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 引き算後の全データのヒストグラム
axes[0, 0].hist(all_subtracted_data, bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Subtracted Data Distribution (Histogram)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(np.min(all_subtracted_data), color='r', linestyle='--', 
                   label=f'Min: {np.min(all_subtracted_data):.2f}')
axes[0, 0].axvline(np.max(all_subtracted_data), color='r', linestyle='--', 
                   label=f'Max: {np.max(all_subtracted_data):.2f}')
axes[0, 0].legend()

# 2. 各ファイルの最大値の分布
file_maxes = [stat['subtracted_max'] for stat in subtracted_stats]
axes[0, 1].hist(file_maxes, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('Max Value')
axes[0, 1].set_ylabel('Number of Files')
axes[0, 1].set_title('Distribution of Max Values per File (Subtracted)')
axes[0, 1].grid(True, alpha=0.3)

# 3. 各ファイルの最小値の分布
file_mins = [stat['subtracted_min'] for stat in subtracted_stats]
axes[1, 0].hist(file_mins, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_xlabel('Min Value')
axes[1, 0].set_ylabel('Number of Files')
axes[1, 0].set_title('Distribution of Min Values per File (Subtracted)')
axes[1, 0].grid(True, alpha=0.3)

# 4. サンプルデータの可視化
sample_idx = 0
original = np.load(npy_files[sample_idx])
subtracted = np.load(os.path.join(subtracted_dir, os.path.basename(npy_files[sample_idx])))
axes[1, 1].plot(original[:1000], alpha=0.7, label='Original', color='blue')
axes[1, 1].plot(subtracted[:1000], alpha=0.7, label='Subtracted', color='red')
axes[1, 1].set_xlabel('Index')
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Sample: Original vs Subtracted')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(subtracted_dir, "subtracted_data_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  グラフを保存: {plot_path}")
plt.close()

# ステップ2: 正規化処理
print("\n" + "="*60)
print("ステップ2: 正規化処理")
print("="*60)

# 正規化用のパラメータ（Min-Max正規化）
global_min = np.min(all_subtracted_data)
global_max = np.max(all_subtracted_data)
print(f"\n正規化パラメータ:")
print(f"  Min: {global_min:.6f}")
print(f"  Max: {global_max:.6f}")
print(f"  範囲: {global_max - global_min:.6f}")

# 正規化と保存
print("\nデータを正規化して保存中...")
normalized_stats = []

for i, stat in enumerate(subtracted_stats):
    file_path = os.path.join(subtracted_dir, stat['filename'])
    subtracted_data = np.load(file_path)
    
    # Min-Max正規化: (x - min) / (max - min) -> [0, 1]
    normalized_data = (subtracted_data - global_min) / (global_max - global_min)
    
    # 統計を記録
    normalized_stats.append({
        'filename': stat['filename'],
        'subtracted_min': stat['subtracted_min'],
        'subtracted_max': stat['subtracted_max'],
        'normalized_min': np.min(normalized_data),
        'normalized_max': np.max(normalized_data),
        'normalized_mean': np.mean(normalized_data),
        'normalized_std': np.std(normalized_data)
    })
    
    # 保存
    output_path = os.path.join(normalized_dir, stat['filename'])
    np.save(output_path, normalized_data)
    
    if (i + 1) % 100 == 0:
        print(f"  処理中: {i + 1}/{len(subtracted_stats)} ファイル")

print(f"\n正規化完了！")
print(f"  保存先: {normalized_dir}")
print(f"  保存ファイル数: {len(normalized_stats)}")

# 正規化後の統計を確認
all_normalized = []
for stat in subtracted_stats:
    normalized_data = np.load(os.path.join(normalized_dir, stat['filename']))
    all_normalized.append(normalized_data.flatten())

all_normalized_data = np.concatenate(all_normalized)
print(f"\n正規化後の統計:")
print(f"  最小値: {np.min(all_normalized_data):.6f}")
print(f"  最大値: {np.max(all_normalized_data):.6f}")
print(f"  平均値: {np.mean(all_normalized_data):.6f}")
print(f"  標準偏差: {np.std(all_normalized_data):.6f}")

# 統計情報をCSVに保存
df_normalized = pd.DataFrame(normalized_stats)
csv_path = os.path.join(normalized_dir, "normalization_statistics.csv")
df_normalized.to_csv(csv_path, index=False)
print(f"\n正規化統計情報を保存: {csv_path}")

# 正規化後のデータの可視化
print("\n正規化後のグラフを作成中...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 正規化後の全データのヒストグラム
axes[0, 0].hist(all_normalized_data, bins=100, edgecolor='black', alpha=0.7, color='purple')
axes[0, 0].set_xlabel('Normalized Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Normalized Data Distribution')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(0, color='r', linestyle='--', label='Min: 0')
axes[0, 0].axvline(1, color='r', linestyle='--', label='Max: 1')
axes[0, 0].legend()

# 2. 引き算後と正規化後の比較（サンプル）
sample_idx = 0
subtracted_sample = np.load(os.path.join(subtracted_dir, subtracted_stats[sample_idx]['filename']))
normalized_sample = np.load(os.path.join(normalized_dir, subtracted_stats[sample_idx]['filename']))
axes[0, 1].plot(subtracted_sample[:1000], alpha=0.7, label='After Subtraction', color='blue')
axes[0, 1].set_xlabel('Index')
axes[0, 1].set_ylabel('Value')
axes[0, 1].set_title('After Subtraction (Sample)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(normalized_sample[:1000], alpha=0.7, label='After Normalization', color='red')
axes[1, 0].set_xlabel('Index')
axes[1, 0].set_ylabel('Normalized Value')
axes[1, 0].set_title('After Normalization (Sample)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 3. 引き算後と正規化後の分布比較
axes[1, 1].hist(all_subtracted_data, bins=100, alpha=0.5, label='After Subtraction', color='blue', density=True)
axes[1, 1].hist(all_normalized_data, bins=100, alpha=0.5, label='After Normalization', color='red', density=True)
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Distribution Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
normalized_plot_path = os.path.join(normalized_dir, "normalized_data_analysis.png")
plt.savefig(normalized_plot_path, dpi=150, bbox_inches='tight')
print(f"  グラフを保存: {normalized_plot_path}")
plt.close()

print("\n" + "="*60)
print("処理完了！")
print("="*60)
print(f"引き算結果: {subtracted_dir}")
print(f"正規化結果: {normalized_dir}")
