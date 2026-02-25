#!/bin/bash
# Min-Max正規化データの確認用スクリプト

echo "=========================================="
echo "Min-Max正規化データの確認"
echo "=========================================="
echo ""

# 1. 可視化ファイルの場所を表示
echo "【可視化ファイルの場所】"
echo "/home/nishioka/GNN/GNN_hole_2026/normalization_comparison_visualization"
echo ""

# 2. 可視化ファイル数を表示
VIS_DIR="/home/nishioka/GNN/GNN_hole_2026/normalization_comparison_visualization"
FILE_COUNT=$(ls -1 "$VIS_DIR"/*.png 2>/dev/null | wc -l)
echo "可視化ファイル数: $FILE_COUNT"
echo ""

# 3. 可視化ファイルの一覧（最初の10個）
echo "【可視化ファイル一覧（最初の10個）】"
ls -1 "$VIS_DIR"/*.png | head -10 | nl
echo ""

# 4. Min-Max正規化データの統計を確認
echo "【Min-Max正規化データの統計確認】"
python3 << 'EOF'
import numpy as np
import os

minmax_dir = '/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_normalized'
files = sorted([f for f in os.listdir(minmax_dir) if f.endswith('.npy')])

print(f"総ファイル数: {len(files)}")
print("\nサンプルファイルの統計（最初の5ファイル）:")
print("=" * 80)

for i, filename in enumerate(files[:5], 1):
    data = np.load(os.path.join(minmax_dir, filename))
    print(f"{i}. {filename}")
    print(f"   最小値: {np.min(data):.6f}")
    print(f"   最大値: {np.max(data):.6f}")
    print(f"   平均値: {np.mean(data):.6f}")
    print(f"   標準偏差: {np.std(data):.6f}")
    print()

# 全データの統計
print("全データの統計:")
all_mins = []
all_maxs = []
for filename in files[:100]:  # サンプルとして100ファイル
    data = np.load(os.path.join(minmax_dir, filename))
    all_mins.append(np.min(data))
    all_maxs.append(np.max(data))

print(f"  最小値の範囲: {min(all_mins):.6f} ~ {max(all_mins):.6f}")
print(f"  最大値の範囲: {min(all_maxs):.6f} ~ {max(all_maxs):.6f}")
EOF

echo ""
echo "=========================================="
echo "可視化ファイルを開くには:"
echo "  xdg-open $VIS_DIR/Defect_L10_B100_el1165_H2_W2_comparison.png"
echo "  または"
echo "  eog $VIS_DIR/Defect_L10_B100_el1165_H2_W2_comparison.png"
echo "=========================================="
