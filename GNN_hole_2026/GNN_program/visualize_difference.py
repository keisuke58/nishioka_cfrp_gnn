"""
Visualization script for difference between defect data and defect-free data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from matplotlib import rcParams
import warnings
import random
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
difference_cmap = 'RdBu_r'  # 差分データ用：赤青の 'RdBu_r'（反転）

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
hole_set = set(hole_elements)


def reshape_data_to_grid(data_1d):
    """
    1次元データをグリッドにマッピングして2層に分割
    
    Args:
        data_1d: 1次元配列（13942要素）
    
    Returns:
        tuple: (layer1_grid, layer2_grid) - 各層のグリッドデータ
    """
    layer1_data = data_1d[:vertices_per_layer]
    layer2_data = data_1d[vertices_per_layer:]
    
    layer1_full = np.empty(total_elements)
    layer1_full[:] = np.nan
    layer2_full = np.empty(total_elements)
    layer2_full[:] = np.nan
    
    # Layer 1をグリッドにマッピング
    data_index = 0
    for j in range(total_elements):
        if j not in hole_set and data_index < vertices_per_layer:
            layer1_full[j] = layer1_data[data_index]
            data_index += 1
    
    # Layer 2をグリッドにマッピング
    data_index = 0
    for j in range(total_elements):
        if j not in hole_set and data_index < vertices_per_layer:
            layer2_full[j] = layer2_data[data_index]
            data_index += 1
    
    # リシェイプと回転
    layer1_reshaped = np.flipud(np.rot90(layer1_full.reshape((num_rows, num_cols)), k=1))
    layer2_reshaped = np.flipud(np.rot90(layer2_full.reshape((num_rows, num_cols)), k=1))
    
    return layer1_reshaped, layer2_reshaped


def visualize_difference(difference_file, reference_file, output_dir, original_file=None):
    """
    Visualize difference between defect data and defect-free data
    
    Args:
        difference_file: Path to difference data file (subtracted data)
        reference_file: Path to defect-free data file (reference data)
        output_dir: Output directory
        original_file: Path to original defect data file (optional, reconstructed if not provided)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference data (defect-free data)
    print(f"Loading reference data: {reference_file}")
    reference_data = np.load(reference_file)[:13942]
    ref_layer1, ref_layer2 = reshape_data_to_grid(reference_data)
    
    # Load difference data (subtracted data)
    print(f"Loading difference data: {difference_file}")
    difference_data = np.load(difference_file)[:13942]
    diff_layer1, diff_layer2 = reshape_data_to_grid(difference_data)
    
    # Get original defect data
    if original_file and os.path.exists(original_file):
        print(f"Loading original data: {original_file}")
        original_data = np.load(original_file)[:13942]
    else:
        # Reconstruct original data from difference (difference + reference)
        print("Reconstructing original data from difference...")
        original_data = difference_data + reference_data
    orig_layer1, orig_layer2 = reshape_data_to_grid(original_data)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Layer 1 (Bottom Layer)
    # 1. Original Defect Data
    ax1 = axes[0, 0]
    im1 = ax1.imshow(orig_layer1, cmap=dpsss_cmap, aspect="equal", interpolation='none')
    ax1.set_title("Layer 1 - Original Defect Data", fontsize=12, pad=20)
    plt.colorbar(im1, ax=ax1, label="DSPSS Value", fraction=0.025, pad=0.04)
    
    # 2. Reference (No Defect)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(ref_layer1, cmap=dpsss_cmap, aspect="equal", interpolation='none')
    ax2.set_title("Layer 1 - Reference (No Defect)", fontsize=12, pad=20)
    plt.colorbar(im2, ax=ax2, label="DSPSS Value", fraction=0.025, pad=0.04)
    
    # 3. Difference Data
    ax3 = axes[0, 2]
    vmax_diff1 = max(np.nanmax(np.abs(diff_layer1)), 1e-6)
    im3 = ax3.imshow(diff_layer1, cmap=difference_cmap, aspect="equal", 
                     interpolation='none', vmin=-vmax_diff1, vmax=vmax_diff1)
    ax3.set_title("Layer 1 - Difference\n(Difference = Original - Reference)", fontsize=12, pad=20)
    plt.colorbar(im3, ax=ax3, label="Difference", fraction=0.025, pad=0.04)
    
    # Layer 2 (Upper Layer)
    # 4. Original Defect Data
    ax4 = axes[1, 0]
    im4 = ax4.imshow(orig_layer2, cmap=dpsss_cmap, aspect="equal", interpolation='none')
    ax4.set_title("Layer 2 - Original Defect Data", fontsize=12, pad=20)
    plt.colorbar(im4, ax=ax4, label="DSPSS Value", fraction=0.025, pad=0.04)
    
    # 5. Reference (No Defect)
    ax5 = axes[1, 1]
    im5 = ax5.imshow(ref_layer2, cmap=dpsss_cmap, aspect="equal", interpolation='none')
    ax5.set_title("Layer 2 - Reference (No Defect)", fontsize=12, pad=20)
    plt.colorbar(im5, ax=ax5, label="DSPSS Value", fraction=0.025, pad=0.04)
    
    # 6. Difference Data
    ax6 = axes[1, 2]
    vmax_diff2 = max(np.nanmax(np.abs(diff_layer2)), 1e-6)
    im6 = ax6.imshow(diff_layer2, cmap=difference_cmap, aspect="equal", 
                     interpolation='none', vmin=-vmax_diff2, vmax=vmax_diff2)
    ax6.set_title("Layer 2 - Difference\n(Difference = Original - Reference)", fontsize=12, pad=20)
    plt.colorbar(im6, ax=ax6, label="Difference", fraction=0.025, pad=0.04)
    
    # Add statistics information
    stats_text = f"""
Statistics (Layer 1):
  Original Data: min={np.nanmin(orig_layer1):.3f}, max={np.nanmax(orig_layer1):.3f}, mean={np.nanmean(orig_layer1):.3f}
  Reference Data: min={np.nanmin(ref_layer1):.3f}, max={np.nanmax(ref_layer1):.3f}, mean={np.nanmean(ref_layer1):.3f}
  Difference: min={np.nanmin(diff_layer1):.3f}, max={np.nanmax(diff_layer1):.3f}, mean={np.nanmean(diff_layer1):.3f}

Statistics (Layer 2):
  Original Data: min={np.nanmin(orig_layer2):.3f}, max={np.nanmax(orig_layer2):.3f}, mean={np.nanmean(orig_layer2):.3f}
  Reference Data: min={np.nanmin(ref_layer2):.3f}, max={np.nanmax(ref_layer2):.3f}, mean={np.nanmean(ref_layer2):.3f}
  Difference: min={np.nanmin(diff_layer2):.3f}, max={np.nanmax(diff_layer2):.3f}, mean={np.nanmean(diff_layer2):.3f}
"""
    
    fig.suptitle(f"Difference Visualization\n{os.path.basename(difference_file)}", 
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_filename = f"{os.path.splitext(os.path.basename(difference_file))[0]}_difference_visualization.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {output_path}")
    print(stats_text)


def find_original_file(filename, original_dirs):
    """
    元の欠陥データファイルを検索
    
    Args:
        filename: 差分データファイル名
        original_dirs: 元のデータが保存されている可能性のあるディレクトリのリスト
    
    Returns:
        元のデータファイルのパス（見つからない場合はNone）
    """
    for orig_dir in original_dirs:
        if not os.path.exists(orig_dir):
            continue
        orig_path = os.path.join(orig_dir, filename)
        if os.path.exists(orig_path):
            return orig_path
    return None


def visualize_multiple_files(difference_data_dir, reference_file, output_dir, max_files=10, original_dirs=None):
    """
    Visualize multiple difference data files
    
    Args:
        difference_data_dir: Directory containing difference data files
        reference_file: Path to defect-free data file
        output_dir: Output directory
        max_files: Maximum number of files to visualize
        original_dirs: List of directories where original data might be stored
    """
    # Get difference data files
    if os.path.isdir(difference_data_dir):
        all_files = sorted([f for f in os.listdir(difference_data_dir) 
                           if f.startswith("Defect_L") and f.endswith(".npy")])
        # Randomly select files
        if len(all_files) > max_files:
            difference_files = random.sample(all_files, max_files)
        else:
            difference_files = all_files
        base_dir = difference_data_dir
    else:
        # Single file case
        difference_files = [os.path.basename(difference_data_dir)]
        base_dir = os.path.dirname(difference_data_dir)
    
    print(f"Number of files to visualize: {len(difference_files)}")
    
    for i, filename in enumerate(difference_files):
        print(f"\nProcessing ({i+1}/{len(difference_files)}): {filename}")
        difference_file_path = os.path.join(base_dir, filename)
        
        if not os.path.exists(difference_file_path):
            print(f"  Warning: File does not exist: {difference_file_path}")
            continue
        
        # Search for original data file
        original_file = None
        if original_dirs:
            original_file = find_original_file(filename, original_dirs)
            if original_file:
                print(f"  Original data found: {original_file}")
        
        try:
            visualize_difference(difference_file_path, reference_file, output_dir, original_file)
        except Exception as e:
            print(f"  Error: Failed to process {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nVisualization complete! Output directory: {output_dir}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize difference between defect data and defect-free data')
    parser.add_argument('--difference_dir', type=str, 
                       default='/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore/test',
                       help='Directory containing difference data files')
    parser.add_argument('--reference_file', type=str,
                       default='/home/nishioka/GNN/GNN_hole_2026/DSPSS_8x8/hole_no_defect_original.npy',
                       help='Path to defect-free data (reference data) file')
    parser.add_argument('--output_dir', type=str,
                       default='/home/nishioka/GNN/GNN_hole_2026/difference_visualizations',
                       help='Output directory for visualization results')
    parser.add_argument('--max_files', type=int, default=10,
                       help='Maximum number of files to visualize')
    parser.add_argument('--original_dirs', type=str, nargs='+', default=None,
                       help='List of directories where original defect data might be stored (optional)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for file selection')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    print("=" * 80)
    print("Difference Visualization Script")
    print("=" * 80)
    print(f"Difference data directory: {args.difference_dir}")
    print(f"Reference data file: {args.reference_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Maximum number of files: {args.max_files}")
    if args.original_dirs:
        print(f"Original data directories: {args.original_dirs}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print("=" * 80)
    
    if not os.path.exists(args.reference_file):
        print(f"Error: Reference data file does not exist: {args.reference_file}")
        return
    
    visualize_multiple_files(args.difference_dir, args.reference_file, 
                           args.output_dir, args.max_files, args.original_dirs)


if __name__ == "__main__":
    main()
