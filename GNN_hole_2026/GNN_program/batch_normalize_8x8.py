"""
8x8版のディレクトリを外れ値除去して正規化するバッチスクリプト
2x2/4x4と同じ処理を適用
"""

import subprocess
import sys
import os

def main():
    base_dir = "/home/nishioka/GNN/GNN_hole_2026"
    
    # 8x8版のディレクトリ
    input_dir = os.path.join(base_dir, "Defect_hole_8x8_Region1_21_npy_subtracted")
    
    if not os.path.exists(input_dir):
        print(f"エラー: ディレクトリが存在しません: {input_dir}")
        print("まず convert_8x8_to_standard_format.py を実行してください")
        return
    
    script_path = os.path.join(os.path.dirname(__file__), "normalize_with_outlier_removal.py")
    
    # 3つの正規化方法を実行
    normalizations = [
        {'name': 'minmax', 'normalization': 'minmax'},
        {'name': 'zscore', 'normalization': 'zscore'},
        {'name': 'robust_zscore', 'normalization': 'robust_zscore'}
    ]
    
    for norm_info in normalizations:
        norm_name = norm_info['name']
        normalization = norm_info['normalization']
        
        print(f"\n{'='*60}")
        print(f"8x8版 - {norm_name.upper()} 正規化を処理中...")
        print(f"{'='*60}\n")
        
        # パーセンタイル法 + 正規化
        cmd = [
            sys.executable,
            script_path,
            '--input_dir', input_dir,
            '--method', 'percentile',
            '--normalization', normalization,
            '--sample_size', '1000'  # 統計量計算用のサンプルサイズ
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"\n8x8版 - {norm_name.upper()} の処理が完了しました")
        except subprocess.CalledProcessError as e:
            print(f"\nエラー: 8x8版 - {norm_name.upper()} の処理に失敗しました: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("全ての処理が完了しました!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
