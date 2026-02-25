"""
可視化で使用されているファイルの対応を詳細に検証するスクリプト
可視化結果の各ファイルについて、データファイルとラベルファイルが正しく対応しているかを確認
"""

import os
import re
from pathlib import Path

def extract_all_components(file_name):
    """
    ファイル名からすべてのコンポーネントを抽出
    Defect_L10_B102_el1169_H2_W2.npy -> (10, 102, 1169, 2, 2)
    """
    # パターン: Defect_L<L>_B<B>_el<el>_H<H>_W<W>.npy
    pattern = r'Defect_L(\d+)_B(\d+)_el(\d+)_H(\d+)_W(\d+)'
    match = re.search(pattern, file_name)
    if match:
        return {
            'L': int(match.group(1)),
            'B': int(match.group(2)),
            'el': int(match.group(3)),
            'H': int(match.group(4)),
            'W': int(match.group(5)),
            'full_match': match.group(0)
        }
    return None

def get_base_name(file_name):
    """ファイル名からベース名を取得（拡張子と_19label、_predを除く）"""
    base = file_name.replace('_19label.npy', '').replace('_pred.npy', '').replace('.npy', '')
    return base

def verify_visualization_files(visualization_dir, data_dir, label_dir):
    """
    可視化結果のファイルについて、データファイルとラベルファイルの対応を検証
    """
    print("=" * 80)
    print("可視化ファイル対応検証スクリプト")
    print("=" * 80)
    
    # 可視化画像ファイルを取得
    print(f"\n[1] 可視化画像ファイルを読み込み中: {visualization_dir}")
    vis_files = []
    if os.path.exists(visualization_dir):
        vis_files = [f for f in os.listdir(visualization_dir) if f.endswith('_visualization.png')]
        vis_files = sorted(vis_files)
    print(f"  見つかった可視化画像数: {len(vis_files)}")
    
    if len(vis_files) == 0:
        print("  エラー: 可視化画像が見つかりません")
        return
    
    # 各可視化画像について検証
    print("\n" + "=" * 80)
    print("[2] 各可視化画像のファイル対応検証")
    print("=" * 80)
    
    all_correct = True
    mismatches = []
    
    for vis_file in vis_files:
        # 可視化画像ファイル名からベース名を取得
        base_name = vis_file.replace('_visualization.png', '')
        
        # データファイルとラベルファイルのパス
        data_file = os.path.join(data_dir, f"{base_name}.npy")
        label_file = os.path.join(label_dir, f"{base_name}_19label.npy")
        
        # ファイルの存在確認
        data_exists = os.path.exists(data_file)
        label_exists = os.path.exists(label_file)
        
        if not data_exists or not label_exists:
            print(f"\n❌ {base_name}")
            if not data_exists:
                print(f"   データファイルが見つかりません: {data_file}")
            if not label_exists:
                print(f"   ラベルファイルが見つかりません: {label_file}")
            all_correct = False
            mismatches.append({
                'base_name': base_name,
                'data_exists': data_exists,
                'label_exists': label_exists
            })
            continue
        
        # ファイル名のコンポーネントを抽出
        data_comp = extract_all_components(base_name + '.npy')
        label_comp = extract_all_components(base_name + '_19label.npy')
        
        if data_comp and label_comp:
            # すべてのコンポーネントが一致しているか確認
            is_match = (
                data_comp['L'] == label_comp['L'] and
                data_comp['B'] == label_comp['B'] and
                data_comp['el'] == label_comp['el'] and
                data_comp['H'] == label_comp['H'] and
                data_comp['W'] == label_comp['W']
            )
            
            if is_match:
                print(f"✓ {base_name}")
                print(f"  データ: L={data_comp['L']}, B={data_comp['B']}, el={data_comp['el']}, H={data_comp['H']}, W={data_comp['W']}")
                print(f"  ラベル: L={label_comp['L']}, B={label_comp['B']}, el={label_comp['el']}, H={label_comp['H']}, W={label_comp['W']}")
            else:
                print(f"\n❌ {base_name} - コンポーネント不一致")
                print(f"  データ: L={data_comp['L']}, B={data_comp['B']}, el={data_comp['el']}, H={data_comp['H']}, W={data_comp['W']}")
                print(f"  ラベル: L={label_comp['L']}, B={label_comp['B']}, el={label_comp['el']}, H={label_comp['H']}, W={label_comp['W']}")
                
                # どの項目が不一致か表示
                mismatches_list = []
                if data_comp['L'] != label_comp['L']:
                    mismatches_list.append(f"L({data_comp['L']} vs {label_comp['L']})")
                if data_comp['B'] != label_comp['B']:
                    mismatches_list.append(f"B({data_comp['B']} vs {label_comp['B']})")
                if data_comp['el'] != label_comp['el']:
                    mismatches_list.append(f"el({data_comp['el']} vs {label_comp['el']})")
                if data_comp['H'] != label_comp['H']:
                    mismatches_list.append(f"H({data_comp['H']} vs {label_comp['H']})")
                if data_comp['W'] != label_comp['W']:
                    mismatches_list.append(f"W({data_comp['W']} vs {label_comp['W']})")
                
                if mismatches_list:
                    print(f"  不一致項目: {', '.join(mismatches_list)}")
                
                all_correct = False
                mismatches.append({
                    'base_name': base_name,
                    'data_comp': data_comp,
                    'label_comp': label_comp,
                    'mismatches': mismatches_list
                })
        else:
            print(f"\n⚠ {base_name} - ファイル名パターンが一致しません")
            if not data_comp:
                print(f"  データファイル名のパターンが一致しません")
            if not label_comp:
                print(f"  ラベルファイル名のパターンが一致しません")
            all_correct = False
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("[3] 検証結果サマリー")
    print("=" * 80)
    print(f"検証した可視化画像数: {len(vis_files)}")
    print(f"完全一致: {len(vis_files) - len(mismatches)}")
    print(f"不一致/問題あり: {len(mismatches)}")
    
    if all_correct:
        print("\n✓ すべてのファイルが正しく対応しています！")
    else:
        print(f"\n❌ {len(mismatches)} 件の不一致/問題が見つかりました")
    
    return all_correct, mismatches


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='可視化ファイルの対応を詳細に検証')
    parser.add_argument('--visualization_dir', type=str, required=True,
                       help='可視化画像のディレクトリ')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='データファイルのディレクトリ')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='ラベルファイルのディレクトリ')
    
    args = parser.parse_args()
    
    verify_visualization_files(args.visualization_dir, args.data_dir, args.label_dir)


if __name__ == "__main__":
    main()
