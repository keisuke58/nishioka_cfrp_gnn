"""
入力データファイルとラベルファイルの対応を詳細に検証するスクリプト
L, B, el, H, Wのすべての値が一致しているかを確認
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_all_components(file_name):
    """
    ファイル名からすべてのコンポーネントを抽出
    Defect_L10_B102_el1169_H2_W2.npy -> (10, 102, 1169, 2, 2)
    """
    # Defectパターン: elあり/なし両対応
    # 例:
    #   Defect_L10_B102_el1169_H2_W2.npy
    #   Defect_L10_B10_H8_W4.npy  (elなし)
    defect_pattern = r'Defect_L(\d+)_B(\d+)(?:_el(\d+))?_H(\d+)_W(\d+)'
    m = re.search(defect_pattern, file_name)
    if m:
        el = m.group(3)
        return {
            'type': 'Defect',
            'L': int(m.group(1)),
            'B': int(m.group(2)),
            'el': int(el) if el is not None else None,
            'H': int(m.group(4)),
            'W': int(m.group(5)),
            'full_match': m.group(0),
        }

    # NoiseDefectFreeパターン（L/B/el/H/Wが無いのでIDのみ）
    ndf_pattern = r'NoiseDefectFree_(\d+)'
    m = re.search(ndf_pattern, file_name)
    if m:
        return {
            'type': 'NoiseDefectFree',
            'id': int(m.group(1)),
            'full_match': m.group(0),
        }

    return None


def _defect_keys(comp):
    """Defect用の検索キーを生成（H/W入れ替え、el無視も含む）"""
    if not comp or comp.get("type") != "Defect":
        return []
    L = comp["L"]
    B = comp["B"]
    el = comp.get("el", None)
    H = comp["H"]
    W = comp["W"]
    keys = []
    # exact (with el)
    keys.append(("Defect", L, B, el, H, W))
    keys.append(("Defect", L, B, el, W, H))  # swapped
    # ignore el
    keys.append(("Defect_noel", L, B, H, W))
    keys.append(("Defect_noel", L, B, W, H))
    return keys

def get_base_name(file_name):
    """ファイル名からベース名を取得（拡張子と_19labelを除く）"""
    base = file_name.replace('_19label.npy', '').replace('_pred.npy', '').replace('.npy', '')
    return base

def verify_file_matching(data_dir, label_dir, predict_dir=None):
    """
    データファイルとラベルファイルの対応を検証
    """
    print("=" * 80)
    print("ファイル対応検証スクリプト")
    print("=" * 80)
    
    # データファイルを読み込み
    print(f"\n[1] データファイルを読み込み中: {data_dir}")
    data_files = []
    if os.path.exists(data_dir):
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.npy') and not f.endswith('_pred.npy') and not f.endswith('_19label.npy')]
        data_files = sorted(data_files)
    print(f"  見つかったデータファイル数: {len(data_files)}")
    
    # ラベルファイルを読み込み
    print(f"\n[2] ラベルファイルを読み込み中: {label_dir}")
    label_files = []
    if os.path.exists(label_dir):
        label_files = [f for f in os.listdir(label_dir) if f.endswith('_19label.npy')]
        label_files = sorted(label_files)
    print(f"  見つかったラベルファイル数: {len(label_files)}")
    
    # 予測ファイルを読み込み（オプション）
    pred_files = []
    if predict_dir:
        print(f"\n[3] 予測ファイルを読み込み中: {predict_dir}")
        predictions_folder = os.path.join(predict_dir, 'predictions')
        if os.path.exists(predictions_folder):
            pred_files = [f for f in os.listdir(predictions_folder) if f.endswith('_pred.npy')]
            pred_files = sorted(pred_files)
        print(f"  見つかった予測ファイル数: {len(pred_files)}")
    
    # データファイルをインデックス化（ベース名で）
    data_index = {}
    data_components = {}
    for data_file in data_files:
        base_name = get_base_name(data_file)
        data_index[base_name] = os.path.join(data_dir, data_file)
        components = extract_all_components(data_file)
        if components:
            data_components[base_name] = components
    
    # ラベルファイルをインデックス化（ベース名で + コンポーネントキーでも）
    label_index = {}
    label_components = {}
    label_key_index = {}         # ("Defect", L,B,el,H,W) -> base_name
    label_key_index_noel = {}    # ("Defect_noel", L,B,H,W) -> base_name (ambiguousは最初を採用)
    for label_file in label_files:
        base_name = get_base_name(label_file)
        label_index[base_name] = os.path.join(label_dir, label_file)
        components = extract_all_components(label_file)
        if components:
            label_components[base_name] = components
            if components.get("type") == "Defect":
                # exact key (el may be None)
                k = ("Defect", components["L"], components["B"], components.get("el", None), components["H"], components["W"])
                label_key_index[k] = base_name
                # ignore-el key
                k2 = ("Defect_noel", components["L"], components["B"], components["H"], components["W"])
                # 既に存在する場合は上書きしない（複数候補がある可能性）
                if k2 not in label_key_index_noel:
                    label_key_index_noel[k2] = base_name
    
    # 予測ファイルをインデックス化（オプション）
    pred_index = {}
    pred_components = {}
    if predict_dir:
        predictions_folder = os.path.join(predict_dir, 'predictions')
        for pred_file in pred_files:
            base_name = get_base_name(pred_file)
            pred_index[base_name] = os.path.join(predictions_folder, pred_file)
            components = extract_all_components(pred_file)
            if components:
                pred_components[base_name] = components
    
    # マッチング検証
    print("\n" + "=" * 80)
    print("[4] ファイル対応の検証")
    print("=" * 80)
    
    # データファイルごとにラベルファイルを検証
    matched_pairs = []
    mismatched_pairs = []
    missing_labels = []
    matched_by_basename_only = 0
    matched_by_swap_or_fuzzy = 0
    
    for base_name, data_path in data_index.items():
        data_comp = data_components.get(base_name)
        label_path = label_index.get(base_name)
        label_comp = label_components.get(base_name) if label_path else None

        # ベース名で見つからない場合、Defectについては H/W 入れ替えや el 無視で探索
        if (label_path is None) and data_comp and data_comp.get("type") == "Defect":
            found_base = None
            for k in _defect_keys(data_comp):
                if k[0] == "Defect":
                    if k in label_key_index:
                        found_base = label_key_index[k]
                        break
                elif k[0] == "Defect_noel":
                    if k in label_key_index_noel:
                        found_base = label_key_index_noel[k]
                        break
            if found_base is not None:
                label_path = label_index.get(found_base)
                label_comp = label_components.get(found_base) if label_path else None
                matched_by_swap_or_fuzzy += 1
        
        if label_path:
            # コンポーネントが取れない（elなし/NoiseDefectFree等）の場合でも、
            # ラベルファイルが存在するなら「ラベル無し」にはしない
            if not (data_comp and label_comp):
                matched_pairs.append({
                    'base_name': base_name,
                    'data': data_path,
                    'label': label_path,
                    'components': data_comp
                })
                matched_by_basename_only += 1
                continue
            
            # NoiseDefectFree等: 種別が一致していればベース名一致でOK
            if data_comp.get("type") != "Defect" or label_comp.get("type") != "Defect":
                if data_comp.get("type") == label_comp.get("type"):
                    matched_pairs.append({
                        'base_name': base_name,
                        'data': data_path,
                        'label': label_path,
                        'components': data_comp
                    })
                    continue
                else:
                    mismatched_pairs.append({
                        'base_name': base_name,
                        'data': data_path,
                        'label': label_path,
                        'data_comp': data_comp,
                        'label_comp': label_comp
                    })
                    continue

        if label_path and data_comp and label_comp and data_comp.get("type") == "Defect" and label_comp.get("type") == "Defect":
            # すべてのコンポーネントが一致しているか確認
            is_match = (
                data_comp['L'] == label_comp['L'] and
                data_comp['B'] == label_comp['B'] and
                data_comp.get('el', None) == label_comp.get('el', None) and
                data_comp['H'] == label_comp['H'] and
                data_comp['W'] == label_comp['W']
            )
            
            if is_match:
                matched_pairs.append({
                    'base_name': base_name,
                    'data': data_path,
                    'label': label_path,
                    'components': data_comp
                })
            else:
                mismatched_pairs.append({
                    'base_name': base_name,
                    'data': data_path,
                    'label': label_path,
                    'data_comp': data_comp,
                    'label_comp': label_comp
                })
        else:
            missing_labels.append({
                'base_name': base_name,
                'data': data_path,
                'data_comp': data_comp
            })
    
    # 結果を表示
    print(f"\n✓ 完全一致したペア: {len(matched_pairs)}")
    print(f"✗ 不一致があったペア: {len(mismatched_pairs)}")
    print(f"⚠ ラベルが見つからないデータファイル: {len(missing_labels)}")
    print(f"  - 参考: ベース名一致のみでOKとしたペア: {matched_by_basename_only}")
    print(f"  - 参考: H/W入れ替え・el無視などで救済したペア: {matched_by_swap_or_fuzzy}")
    
    # 不一致の詳細を表示
    if mismatched_pairs:
        print("\n" + "-" * 80)
        print("【不一致の詳細】")
        print("-" * 80)
        for i, pair in enumerate(mismatched_pairs[:20]):  # 最大20件表示
            print(f"\n[{i+1}] ベース名: {pair['base_name']}")
            print(f"    データファイル: {os.path.basename(pair['data'])}")
            print(f"    ラベルファイル: {os.path.basename(pair['label'])}")
            print(f"    データ: L={pair['data_comp']['L']}, B={pair['data_comp']['B']}, el={pair['data_comp']['el']}, H={pair['data_comp']['H']}, W={pair['data_comp']['W']}")
            print(f"    ラベル: L={pair['label_comp']['L']}, B={pair['label_comp']['B']}, el={pair['label_comp']['el']}, H={pair['label_comp']['H']}, W={pair['label_comp']['W']}")
            
            # どの項目が不一致か表示
            mismatches = []
            if pair['data_comp']['L'] != pair['label_comp']['L']:
                mismatches.append(f"L({pair['data_comp']['L']} vs {pair['label_comp']['L']})")
            if pair['data_comp']['B'] != pair['label_comp']['B']:
                mismatches.append(f"B({pair['data_comp']['B']} vs {pair['label_comp']['B']})")
            if pair['data_comp']['el'] != pair['label_comp']['el']:
                mismatches.append(f"el({pair['data_comp']['el']} vs {pair['label_comp']['el']})")
            if pair['data_comp']['H'] != pair['label_comp']['H']:
                mismatches.append(f"H({pair['data_comp']['H']} vs {pair['label_comp']['H']})")
            if pair['data_comp']['W'] != pair['label_comp']['W']:
                mismatches.append(f"W({pair['data_comp']['W']} vs {pair['label_comp']['W']})")
            
            if mismatches:
                print(f"    ❌ 不一致項目: {', '.join(mismatches)}")
        
        if len(mismatched_pairs) > 20:
            print(f"\n    ... 他 {len(mismatched_pairs) - 20} 件の不一致があります")
    
    # ラベルが見つからないデータファイルの詳細
    if missing_labels:
        print("\n" + "-" * 80)
        print("【ラベルが見つからないデータファイル】")
        print("-" * 80)
        for i, item in enumerate(missing_labels[:10]):  # 最大10件表示
            print(f"\n[{i+1}] {os.path.basename(item['data'])}")
            if item['data_comp']:
                if item["data_comp"].get("type") == "Defect":
                    print(
                        f"    L={item['data_comp']['L']}, B={item['data_comp']['B']}, "
                        f"el={item['data_comp'].get('el', None)}, H={item['data_comp']['H']}, W={item['data_comp']['W']}"
                    )
                else:
                    print(f"    type={item['data_comp'].get('type')}, id={item['data_comp'].get('id', None)}")
        
        if len(missing_labels) > 10:
            print(f"\n    ... 他 {len(missing_labels) - 10} 件のデータファイルにラベルがありません")
    
    # 予測ファイルとの対応も検証（オプション）
    if predict_dir and pred_index:
        print("\n" + "=" * 80)
        print("[5] 予測ファイルとの対応検証")
        print("=" * 80)
        
        pred_matched = 0
        pred_mismatched = []
        pred_missing = []
        
        for base_name, pred_path in pred_index.items():
            pred_comp = pred_components.get(base_name)
            data_path = data_index.get(base_name)
            data_comp = data_components.get(base_name)
            
            if data_path and pred_comp and data_comp:
                is_match = (
                    data_comp['L'] == pred_comp['L'] and
                    data_comp['B'] == pred_comp['B'] and
                    data_comp['el'] == pred_comp['el'] and
                    data_comp['H'] == pred_comp['H'] and
                    data_comp['W'] == pred_comp['W']
                )
                
                if is_match:
                    pred_matched += 1
                else:
                    pred_mismatched.append({
                        'base_name': base_name,
                        'data': data_path,
                        'pred': pred_path,
                        'data_comp': data_comp,
                        'pred_comp': pred_comp
                    })
            else:
                pred_missing.append({
                    'base_name': base_name,
                    'pred': pred_path
                })
        
        print(f"\n✓ 予測ファイルと完全一致: {pred_matched}")
        print(f"✗ 予測ファイルと不一致: {len(pred_mismatched)}")
        print(f"⚠ 対応するデータファイルが見つからない予測ファイル: {len(pred_missing)}")
        
        if pred_mismatched:
            print("\n【予測ファイルとの不一致の詳細】")
            for i, item in enumerate(pred_mismatched[:10]):
                print(f"\n[{i+1}] {item['base_name']}")
                print(f"    データ: L={item['data_comp']['L']}, B={item['data_comp']['B']}, el={item['data_comp']['el']}, H={item['data_comp']['H']}, W={item['data_comp']['W']}")
                print(f"    予測: L={item['pred_comp']['L']}, B={item['pred_comp']['B']}, el={item['pred_comp']['el']}, H={item['pred_comp']['H']}, W={item['pred_comp']['W']}")
    
    # 統計情報
    print("\n" + "=" * 80)
    print("[6] 統計情報")
    print("=" * 80)
    print(f"データファイル総数: {len(data_files)}")
    print(f"ラベルファイル総数: {len(label_files)}")
    if predict_dir:
        print(f"予測ファイル総数: {len(pred_files)}")
    print(f"完全一致ペア数: {len(matched_pairs)}")
    print(f"不一致ペア数: {len(mismatched_pairs)}")
    print(f"ラベルなしデータファイル数: {len(missing_labels)}")
    
    return {
        'matched_pairs': matched_pairs,
        'mismatched_pairs': mismatched_pairs,
        'missing_labels': missing_labels
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='データファイルとラベルファイルの対応を詳細に検証')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='データファイルのディレクトリ')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='ラベルファイルのディレクトリ')
    parser.add_argument('--predict_dir', type=str, default=None,
                       help='予測ファイルのディレクトリ（オプション）')
    
    args = parser.parse_args()
    
    verify_file_matching(args.data_dir, args.label_dir, args.predict_dir)


if __name__ == "__main__":
    main()
