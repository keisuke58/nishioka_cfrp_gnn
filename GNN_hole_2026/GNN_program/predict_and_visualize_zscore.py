"""
指定されたモデルを使用して予測を実行し、結果を可視化するスクリプト
モデル: GATModel_20260110_211113_Best_Epoch100_ValLoss0.7966_MacroF10.2868.pth
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse
import datetime
from pathlib import Path
import json
import sys

# モデル定義（GNN_zscore_sub.pyと同じ構造）
def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def edge_dropout(edge_index, drop_prob=0.0002):
    """エッジドロップアウト（トレーニング時のみ使用）"""
    if drop_prob == 0.0:
        return edge_index
    mask = torch.rand(edge_index.size(1)) > drop_prob
    edge_index_dropped = edge_index[:, mask]
    return edge_index_dropped

class GATModel(torch.nn.Module):
    def __init__(self, hidden_channels=8, num_classes=19):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(4, hidden_channels, heads=4, concat=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels * 4)
        
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels * 2, heads=4, concat=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels * 8)
        
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=4, concat=True)
        self.batch_norm3 = nn.BatchNorm1d(hidden_channels * 4)
        
        self.fc = nn.Linear(hidden_channels * 4, num_classes)
        self.dropout = nn.Dropout(p=0.0002)

        # プロジェクションレイヤー
        self.proj1 = nn.Linear(4, hidden_channels * 4)
        self.proj2 = nn.Linear(hidden_channels * 4, hidden_channels * 8)
        self.proj3 = nn.Linear(hidden_channels * 8, hidden_channels * 4)

        self.apply(initialize_weights)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # エッジドロップアウトの適用（トレーニング時のみ）
        if self.training:
            edge_index = edge_dropout(edge_index, drop_prob=0.0002)

        # Layer 1 with residual connection
        x_res = self.proj1(x)
        x = F.relu(self.conv1(x, edge_index)) + x_res
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        # Layer 2 with residual connection
        x_res = self.proj2(x)
        x = F.relu(self.conv2(x, edge_index)) + x_res
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        # Layer 3 with residual connection
        x_res = self.proj3(x)
        x = F.relu(self.conv3(x, edge_index)) + x_res
        x = self.batch_norm3(x)
        x = self.dropout(x)
        
        # Fully connected layer
        x = self.fc(x)
        return x


def prepare_prediction_data(data_folder, x_coords, y_coords, z_coords, edge_index, num_nodes=13942, max_files=None):
    """
    予測用データを準備
    
    Args:
        data_folder: データフォルダのパス
        x_coords, y_coords, z_coords: 座標データ
        edge_index: エッジインデックス
        num_nodes: ノード数
        max_files: 最大ファイル数（Noneの場合は全ファイル）
    """
    import random
    import re
    
    data_list = []
    filenames = []
    
    # データファイルを取得
    all_data_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
    
    # Lの値でグループ化
    files_by_L = {}
    for f in all_data_files:
        match = re.search(r'L(\d+)_', f)
        if match:
            L_value = int(match.group(1))
            if L_value not in files_by_L:
                files_by_L[L_value] = []
            files_by_L[L_value].append(f)
    
    print(f"Found {len(all_data_files)} data files in {data_folder}")
    print(f"Files grouped by L value: {len(files_by_L)} different L values")
    for L_val in sorted(files_by_L.keys()):
        print(f"  L{L_val}: {len(files_by_L[L_val])} files")
    
    # max_filesが指定されている場合、各Lから均等にサンプリング
    if max_files is not None and max_files > 0:
        selected_files = []
        num_L_values = len(files_by_L)
        files_per_L = max(1, max_files // num_L_values)  # 各Lから最低1ファイル
        
        for L_val in sorted(files_by_L.keys()):
            L_files = sorted(files_by_L[L_val])  # 各L内でソート
            # 各Lから均等にサンプリング（ランダムではなく順番に）
            selected_from_L = L_files[:min(files_per_L, len(L_files))]
            selected_files.extend(selected_from_L)
        
        # まだmax_filesに達していない場合、残りをランダムに追加
        if len(selected_files) < max_files:
            remaining_files = [f for f in all_data_files if f not in selected_files]
            random.shuffle(remaining_files)
            selected_files.extend(remaining_files[:max_files - len(selected_files)])
        
        all_data_files = selected_files
        print(f"Selected {len(all_data_files)} files (balanced across L values)")
    else:
        # max_filesが指定されていない場合は全ファイルをソート
        all_data_files = sorted(all_data_files)
    
    total_files = len(all_data_files)
    for idx, data_file in enumerate(all_data_files):
        data_file_path = os.path.join(data_folder, data_file)
        
        if not os.path.exists(data_file_path):
            print(f"警告: データファイルが存在しません: {data_file_path}")
            continue

        # データをロード
        try:
            values = np.load(data_file_path)
            # ノード数を確認・調整
            if len(values) < num_nodes:
                print(f"警告: {data_file} のノード数が不足しています ({len(values)} < {num_nodes})")
                # ゼロパディング
                values = np.pad(values, (0, num_nodes - len(values)), mode='constant')
            elif len(values) > num_nodes:
                values = values[:num_nodes]
            
            # ノード特徴量を作成 [x, y, z, value]
            node_features = np.vstack((x_coords, y_coords, z_coords, values)).T
            x = torch.tensor(node_features, dtype=torch.float)
            
            # ラベルは不要なのでダミーを作成（Dataオブジェクトに必要）
            y = torch.zeros(num_nodes, dtype=torch.long)
            
        except Exception as e:
            print(f"エラー: {data_file} の読み込みに失敗しました: {e}")
            continue

        # データリストに追加
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
        filenames.append(data_file)
        
        if (idx + 1) % 100 == 0:
            print(f"  処理中: {idx + 1}/{total_files} ファイル")

    print(f"準備完了: {len(data_list)} ファイル")
    return data_list, filenames


def predict(model, data_loader, device):
    """
    モデルを使用して予測を実行
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1).cpu().numpy()
            all_predictions.extend(pred)
            
            if batch_idx == 0:
                print(f"  デバッグ: 最初のバッチの予測結果")
                print(f"    出力形状: {out.shape}")
                print(f"    予測クラス: {np.unique(pred, return_counts=True)}")
    
    return np.array(all_predictions)


def main():
    parser = argparse.ArgumentParser(description='保存されたモデルで予測を実行し、可視化')
    parser.add_argument('--model_path', type=str, 
                       default='/home/nishioka/GNN/GNN_hole_2026/GNN_model/19classmodel_hole_zscore/GATModel_20260110_211113_Best_Epoch100_ValLoss0.7966_MacroF10.2868.pth',
                       help='モデルファイルのパス')
    parser.add_argument('--input_data_folder', type=str, 
                       default='/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore/test',
                       help='予測するデータフォルダのパス')
    parser.add_argument('--output_folder', type=str, default=None,
                       help='予測結果の保存先フォルダ（指定しない場合は自動生成）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='バッチサイズ')
    parser.add_argument('--hidden_channels', type=int, default=16,
                       help='モデルの隠れ層チャネル数（デフォルト: 16）')
    parser.add_argument('--num_classes', type=int, default=19,
                       help='クラス数')
    parser.add_argument('--num_nodes', type=int, default=13942,
                       help='ノード数（デフォルト: 13942）')
    parser.add_argument('--max_files', type=int, default=None,
                       help='処理する最大ファイル数（Noneの場合は全ファイル）')
    parser.add_argument('--max_visualize', type=int, default=50,
                       help='可視化する最大ファイル数（デフォルト: 50）')
    parser.add_argument('--label_dir', type=str, 
                       default='/home/nishioka/GNN/GNN_hole_2026/all_19class_label',
                       help='可視化用のラベルファイルディレクトリ')
    parser.add_argument('--dpsss_dir', type=str,
                       default='/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore',
                       help='可視化用のDSPSSデータディレクトリ')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # 座標データとエッジデータを読み込み
    print("\n座標データとエッジデータを読み込み中...")
    x_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
    y_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
    z_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")
    
    edges = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy")
    edge_index = torch.tensor(edges.T, dtype=torch.long).to(device)
    
    print(f"  ノード数: {len(x_coords)}")
    print(f"  エッジ数: {edge_index.shape[1]}")
    
    # モデルを読み込み
    print(f"\nモデルを読み込み中: {args.model_path}")
    model = GATModel(hidden_channels=args.hidden_channels, num_classes=args.num_classes).to(device)
    
    # モデルの重みを読み込み
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 直接state_dictの場合
        model.load_state_dict(checkpoint)
    
    print("モデルの読み込み完了")
    
    # 予測データを準備
    print(f"\n予測データを準備中: {args.input_data_folder}")
    data_list, filenames = prepare_prediction_data(
        args.input_data_folder,
        x_coords, y_coords, z_coords,
        edge_index,
        num_nodes=args.num_nodes,
        max_files=args.max_files
    )
    
    if len(data_list) == 0:
        print("エラー: 予測可能なデータがありません")
        return
    
    # DataLoaderを作成
    data_loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=False)
    
    # 予測を実行
    print("\n予測を実行中...")
    all_predictions = predict(model, data_loader, device)
    
    # デバッグ: 予測結果の統計を表示
    print(f"\n予測結果の統計:")
    print(f"  総予測数: {len(all_predictions)}")
    unique, counts = np.unique(all_predictions, return_counts=True)
    print(f"  ユニークな値: {dict(zip(unique, counts))}")
    print(f"  最小値: {np.min(all_predictions)}, 最大値: {np.max(all_predictions)}")
    
    # 出力フォルダを設定
    if args.output_folder is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_folder_name = Path(args.input_data_folder).name
        args.output_folder = f"/home/nishioka/GNN/GNN_hole_2026/Predict_data/Pred_{input_folder_name}_{timestamp}"
    
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"\n予測結果を保存中: {args.output_folder}")
    
    # メタデータを保存
    metadata = {
        'model_path': args.model_path,
        'input_data_folder': args.input_data_folder,
        'num_files': len(filenames),
        'num_nodes': args.num_nodes,
        'num_classes': args.num_classes,
        'hidden_channels': args.hidden_channels
    }
    metadata_path = os.path.join(args.output_folder, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # npyファイルを1つのフォルダにまとめて保存
    npy_folder = os.path.join(args.output_folder, 'predictions')
    os.makedirs(npy_folder, exist_ok=True)
    
    # ファイルごとに予測結果を保存
    num_nodes_per_file = args.num_nodes
    for i, filename in enumerate(filenames):
        start_idx = i * num_nodes_per_file
        end_idx = start_idx + num_nodes_per_file
        
        if end_idx > len(all_predictions):
            print(f"警告: {filename} の予測データが不足しています")
            continue
        
        sample_preds = all_predictions[start_idx:end_idx]
        
        base_filename = os.path.splitext(filename)[0]
        pred_filename = f"{base_filename}_pred.npy"
        output_path = os.path.join(npy_folder, pred_filename)
        
        np.save(output_path, sample_preds)
        
        if (i + 1) % 100 == 0:
            print(f"  保存中: {i + 1}/{len(filenames)} ファイル")
    
    print(f"\n予測完了!")
    print(f"  処理ファイル数: {len(filenames)}")
    print(f"  保存先: {args.output_folder}")
    
    # 可視化を実行
    print("\n可視化を実行中...")
    try:
        # 可視化スクリプトのパス
        visualize_script_path = os.path.join(os.path.dirname(__file__), 'visualize_predictions.py')
        if os.path.exists(visualize_script_path):
            # 可視化スクリプトをインポート
            sys.path.insert(0, os.path.dirname(visualize_script_path))
            from visualize_predictions import find_matching_files, visualize_comparison
            
            # ファイルの対応付け
            file_triplets = find_matching_files(
                args.output_folder, 
                args.label_dir, 
                args.dpsss_dir, 
                input_data_dir=args.input_data_folder
            )
            
            if len(file_triplets) > 0:
                # 可視化結果の保存先
                vis_output_dir = os.path.join(args.output_folder, 'visualizations')
                # 可視化を実行
                visualize_comparison(
                    file_triplets, 
                    vis_output_dir, 
                    args.max_visualize, 
                    input_data_dir=args.input_data_folder
                )
                print(f"\n可視化完了! 結果は {vis_output_dir} に保存されました")
            else:
                print("警告: 可視化用の対応ファイルが見つかりませんでした")
        else:
            print(f"警告: 可視化スクリプトが見つかりません: {visualize_script_path}")
            print("可視化スクリプトを直接実行してください:")
            print(f"  python {visualize_script_path} --predict_dir {args.output_folder}")
    except Exception as e:
        import traceback
        print(f"可視化中にエラーが発生しました: {e}")
        traceback.print_exc()
        print("可視化スクリプトを直接実行してください:")
        print(f"  python {os.path.join(os.path.dirname(__file__), 'visualize_predictions.py')} --predict_dir {args.output_folder}")


if __name__ == "__main__":
    main()
