"""
保存されたモデルを使用して予測を実行するスクリプト
モデル: GATModel_20250126_173116_Final.pth
入力: 別のデータセット（正規化済みDSPSSデータ）
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

# モデル定義（GNN_88_sub.pyと同じ構造）
def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

class GATModel(torch.nn.Module):
    def __init__(self, hidden_channels=64, num_classes=19):
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

        # Layer 1 with residual connection
        x_residual = x
        x = F.relu(self.conv1(x, edge_index))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        if x_residual.size(1) != x.size(1):
            x_residual = self.proj1(x_residual)
        x = x + x_residual

        # Layer 2 with residual connection
        x_residual = x
        x = F.relu(self.conv2(x, edge_index))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        if x_residual.size(1) != x.size(1):
            x_residual = self.proj2(x_residual)
        x = x + x_residual

        # Layer 3 with residual connection
        x_residual = x
        x = F.relu(self.conv3(x, edge_index))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        if x_residual.size(1) != x.size(1):
            x_residual = self.proj3(x_residual)
        x = x + x_residual

        # Fully connected layer
        x = self.fc(x)
        return x


def prepare_prediction_data(data_folder, x_coords, y_coords, z_coords, edge_index, num_nodes=13942, random_sample=None, random_seed=42):
    """
    予測用データを準備（ラベルファイルは不要）
    
    Args:
        data_folder: データフォルダのパス
        x_coords, y_coords, z_coords: 座標データ
        edge_index: エッジインデックス
        num_nodes: ノード数
        random_sample: ランダムに選択するファイル数（Noneの場合は全ファイル）
        random_seed: ランダムシード
    """
    import random
    data_list = []
    filenames = []
    
    # データファイルを取得
    all_data_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')])
    
    print(f"Found {len(all_data_files)} data files in {data_folder}")
    
    # ランダムサンプリング
    if random_sample is not None and random_sample > 0:
        random.seed(random_seed)
        data_files = random.sample(all_data_files, min(random_sample, len(all_data_files)))
        print(f"ランダムに {len(data_files)} ファイルを選択しました（シード: {random_seed}）")
    else:
        data_files = all_data_files
    
    total_files = len(data_files)
    for data_file in data_files:
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
        
        if len(data_list) % 100 == 0:
            print(f"  処理中: {len(data_list)}/{total_files} ファイル")

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
            
            # デバッグ: 最初のバッチの出力を確認
            if batch_idx == 0:
                print(f"  デバッグ: 最初のバッチの予測結果")
                print(f"    出力形状: {out.shape}")
                print(f"    予測クラス: {np.unique(pred, return_counts=True)}")
                print(f"    出力の最大値: {out.max(dim=1)[0].cpu().numpy()[:10]}")
                print(f"    出力の最小値: {out.min(dim=1)[0].cpu().numpy()[:10]}")
    
    return np.array(all_predictions)


def main():
    parser = argparse.ArgumentParser(description='保存されたモデルで予測を実行')
    parser.add_argument('--model_path', type=str, 
                       default='/home/nishioka/GNN/GNN_hole/GNN_model/19classmodel_hole/GATModel_20250126_173116_Final.pth',
                       help='モデルファイルのパス')
    parser.add_argument('--input_data_folder', type=str, required=True,
                       help='予測するデータフォルダのパス（正規化済みDSPSSデータ）')
    parser.add_argument('--output_folder', type=str, default=None,
                       help='予測結果の保存先フォルダ（指定しない場合は自動生成）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='バッチサイズ')
    parser.add_argument('--hidden_channels', type=int, default=64,
                       help='モデルの隠れ層チャネル数')
    parser.add_argument('--num_classes', type=int, default=19,
                       help='クラス数')
    parser.add_argument('--num_nodes', type=int, default=13942,
                       help='ノード数（デフォルト: 13942）')
    parser.add_argument('--random_sample', type=int, default=50,
                       help='ランダムに選択するファイル数（デフォルト: 10）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='ランダムシード（デフォルト: 42）')
    parser.add_argument('--visualize', action='store_true',
                       help='予測後に可視化を実行')
    parser.add_argument('--label_dir', type=str, 
                       default='/home/nishioka/GNN/GNN_hole_2026/Def4x4_19class_label',
                       help='可視化用のラベルファイルディレクトリ')
    parser.add_argument('--dpsss_dir', type=str,
                       default='/home/nishioka/GNN/GNN_hole_2026/Defect_hole_4x4_Region1_21_npy_subtracted_normalized',
                       help='可視化用のDSPSSデータディレクトリ')
    parser.add_argument('--max_visualize', type=int, default=50,
                       help='可視化する最大ファイル数（デフォルト: 10）')
    
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
        random_sample=args.random_sample,
        random_seed=args.random_seed
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
    print(f"  ユニークな値: {np.unique(all_predictions, return_counts=True)}")
    print(f"  最小値: {np.min(all_predictions)}, 最大値: {np.max(all_predictions)}")
    if len(all_predictions) > 0:
        print(f"  最初の10個の予測: {all_predictions[:10]}")
    
    # 出力フォルダを設定
    if args.output_folder is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_folder_name = Path(args.input_data_folder).name
        args.output_folder = f"/home/nishioka/GNN/GNN_hole_2026/Predict_data/Pred_{input_folder_name}_{timestamp}"
    
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"\n予測結果を保存中: {args.output_folder}")
    
    # メタデータを保存（入力データフォルダのパス）
    metadata = {
        'input_data_folder': args.input_data_folder,
        'num_files': len(filenames),
        'num_nodes': args.num_nodes,
        'num_classes': args.num_classes
    }
    import json
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
        
        # デバッグ: 最初のファイルの予測結果を確認
        if i == 0:
            print(f"\nデバッグ: 最初のファイルの予測結果")
            print(f"  start_idx: {start_idx}, end_idx: {end_idx}")
            print(f"  sample_predsの形状: {sample_preds.shape}")
            print(f"  sample_predsのユニーク値: {np.unique(sample_preds, return_counts=True)}")
            print(f"  sample_predsの最初の10個: {sample_preds[:10]}")
        
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
    if args.visualize:
        print("\n可視化を実行中...")
        try:
            # 可視化スクリプトをインポート
            import sys
            visualize_script_path = os.path.join(os.path.dirname(__file__), 'visualize_predictions.py')
            if os.path.exists(visualize_script_path):
                from visualize_predictions import find_matching_files, visualize_comparison
                
                # ファイルの対応付け（入力データフォルダを優先的に使用）
                file_triplets = find_matching_files(args.output_folder, args.label_dir, args.dpsss_dir, input_data_dir=args.input_data_folder)
                
                if len(file_triplets) > 0:
                    # 可視化結果の保存先
                    vis_output_dir = os.path.join(args.output_folder, 'visualizations')
                    # 入力データフォルダをDSPSSディレクトリとして使用
                    visualize_comparison(file_triplets, vis_output_dir, args.max_visualize, input_data_dir=args.input_data_folder)
                    print(f"\n可視化完了! 結果は {vis_output_dir} に保存されました")
                else:
                    print("警告: 可視化用の対応ファイルが見つかりませんでした")
            else:
                print(f"警告: 可視化スクリプトが見つかりません: {visualize_script_path}")
                print("可視化スクリプトを直接実行してください:")
                print(f"  python {visualize_script_path} --predict_dir {args.output_folder}")
        except Exception as e:
            print(f"可視化中にエラーが発生しました: {e}")
            print("可視化スクリプトを直接実行してください:")
            print(f"  python {os.path.join(os.path.dirname(__file__), 'visualize_predictions.py')} --predict_dir {args.output_folder}")


if __name__ == "__main__":
    main()
