import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import datetime
from torch.utils.data import DistributedSampler

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ----------------------------
# モデル定義
# ----------------------------
class GCNModel(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_classes=18):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.fc(x)
        return x  # log_softmaxを削除し、生のロジットを返す

# ----------------------------
# ペアリング関数
# ----------------------------
def extract_layer_block(file_name):
    """ファイル名から層とブロック番号を抽出"""
    try:
        # ファイル名から 'L' と 'B' に続く数字を抽出
        layer_block_str = re.search(r'L(\d+)B(\d+)', file_name).groups()
        layer = int(layer_block_str[0])
        block = int(layer_block_str[1])
        return (layer, block)
    except AttributeError:
        print(f"Invalid file name format: {file_name}")
        return None

def create_data_label_pairs(data_files, label_files):
    """データファイルとラベルファイルのペアを作成"""
    data_label_pairs = {}
    for data_file in data_files:
        layer_block = extract_layer_block(data_file)
        if layer_block:
            data_label_pairs[layer_block] = {"data": data_file}

    for label_file in label_files:
        layer_block = extract_layer_block(label_file)
        if layer_block and layer_block in data_label_pairs:
            data_label_pairs[layer_block]["label"] = label_file

    # 有効なペアのみ取得
    valid_pairs = [(v["data"], v["label"]) for k, v in data_label_pairs.items() if "label" in v]
    return valid_pairs

# ----------------------------
# データ準備関数
# ----------------------------
def prepare_data(pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index):
    data_list = []
    for data_file, label_file in pairs:
        data_file_path = os.path.join(standardized_data_folder, data_file)
        label_file_path = os.path.join(label_data_folder, label_file)

        # Load data and labels
        values = np.load(data_file_path)[:3654]
        label = np.load(label_file_path)[:3654]

        # Create node features
        node_features = np.vstack((x_coords, y_coords, z_coords, values)).T
        x = torch.tensor(node_features, dtype=torch.float)

        # ターゲットがワンホットエンコーディングの場合、クラスインデックスに変換
        y = torch.argmax(torch.tensor(label, dtype=torch.float), dim=1).long()

        # デバッグ用にターゲットの型とサンプルを出力
        print(f"Data file: {data_file}, Label file: {label_file}")
        print(f"y type: {y.dtype}, y sample: {y[:5]}")
        print(f"Unique labels in y: {torch.unique(y)}")

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

# ----------------------------
# シード設定関数
# ----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPUの場合
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# プロセスグループの初期化
# ----------------------------
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

# ----------------------------
# プロセスグループのクリーンアップ
# ----------------------------
def cleanup():
    dist.destroy_process_group()

# ----------------------------
# メイン関数
# ----------------------------
def main(args):
    # シード設定
    set_seed(42)

    # 環境変数からランクと世界サイズを取得
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')

    if rank == 0:
        print(f"Rank: {rank}, World Size: {world_size}, Master Addr: {master_addr}, Master Port: {master_port}")

    # プロセスグループの初期化
    setup(rank, world_size)

    # デバイス設定
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank} using device: {device}")

    # データフォルダの設定
    standardized_data_folder = "/home/nishioka/GNN/Defect_4x4_Normalized1"
    label_data_folder = "/home/nishioka/GNN/DefectClass_OneHot_test1"

    # 座標データの読み込み
    x_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/x_2layer_normalized.npy")[:3654]
    y_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/y_2layer_normalized.npy")[:3654]
    z_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/z_2layer_normalized.npy")[:3654]

    # エッジ情報の読み込み
    edges = np.load("/home/nishioka/GNN/BasicdataforGNN/edges_2layer.npy")
    edge_index = torch.tensor(edges.T, dtype=torch.long).to(device)

    # データファイルとラベルファイルのリストを取得
    data_files = [f for f in os.listdir(standardized_data_folder) if f.startswith("Normalized1_Defect4x4_ELNOD")]
    label_files = [f for f in os.listdir(label_data_folder) if f.startswith("DefectClass_L")]

    # データとラベルのペア作成（以前の方法を使用）
    pairs = create_data_label_pairs(data_files, label_files)
    print(f"Total valid pairs: {len(pairs)}")

    # モデルの初期化
    model = GCNModel(hidden_channels=args.hidden_channels, num_classes=18).to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # クロスエントロピー損失
    loss_fn = nn.CrossEntropyLoss().to(device)

    # K-Fold クロスバリデーションの設定
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_fold_metrics = []
    for train_index, val_index in kf.split(pairs):
        if rank == 0:
            print(f"Starting Fold {fold}")

        train_pairs = [pairs[i] for i in train_index]
        val_pairs = [pairs[i] for i in val_index]

        # データ準備
        train_dataset = prepare_data(train_pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index)
        val_dataset = prepare_data(val_pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index)

        # DistributedSamplerの使用
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # DataLoaderの設定
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

        # オプティマイザの設定
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # トレーニングループ
        best_val_loss = float('inf')
        counter = 0
        for epoch in range(1, args.epochs + 1):
            ddp_model.train()
            train_sampler.set_epoch(epoch)
            total_loss = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = ddp_model(batch)
                y = batch.y

                # デバッグ用に出力とターゲットの型と形状を出力
                if rank == 0:
                    print(f"out dtype: {out.dtype}, out shape: {out.shape}")
                    print(f"y dtype: {y.dtype}, y shape: {y.shape}")
                    print(f"Unique labels in y: {torch.unique(y)}")

                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 全プロセスで損失を平均
            avg_train_loss = torch.tensor(total_loss / len(train_loader), dtype=torch.float).to(device)
            dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
            avg_train_loss /= world_size

            # 検証
            ddp_model.eval()
            val_loss = 0.0
            correct = 0
            total_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = ddp_model(batch)
                    y = batch.y

                    loss = loss_fn(out, y)
                    val_loss += loss.item()

                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total_samples += y.size(0)

            # 検証損失と精度の平均
            avg_val_loss = torch.tensor(val_loss / len(val_loader), dtype=torch.float).to(device)
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
            avg_val_loss /= world_size

            # 正解数の合計
            total_correct = torch.tensor(correct, dtype=torch.float).to(device)
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            total_correct /= world_size

            # 全サンプル数の合計
            total_samples_tensor = torch.tensor(total_samples, dtype=torch.float).to(device)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            total_samples = total_samples_tensor.item() / world_size

            val_accuracy = total_correct.item() / total_samples_tensor.item()

            # Early Stopping
            if avg_val_loss.item() < best_val_loss:
                best_val_loss = avg_val_loss.item()
                counter = 0
                # モデルの保存（rank 0のみ）
                if rank == 0:
                    torch.save(ddp_model.module.state_dict(), f"/home/nishioka/GNN/GNNmodel/18classmodel/{type(model).__name__}_{timestamp}best_model_fold_{fold}.pth")
            else:
                counter += 1
                if counter >= args.patience:
                    if rank == 0:
                        print(f"Early stopping triggered for Fold {fold} at epoch {epoch}")
                    break

            if rank == 0:
                print(f'Fold {fold}, Epoch {epoch}, Train Loss: {avg_train_loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}')

        # ベストメトリクスの収集（rank 0のみ）
        if rank == 0:
            all_fold_metrics.append({
                "fold": fold,
                "best_val_loss": best_val_loss,
                "val_accuracy": val_accuracy
            })
            print(f"Completed Fold {fold}")
        fold += 1

    # 同期ポイント
    dist.barrier()

    # テストフェーズ（rank 0のみ）
    if rank == 0:
        # 全てのデータを使用して最終トレーニング
        print("Starting final training on all data...")

        train_val_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
        train_val_dataset = prepare_data(train_val_pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index)
        test_dataset = prepare_data(test_pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index)

        # DataLoaderの設定（DistributedSamplerは使用しない）
        final_train_loader = DataLoader(train_val_dataset, batch_size=args.batch_size, shuffle=True)
        final_test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # モデルのロード（ベストモデル）
        model_path = f"/home/nishioka/GNN/GNNmodel/18classmodel/{type(model).__name__}_{timestamp}best_model_fold_{all_fold_metrics[0]['fold']}.pth"  # 最初のフォールドのベストモデルを使用
        ddp_model.module.load_state_dict(torch.load(model_path))
        ddp_model.eval()

        # テスト
        correct = 0
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in final_test_loader:
                batch = batch.to(device)
                out = ddp_model(batch)
                y = batch.y

                loss = loss_fn(out, y)
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_samples += y.size(0)

        test_loss = total_loss / len(final_test_loader)
        test_accuracy = correct / total_samples
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # クリーンアップ
    cleanup()

# ----------------------------
# エントリーポイント
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training Script')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=300, help='Early stopping patience') 

    args = parser.parse_args()

    main(args)
