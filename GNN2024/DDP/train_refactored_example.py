"""
リファクタリング後のトレーニングスクリプトの例
共通モジュールを使用してコードを簡潔に
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import argparse
import datetime

# 共通モジュールのインポート
from common.models import GCNModel, GATModel
from common.data_utils import create_data_label_pairs, prepare_data, compute_class_weights
from common.training_utils import set_seed, setup, cleanup, get_distributed_info
from common.metrics import calculate_metrics
from common.config import Config

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def main(args):
    """メイン関数"""
    # シード設定
    set_seed(args.seed)
    
    # 分散学習の設定
    rank, world_size, master_addr, master_port = get_distributed_info()
    
    if rank == 0:
        print(f"Rank: {rank}, World Size: {world_size}, Master Addr: {master_addr}, Master Port: {master_port}")
    
    # プロセスグループの初期化
    setup(rank, world_size)
    
    # デバイス設定
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank} using device: {device}")
    
    # 設定の読み込み
    if args.num_classes == 18:
        config = Config.for_18_classes()
    elif args.num_classes == 10:
        config = Config.for_10_classes()
    else:
        config = Config.default()
        config.model.num_classes = args.num_classes
    
    # ハイパーパラメータの上書き
    config.model.hidden_channels = args.hidden_channels
    config.model.model_type = args.model_type
    config.training.learning_rate = args.learning_rate
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.weight_decay = args.weight_decay
    config.training.patience = args.patience
    
    # データの読み込み
    x_coords = np.load(config.data.x_coords_path)[:config.data.max_nodes]
    y_coords = np.load(config.data.y_coords_path)[:config.data.max_nodes]
    z_coords = np.load(config.data.z_coords_path)[:config.data.max_nodes]
    edges = np.load(config.data.edges_path)
    edge_index = torch.tensor(edges.T, dtype=torch.long).to(device)
    
    # データファイルとラベルファイルのリストを取得
    data_files = [
        f for f in os.listdir(config.data.standardized_data_folder) 
        if f.startswith(config.data.data_file_prefix)
    ]
    label_files = [
        f for f in os.listdir(config.data.label_data_folder) 
        if f.startswith(config.data.label_file_prefix)
    ]
    
    # ペア作成
    pairs = create_data_label_pairs(data_files, label_files)
    if rank == 0:
        print(f"Total valid pairs: {len(pairs)}")
    
    # モデルの初期化
    if config.model.model_type == "GAT":
        model = GATModel(
            hidden_channels=config.model.hidden_channels,
            num_classes=config.model.num_classes,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout
        ).to(device)
    else:
        model = GCNModel(
            hidden_channels=config.model.hidden_channels,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout
        ).to(device)
    
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # K-Fold クロスバリデーション
    kf = KFold(n_splits=config.training.k_folds, shuffle=True, random_state=config.training.seed)
    fold = 1
    all_fold_metrics = []
    
    for train_index, val_index in kf.split(pairs):
        if rank == 0:
            print(f"Starting Fold {fold}")
        
        train_pairs = [pairs[i] for i in train_index]
        val_pairs = [pairs[i] for i in val_index]
        
        # データ準備（クラス重みが必要な場合は return_class_weights=True）
        train_dataset = prepare_data(
            train_pairs, 
            config.data.standardized_data_folder,
            config.data.label_data_folder,
            x_coords, y_coords, z_coords, edge_index,
            max_nodes=config.data.max_nodes,
            return_class_weights=False
        )
        val_dataset = prepare_data(
            val_pairs,
            config.data.standardized_data_folder,
            config.data.label_data_folder,
            x_coords, y_coords, z_coords, edge_index,
            max_nodes=config.data.max_nodes,
            return_class_weights=False
        )
        
        # DistributedSamplerの設定
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        
        # DataLoaderの設定
        train_loader = DataLoader(
            train_dataset, batch_size=config.training.batch_size, sampler=train_sampler
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.training.batch_size, sampler=val_sampler
        )
        
        # オプティマイザの設定
        optimizer = torch.optim.Adam(
            ddp_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # トレーニングループ
        best_val_loss = float('inf')
        counter = 0
        
        for epoch in range(1, config.training.epochs + 1):
            ddp_model.train()
            train_sampler.set_epoch(epoch)
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = ddp_model(batch)
                y = batch.y
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 損失の平均化
            avg_train_loss = torch.tensor(total_loss / len(train_loader), dtype=torch.float).to(device)
            dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
            avg_train_loss /= world_size
            
            # 検証
            ddp_model.eval()
            val_loss = 0.0
            correct = 0
            total_samples = 0
            all_preds = []
            all_labels = []
            
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
                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(y.cpu().numpy())
            
            # 検証損失と精度の平均化
            avg_val_loss = torch.tensor(val_loss / len(val_loader), dtype=torch.float).to(device)
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
            avg_val_loss /= world_size
            
            total_correct = torch.tensor(correct, dtype=torch.float).to(device)
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            total_correct /= world_size
            
            total_samples_tensor = torch.tensor(total_samples, dtype=torch.float).to(device)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            total_samples = total_samples_tensor.item() / world_size
            
            val_accuracy = total_correct.item() / total_samples_tensor.item()
            
            # Early Stopping
            if avg_val_loss.item() < best_val_loss:
                best_val_loss = avg_val_loss.item()
                counter = 0
                if rank == 0:
                    model_save_path = os.path.join(
                        config.training.model_save_dir,
                        f"{type(model).__name__}_{timestamp}_best_model_fold_{fold}.pth"
                    )
                    os.makedirs(config.training.model_save_dir, exist_ok=True)
                    torch.save(ddp_model.module.state_dict(), model_save_path)
            else:
                counter += 1
                if counter >= config.training.patience:
                    if rank == 0:
                        print(f"Early stopping triggered for Fold {fold} at epoch {epoch}")
                    break
            
            if rank == 0 and epoch % 10 == 0:
                print(f'Fold {fold}, Epoch {epoch}, Train Loss: {avg_train_loss.item():.4f}, '
                      f'Val Loss: {avg_val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}')
        
        if rank == 0:
            all_fold_metrics.append({
                "fold": fold,
                "best_val_loss": best_val_loss,
                "val_accuracy": val_accuracy
            })
            print(f"Completed Fold {fold}")
        fold += 1
    
    dist.barrier()
    
    # 最終テスト（rank 0のみ）
    if rank == 0:
        print("Starting final evaluation on test data...")
        
        train_val_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=config.training.seed)
        test_dataset = prepare_data(
            test_pairs,
            config.data.standardized_data_folder,
            config.data.label_data_folder,
            x_coords, y_coords, z_coords, edge_index,
            max_nodes=config.data.max_nodes,
            return_class_weights=False
        )
        
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
        
        # ベストモデルのロード
        model_path = os.path.join(
            config.training.model_save_dir,
            f"{type(model).__name__}_{timestamp}_best_model_fold_{all_fold_metrics[0]['fold']}.pth"
        )
        ddp_model.module.load_state_dict(torch.load(model_path))
        ddp_model.eval()
        
        # テスト評価
        correct = 0
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = ddp_model(batch)
                y = batch.y
                loss = loss_fn(out, y)
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_samples += y.size(0)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        test_loss = total_loss / len(test_loader)
        test_accuracy = correct / total_samples
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # メトリクス計算
        precision, recall, f1 = calculate_metrics(
            all_labels, all_preds, 
            num_classes=config.model.num_classes,
            show_plot=True
        )
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        # 最終モデルの保存
        final_model_path = os.path.join(
            config.training.model_save_dir,
            f"{type(model).__name__}_{timestamp}_Final.pth"
        )
        torch.save(ddp_model.module.state_dict(), final_model_path)
    
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refactored Distributed Training Script')
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'GAT'],
                       help='Model type: GCN or GAT')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels')
    parser.add_argument('--num_classes', type=int, default=19, help='Number of classes')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    main(args)