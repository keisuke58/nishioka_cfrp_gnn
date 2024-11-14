import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import datetime
from torch.utils.data import DistributedSampler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score, roc_auc_score, roc_curve, auc
import optuna
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from Loss.focal_loss import FocalLoss
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import warnings
import optuna
warnings.filterwarnings("ignore")



timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# モデル定義 (GAT)
# ----------------------------

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class GATModel(torch.nn.Module):
    def __init__(self, hidden_channels=64, num_classes=19):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(4, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels * 2, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=4, concat=True)
        self.conv4 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.fc = nn.Linear(hidden_channels * 4, num_classes)
        self.dropout = nn.Dropout(p=0.2)

        self.apply(initialize_weights)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.fc(x)

        x = F.softmax(x, dim=-1)

        return x

# ----------------------------
# サンプル数に基づく重み付けを計算する関数
# ----------------------------


# gamma = 1.5
gamma = 1.75
# gamma = 2.0
# gamma = 3.0
# gamma = 5.0
# gamma = 10.0

def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # 正規化して合計が1になるようにする
    multiplier = 1
    class_weights[1:] *= multiplier

    # print("Class weights:", class_weights)
    
    return torch.tensor(class_weights, dtype=torch.float)

def evaluate_and_visualize(final_test_loader, final_train_loader, ddp_model, device, class_weights, test_pairs, train_pairs):
    ddp_model.eval()# モデルを評価モードに切り替え
    all_preds = []
    all_labels = []
    all_probs = []  # 予測確率を保存するリストを追加
    total_loss = 0.0
    correct = 0
    total_samples = 0

    
    with torch.no_grad():  # 評価時は勾配計算を無効化
        for batch in final_test_loader:
            batch = batch.to(device)
            out = ddp_model(batch)
            y = batch.y

            # 予測確率を保存
            all_probs.extend(F.softmax(out, dim=1).cpu().numpy())  # softmaxで確率を取得して保存
            all_labels.extend(y.cpu().numpy())

            # クラスごとの予測
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())  # 各ノードの予測を保存
            correct += (pred == y).sum().item()
            total_samples += y.size(0)

    test_accuracy = correct / total_samples
    avg_test_loss = total_loss / len(final_test_loader)

    # --- 出力ディレクトリの作成 ---
    output_dir_test_predict = f"/home/nishioka/GNN/Predict_data/Predict19Class4x4_Test{timestamp}"
    os.makedirs(output_dir_test_predict, exist_ok=True)
    
    # --- テストデータファイルごとに3654個の予測結果を保存 ---
    num_nodes_per_sample = 3654  # 各サンプルごとのデータ数を指定

    # テストデータごとの予測結果を保存
    for i, filename_tuple in enumerate(test_pairs):
        filename = filename_tuple[0] if isinstance(filename_tuple, tuple) else filename_tuple
        start_idx = i * num_nodes_per_sample
        end_idx = start_idx + num_nodes_per_sample
        sample_preds = np.array(all_preds[start_idx:end_idx])  # 各ファイルに対応する3654個のノードの予測を抽出
        
        # Format the filename to have a "_pred.npy" suffix
        base_filename = os.path.splitext(filename)[0]  # Remove the original file extension
        pred_filename = f"{base_filename}_pred.npy"
        
        np.save(f"{output_dir_test_predict}/{pred_filename}", sample_preds)
        
        # if i < 3:
        # print(f"Saving predictions for {filename} as {pred_filename}, Shape of predictions: {sample_preds.shape}")
    
    print(f"Test data predictions saved for all test samples in: {output_dir_test_predict}")   


    # --- 学習データ評価 ---
    all_preds = []  # リセット
    all_labels = []
    all_probs = []
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in final_train_loader:
            batch = batch.to(device)
            out = ddp_model(batch)
            y = batch.y

            all_probs.extend(F.softmax(out, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())  # 各ノードの予測を保存
            correct += (pred == y).sum().item()
            total_samples += y.size(0)

    train_accuracy = correct / total_samples
    avg_train_loss = total_loss / len(final_train_loader)

    # --- 出力ディレクトリの作成 ---
    output_dir_train_predict = f"/home/nishioka/GNN/Predict_data/Predict19Class4x4_Train{timestamp}"
    os.makedirs(output_dir_train_predict, exist_ok=True)
    
    # --- 学習データファイルごとに3654個の予測結果を保存 ---
    num_nodes_per_sample = 3654  # 各サンプルごとのデータ数を指定

    # 学習データごとの予測結果を保存
    for i, filename_tuple in enumerate(train_pairs):
        filename = filename_tuple[0] if isinstance(filename_tuple, tuple) else filename_tuple
        start_idx = i * num_nodes_per_sample
        end_idx = start_idx + num_nodes_per_sample
        sample_preds = np.array(all_preds[start_idx:end_idx])  # 各ファイルに対応する3654個のノードの予測を抽出
        
        # Format the filename to have a "_pred.npy" suffix
        base_filename = os.path.splitext(filename)[0]  # Remove the original file extension
        pred_filename = f"{base_filename}_train_pred.npy"
        
        np.save(f"{output_dir_train_predict}/{pred_filename}", sample_preds)
    
    print(f"Training data predictions saved for all training samples in: {output_dir_train_predict}")


    output_dir1 = f"/home/nishioka/GNN/Predict_data/Predict19Class4x4_Test{timestamp}/pred_data_{timestamp}"
    os.makedirs(output_dir1, exist_ok=True)
    
    # 予測値・ラベル・確率を保存し、形状を確認
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)
    
    # それぞれの形状を確認して出力
    print("Shape of all_preds:", all_preds_np.shape)
    print("Shape of all_labels:", all_labels_np.shape)
    print("Shape of all_probs:", all_probs_np.shape)
    
    np.save(os.path.join(output_dir1, "all_preds.npy"), all_preds_np)
    np.save(os.path.join(output_dir1, "all_labels.npy"), all_labels_np)
    np.save(os.path.join(output_dir1, "all_probs.npy"), all_probs_np)

    
    # Confusion Matrix と Classification Report の表示
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    class_report = classification_report(all_labels, all_preds, zero_division=0)

    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"Balanced Accuracy: {balanced_acc}")
    
    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(all_labels, all_preds)
    print(f"Matthews Correlation Coefficient: {mcc}")

    # ROC AUCスコア計算
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        roc_auc = "ROC AUC calculation failed due to missing classes in prediction."
    
    print(f"ROC AUC Score: {roc_auc}")

    # 混同行列の可視化
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(19), yticklabels=range(19))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print(f"\nClassification Report:\n{class_report}")
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"ROC AUC Score: {roc_auc}")


    # Directory for saving images
    output_dir = f"/home/nishioka/GNN/Predict_truth/pred_vs_true_images_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
        
    # 1. 予測値 vs. 正解ラベルの散布図
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(all_labels)), all_labels, color="blue", label="Actual Labels", alpha=0.6)
    plt.scatter(range(len(all_preds)), all_preds, color="red", marker="x", label="Predicted Labels", alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("Predicted vs. Actual Class Labels")
    plt.legend()
    plt.savefig(f"{output_dir}/pred_vs_true_class_labels{timestamp}.png")  # Save as PNG
    plt.close()

    # 2. 混同行列のヒートマップ
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(19), yticklabels=range(19))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix{timestamp}.png")
    plt.close()

    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(all_labels, all_preds)
    # Apply log scale to avoid overwhelming contrast
    log_cm = np.log1p(cm)  # Use log1p to handle large values in a more balanced way
    # Set up the figure and heatmap for the PDF
    ax = sns.heatmap(log_cm, annot=cm, fmt="d", cmap="Blues", 
                     cbar_kws={'label': 'Counts (Log Scale)'}, vmin=0, vmax=7, square=True, 
                     annot_kws={"size": 8, "weight": "bold"}, linewidths=0.01, linecolor="gray")
    
    # Highlight the diagonal with red borders for correct classifications
    for i in range(cm.shape[0]):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    # Add labels and title
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix 2", fontsize=15)
    plt.savefig(f"{output_dir}/confusion_matrix2{timestamp}.png")

    # # 3. ROC曲線
    # if len(set(all_labels)) > 2:  # マルチクラスの場合
    #     fpr = {}
    #     tpr = {}
    #     roc_auc = {}
    #     for i in range(len(set(all_labels))):  # クラス数だけループ
    #         fpr[i], tpr[i], _ = roc_curve([1 if x == i else 0 for x in all_labels],
    #                                       [prob[i] for prob in all_probs])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #         plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    # else:  # 2クラスの場合
    #     fpr, tpr, _ = roc_curve(all_labels, all_probs)
    #     roc_auc = auc(fpr, tpr)
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic (ROC) Curve")
    # plt.legend(loc="lower right")
    # pdf.savefig()  # PDFに保存
    # plt.close()

    # 4. クラスごとのF1スコアの棒グラフ
    class_report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    f1_scores = [class_report[str(i)]['f1-score'] for i in range(len(class_report) - 3)]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(f1_scores)), f1_scores)
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Class")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 1刻みで表示
    plt.savefig(f"{output_dir}/f1score_class{timestamp}.png")
    plt.close()

    # # 5. クラスごとの精度（precision）の棒グラフ
    # precisions = [class_report[str(i)]['precision'] for i in range(len(class_report) - 3)]
    # plt.figure(figsize=(12, 6))
    # plt.bar(range(len(precisions)), precisions, color="purple")
    # plt.xlabel("Class")
    # plt.ylabel("Precision")
    # plt.title("Precision by Class")
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 1刻みで表示
    # pdf.savefig()  # PDFに保存
    # plt.close()

    # # 6. クラスごとのサポート（サンプル数）の棒グラフ
    # supports = [class_report[str(i)]['support'] for i in range(len(class_report) - 3)]
    # plt.figure(figsize=(12, 6))
    # plt.bar(range(len(supports)), supports, color="orange")
    # plt.xlabel("Class")
    # plt.ylabel("Support (Sample Count)")
    # plt.title("Support by Class")
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 1刻みで表示
    # pdf.savefig()  # PDFに保存
    # plt.close()
    
    print(f"Images have been saved in {output_dir}")


    # # 予測値 vs. 正解ラベルの散布図
    # plt.figure(figsize=(12, 6))
    # plt.scatter(range(len(all_labels)), all_labels, color="blue", label="Actual Labels", alpha=0.6)
    # plt.scatter(range(len(all_preds)), all_preds, color="red", marker="x", label="Predicted Labels", alpha=0.6)
    # plt.xlabel("Sample Index")
    # plt.ylabel("Class Label")
    # plt.title("Predicted vs. Actual Class Labels")
    # plt.legend()
    # plt.show()

    # plt.savefig(f"/home/nishioka/GNN/Predict_truth/pred_vs_true_class_labels{timestamp}.png")  # 現在のディレクトリに画像として保存


# ----------------------------
# データペアリング関数
# ----------------------------
def extract_layer_block(file_name):
    """ファイル名から層とブロック番号を抽出"""
    try:
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

    valid_pairs = [(v["data"], v["label"]) for k, v in data_label_pairs.items() if "label" in v]
    return valid_pairs


# ----------------------------
# データ準備関数
# ----------------------------
def prepare_data(pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index):
    data_list = []
    labels = []
    
    # 3つのペアを確認用に出力するためのカウンタ
    pair_counter = 0

    for data_file, label_file in pairs:
        data_file_path = os.path.join(standardized_data_folder, data_file)
        label_file_path = os.path.join(label_data_folder, label_file)

        values = np.load(data_file_path)[:3654]
        label = np.load(label_file_path)[:3654]

        node_features = np.vstack((x_coords, y_coords, z_coords, values)).T
        x = torch.tensor(node_features, dtype=torch.float)

        y = torch.argmax(torch.tensor(label, dtype=torch.float), dim=1).long()

        labels.extend(y.tolist())

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

        # # 確認用: 3つのペアとデータ形状を出力
        # if pair_counter < 3:
        #     print(f"Pair {pair_counter + 1}:")
        #     print(f"Data file: {data_file_path}")
        #     print(f"Label file: {label_file_path}")
        #     print(f"x shape: {x.shape}")  # xの形状
        #     print(f"y shape: {y.shape}")  # yの形状
        #     pair_counter += 1

    class_weights = compute_class_weights(np.array(labels))

    return data_list, class_weights

# ----------------------------
# Optunaのobjective関数
# ----------------------------
def objective(trial):
    # Optunaによるハイパーパラメータ探索
    hidden_channels = trial.suggest_int('hidden_channels', 32, 128, step=16)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)  # ドロップアウト率の探索
    
    # その他の設定
    patience = 30
    epochs = 150
    
    set_seed(42)  # 再現性のためのシード設定
    
    # Modelの再定義
    model = GATModel(hidden_channels=hidden_channels, num_classes=19).to(device)
    model.dropout = nn.Dropout(p=dropout_rate)  # ドロップアウト率を設定
    ddp_model = DDP(model, device_ids=[rank])

    # データローダーの準備（コードをmain関数からコピーしてくる）
    # 略
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = FocalLoss(weights=class_weights, gamma=gamma, reduction='mean').to(device)

    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(1, epochs + 1):
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

        avg_train_loss = total_loss / len(train_loader)
        
        # 検証ステップ（main関数から評価部分をコピー）
        # 略

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

# ----------------------------
# OptunaのStudy作成と最適化
# ----------------------------
if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')  # 目的は検証損失の最小化
    study.optimize(objective, n_trials=50, timeout=3600)  # 試行回数や時間制限を設定

    # 結果の出力
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")




# ----------------------------
# シード設定関数
# ----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    set_seed(42)

    # Initialize process group
    dist.init_process_group(
        backend='nccl',         # 'nccl' for GPUs, 'gloo' for CPUs
        init_method='env://',   # initialization method
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE'])
    )

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')

    if rank == 0:
        print(f"Rank: {rank}, World Size: {world_size}, Master Addr: {master_addr}, Master Port: {master_port}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank} using device: {device}")

    # Data preparation and model definition
    standardized_data_folder = "/home/nishioka/GNN/Defect_4x4_Normalized1"
    label_data_folder = "/home/nishioka/GNN/Defect19Class_OneHot_test3"

    x_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/x_2layer_normalized.npy")[:3654]
    y_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/y_2layer_normalized.npy")[:3654]
    z_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/z_2layer_normalized.npy")[:3654]

    edges = np.load("/home/nishioka/GNN/BasicdataforGNN/edges_2layer.npy")
    edge_index = torch.tensor(edges.T, dtype=torch.long).to(device)

    data_files = [f for f in os.listdir(standardized_data_folder) if f.startswith("Normalized1_Defect4x4_ELNOD")]
    label_files = [f for f in os.listdir(label_data_folder) if f.startswith("Defect19Class_L")]

    pairs = create_data_label_pairs(data_files, label_files)
    # print(f"Total valid pairs: {len(pairs)}")

    model = GATModel(hidden_channels=args.hidden_channels, num_classes=19).to(device)
    
    # Create the DDP model after initializing the process group
    ddp_model = DDP(model, device_ids=[rank])

    # KFold Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_fold_metrics = []
    
    for train_index, val_index in kf.split(pairs):
        if rank == 0:
            print(f"Starting Fold {fold}")

        train_pairs = [pairs[i] for i in train_index]
        val_pairs = [pairs[i] for i in val_index]

        # 修正ポイント：class_weightsを受け取る
        train_dataset, class_weights = prepare_data(train_pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index)
        val_dataset, _ = prepare_data(val_pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # 修正ポイント：class_weightsを使用してloss_fnを定義
        # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
        # loss_fn = FocalLoss(apply_nonlin=F.softmax, alpha=class_weights, gamma=2, size_average=True).to(device)
        # loss_fn = FocalLoss(alpha=0.25, gamma=gamma, reduction='mean').to(device)
        # loss_fn = FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean').to(device)
        loss_fn = FocalLoss(weights=class_weights, gamma=gamma, reduction='mean').to(device)



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

                loss = loss_fn(out, y)
                # loss = loss.mean()  # 勾配計算の前に平均を取る reduction='none' の場合
                loss.backward()
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Layer {name} grad: {param.grad.abs().mean()}")
                #     else:
                #         print(f"Layer {name} has no gradient.")

                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = torch.tensor(total_loss / len(train_loader), dtype=torch.float).to(device)
            dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
            avg_train_loss /= world_size

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

            avg_val_loss = torch.tensor(val_loss / len(val_loader), dtype=torch.float).to(device)
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
            avg_val_loss /= world_size

            total_correct = torch.tensor(correct, dtype=torch.float).to(device)
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            
            val_accuracy = total_correct / total_samples / world_size

            # ここに10エポックごとの出力を追加
            if epoch % 10 == 0:
                print(f"\n--- Epoch {epoch} ---")
                print(f"Class Weights: {class_weights.cpu().numpy()}")
                
                for param_group in optimizer.param_groups:
                    print(f"Learning Rate: {param_group['lr']}")
        
                print(f"Train Loss: {avg_train_loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}")
        
                if rank == 0 and all_preds and all_labels:
                    all_preds_np = np.concatenate(all_preds)
                    all_labels_np = np.concatenate(all_labels)
                    precision, recall, f1, support = precision_recall_fscore_support(all_labels_np, all_preds_np, zero_division=0)
                    print(f"Per-Class Precision: {precision}")
                    print(f"Per-Class Recall: {recall}")
                    print(f"Per-Class F1-Score: {f1}")
                    print(f"Support per Class: {support}")
        
                    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                        all_labels_np, all_preds_np, average='weighted', zero_division=0)
                    print(f"Weighted Precision: {weighted_precision:.4f}, Weighted Recall: {weighted_recall:.4f}, Weighted F1-Score: {weighted_f1:.4f}")

            if rank == 0:
                if all_preds and all_labels:
                    all_preds = np.concatenate(all_preds)
                    all_labels = np.concatenate(all_labels)
                    cm = confusion_matrix(all_labels, all_preds)
                    # plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])  # この関数が定義されていない場合はコメントアウト

                    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
                    recall = recall_score(all_labels, all_preds, average='weighted')
                    f1 = f1_score(all_labels, all_preds, average='weighted')
                    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

            if avg_val_loss.item() < best_val_loss:
                best_val_loss = avg_val_loss.item()
                counter = 0
                if rank == 0:
                    torch.save(ddp_model.module.state_dict(), f"/home/nishioka/GNN/GNNmodel/19classmodel/{type(model).__name__}_{timestamp}best_model_fold_{fold}.pth")
            else:
                counter += 1
                if counter >= args.patience:
                    if rank == 0:
                        print(f"Early stopping triggered for Fold {fold} at epoch {epoch}")
                    break

            if rank == 0:
                print(f'Fold {fold}, Epoch {epoch}, Train Loss: {avg_train_loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}')

        if rank == 0:
            all_fold_metrics.append({
                "fold": fold,
                "best_val_loss": best_val_loss,
                "val_accuracy": val_accuracy
            })
            print(f"Completed Fold {fold}")
        fold += 1

    dist.barrier()

    if rank == 0:
        print("Starting final training on all data...")

        train_val_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
        # 修正ポイント：class_weightsを受け取る
        train_val_dataset, class_weights = prepare_data(train_val_pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index)
        test_dataset, _ = prepare_data(test_pairs, standardized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index)

        final_train_loader = DataLoader(train_val_dataset, batch_size=args.batch_size, shuffle=True)
        final_test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model_path = f"/home/nishioka/GNN/GNNmodel/19classmodel/{type(model).__name__}_{timestamp}best_model_fold_{all_fold_metrics[0]['fold']}.pth"
        ddp_model.module.load_state_dict(torch.load(model_path))
        ddp_model.eval()

        print("Evaluating and Visualizing on Test Data")

        evaluate_and_visualize(final_test_loader, final_train_loader, ddp_model, device, class_weights, test_pairs, train_pairs)


        # 修正ポイント：loss_fnを定義
        # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
        # loss_fn = FocalLoss(apply_nonlin=F.softmax, alpha=class_weights, gamma=2, size_average=True).to(device)
        # loss_fn = FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean').to(device)
        loss_fn = FocalLoss(weights=class_weights, gamma=gamma, reduction='mean').to(device)
        # loss_fn = FocalLoss(alpha=0.25, gamma=gamma, reduction='mean').to(device)


        correct = 0
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
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

                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        test_loss = total_loss / len(final_test_loader)
        test_accuracy = correct / total_samples

        # Flatten predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate Precision, Recall, F1-Score
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Precision, Recall, F1-Score の計算と混同行列の表示
        # precision, recall, f1 = calculate_metrics(all_labels, all_preds)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # 最後に使用したモデルを保存
        torch.save(ddp_model.module.state_dict(), f"/home/nishioka/GNN/GNNmodel/19classmodel/{type(model).__name__}_{timestamp}_Final.pth")

    cleanup()

# ----------------------------
# エントリーポイント
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training Script')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience') 

    args = parser.parse_args()

    main(args)