import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GCNConv
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ----------------------------
# Graph Autoencoder (GAE) モデル定義
# ----------------------------
class GraphAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = F.relu(self.encoder(x, edge_index))
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

# ----------------------------
# 欠陥予測用 GNN モデル定義 (GAT)
# ----------------------------

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
            
class DefectPredictionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(DefectPredictionGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x

        self.apply(initialize_weights)


# ----------------------------
# 異常検知と欠陥予測のステップ
# ----------------------------
def anomaly_detection(data_loader, autoencoder, device):
    autoencoder.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            x_reconstructed, _ = autoencoder(data)
            loss = F.mse_loss(x_reconstructed, data.x, reduction='none').mean(dim=1)
            reconstruction_errors.append(loss.cpu().numpy())
    return np.concatenate(reconstruction_errors)

def label_defects_based_on_anomalies(reconstruction_errors, threshold):
    return (reconstruction_errors > threshold).astype(int)  # 1: 異常（欠陥）, 0: 正常

def load_node_features(standardized_data_folder, data_file):
    # 主応力和データの読み込み
    file_path = os.path.join(standardized_data_folder, data_file)
    principal_stress_sum = np.load(file_path)[:3654]  # 主応力和

    # ノードの座標 (必要に応じて追加)
    x_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/x_2layer_normalized.npy")[:3654]
    y_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/y_2layer_normalized.npy")[:3654]
    z_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/z_2layer_normalized.npy")[:3654]

    # 特徴量を結合 (例: x, y, z, 主応力和)
    node_features = np.vstack((x_coords, y_coords, z_coords, principal_stress_sum)).T

    return node_features

def prepare_data(node_features, edge_index):
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 4  # 座標 (x, y, z) と主応力和を含むので 4 次元
    hidden_dim = 64
    num_classes = 2  # 欠陥: 1、正常: 0

    # データフォルダとファイルの指定
    standardized_data_folder = "/home/nishioka/GNN/Defect_4x4_Normalized1"
    data_files = [f for f in os.listdir(standardized_data_folder) if f.endswith('.npy')]
    
    # エッジ情報のロード
    edges = np.load("/home/nishioka/GNN/BasicdataforGNN/edges_2layer.npy")
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # データの準備
    data_list = []
    for data_file in data_files:
        node_features = load_node_features(standardized_data_folder, data_file)
        data = prepare_data(node_features, edge_index)
        data_list.append(data)

    data_loader = DataLoader(data_list, batch_size=64)

    # Autoencoder モデルで異常検知
    autoencoder = GraphAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    autoencoder.load_state_dict(torch.load("path/to/autoencoder/model.pth"))  # トレーニング済みモデルのロード
    reconstruction_errors = anomaly_detection(data_loader, autoencoder, device)
    threshold = np.mean(reconstruction_errors) + np.std(reconstruction_errors)
    defect_labels = label_defects_based_on_anomalies(reconstruction_errors, threshold)

    # 欠陥予測用データの作成
    for i, data in enumerate(data_list):
        data.y = torch.tensor(defect_labels[i], dtype=torch.long)  # 異常検知結果をラベルとして設定

    # GNN モデルのトレーニング
    train_loader = DataLoader(data_list, batch_size=64, shuffle=True)
    gnn_model = DefectPredictionGNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # トレーニングループ
    epochs = 50
    for epoch in range(epochs):
        gnn_model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = gnn_model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # 評価と可視化（必要に応じて追加）
    gnn_model.eval()
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            pred = gnn_model(batch).argmax(dim=1)
            print("Predicted Defect Labels:", pred.cpu().numpy())

if __name__ == "__main__":
    main()

