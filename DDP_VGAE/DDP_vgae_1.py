import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt

# ----------------------------
# モデル定義 (Graph Autoencoder)
# ----------------------------
class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Encode
        z = F.relu(self.encoder(x, edge_index))
        # Decode
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

# ----------------------------
# モデルのトレーニング関数
# ----------------------------
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_reconstructed, _ = model(data)
        loss = criterion(x_reconstructed, data.x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# ----------------------------
# モデルの評価関数
# ----------------------------
def evaluate(model, data_loader, device):
    model.eval()
    all_losses = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            x_reconstructed, _ = model(data)
            loss = F.mse_loss(x_reconstructed, data.x, reduction='none')
            sample_loss = loss.mean(dim=1).cpu().numpy()  # 各ノードの再構成誤差
            all_losses.extend(sample_loss)
    return all_losses

# ----------------------------
# メイン関数
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 4  # 特徴量の数 (例: x, y, z, value)
    hidden_dim = 64

    # データのロード (ノード特徴量とエッジインデックス)
    x_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/x_2layer_normalized.npy")[:3654]
    y_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/y_2layer_normalized.npy")[:3654]
    z_coords = np.load("/home/nishioka/GNN/BasicdataforGNN/z_2layer_normalized.npy")[:3654]
    values = np.random.rand(3654)  # ダミーデータとしてランダムな特徴量を生成
    node_features = np.vstack((x_coords, y_coords, z_coords, values)).T

    edges = np.load("/home/nishioka/GNN/BasicdataforGNN/edges_2layer.npy")
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    data_list = [data]  # データセットをリストで格納
    data_loader = DataLoader(data_list, batch_size=1)

    model = GraphAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # トレーニング
    epochs = 100
    for epoch in range(epochs):
        train_loss = train(model, data_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

    # 評価
    reconstruction_errors = evaluate(model, data_loader, device)
    print("Reconstruction Errors:", reconstruction_errors)

    # 異常検知のしきい値設定 (例: 平均 + 標準偏差)
    threshold = np.mean(reconstruction_errors) + np.std(reconstruction_errors)
    anomalies = [i for i, error in enumerate(reconstruction_errors) if error > threshold]

    print(f"Anomalies detected at node indices: {anomalies}")

    # 可視化
    plt.hist(reconstruction_errors, bins=50, color='blue', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='--', label='Anomaly Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Reconstruction Errors')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
