"""予測結果をテストするスクリプト"""
import torch
import numpy as np
from predict_with_saved_model import GATModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# モデルを読み込み
model_path = '/home/nishioka/GNN/GNN_hole/GNN_model/19classmodel_hole/GATModel_20250126_173116_Final.pth'
model = GATModel(hidden_channels=64, num_classes=19).to(device)

checkpoint = torch.load(model_path, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# テストデータを作成
from torch_geometric.data import Data
x = torch.randn(100, 4).to(device)
edge_index = torch.randint(0, 100, (2, 200)).to(device)
test_data = Data(x=x, edge_index=edge_index).to(device)

# 予測を実行
with torch.no_grad():
    out = model(test_data)
    pred = out.argmax(dim=1).cpu().numpy()

print(f"出力形状: {out.shape}")
print(f"出力の統計: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")
print(f"予測クラス: {np.unique(pred, return_counts=True)}")
print(f"予測結果の最初の10個: {pred[:10]}")
