"""GNNモデル定義"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCNModel(torch.nn.Module):
    """Graph Convolutional Network モデル"""
    def __init__(self, hidden_channels=128, num_classes=19, dropout=0.2):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.fc(x)
        return x


class GATModel(torch.nn.Module):
    """Graph Attention Network モデル"""
    def __init__(self, hidden_channels=64, num_classes=19, num_heads=4, dropout=0.2):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(4, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels * 2, heads=num_heads, concat=True)
        self.conv3 = GATConv(hidden_channels * 2 * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.fc = nn.Linear(hidden_channels * num_heads, num_classes)
        self.dropout = nn.Dropout(p=dropout)

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
        return x