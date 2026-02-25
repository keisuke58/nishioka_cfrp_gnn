"""GNNモデル定義"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GCNModel(torch.nn.Module):
    """Graph Convolutional Network モデル"""
    def __init__(self, hidden_channels=128, num_classes=19, dropout=0.2):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        
        # Xavier初期化を適用
        self.init_weights()

    def init_weights(self):
        """Xavier初期化を適用"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, GCNConv):
                # GCNConvの内部LinearレイヤーにXavier初期化を適用
                if hasattr(module, 'lin') and module.lin is not None:
                    nn.init.xavier_uniform_(module.lin.weight)
                    if module.lin.bias is not None:
                        nn.init.constant_(module.lin.bias, 0)

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
        
        # Xavier初期化を適用
        self.init_weights()

    def init_weights(self):
        """Xavier初期化を適用"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, GATConv):
                # GATConvの内部LinearレイヤーにXavier初期化を適用
                if hasattr(module, 'lin_src') and module.lin_src is not None:
                    nn.init.xavier_uniform_(module.lin_src.weight)
                    if module.lin_src.bias is not None:
                        nn.init.constant_(module.lin_src.bias, 0)
                if hasattr(module, 'lin_dst') and module.lin_dst is not None:
                    nn.init.xavier_uniform_(module.lin_dst.weight)
                    if module.lin_dst.bias is not None:
                        nn.init.constant_(module.lin_dst.bias, 0)
                # Attentionの重みも初期化
                if hasattr(module, 'att_src') and module.att_src is not None:
                    nn.init.xavier_uniform_(module.att_src)
                if hasattr(module, 'att_dst') and module.att_dst is not None:
                    nn.init.xavier_uniform_(module.att_dst)

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


class GATModelWithCrossEdges(torch.nn.Module):
    """Cross-edge対応のGATモデル（Yehia方式）
    
    Cross-edgeを既存のエッジと統合して使用します。
    エッジタイプの区別は行わず、すべてのエッジを同等に扱います。
    """
    def __init__(
        self, 
        hidden_channels=64, 
        num_classes=19, 
        num_heads=4, 
        dropout=0.2,
        use_edge_type=False  # エッジタイプを使うか（将来の拡張用）
    ):
        super(GATModelWithCrossEdges, self).__init__()
        self.use_edge_type = use_edge_type
        
        self.conv1 = GATConv(4, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels * 2, heads=num_heads, concat=True)
        self.conv3 = GATConv(hidden_channels * 2 * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.fc = nn.Linear(hidden_channels * num_heads, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        
        # エッジタイプ埋め込み（将来の拡張用、現在は未使用）
        if use_edge_type:
            self.edge_type_embedding = nn.Embedding(3, hidden_channels)  # タイプ0, 1, 2
        
        # Xavier初期化を適用
        self.init_weights()

    def init_weights(self):
        """Xavier初期化を適用"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                # EmbeddingレイヤーにもXavier初期化を適用
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, GATConv):
                # GATConvの内部LinearレイヤーにXavier初期化を適用
                if hasattr(module, 'lin_src') and module.lin_src is not None:
                    nn.init.xavier_uniform_(module.lin_src.weight)
                    if module.lin_src.bias is not None:
                        nn.init.constant_(module.lin_src.bias, 0)
                if hasattr(module, 'lin_dst') and module.lin_dst is not None:
                    nn.init.xavier_uniform_(module.lin_dst.weight)
                    if module.lin_dst.bias is not None:
                        nn.init.constant_(module.lin_dst.bias, 0)
                # Attentionの重みも初期化
                if hasattr(module, 'att_src') and module.att_src is not None:
                    nn.init.xavier_uniform_(module.att_src)
                if hasattr(module, 'att_dst') and module.att_dst is not None:
                    nn.init.xavier_uniform_(module.att_dst)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # エッジタイプがある場合は特徴量に追加（将来の拡張用）
        # 現在の実装では、cross-edgeは既存エッジと統合されているため、
        # 通常のGATConvで処理される
        if self.use_edge_type and hasattr(data, 'edge_type'):
            # 将来の実装: エッジタイプに応じた重み付けなど
            pass

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.fc(x)
        return x


class GATMultiTaskModel(torch.nn.Module):
    """Multi-task GAT: 位置分類（ノード単位）+ 欠陥サイズ分類（グラフ単位）

    共有GATバックボーン（4層）から2つのヘッドに分岐:
      - head_location: Linear → [N, num_classes] （ノード単位、既存GATModelと同等）
      - head_size: global_mean_pool → Linear → [G, num_size_classes] （グラフ単位）
    """

    def __init__(
        self,
        hidden_channels=64,
        num_classes=19,
        num_size_classes=3,
        num_heads=4,
        dropout=0.2,
    ):
        super(GATMultiTaskModel, self).__init__()
        self.conv1 = GATConv(4, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels * 2, heads=num_heads, concat=True)
        self.conv3 = GATConv(hidden_channels * 2 * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.dropout = nn.Dropout(p=dropout)

        backbone_out = hidden_channels * num_heads
        self.head_location = nn.Linear(backbone_out, num_classes)
        self.head_size = nn.Linear(backbone_out, num_size_classes)

        self.init_weights()

    def init_weights(self):
        """Xavier初期化を適用"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, GATConv):
                if hasattr(module, 'lin_src') and module.lin_src is not None:
                    nn.init.xavier_uniform_(module.lin_src.weight)
                    if module.lin_src.bias is not None:
                        nn.init.constant_(module.lin_src.bias, 0)
                if hasattr(module, 'lin_dst') and module.lin_dst is not None:
                    nn.init.xavier_uniform_(module.lin_dst.weight)
                    if module.lin_dst.bias is not None:
                        nn.init.constant_(module.lin_dst.bias, 0)
                if hasattr(module, 'att_src') and module.att_src is not None:
                    nn.init.xavier_uniform_(module.att_src)
                if hasattr(module, 'att_dst') and module.att_dst is not None:
                    nn.init.xavier_uniform_(module.att_dst)

    def _backbone(self, x, edge_index):
        """共有GATバックボーン"""
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index))
        return x

    def forward(self, data, multitask=True):
        """
        Args:
            data: PyG Data / Batch オブジェクト
            multitask: Falseの場合、既存GATModelと同じ単一テンソル出力

        Returns:
            multitask=True:  dict {"location": [N, 19], "size": [G, 3]}
            multitask=False: Tensor [N, 19] （後方互換）
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        shared = self._backbone(x, edge_index)

        location_out = self.head_location(shared)

        if not multitask:
            return location_out

        if batch is None:
            batch = torch.zeros(shared.size(0), dtype=torch.long, device=shared.device)

        graph_emb = global_mean_pool(shared, batch)
        size_out = self.head_size(graph_emb)

        return {"location": location_out, "size": size_out}