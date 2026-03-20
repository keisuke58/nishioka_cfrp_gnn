"""Two-stage GNN models for CFRP defect detection.

Addresses extreme class imbalance (1:170 defect-to-normal ratio) by splitting
the 19-class problem into two stages:
  Stage 1: Binary classification (defect vs no-defect)
  Stage 2: 18-class location classification (defect nodes only, classes 1-18)

Usage:
    from gnn_common.models_twostage import BinaryDefectModel, LocationClassifierModel, TwoStageInference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class BinaryDefectModel(nn.Module):
    """Stage 1: Binary classifier (defect vs no-defect).

    Uses the same 4-layer GATConv backbone as GATModel but outputs 2 classes:
      class 0 = no defect (original class 0)
      class 1 = defect (original classes 1-18)

    Args:
        hidden_channels: Hidden channel size (default: 64)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.2)
        input_channels: Number of input features per node (default: 4)
    """

    def __init__(self, hidden_channels=64, num_heads=4, dropout=0.2, input_channels=4):
        super(BinaryDefectModel, self).__init__()
        self.conv1 = GATConv(input_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels * 2, heads=num_heads, concat=True)
        self.conv3 = GATConv(hidden_channels * 2 * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.fc = nn.Linear(hidden_channels * num_heads, 2)
        self.dropout = nn.Dropout(p=dropout)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization (same pattern as GATModel)."""
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

    def forward(self, data):
        """Forward pass returning binary logits [N, 2]."""
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


class LocationClassifierModel(nn.Module):
    """Stage 2: 18-class location classifier for defect nodes only.

    Classifies defect nodes into 18 location classes (originally classes 1-18,
    remapped to 0-17 for this model).

    The architecture is intentionally smaller than Stage 1 since it operates
    on the full graph but is trained/evaluated only on defect nodes (much fewer).

    Args:
        hidden_channels: Hidden channel size (default: 64)
        num_location_classes: Number of location classes (default: 18)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.2)
        input_channels: Number of input features per node (default: 4)
    """

    def __init__(self, hidden_channels=64, num_location_classes=18, num_heads=4,
                 dropout=0.2, input_channels=4):
        super(LocationClassifierModel, self).__init__()
        self.num_location_classes = num_location_classes
        self.conv1 = GATConv(input_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels * 2, heads=num_heads, concat=True)
        self.conv3 = GATConv(hidden_channels * 2 * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.fc = nn.Linear(hidden_channels * num_heads, num_location_classes)
        self.dropout = nn.Dropout(p=dropout)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization."""
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

    def forward(self, data):
        """Forward pass on full graph, returning logits [N, 18].

        Note: During training, loss is computed only on defect node indices.
        During inference, TwoStageInference handles the masking.
        """
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


class TwoStageInference(nn.Module):
    """Combined two-stage inference pipeline.

    Stage 1 (binary_model) predicts defect vs no-defect for all nodes.
    Stage 2 (location_model) classifies defect nodes into 18 location classes.
    Final predictions are mapped back to the original 19-class label space.

    Args:
        binary_model: Trained BinaryDefectModel
        location_model: Trained LocationClassifierModel
        binary_threshold: Probability threshold for defect detection (default: 0.5)
    """

    def __init__(self, binary_model, location_model, binary_threshold=0.5):
        super(TwoStageInference, self).__init__()
        self.binary_model = binary_model
        self.location_model = location_model
        self.binary_threshold = binary_threshold

    @torch.no_grad()
    def forward(self, data):
        """Run two-stage inference.

        Returns:
            predictions_19: [N] tensor with predicted class labels in 0-18 space
            binary_probs: [N] tensor with defect probabilities from Stage 1
        """
        self.binary_model.eval()
        self.location_model.eval()

        # Stage 1: Binary classification
        binary_logits = self.binary_model(data)  # [N, 2]
        binary_probs = F.softmax(binary_logits, dim=1)
        defect_probs = binary_probs[:, 1]  # probability of being defect
        is_defect = defect_probs >= self.binary_threshold  # [N] bool

        # Initialize all predictions as class 0 (no defect)
        predictions_19 = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        if is_defect.any():
            # Stage 2: Location classification on full graph
            location_logits = self.location_model(data)  # [N, 18]
            # Only use predictions for nodes classified as defect
            defect_preds = location_logits[is_defect].argmax(dim=1)  # classes 0-17
            # Map back to original 19-class space: class k -> class k+1
            predictions_19[is_defect] = defect_preds + 1

        return predictions_19, defect_probs

    @torch.no_grad()
    def predict_with_logits(self, data):
        """Run two-stage inference returning full 19-class logits.

        Useful for computing per-class probabilities and confidence scores.

        Returns:
            logits_19: [N, 19] combined logits in 19-class space
            binary_probs: [N] defect probabilities from Stage 1
        """
        self.binary_model.eval()
        self.location_model.eval()

        N = data.x.size(0)
        device = data.x.device

        # Stage 1
        binary_logits = self.binary_model(data)  # [N, 2]
        binary_probs = F.softmax(binary_logits, dim=1)
        defect_probs = binary_probs[:, 1]

        # Build 19-class logits: class 0 gets the no-defect logit,
        # classes 1-18 get location logits scaled by defect probability
        logits_19 = torch.full((N, 19), -1e6, device=device)
        logits_19[:, 0] = binary_logits[:, 0]  # no-defect logit

        # Stage 2 on full graph
        location_logits = self.location_model(data)  # [N, 18]
        # Place location logits into columns 1-18, scaled by binary confidence
        logits_19[:, 1:19] = location_logits + binary_logits[:, 1:2]

        return logits_19, defect_probs


def labels_to_binary(labels):
    """Convert 19-class labels to binary labels (0=no-defect, 1=defect).

    Args:
        labels: Tensor or ndarray of labels in 0-18 range

    Returns:
        Binary labels: 0 where original==0, 1 where original>0
    """
    if isinstance(labels, torch.Tensor):
        return (labels > 0).long()
    else:
        import numpy as np
        return (np.asarray(labels) > 0).astype(np.int64)


def labels_to_location(labels):
    """Convert 19-class labels to 18-class location labels (shift by -1).

    Only valid for defect nodes (labels > 0). Class 1 -> 0, Class 2 -> 1, ..., Class 18 -> 17.

    Args:
        labels: Tensor or ndarray of labels in 1-18 range (defect nodes only)

    Returns:
        Location labels in 0-17 range
    """
    if isinstance(labels, torch.Tensor):
        return labels - 1
    else:
        import numpy as np
        return np.asarray(labels) - 1
