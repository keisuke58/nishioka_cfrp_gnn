import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Focal Loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply softmax if necessary
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Compute the focal loss
        pt = (inputs * targets_one_hot).sum(dim=1)  # Probability of the true class
        focal_term = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_term * torch.log(pt)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ----------------------------
# Usage within the training loop
# ----------------------------
# Initialize focal loss with desired alpha and gamma values
# loss_fn = FocalLoss(alpha=0.25, gamma=2.0).to(device)

# Example usage in your train/validation loop:
# loss = loss_fn(outputs, targets)
