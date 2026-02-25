import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union

class FocalLossWithAttention(nn.Module):
    """
    Focal Loss with Attention Weights and Class Weights (Alpha).
    """
    def __init__(self, gamma, alpha=None, reduction: str = 'mean', ignore_index=-100, eps=1e-16, att_lambda=0.001):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(f'Reduction {reduction} not implemented.')
        
        self.gamma = gamma
        self.alpha = alpha  # --- ⭐ クラス重み (Alpha) を導入 ---
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.eps = eps
        self.att_lambda = att_lambda  # Attentionの重み調整

    def _get_alpha(self, target: Tensor) -> Tensor:
        """
        Alpha (クラス重み) を取得
        """
        if self.alpha is None:
            return torch.ones(target.shape[0], device=target.device)
        alpha = self.alpha.to(target.device)
        alpha = target * alpha
        return alpha.sum(dim=-1)

    def _process_target(self, target: Tensor, num_classes: int, mask: Tensor) -> Tensor:
        target = target * (target != self.ignore_index)
        target = target.view(-1)
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x]).permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(self, target: Tensor, x: Tensor, mask: Tensor) -> Tensor:
        p = (target * x).sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor, attention_weights: Tensor) -> Tensor:
        # --- 確率チェック ---
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError("Predictions must be between [0,1]")

        mask = target == self.ignore_index
        mask = mask.view(-1)

        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes, mask)

        # --- ⭐ Alpha適用（クラス重み） ---
        alpha = self._get_alpha(target)

        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)

        # --- 🔥 Attention Weightsの適用 ---
        attention_weights = attention_weights.view(-1)
        if attention_weights.shape != focal.shape:
            raise ValueError(f"Attention shape {attention_weights.shape} != Focal Loss shape {focal.shape}")

        # --- 💡 Alpha（クラス重み）とAttention Weightの適用 ---
        loss = alpha * (focal ** self.gamma) * nll * (1 + self.att_lambda * attention_weights)

        return self._reduce(loss, mask, alpha)

    def _reduce(self, x: Tensor, mask: Tensor, alpha: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * alpha).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x