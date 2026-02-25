import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union


class FocalLoss(nn.Module):
    """Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2

    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        # Reduction method should be 'mean', 'none' or 'sum'
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        # weights should be a Tensor or None
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        # Setting instance variables for focal loss parameters
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    # # Method to get the weights for each class based on target values
    # def _get_weights(self, target: Tensor) -> Tensor:
    #     if self.weights is None:
    #         return torch.ones(target.shape[0])
    #     # Apply weights to each class in the target
    #     weights = target * self.weights
    #     return weights.sum(dim=-1)

    # 重みを計算するための修正版メソッド
    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            # ターゲットテンソルと同じデバイスに全ての重みを 1 で初期化
            return torch.ones(target.shape[0], device=target.device)
        # ターゲットと同じデバイスに重みを転送
        weights = self.weights.to(target.device)
        weights = target * weights
        return weights.sum(dim=-1)

    # Method to process target tensor and convert it to one-hot format
    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
            ) -> Tensor:
        # Convert ignore_index elements to zero to avoid error in one_hot conversion
        target = target * (target != self.ignore_index) 
        # Reshape target to 1D for one-hot encoding
        target = target.view(-1)
        # Convert target tensor to one-hot format
        return one_hot(target, num_classes=num_classes)

    # Method to process the predictions
    def _process_preds(self, x: Tensor) -> Tensor:
        # If the input tensor is 1D, convert it to 2D format for binary classification
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)  # Transpose the tensor to make it (N, C)
            return x
        # Reshape input tensor to ensure compatibility
        return x.view(-1, x.shape[-1])

    # Method to calculate p_t (probability of the true class)
    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        # Calculate probability for each target class
        p = target * x
        p = p.sum(dim=-1)  # Sum over classes to get the probability of the true class
        p = p * ~mask  # Apply mask to ignore specific elements
        return p

    # Forward method to compute the loss
    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        # 成功版と同じく確率（0-1の範囲）を受け取る実装
        # Ensure all input values are between 0 and 1 (valid probabilities)
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The predictions values should be between 0 and 1, \
                make sure to pass the values to sigmoid for binary \
                classification or softmax for multi-class classification'
        )
        # Create a mask for ignored indices
        mask = target == self.ignore_index
        mask = mask.view(-1)
        # Process predictions and targets to required format
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes, mask)
        # Get class weights if available
        weights = self._get_weights(target).to(x.device)
        # Calculate probability of true class
        pt = self._calc_pt(target, x, mask)
        # Calculate focal term (1 - p_t)
        focal = 1 - pt
        # Calculate negative log likelihood (NLL) for the true class probabilities
        nll = -torch.log(self.eps + pt)
        # Apply mask to set ignored indices to zero
        nll = nll.masked_fill(mask, 0)
        # Calculate focal loss with weighting and focusing parameter gamma
        loss = weights * (focal ** self.gamma) * nll
        # Reduce the loss based on reduction type
        return self._reduce(loss, mask, weights)

    # Method to reduce the loss tensor based on reduction type (mean, sum, none)
    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            # Calculate mean loss, taking into account ignored elements
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            # Sum all the loss values
            return x.sum()
        else:
            # Return the loss as is, without reduction
            return x

# 日本語の要約：
# このコードはFocal Lossという損失関数を実装しています。Focal Lossは、難しいサンプル（分類が困難なサンプル）に対してより大きな重みを与えることで、学習の際のクラス不均衡の影響を軽減することを目的としています。
# Focal Lossは、各サンプルの損失を、そのサンプルの予測確率と対象クラスに基づいて調整します。
# `gamma`はフォーカスパラメータで、難しいサンプルの影響を増強します。クラスの重み付けや、無視すべきターゲット（`ignore_index`）の設定も可能です。
# 主なメソッドには、ターゲットのワンホット変換、予測値の処理、損失の計算、そして損失のリダクション（平均、合計、リダクションなし）などがあります。
