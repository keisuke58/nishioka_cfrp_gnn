"""損失関数（GNN_zscore_sub_noise_defect_free.pyを基に）"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerAwareLoss(nn.Module):
    """
    層ごとのラベル制約を追加する損失関数ラッパー
    Layer 1 (前半): 0, 1~9 のみ
    Layer 2 (後半): 0, 10~18 のみ
    
    データリークではない理由: これはデータの構造的制約であり、物理的な意味に基づく
    """
    def __init__(self, base_loss_fn, vertices_per_layer=6971, constraint_weight=1.0):
        """
        Args:
            base_loss_fn: ベースとなる損失関数（LogitAdjustLoss, FocalLossLogSoftmax, CrossEntropyLossなど）
            vertices_per_layer: 1層あたりのノード数（デフォルト: 6971）
            constraint_weight: 制約違反のペナルティ重み（デフォルト: 1.0）
        """
        super(LayerAwareLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.vertices_per_layer = vertices_per_layer
        self.constraint_weight = constraint_weight
    
    def forward(self, logits, target, batch=None):
        """
        Args:
            logits: [N, num_classes] のロジット
            target: [N] のクラスインデックス
            batch: [N] のバッチインデックス（各グラフのノードがどのグラフに属するか）
                   バッチサイズが1の場合はNoneでも可
        """
        # ベース損失を計算
        base_loss = self.base_loss_fn(logits, target)
        
        # 層ごとの制約違反ペナルティを計算
        constraint_loss = self._compute_constraint_penalty(logits, batch)
        
        return base_loss + self.constraint_weight * constraint_loss
    
    def _compute_constraint_penalty(self, logits, batch):
        """
        層ごとのラベル制約違反に対するペナルティを計算
        """
        # バッチが指定されていない場合、単一グラフと仮定
        if batch is None:
            # 単一グラフの場合、最初のvertices_per_layerがLayer 1、残りがLayer 2
            N = logits.size(0)
            if N <= self.vertices_per_layer:
                # ノード数が少ない場合は制約を適用しない
                return torch.tensor(0.0, device=logits.device)
            
            layer1_mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
            layer1_mask[:self.vertices_per_layer] = True
            layer2_mask = ~layer1_mask
        else:
            # 複数グラフの場合、各グラフ内でLayer 1とLayer 2を分離
            # batch内の各グラフについて、最初のvertices_per_layerがLayer 1
            unique_batches = torch.unique(batch)
            layer1_mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
            layer2_mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
            
            for b in unique_batches:
                batch_mask = (batch == b)
                batch_indices = torch.where(batch_mask)[0]
                if len(batch_indices) > self.vertices_per_layer:
                    layer1_indices = batch_indices[:self.vertices_per_layer]
                    layer2_indices = batch_indices[self.vertices_per_layer:]
                    layer1_mask[layer1_indices] = True
                    layer2_mask[layer2_indices] = True
        
        # 予測確率を計算
        probs = F.softmax(logits, dim=1)
        
        # Layer 1の制約違反: 10以上のクラスの予測確率にペナルティ
        layer1_logits = logits[layer1_mask]
        layer1_probs = probs[layer1_mask]
        if layer1_logits.size(0) > 0:
            # クラス10~18の予測確率の合計にペナルティ
            invalid_classes_layer1 = torch.arange(10, 19, device=logits.device)
            layer1_penalty = layer1_probs[:, invalid_classes_layer1].sum(dim=1).mean()
        else:
            layer1_penalty = torch.tensor(0.0, device=logits.device)
        
        # Layer 2の制約違反: 1~9のクラスの予測確率にペナルティ
        layer2_logits = logits[layer2_mask]
        layer2_probs = probs[layer2_mask]
        if layer2_logits.size(0) > 0:
            # クラス1~9の予測確率の合計にペナルティ
            invalid_classes_layer2 = torch.arange(1, 10, device=logits.device)
            layer2_penalty = layer2_probs[:, invalid_classes_layer2].sum(dim=1).mean()
        else:
            layer2_penalty = torch.tensor(0.0, device=logits.device)
        
        return layer1_penalty + layer2_penalty


class LogitAdjustLoss(nn.Module):
    """
    LogitAdjust Loss: logitsにtau*log(pi)を加算してクラス不均衡を補正
    Focal Lossより「全部0」崩壊を防ぎつつminorityを押し上げる効果が高い
    
    Reference: "Long-tail learning via logit adjustment" (Menon et al., 2020)
    
    Note: class_priorは必須（バッチ推定はDDP+不均衡で揺れて危険なため禁止）
    """
    def __init__(self, class_prior: torch.Tensor, tau=1.0, reduction='mean'):
        """
        Args:
            class_prior: 各クラスの事前確率（必須、train全体から計算した固定値）
            tau: 調整強度（デフォルト: 1.0, 推奨範囲: 1.0~2.0）
            reduction: 'mean' or 'sum'
        """
        super(LogitAdjustLoss, self).__init__()
        self.tau = float(tau)
        self.reduction = reduction
        # 固定priorを登録（DDP安全）
        self.register_buffer('log_pi', torch.log(class_prior + 1e-8))
    
    def forward(self, logits, target):
        """
        Args:
            logits: [N, num_classes] のロジット
            target: [N] のクラスインデックス
        """
        # Logit adjustment: logits + tau * log(pi)
        adjusted_logits = logits + self.tau * self.log_pi.unsqueeze(0)
        
        # Cross-entropy loss
        return F.cross_entropy(adjusted_logits, target, reduction=self.reduction)


class FocalLossLogSoftmax(nn.Module):
    """Focal Loss using log_softmax for numerical stability"""
    def __init__(self, weights=None, gamma=3.0, reduction='mean'):
        super(FocalLossLogSoftmax, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weights is not None:
            self.register_buffer('weights', weights)
        else:
            self.weights = None
    
    def forward(self, logits, target):
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # Get log probability of the correct class
        target_one_hot = F.one_hot(target, num_classes=logits.size(1))
        log_probs_target = (log_probs * target_one_hot).sum(dim=1)
        probs_target = (probs * target_one_hot).sum(dim=1)
        
        # Focal weight: (1 - p)^gamma
        focal_weight = (1.0 - probs_target) ** self.gamma
        
        # Weighted negative log likelihood
        loss = -focal_weight * log_probs_target
        
        # Apply class weights if provided
        if self.weights is not None:
            class_weights = self.weights[target]
            loss = loss * class_weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss