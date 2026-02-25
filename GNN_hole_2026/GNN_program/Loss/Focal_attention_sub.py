class FocalLossWithAttention(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True, att_lambda=0.001):
        super(FocalLossWithAttention, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.att_lambda = att_lambda  # 🔥 Attention正則化の強度

    def forward(self, logit, target, att_weights=None):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit, dim=1)

        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = torch.squeeze(target, 1).view(-1)

        alpha = self.alpha if self.alpha is not None else torch.ones(num_class, 1).to(logit.device)

        idx = target.cpu().long()
        one_hot_key = torch.zeros(target.size(0), num_class).to(logit.device).scatter_(1, idx.unsqueeze(1), 1)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        alpha = alpha[idx].squeeze()

        # 🎯 **Focal Loss**
        focal_loss = -1 * alpha * (1 - pt) ** self.gamma * logpt
        focal_loss = focal_loss.mean() if self.size_average else focal_loss.sum()

        # 🔥 **Attention正則化** (L1正則化)
        att_reg_loss = 0
        if att_weights is not None:
            att_reg_loss = self.att_lambda * torch.mean(torch.abs(att_weights))

        # ✅ **総合損失** = Focal Loss + Attention正則化
        total_loss = focal_loss + att_reg_loss

        return total_loss