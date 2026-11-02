"""
Loss Functions for Robust Plant Disease Training
Implements Label Smoothing, Focal Loss, and Consistency Loss for stability under image corruptions.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    Prevents overconfidence on degraded inputs and stabilizes gradient updates.
    """
    def __init__(self, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        self.smoothing = smoothing

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        
        # When targets are 1D class indices
        if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
            targets = targets.view(-1)
            loss = F.cross_entropy(preds, targets, label_smoothing=self.smoothing)
            return loss

        # When targets are one-hot / smoothed distributions (e.g. from CutMix/MixUp)
        smooth_targets = targets * (1.0 - self.smoothing) + self.smoothing / num_classes
        loss = (-smooth_targets * log_preds).sum(dim=-1).mean()
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard degraded samples.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class ConsistencyRegularizationLoss(nn.Module):
    """
    Enforces that predictions on a clean image and its corrupted counterpart
    maintain consistent probability distributions using symmetric KL divergence / JS divergence.
    """
    def __init__(self, base_criterion: Optional[nn.Module] = None, lambda_cons: float = 1.0):
        super(ConsistencyRegularizationLoss, self).__init__()
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()
        self.lambda_cons = lambda_cons

    def forward(
        self,
        clean_logits: torch.Tensor,
        corrupted_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        base_loss = self.base_criterion(clean_logits, targets)
        
        # Jensen-Shannon Divergence
        p_clean = F.softmax(clean_logits, dim=-1)
        p_corrupt = F.softmax(corrupted_logits, dim=-1)
        m = 0.5 * (p_clean + p_corrupt)
        
        kl_clean = F.kl_div(F.log_softmax(clean_logits, dim=-1), m, reduction="batchmean")
        kl_corrupt = F.kl_div(F.log_softmax(corrupted_logits, dim=-1), m, reduction="batchmean")
        js_div = 0.5 * (kl_clean + kl_corrupt)
        
        return base_loss + self.lambda_cons * js_div
