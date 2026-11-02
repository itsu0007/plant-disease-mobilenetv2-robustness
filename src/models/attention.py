"""
Attention Modules for Robust Feature Enhancement
Implements CBAM (Convolutional Block Attention Module) and Squeeze-and-Excitation (SE)
to help MobileNetV2 focus on pathological lesion patterns while suppressing noise/blur artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention Module:
    Aggregates spatial information using both Average Pooling and Max Pooling,
    passing through a shared MLP to compute channel importance weights.
    """
    def __init__(self, in_planes: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_dim = max(4, in_planes // reduction_ratio)
        self.fc1 = nn.Conv2d(in_planes, reduced_dim, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_dim, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module:
    Aggregates channel information using AvgPool and MaxPool across channels,
    convolving with a 7x7 filter to compute spatial lesion saliency masks.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(combined)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module:
    Sequential Channel Attention followed by Spatial Attention.
    Significantly improves robustness under real-world quality variations.
    """
    def __init__(self, in_planes: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction_ratio=reduction_ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel gating
        out = x * self.ca(x)
        # Spatial gating
        out = out * self.sa(out)
        return out

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Block.
    Lightweight channel recalibration mechanism.
    """
    def __init__(self, in_planes: int, reduction_ratio: int = 16):
        super(SEBlock, self).__init__()
        reduced_dim = max(4, in_planes // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
