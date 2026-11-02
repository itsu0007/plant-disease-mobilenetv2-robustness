"""
MobileNetV2 Robust Architecture for Plant Disease Classification
Implements MobileNetV2 backbone with attention gating, robust multi-layer classification head,
and hooks for Grad-CAM explainability under corrupted image conditions.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

from .attention import CBAMBlock, SEBlock

class RobustMobileNetV2(nn.Module):
    """
    MobileNetV2 architecture with optional CBAM/SE attention block and robust classification head.
    Optimized for low-latency edge deployment and high robustness against real-world corruptions.
    """
    def __init__(
        self,
        num_classes: int = 35,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        attention_type: str = "cbam",
        head_hidden_dim: int = 512,
        use_head_batchnorm: bool = True
    ):
        super(RobustMobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.attention_type = attention_type.lower()
        
        # Load backbone
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v2(weights=weights)
        self.features = base_model.features
        
        # MobileNetV2 output feature channel dimension is 1280
        in_features = 1280
        
        # Optional Attention Block
        if self.use_attention:
            if self.attention_type == "cbam":
                self.attention = CBAMBlock(in_planes=in_features)
            elif self.attention_type == "se":
                self.attention = SEBlock(in_planes=in_features)
            else:
                self.attention = nn.Identity()
        else:
            self.attention = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Robust Multi-stage Classifier Head
        head_layers = []
        if head_hidden_dim and head_hidden_dim > 0:
            head_layers.append(nn.Linear(in_features, head_hidden_dim))
            if use_head_batchnorm:
                head_layers.append(nn.BatchNorm1d(head_hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(p=dropout_rate))
            head_layers.append(nn.Linear(head_hidden_dim, num_classes))
        else:
            head_layers.append(nn.Dropout(p=dropout_rate))
            head_layers.append(nn.Linear(in_features, num_classes))
            
        self.classifier = nn.Sequential(*head_layers)

        # Storage for Grad-CAM activation and gradient caching
        self._feature_maps = None
        self._gradients = None

    def get_last_conv_layer(self) -> nn.Module:
        """Returns the last convolutional layer (layer 18 in features) for Grad-CAM hook."""
        return self.features[-1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature maps through backbone and attention."""
        feats = self.features(x)
        if self.use_attention:
            feats = self.attention(feats)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning classification logits."""
        feats = self.forward_features(x)
        pooled = self.pool(feats)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits

    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze backbone feature extraction weights."""
        for param in self.features.parameters():
            param.requires_grad = not freeze

    def get_parameter_summary(self) -> Dict[str, Union[int, float]]:
        """Return total, trainable, and frozen parameter counts and model size."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        size_mb = total * 4 / (1024 * 1024)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "size_mb": round(size_mb, 2)
        }
