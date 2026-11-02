"""
Data Augmentation & Robust Degradation Pipeline
Implements standard baseline data augmentations and robust training augmentations
that incorporate realistic camera corruptions, CutMix, and MixUp.
"""

from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from ..corruptions.degradation_engine import apply_random_corruption

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class RealWorldCorruptionTransform:
    """
    Torchvision-compatible transform that injects realistic quality degradations
    (blur, noise, exposure, compression) with a specified probability.
    """
    def __init__(
        self,
        prob: float = 0.70,
        severity_range: Tuple[int, int] = (1, 4),
        allowed_corruptions: Optional[List[str]] = None
    ):
        self.prob = prob
        self.severity_range = severity_range
        self.allowed_corruptions = allowed_corruptions

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.prob:
            corrupted, _, _ = apply_random_corruption(
                img,
                severities=self.severity_range,
                allowed_corruptions=self.allowed_corruptions,
                return_pil=True
            )
            return corrupted
        return img

def get_baseline_transforms(img_size: int = 224, is_training: bool = True) -> T.Compose:
    """
    Standard data augmentation pipeline used for baseline model.
    Standard geometry transforms (flips, slight rotations) without degradation robustness.
    """
    if is_training:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

def get_robust_transforms(
    img_size: int = 224,
    corruption_prob: float = 0.75,
    severity_range: Tuple[int, int] = (1, 4)
) -> T.Compose:
    """
    Robust training pipeline that incorporates real-world image quality degradation
    simulation on top of geometric and color variations.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=25),
        RealWorldCorruptionTransform(prob=corruption_prob, severity_range=severity_range),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def cutmix_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies CutMix augmentation to a batch of images and labels.
    Returns: mixed_images, targets_a, targets_b, lambda_val
    """
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = images.size()
    indices = torch.randperm(batch_size)
    
    # Calculate bounding box coordinates
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    # Uniform center
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    mixed_images = images.clone()
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exact pixel ratio
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    return mixed_images, targets, targets[indices], lam

def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp augmentation to a batch of images and labels.
    Returns: mixed_images, targets_a, targets_b, lambda_val
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    indices = torch.randperm(batch_size)
    
    mixed_images = lam * images + (1.0 - lam) * images[indices]
    return mixed_images, targets, targets[indices], lam
