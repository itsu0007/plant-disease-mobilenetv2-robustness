"""
Plant Leaf Dataset Loader & Splitter
Handles discovery, dataset indexing, stratified train/val/test splitting, and DataLoader creation.
Supports both real datasets (e.g. PlantVillage) and synthetic fallback datasets.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from .augmentations import get_baseline_transforms, get_robust_transforms
from .synthetic_data import create_synthetic_dataset
from ..utils.plant_classes import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class PlantDiseaseDataset(Dataset):
    """
    Standard PyTorch dataset for plant disease leaf classification.
    Reads images from disk and applies specified torchvision transformation.
    """
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[torch.nn.Module] = None
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    tensor_img = self.transform(img)
                else:
                    tensor_img = img
                return tensor_img, label, str(path)
        except Exception as e:
            # Fallback to zero tensor if image corrupted
            print(f"[Warning] Failed to read {path}: {e}")
            dummy = torch.zeros(3, 224, 224)
            return dummy, label, str(path)

def discover_dataset_samples(
    data_dir: Union[str, Path],
    class_to_idx: Optional[Dict[str, int]] = None
) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
    """
    Scan directory structure for class folders and collect all image paths.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return [], {}

    c2i = class_to_idx or CLASS_TO_IDX
    samples = []
    class_counts = {}

    for folder in sorted(data_path.iterdir()):
        if folder.is_dir():
            cls_name = folder.name
            if cls_name in c2i:
                label_idx = c2i[cls_name]
                img_files = [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
                class_counts[cls_name] = len(img_files)
                for p in img_files:
                    samples.append((p, label_idx))
                    
    return samples, class_counts

def get_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    img_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.15,
    use_robust_aug: bool = False,
    corruption_prob: float = 0.70,
    num_workers: int = 0,
    seed: int = 42,
    synthetic_fallback_dir: Optional[Union[str, Path]] = None,
    samples_per_class_if_synthetic: int = 15
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train, validation, and test DataLoaders with stratified splits.
    If the data directory is missing or empty, automatically prepares synthetic sample data.
    """
    data_path = Path(data_dir)
    samples, counts = discover_dataset_samples(data_path)

    # Automatic fallback to synthetic data if empty or not found
    if len(samples) == 0:
        fallback_dir = Path(synthetic_fallback_dir or "data/synthetic_sample")
        print(f"[Dataset] No images found in '{data_dir}'. Generating synthetic dataset in '{fallback_dir}'...")
        counts = create_synthetic_dataset(
            output_dir=fallback_dir,
            samples_per_class=samples_per_class_if_synthetic,
            img_size=(img_size, img_size)
        )
        samples, counts = discover_dataset_samples(fallback_dir)
        print(f"[Dataset] Generated {len(samples)} synthetic images across {len(counts)} classes.")

    # Extract labels for stratified splitting
    labels = [s[1] for s in samples]
    
    # Train / Val / Test split
    val_test_ratio = val_split + test_split
    train_samples, val_test_samples, _, val_test_labels = train_test_split(
        samples, labels, test_size=val_test_ratio, random_state=seed, stratify=labels
    )
    
    test_relative_ratio = test_split / val_test_ratio
    val_samples, test_samples = train_test_split(
        val_test_samples, test_size=test_relative_ratio, random_state=seed, stratify=val_test_labels
    )

    # Transforms
    if use_robust_aug:
        train_transform = get_robust_transforms(img_size=img_size, corruption_prob=corruption_prob)
    else:
        train_transform = get_baseline_transforms(img_size=img_size, is_training=True)

    val_test_transform = get_baseline_transforms(img_size=img_size, is_training=False)

    train_ds = PlantDiseaseDataset(train_samples, transform=train_transform)
    val_ds = PlantDiseaseDataset(val_samples, transform=val_test_transform)
    test_ds = PlantDiseaseDataset(test_samples, transform=val_test_transform)

    # Multi-processing DataLoader setup
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, counts
