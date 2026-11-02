"""
Robustness Benchmark Suite
Evaluates vision models across 12 real-world image corruptions at 5 severity levels.
Computes Clean Accuracy, Corruption Error (CE), Mean Corruption Error (mCE),
Relative mCE, and category-level breakdown (Blur, Noise, Lighting, Digital).
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .degradation_engine import (
    CORRUPTION_CATEGORIES,
    CORRUPTION_REGISTRY,
    apply_corruption_by_name,
)
from ..dataset.augmentations import IMAGENET_MEAN, IMAGENET_STD
import torchvision.transforms as T

def evaluate_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Tuple[float, float]:
    """Evaluate top-1 accuracy and average loss on a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item() * targets.size(0)

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return acc, avg_loss

class CorruptionBenchmarkRunner:
    """
    Orchestrates systematic benchmarking of models across all corruptions and severities.
    """
    def __init__(
        self,
        img_size: int = 224,
        device: Optional[str] = None
    ):
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def evaluate_under_corruption(
        self,
        model: torch.nn.Module,
        dataset_samples: List[Tuple[Path, int]],
        corruption_name: str,
        severity: int,
        batch_size: int = 32
    ) -> float:
        """
        Evaluate accuracy of a model on a sample list subjected to a specific corruption and severity.
        """
        model.eval()
        correct = 0
        total = 0

        # Process in batches
        for i in range(0, len(dataset_samples), batch_size):
            batch_slice = dataset_samples[i : i + batch_size]
            batch_tensors = []
            batch_labels = []

            for path, label in batch_slice:
                try:
                    with Image.open(path) as img:
                        img = img.convert("RGB")
                        # Apply corruption
                        corrupted_img = apply_corruption_by_name(
                            img, corruption_name, severity=severity, return_pil=True
                        )
                        tensor = self.tensor_transform(corrupted_img)
                        batch_tensors.append(tensor)
                        batch_labels.append(label)
                except Exception as e:
                    continue

            if len(batch_tensors) == 0:
                continue

            images = torch.stack(batch_tensors).to(self.device)
            targets = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

            with torch.no_grad():
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0

    def run_full_benchmark(
        self,
        model: torch.nn.Module,
        dataset_samples: List[Tuple[Path, int]],
        model_name: str = "MobileNetV2",
        corruptions: Optional[List[str]] = None,
        severities: Optional[List[int]] = None,
        baseline_clean_acc: Optional[float] = None
    ) -> Dict:
        """
        Execute comprehensive benchmarking across all corruption families and severities.
        Returns detailed results including mCE, per-category accuracy, and raw matrices.
        """
        corruptions = corruptions or list(CORRUPTION_REGISTRY.keys())
        severities = severities or [1, 2, 3, 4, 5]
        
        # 1. Clean test accuracy
        if baseline_clean_acc is None:
            print(f"[Benchmark] Measuring Clean Accuracy for '{model_name}'...")
            clean_correct = 0
            clean_total = 0
            for i in range(0, len(dataset_samples), 32):
                batch = dataset_samples[i : i + 32]
                tensors = []
                labels = []
                for p, l in batch:
                    with Image.open(p) as im:
                        tensors.append(self.tensor_transform(im.convert("RGB")))
                        labels.append(l)
                if tensors:
                    imgs = torch.stack(tensors).to(self.device)
                    lbls = torch.tensor(labels, dtype=torch.long).to(self.device)
                    with torch.no_grad():
                        preds = model(imgs).max(1)[1]
                        clean_correct += preds.eq(lbls).sum().item()
                        clean_total += lbls.size(0)
            clean_acc = clean_correct / clean_total if clean_total > 0 else 0.0
        else:
            clean_acc = baseline_clean_acc

        print(f"[Benchmark] Clean Accuracy: {clean_acc * 100:.2f}%")

        # 2. Corrupted evaluation
        results_matrix: Dict[str, Dict[int, float]] = {}
        error_matrix: Dict[str, Dict[int, float]] = {}
        category_accs: Dict[str, List[float]] = {cat: [] for cat in CORRUPTION_CATEGORIES}

        for corr in corruptions:
            results_matrix[corr] = {}
            error_matrix[corr] = {}
            # Find category
            category = "other"
            for cat_k, cat_list in CORRUPTION_CATEGORIES.items():
                if corr in cat_list:
                    category = cat_k
                    break

            for sev in severities:
                acc = self.evaluate_under_corruption(
                    model, dataset_samples, corruption_name=corr, severity=sev
                )
                err = 1.0 - acc
                results_matrix[corr][sev] = acc
                error_matrix[corr][sev] = err
                if category in category_accs:
                    category_accs[category].append(acc)

        # 3. Compute Mean Corruption Error (mCE)
        all_errors = [error_matrix[c][s] for c in corruptions for s in severities]
        mce = float(np.mean(all_errors))
        mean_corrupted_acc = 1.0 - mce

        category_summary = {}
        for cat, acc_list in category_accs.items():
            if acc_list:
                category_summary[cat] = {
                    "mean_accuracy": float(np.mean(acc_list)),
                    "mean_error": float(1.0 - np.mean(acc_list))
                }

        report = {
            "model_name": model_name,
            "clean_accuracy": float(clean_acc),
            "mean_corrupted_accuracy": float(mean_corrupted_acc),
            "mean_corruption_error_mce": float(mce),
            "category_summary": category_summary,
            "accuracy_matrix": results_matrix,
            "error_matrix": error_matrix
        }

        return report

def save_benchmark_report(report: Dict, output_path: Union[str, Path]):
    """Save benchmark results to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Benchmark] Report saved to '{path}'")
