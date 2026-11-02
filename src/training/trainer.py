"""
Robust Training Engine
Implements training pipeline with Automatic Mixed Precision (AMP), Cosine Annealing,
Early Stopping, Checkpointing, and CutMix/MixUp support.
"""

import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import LabelSmoothingCrossEntropy
from ..dataset.augmentations import cutmix_batch, mixup_batch

class ModelTrainer:
    """
    Handles training, validation, early stopping, and checkpoint management.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        mixed_precision: bool = True,
        use_cutmix: bool = False,
        cutmix_prob: float = 0.3,
        use_mixup: bool = False,
        mixup_prob: float = 0.2,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        model_name: str = "mobilenetv2_robust"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # Optimizer fallback
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=0.0005, weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.mixed_precision = mixed_precision and (self.device == "cuda")
        self.scaler = torch.amp.GradScaler('cuda') if self.mixed_precision else None
        
        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.use_mixup = use_mixup
        self.mixup_prob = mixup_prob

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": []
        }

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Runs a single epoch of training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for images, targets, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            apply_cm = self.use_cutmix and (np.random.rand() < self.cutmix_prob)
            apply_mu = self.use_mixup and (np.random.rand() < self.mixup_prob) and not apply_cm

            self.optimizer.zero_grad()

            if apply_cm:
                images, targets_a, targets_b, lam = cutmix_batch(images, targets)
            elif apply_mu:
                images, targets_a, targets_b, lam = mixup_batch(images, targets)

            # Mixed precision forward & backward
            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    if apply_cm or apply_mu:
                        loss = lam * self.criterion(outputs, targets_a) + (1.0 - lam) * self.criterion(outputs, targets_b)
                    else:
                        loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if apply_cm or apply_mu:
                    loss = lam * self.criterion(outputs, targets_a) + (1.0 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.1f}%"})

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """Runs validation step."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, targets, _ in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                if self.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        loss = val_criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = val_criterion(outputs, targets)

                running_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        return val_loss, val_acc

    def fit(
        self,
        epochs: int = 25,
        early_stopping_patience: int = 8,
        save_best_only: bool = True
    ) -> Dict[str, List[float]]:
        """
        Executes full training loop across epochs with early stopping.
        """
        best_val_acc = 0.0
        patience_counter = 0
        best_ckpt_path = self.checkpoint_dir / f"{self.model_name}.pth"

        print(f"\n=======================================================")
        print(f"Starting Training: '{self.model_name}' on {self.device.upper()}")
        print(f"Epochs: {epochs} | AMP: {self.mixed_precision} | Early Stopping: {early_stopping_patience}")
        print(f"=======================================================\n")

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            # Checkpoint save
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "val_loss": val_loss,
                    "model_name": self.model_name,
                    "history": self.history
                }, best_ckpt_path)
                print(f"  --> Saved new best checkpoint: {best_ckpt_path} (Val Acc: {best_val_acc*100:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n[EarlyStopping] Validation accuracy did not improve for {early_stopping_patience} epochs. Stopping early.")
                    break

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.2f} minutes. Best Val Acc: {best_val_acc*100:.2f}%")

        # Save training history JSON
        history_file = self.checkpoint_dir / f"{self.model_name}_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
