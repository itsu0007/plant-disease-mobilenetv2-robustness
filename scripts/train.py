"""
CLI Script: Train Baseline or Robust MobileNetV2 Model
Executes training with full AMP acceleration, early stopping, and metric logging.
"""

import argparse
from pathlib import Path
import sys
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.plant_dataset import get_dataloaders
from src.models.model_factory import build_baseline_model, build_robust_model
from src.training.losses import LabelSmoothingCrossEntropy
from src.training.trainer import ModelTrainer
from src.utils.config_parser import load_yaml_config
from src.utils.plant_classes import NUM_CLASSES

def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on Plant Disease Dataset")
    parser.add_argument("--mode", type=str, choices=["robust", "baseline"], default="robust",
                        help="Model training regime: 'robust' (with quality corruption augs) or 'baseline'")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Peak learning rate")
    parser.add_argument("--data_dir", type=str, default="data/plant_dataset", help="Dataset directory")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    args = parser.parse_args()

    # Load configuration
    config = load_yaml_config(args.config) if Path(args.config).exists() else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"Training Mode: {args.mode.upper()}")
    print(f"Device: {device.upper()} | PyTorch Version: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    # Prepare DataLoaders
    use_robust_aug = (args.mode == "robust")
    train_loader, val_loader, test_loader, counts = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_robust_aug=use_robust_aug,
        corruption_prob=0.75 if use_robust_aug else 0.0,
        synthetic_fallback_dir="data/synthetic_sample"
    )

    # Build Model
    if args.mode == "baseline":
        model = build_baseline_model(num_classes=NUM_CLASSES, pretrained=True)
        model_name = "mobilenetv2_baseline"
        criterion = torch.nn.CrossEntropyLoss()
        use_cutmix = False
        use_mixup = False
    else:
        model = build_robust_model(num_classes=NUM_CLASSES, pretrained=True, attention_type="cbam")
        model_name = "mobilenetv2_robust"
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        use_cutmix = True
        use_mixup = True

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Trainer instance
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=True,
        use_cutmix=use_cutmix,
        use_mixup=use_mixup,
        checkpoint_dir=args.checkpoints_dir,
        model_name=model_name
    )

    # Train
    history = trainer.fit(epochs=args.epochs, early_stopping_patience=6)
    print(f"[Done] Training finished for {model_name}. Checkpoint in '{args.checkpoints_dir}'.")

if __name__ == "__main__":
    main()
