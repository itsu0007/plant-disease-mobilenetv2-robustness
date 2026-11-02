"""
CLI Script: Generate Sample Synthetic Leaf Dataset
Generates sample leaves across all classes for rapid testing, demonstrations, and training.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.synthetic_data import create_synthetic_dataset
from src.utils.plant_classes import CLASS_NAMES

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic plant disease leaf dataset")
    parser.add_argument("--output_dir", type=str, default="data/plant_dataset",
                        help="Target output folder for dataset")
    parser.add_argument("--samples_per_class", type=int, default=20,
                        help="Number of images per disease class")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image resolution (square)")
    args = parser.parse_args()

    print(f"Generating synthetic dataset in '{args.output_dir}'...")
    print(f"Classes: {len(CLASS_NAMES)} | Samples per class: {args.samples_per_class} | Resolution: {args.img_size}x{args.img_size}")
    
    counts = create_synthetic_dataset(
        output_dir=args.output_dir,
        classes=CLASS_NAMES,
        samples_per_class=args.samples_per_class,
        img_size=(args.img_size, args.img_size)
    )
    
    total = sum(counts.values())
    print(f"Success! Generated {total} synthetic images across {len(counts)} classes.")

if __name__ == "__main__":
    main()
