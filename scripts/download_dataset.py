"""
Dataset Acquisition & Setup Script
Assists in downloading and setting up the PlantVillage dataset or generating
the synthetic reference dataset for rapid training and experimentation.
"""

import argparse
from pathlib import Path
import sys
import urllib.request
import zipfile

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.synthetic_data import create_synthetic_dataset
from src.utils.plant_classes import CLASS_NAMES

def main():
    parser = argparse.ArgumentParser(description="Acquire or generate plant disease leaf dataset")
    parser.add_argument("--data_dir", type=str, default="data/plant_dataset",
                        help="Target dataset directory")
    parser.add_argument("--generate_synthetic", action="store_true", default=True,
                        help="Generate synthetic dataset if raw dataset is not present")
    parser.add_argument("--samples_per_class", type=int, default=25,
                        help="Number of synthetic samples per class")
    args = parser.parse_args()

    target_dir = Path(args.data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("Plant Disease Dataset Setup Assistant")
    print("==================================================")
    print(f"Target Directory: {target_dir.resolve()}")

    # Check if dataset already has folders
    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if len(subdirs) >= 10:
        print(f"[Info] Found {len(subdirs)} existing class directories in '{target_dir}'. Dataset is ready!")
        return

    print("\n[Options for Real World Datasets]:")
    print("1. Kaggle PlantVillage Dataset:")
    print("   Run: kaggle datasets download -d emmarex/plantdisease")
    print("   Extract contents directly into: data/plant_dataset/<class_name>/...")
    print("2. PlantVillage GitHub Mirror:")
    print("   https://github.com/spMohanty/PlantVillage-Dataset\n")

    if args.generate_synthetic:
        print(f"[Generator] Generating {args.samples_per_class} synthetic leaf samples per class...")
        counts = create_synthetic_dataset(
            output_dir=target_dir,
            classes=CLASS_NAMES,
            samples_per_class=args.samples_per_class
        )
        print(f"[Done] Generated {sum(counts.values())} sample images across {len(counts)} disease classes.")
        print("You can now train models and run the web demo immediately!")

if __name__ == "__main__":
    main()
