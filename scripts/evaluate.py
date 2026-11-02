"""
CLI Script: Comprehensive Robustness Benchmarking
Evaluates Baseline vs Robust MobileNetV2 models across real-world image corruptions (Blur, Noise, Lighting, Digital)
at 5 severity levels, outputs mCE metrics, and exports comparison charts.
"""

import argparse
import json
from pathlib import Path
import sys
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions.benchmark_suite import CorruptionBenchmarkRunner, save_benchmark_report
from src.dataset.plant_dataset import discover_dataset_samples
from src.dataset.synthetic_data import create_synthetic_dataset
from src.models.model_factory import load_model_from_checkpoint
from src.utils.plant_classes import CLASS_NAMES, NUM_CLASSES
from src.utils.visualization import plot_robustness_comparison, plot_severity_curves

def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness across corruptions")
    parser.add_argument("--baseline_ckpt", type=str, default="checkpoints/mobilenetv2_baseline.pth",
                        help="Path to baseline model checkpoint")
    parser.add_argument("--robust_ckpt", type=str, default="checkpoints/mobilenetv2_robust.pth",
                        help="Path to robust model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/plant_dataset",
                        help="Path to dataset folder")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save benchmark results and charts")
    parser.add_argument("--max_test_samples", type=int, default=150,
                        help="Maximum test samples to evaluate for benchmark speed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==================================================")
    print("Plant Disease MobileNetV2 Robustness Benchmark")
    print(f"Device: {device.upper()}")
    print("==================================================")

    # 1. Discover or create test samples
    samples, _ = discover_dataset_samples(args.data_dir)
    if len(samples) == 0:
        print(f"[Dataset] No images in '{args.data_dir}'. Generating synthetic benchmark samples...")
        sample_dir = Path("data/synthetic_sample")
        create_synthetic_dataset(sample_dir, samples_per_class=10)
        samples, _ = discover_dataset_samples(sample_dir)

    # Limit sample size if requested for speed
    if args.max_test_samples and len(samples) > args.max_test_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, args.max_test_samples)
    print(f"[Benchmark] Evaluating on {len(samples)} leaf test samples across classes.")

    # 2. Instantiate models
    baseline_model, _ = load_model_from_checkpoint(
        args.baseline_ckpt, model_type="baseline", num_classes=NUM_CLASSES, device=device
    )
    robust_model, _ = load_model_from_checkpoint(
        args.robust_ckpt, model_type="robust", num_classes=NUM_CLASSES, device=device
    )

    runner = CorruptionBenchmarkRunner(img_size=224, device=device)

    # 3. Benchmark Baseline Model
    print("\n>>> Benchmarking Baseline MobileNetV2...")
    baseline_report = runner.run_full_benchmark(
        baseline_model, dataset_samples=samples, model_name="Baseline MobileNetV2"
    )
    save_benchmark_report(baseline_report, output_dir / "baseline_benchmark.json")

    # 4. Benchmark Robust Model
    print("\n>>> Benchmarking Robust MobileNetV2...")
    robust_report = runner.run_full_benchmark(
        robust_model, dataset_samples=samples, model_name="Robust MobileNetV2"
    )
    save_benchmark_report(robust_report, output_dir / "robust_benchmark.json")

    # 5. Print Comparison Summary
    b_clean = baseline_report["clean_accuracy"] * 100
    r_clean = robust_report["clean_accuracy"] * 100
    b_corr = baseline_report["mean_corrupted_accuracy"] * 100
    r_corr = robust_report["mean_corrupted_accuracy"] * 100
    b_mce = baseline_report["mean_corruption_error_mce"] * 100
    r_mce = robust_report["mean_corruption_error_mce"] * 100

    print("\n=======================================================")
    print("ROBUSTNESS BENCHMARK FINAL RESULTS")
    print("=======================================================")
    print(f"Baseline MobileNetV2: Clean: {b_clean:.1f}% | Corrupted: {b_corr:.1f}% | mCE: {b_mce:.1f}%")
    print(f"Robust MobileNetV2:   Clean: {r_clean:.1f}% | Corrupted: {r_corr:.1f}% | mCE: {r_mce:.1f}%")
    acc_gain = r_corr - b_corr
    mce_drop = b_mce - r_mce
    print(f"Robustness Gain under Corruptions: +{acc_gain:.1f}% Accuracy | mCE Reduction: -{mce_drop:.1f}%")
    print("=======================================================\n")

    # 6. Generate Figures
    print("[Visualization] Generating comparison charts...")
    plot_robustness_comparison(baseline_report, robust_report, output_path=output_dir / "robustness_comparison.png")
    
    key_corruptions = ["gaussian_blur", "motion_blur", "gaussian_noise", "overexposure", "jpeg_compression", "low_resolution_downsample"]
    plot_severity_curves(
        baseline_report["accuracy_matrix"],
        robust_report["accuracy_matrix"],
        corruption_names=key_corruptions,
        output_path=output_dir / "severity_curves.png"
    )
    print(f"[Done] Benchmark completed. Figures saved to '{output_dir}'.")

if __name__ == "__main__":
    main()
