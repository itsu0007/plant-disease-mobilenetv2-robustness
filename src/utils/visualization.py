"""
Visualization Utilities
Generates publication-quality charts for robustness comparisons, corruption curves,
confusion matrices, and side-by-side Grad-CAM explainability figures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution and server safety
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns

def plot_robustness_comparison(
    baseline_report: Dict,
    robust_report: Dict,
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plots a multi-panel comparison between Baseline and Robust MobileNetV2 models:
    1. Overall Metrics (Clean Acc, Corrupted Acc, mCE)
    2. Category-level Accuracies (Blur, Noise, Lighting, Digital)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Key Metrics Comparison
    metrics = ["Clean Accuracy", "Corrupted Accuracy", "Mean Corruption Error (mCE)"]
    baseline_vals = [
        baseline_report.get("clean_accuracy", 0.0) * 100,
        baseline_report.get("mean_corrupted_accuracy", 0.0) * 100,
        baseline_report.get("mean_corruption_error_mce", 0.0) * 100
    ]
    robust_vals = [
        robust_report.get("clean_accuracy", 0.0) * 100,
        robust_report.get("mean_corrupted_accuracy", 0.0) * 100,
        robust_report.get("mean_corruption_error_mce", 0.0) * 100
    ]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, baseline_vals, width, label="Baseline MobileNetV2", color="#e74c3c", alpha=0.9)
    rects2 = axes[0].bar(x + width/2, robust_vals, width, label="Robust MobileNetV2 (Ours)", color="#2ecc71", alpha=0.9)
    
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Overall Robustness Benchmark Comparison", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=10)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Add value annotations
    for rect in rects1 + rects2:
        h = rect.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Panel 2: Category Breakdown
    categories = ["blur", "noise", "weather_and_lighting", "digital"]
    cat_labels = ["Blur", "Noise", "Weather/Light", "Digital/JPEG"]
    
    b_cat_acc = [baseline_report.get("category_summary", {}).get(c, {}).get("mean_accuracy", 0.0) * 100 for c in categories]
    r_cat_acc = [robust_report.get("category_summary", {}).get(c, {}).get("mean_accuracy", 0.0) * 100 for c in categories]

    xc = np.arange(len(categories))
    axes[1].bar(xc - width/2, b_cat_acc, width, label="Baseline", color="#e74c3c", alpha=0.9)
    axes[1].bar(xc + width/2, r_cat_acc, width, label="Robust (Ours)", color="#2ecc71", alpha=0.9)
    axes[1].set_ylabel("Mean Accuracy (%)")
    axes[1].set_title("Robustness Across Corruption Categories", fontsize=12, fontweight="bold")
    axes[1].set_xticks(xc)
    axes[1].set_xticklabels(cat_labels)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        print(f"[Visualization] Figure saved to {output_path}")

    return fig

def plot_severity_curves(
    baseline_matrix: Dict[str, Dict[int, float]],
    robust_matrix: Dict[str, Dict[int, float]],
    corruption_names: List[str],
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plots Accuracy vs Degradation Severity (1 to 5) for selected corruptions.
    """
    n_plots = len(corruption_names)
    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    severities = [1, 2, 3, 4, 5]

    for idx, corr_name in enumerate(corruption_names):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        b_acc = [baseline_matrix.get(corr_name, {}).get(s, 0.0) * 100 for s in severities]
        r_acc = [robust_matrix.get(corr_name, {}).get(s, 0.0) * 100 for s in severities]

        ax.plot(severities, b_acc, marker="o", color="#e74c3c", linewidth=2, label="Baseline")
        ax.plot(severities, r_acc, marker="s", color="#2ecc71", linewidth=2, label="Robust (Ours)")
        
        clean_title = corr_name.replace("_", " ").title()
        ax.set_title(clean_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Severity Level (1-5)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_xticks(severities)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend()

    # Hide unused subplots
    for idx in range(n_plots, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig

def plot_confusion_matrix_heatmap(
    conf_matrix: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    title: str = "Plant Disease Confusion Matrix",
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Plots annotated confusion matrix heatmap."""
    cm = np.array(conf_matrix)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Greens",
        xticklabels=[c.split("___")[-1][:12] for c in class_names],
        yticklabels=[c.split("___")[-1][:12] for c in class_names],
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
    return fig
