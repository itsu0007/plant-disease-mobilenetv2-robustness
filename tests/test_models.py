"""
Unit Tests: Model Architecture & Attention Blocks
Verifies that Baseline and Robust MobileNetV2 models construct, run forward passes,
and yield expected tensor shapes.
"""

from pathlib import Path
import sys
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import build_baseline_model, build_robust_model
from src.utils.plant_classes import NUM_CLASSES

def test_baseline_mobilenetv2_forward():
    """Verify baseline MobileNetV2 architecture output."""
    model = build_baseline_model(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, NUM_CLASSES), f"Expected shape (2, {NUM_CLASSES}), got {output.shape}"

def test_robust_mobilenetv2_forward():
    """Verify robust MobileNetV2 with CBAM attention."""
    model = build_robust_model(num_classes=NUM_CLASSES, pretrained=False, attention_type="cbam")
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, NUM_CLASSES), f"Expected shape (2, {NUM_CLASSES}), got {output.shape}"
    
    summary = model.get_parameter_summary()
    assert summary["total_params"] > 2_000_000

if __name__ == "__main__":
    test_baseline_mobilenetv2_forward()
    test_robust_mobilenetv2_forward()
    print("All model architecture tests passed successfully!")
