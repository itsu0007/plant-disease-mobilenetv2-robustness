"""
Unit Tests: Grad-CAM Explainability Engine
Verifies that Grad-CAM hooks correctly capture activation maps, calculate gradients,
and return valid normalized heatmaps.
"""

from pathlib import Path
import sys
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.gradcam import GradCAM, apply_heatmap_overlay
from src.models.model_factory import build_robust_model
from src.utils.plant_classes import NUM_CLASSES

def test_gradcam_generation():
    """Verify Grad-CAM output shape and value normalization."""
    model = build_robust_model(num_classes=NUM_CLASSES, pretrained=False)
    gradcam = GradCAM(model)

    dummy_input = torch.randn(1, 3, 224, 224)
    cam, pred_class, confidence = gradcam.generate_cam(dummy_input)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224)
    assert 0 <= cam.min() <= cam.max() <= 1.0 + 1e-5
    assert 0 <= pred_class < NUM_CLASSES
    assert 0.0 <= confidence <= 1.0

    gradcam.remove_hooks()

def test_heatmap_overlay():
    """Verify heatmap blending onto RGB image."""
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    dummy_heat = np.random.rand(224, 224).astype(np.float32)

    blended = apply_heatmap_overlay(dummy_img, dummy_heat, alpha=0.5)
    assert blended.size == (224, 224)

if __name__ == "__main__":
    test_gradcam_generation()
    test_heatmap_overlay()
    print("All Grad-CAM tests passed successfully!")
