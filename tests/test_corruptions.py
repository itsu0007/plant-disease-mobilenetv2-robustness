"""
Unit Tests: Real-World Degradation Engine
Verifies that all corruptions execute without error and preserve valid image shape and dtype.
"""

from pathlib import Path
import sys
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions.degradation_engine import (
    CORRUPTION_REGISTRY,
    apply_corruption_by_name,
    apply_random_corruption,
)

def test_all_corruptions_execute():
    """Verify that every registered corruption runs on a dummy image across severity levels."""
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    for name in CORRUPTION_REGISTRY.keys():
        for severity in [1, 3, 5]:
            out = apply_corruption_by_name(dummy_img, name, severity=severity)
            assert isinstance(out, np.ndarray), f"Corruption {name} did not return numpy array"
            assert out.shape == (224, 224, 3), f"Corruption {name} changed shape: {out.shape}"
            assert out.dtype == np.uint8, f"Corruption {name} returned wrong dtype: {out.dtype}"
            assert out.min() >= 0 and out.max() <= 255, f"Corruption {name} out of bounds"

def test_random_corruption_selector():
    """Verify random corruption selection."""
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    out, name, sev = apply_random_corruption(dummy_img, severities=(1, 5))
    
    assert name in CORRUPTION_REGISTRY
    assert 1 <= sev <= 5
    assert out.shape == (224, 224, 3)

if __name__ == "__main__":
    test_all_corruptions_execute()
    test_random_corruption_selector()
    print("All corruption tests passed successfully!")
