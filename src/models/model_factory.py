"""
Model Factory & Checkpoint Loader
Provides clean constructor functions to instantiate Baseline and Robust MobileNetV2 variants.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch

from .mobilenet_v2 import RobustMobileNetV2
from ..utils.plant_classes import NUM_CLASSES

def build_baseline_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.2
) -> RobustMobileNetV2:
    """
    Constructs the standard baseline MobileNetV2 architecture.
    Does not use attention or extra deep dense layers.
    """
    return RobustMobileNetV2(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_attention=False,
        head_hidden_dim=0,
        use_head_batchnorm=False
    )

def build_robust_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    attention_type: str = "cbam",
    hidden_dim: int = 512
) -> RobustMobileNetV2:
    """
    Constructs the enhanced robust MobileNetV2 architecture.
    Equipped with CBAM attention gating and batch-normalized dense head.
    """
    return RobustMobileNetV2(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_attention=True,
        attention_type=attention_type,
        head_hidden_dim=hidden_dim,
        use_head_batchnorm=True
    )

def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_type: str = "robust",
    num_classes: int = NUM_CLASSES,
    device: str = "cpu"
) -> Tuple[RobustMobileNetV2, Dict]:
    """
    Instantiate model and load state dictionary from checkpoint.
    """
    if model_type.lower() == "baseline":
        model = build_baseline_model(num_classes=num_classes, pretrained=False)
    else:
        model = build_robust_model(num_classes=num_classes, pretrained=False)

    ckpt_path = Path(checkpoint_path)
    metadata = {}
    if ckpt_path.exists():
        try:
            state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(str(ckpt_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
            metadata = {k: v for k, v in state.items() if k != "state_dict"}
        elif isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
            metadata = {k: v for k, v in state.items() if k != "model"}
        else:
            model.load_state_dict(state)
        print(f"[ModelFactory] Loaded '{model_type}' weights from '{ckpt_path}'")
    else:
        print(f"[ModelFactory] Checkpoint '{ckpt_path}' not found. Using initialized weights.")

    model.to(device)
    model.eval()
    return model, metadata
