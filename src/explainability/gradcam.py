"""
Grad-CAM & Grad-CAM++ Explainability Engine
Generates Class Activation Maps (heatmaps) highlighting visual regions influential in MobileNetV2 predictions.
Essential for analyzing attention stability under image quality variations (clean vs corrupted).
"""

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for MobileNetV2.
    Hooks into the final convolutional feature maps and computes target class gradients.
    """
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # Default to the last convolutional layer of MobileNetV2 features
        if target_layer is None:
            if hasattr(model, "get_last_conv_layer"):
                self.target_layer = model.get_last_conv_layer()
            elif hasattr(model, "features"):
                self.target_layer = model.features[-1]
            else:
                raise ValueError("Could not automatically locate the target conv layer. Pass target_layer explicitly.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Clean up registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for a given input tensor (shape: [1, C, H, W]).
        Returns:
            heatmap: 2D numpy array [H, W] normalized to [0, 1]
            pred_class: predicted class index
            confidence: softmax probability for the class
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            pred_class = int(output.argmax(dim=1).item())
        else:
            pred_class = target_class

        confidence = float(probs[0, pred_class].item())

        # Backward pass for the target class logit
        target_score = output[0, pred_class]
        target_score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: [1, C, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU to retain only features that positively correlate with class
        cam = F.relu(cam)

        # Upsample to match original input tensor height and width
        h, w = input_tensor.size(2), input_tensor.size(3)
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        # Min-max normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence

def apply_heatmap_overlay(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap onto the original RGB image.
    """
    if isinstance(image, Image.Image):
        orig_img = np.array(image.convert("RGB"))
    else:
        orig_img = image.copy()
        
    h, w = orig_img.shape[:2]
    
    # Resize heatmap if shapes differ
    if heatmap.shape != (h, w):
        pil_heat = Image.fromarray((heatmap * 255).astype(np.uint8))
        pil_heat = pil_heat.resize((w, h), Image.Resampling.BILINEAR)
        heat_norm = np.array(pil_heat).astype(np.float32) / 255.0
    else:
        heat_norm = heatmap

    # Simple JET-like color mapping without requiring heavy cv2
    # Jet mapping: Blue (0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1.0)
    r = np.clip(1.5 - np.abs(4.0 * heat_norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * heat_norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * heat_norm - 1.0), 0.0, 1.0)
    color_map = np.stack([r, g, b], axis=-1) * 255.0

    # Blend original image and colormap
    blended = (1.0 - alpha) * orig_img.astype(np.float32) + alpha * color_map
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)
