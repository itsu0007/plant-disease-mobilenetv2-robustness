"""
Real-World Image Quality Degradation Engine
Implements realistic agricultural camera and environmental corruptions across 5 severity levels.
Works with NumPy arrays (HWC, uint8, RGB) and PIL Images.
"""

import io
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageEnhance

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _ensure_numpy_rgb(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Ensure image is an RGB NumPy array of dtype uint8 with values in [0, 255]."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)
    return image.copy()

def _to_pil_if_needed(image: np.ndarray, return_pil: bool) -> Union[np.ndarray, Image.Image]:
    """Convert back to PIL if requested."""
    if return_pil:
        return Image.fromarray(image)
    return image

# ------------------------------------------------------------------------------
# 1. Blur Corruptions
# ------------------------------------------------------------------------------

def apply_gaussian_blur(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate camera out-of-focus Gaussian point spread function."""
    image = _ensure_numpy_rgb(image)
    # Severities: (kernel_size, sigma)
    params = {
        1: (5, 1.0),
        2: (9, 2.0),
        3: (15, 3.5),
        4: (21, 5.0),
        5: (31, 7.5),
    }
    ksize, sigma = params.get(severity, params[3])
    
    # 1D Gaussian kernel convolution along axes
    radius = ksize // 2
    x = np.arange(-radius, radius + 1)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    
    # Separable filtering for speed & no OpenCV dependency requirement
    img_f = image.astype(np.float32)
    filtered = np.zeros_like(img_f)
    pad = radius
    
    # Horizontal pass
    padded_h = np.pad(img_f, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    for i in range(ksize):
        filtered += padded_h[:, i : i + image.shape[1], :] * kernel_1d[i]
        
    # Vertical pass
    padded_v = np.pad(filtered, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
    result = np.zeros_like(img_f)
    for i in range(ksize):
        result += padded_v[i : i + image.shape[0], :, :] * kernel_1d[i]
        
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_motion_blur(image: np.ndarray, severity: int = 1, angle: float = 45.0) -> np.ndarray:
    """Simulate hand shaking or foliage movement motion blur."""
    image = _ensure_numpy_rgb(image)
    sizes = {1: 7, 2: 13, 3: 21, 4: 29, 5: 41}
    ksize = sizes.get(severity, 21)
    
    # Generate linear motion blur kernel at angle
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    rad = np.deg2rad(angle)
    center = ksize // 2
    slope_x = np.cos(rad)
    slope_y = np.sin(rad)
    
    for i in range(-center, center + 1):
        x = int(round(center + i * slope_x))
        y = int(round(center + i * slope_y))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
            
    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0

    # 2D Fourier domain or direct spatial convolution
    pad = ksize // 2
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w, c = image.shape
    out = np.zeros((h, w, c), dtype=np.float32)
    
    for c_idx in range(c):
        for ky in range(ksize):
            for kx in range(ksize):
                w_val = kernel[ky, kx]
                if w_val > 0:
                    out[:, :, c_idx] += padded[ky : ky + h, kx : kx + w, c_idx] * w_val
                    
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_defocus_blur(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate optical lens defocus with circular disc kernel (bokeh blur)."""
    image = _ensure_numpy_rgb(image)
    radii = {1: 2, 2: 4, 3: 6, 4: 9, 5: 13}
    radius = radii.get(severity, 6)
    
    # Create circular disc kernel
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((diameter, diameter), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= kernel.sum()
    
    pad = radius
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w, c = image.shape
    out = np.zeros((h, w, c), dtype=np.float32)
    
    for c_idx in range(c):
        for ky in range(diameter):
            for kx in range(diameter):
                w_val = kernel[ky, kx]
                if w_val > 0:
                    out[:, :, c_idx] += padded[ky : ky + h, kx : kx + w, c_idx] * w_val
                    
    return np.clip(out, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------------------
# 2. Noise Corruptions
# ------------------------------------------------------------------------------

def apply_gaussian_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate high-ISO sensor thermal noise in low-light conditions."""
    image = _ensure_numpy_rgb(image)
    sigmas = {1: 15.0, 2: 30.0, 3: 50.0, 4: 75.0, 5: 110.0}
    sigma = sigmas.get(severity, 50.0)
    
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_shot_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate quantum photon shot noise (Poisson distribution)."""
    image = _ensure_numpy_rgb(image)
    scales = {1: 120.0, 2: 60.0, 3: 30.0, 4: 15.0, 5: 7.0}
    scale = scales.get(severity, 30.0)
    
    # Normalize to [0, 1], sample Poisson, rescale
    img_norm = image.astype(np.float32) / 255.0
    noisy = np.random.poisson(np.clip(img_norm * scale, 0, None)) / scale
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)

def apply_salt_and_pepper_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate analog sensor transmission bit-errors and bad sensor pixels."""
    image = _ensure_numpy_rgb(image)
    amounts = {1: 0.01, 2: 0.03, 3: 0.06, 4: 0.10, 5: 0.18}
    amount = amounts.get(severity, 0.06)
    
    noisy = image.copy()
    h, w, c = image.shape
    num_noise = int(amount * h * w)
    
    # Salt (pure white)
    num_salt = num_noise // 2
    coords_salt_y = np.random.randint(0, h, num_salt)
    coords_salt_x = np.random.randint(0, w, num_salt)
    noisy[coords_salt_y, coords_salt_x, :] = 255
    
    # Pepper (pure black)
    num_pepper = num_noise - num_salt
    coords_pep_y = np.random.randint(0, h, num_pepper)
    coords_pep_x = np.random.randint(0, w, num_pepper)
    noisy[coords_pep_y, coords_pep_x, :] = 0
    
    return noisy

# ------------------------------------------------------------------------------
# 3. Weather & Lighting Corruptions
# ------------------------------------------------------------------------------

def apply_overexposure(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate harsh direct noon sunlight glare and highlight clipping."""
    image = _ensure_numpy_rgb(image)
    factors = {1: 1.25, 2: 1.50, 3: 1.80, 4: 2.20, 5: 2.70}
    factor = factors.get(severity, 1.80)
    
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    enhanced = enhancer.enhance(factor)
    return np.array(enhanced)

def apply_underexposure(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate canopy shading, cloud cover, or dusk lighting."""
    image = _ensure_numpy_rgb(image)
    factors = {1: 0.78, 2: 0.62, 3: 0.48, 4: 0.35, 5: 0.20}
    factor = factors.get(severity, 0.48)
    
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    enhanced = enhancer.enhance(factor)
    return np.array(enhanced)

def apply_shadow_gradient(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate non-uniform cast shadow across the leaf from neighboring foliage."""
    image = _ensure_numpy_rgb(image)
    strengths = {1: 0.30, 2: 0.45, 3: 0.60, 4: 0.75, 5: 0.90}
    strength = strengths.get(severity, 0.60)
    
    h, w, _ = image.shape
    # Diagonal linear gradient mask
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    gradient = (xx + yy) / 2.0  # Range 0 to 1
    
    # Shadow multiplier: 1.0 (unshadowed) down to (1.0 - strength)
    shadow_map = 1.0 - strength * (1.0 - gradient)
    shadow_map = np.clip(shadow_map, 0.05, 1.0)[:, :, np.newaxis]
    
    shaded = image.astype(np.float32) * shadow_map
    return np.clip(shaded, 0, 255).astype(np.uint8)

def apply_color_temperature_shift(image: np.ndarray, severity: int = 1, warm: bool = True) -> np.ndarray:
    """Simulate golden hour sunset or cool blue overcast daylight."""
    image = _ensure_numpy_rgb(image)
    shifts = {1: 20, 2: 40, 3: 65, 4: 90, 5: 125}
    shift = shifts.get(severity, 65)
    
    img_f = image.astype(np.float32)
    if warm:
        # Boost Red, reduce Blue
        img_f[:, :, 0] += shift
        img_f[:, :, 2] -= shift * 0.7
    else:
        # Boost Blue, reduce Red
        img_f[:, :, 2] += shift
        img_f[:, :, 0] -= shift * 0.7
        
    return np.clip(img_f, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------------------
# 4. Digital & Compression Corruptions
# ------------------------------------------------------------------------------

def apply_jpeg_compression(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate aggressive mobile messaging compression (WhatsApp, Telegram, MMS)."""
    image = _ensure_numpy_rgb(image)
    qualities = {1: 65, 2: 45, 3: 25, 4: 15, 5: 6}
    quality = qualities.get(severity, 25)
    
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_pil = Image.open(buffer)
    return np.array(compressed_pil.convert("RGB"))

def apply_low_resolution_downsample(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate low-resolution sensor capture upsampled to model resolution."""
    image = _ensure_numpy_rgb(image)
    factors = {1: 0.70, 2: 0.50, 3: 0.35, 4: 0.22, 5: 0.12}
    factor = factors.get(severity, 0.35)
    
    h, w, _ = image.shape
    new_h = max(8, int(h * factor))
    new_w = max(8, int(w * factor))
    
    pil_img = Image.fromarray(image)
    downsampled = pil_img.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    upsampled = downsampled.resize((w, h), resample=Image.Resampling.BILINEAR)
    return np.array(upsampled)

# ------------------------------------------------------------------------------
# Master Dispatcher & Random Selector
# ------------------------------------------------------------------------------

CORRUPTION_REGISTRY = {
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "defocus_blur": apply_defocus_blur,
    "gaussian_noise": apply_gaussian_noise,
    "shot_noise": apply_shot_noise,
    "salt_and_pepper_noise": apply_salt_and_pepper_noise,
    "overexposure": apply_overexposure,
    "underexposure": apply_underexposure,
    "shadow_gradient": apply_shadow_gradient,
    "color_temperature_shift": apply_color_temperature_shift,
    "jpeg_compression": apply_jpeg_compression,
    "low_resolution_downsample": apply_low_resolution_downsample,
}

CORRUPTION_CATEGORIES = {
    "blur": ["gaussian_blur", "motion_blur", "defocus_blur"],
    "noise": ["gaussian_noise", "shot_noise", "salt_and_pepper_noise"],
    "weather_and_lighting": ["overexposure", "underexposure", "shadow_gradient", "color_temperature_shift"],
    "digital": ["jpeg_compression", "low_resolution_downsample"],
}

def apply_corruption_by_name(
    image: Union[np.ndarray, Image.Image],
    corruption_name: str,
    severity: int = 3,
    return_pil: bool = False
) -> Union[np.ndarray, Image.Image]:
    """Apply a named corruption with severity in [1, 5]."""
    severity = max(1, min(5, severity))
    fn = CORRUPTION_REGISTRY.get(corruption_name)
    if fn is None:
        raise ValueError(f"Unknown corruption: '{corruption_name}'. Choose from: {list(CORRUPTION_REGISTRY.keys())}")
    
    img_np = _ensure_numpy_rgb(image)
    corrupted_np = fn(img_np, severity=severity)
    return _to_pil_if_needed(corrupted_np, return_pil)

def apply_random_corruption(
    image: Union[np.ndarray, Image.Image],
    severities: Tuple[int, int] = (1, 4),
    allowed_corruptions: Optional[List[str]] = None,
    return_pil: bool = False
) -> Tuple[Union[np.ndarray, Image.Image], str, int]:
    """
    Select and apply a random corruption with a random severity level.
    Used for robust data augmentation during training.
    """
    candidates = allowed_corruptions or list(CORRUPTION_REGISTRY.keys())
    chosen_name = np.random.choice(candidates)
    severity = int(np.random.randint(severities[0], severities[1] + 1))
    
    corrupted = apply_corruption_by_name(image, chosen_name, severity=severity, return_pil=return_pil)
    return corrupted, chosen_name, severity
