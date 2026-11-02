"""
Synthetic Leaf Dataset Generator
Generates realistic leaf images with morphological structures (veins, contours, chlorophyll gradients)
and disease-specific pathological lesions (blights, pustules, leaf spots, mildew).
Enables instant end-to-end testing, quick training runs, and local validation without requiring 
immediate multi-gigabyte dataset downloads.
"""

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..utils.plant_classes import CLASS_NAMES, PLANT_DISEASE_METADATA

def _draw_leaf_base(draw: ImageDraw.ImageDraw, width: int, height: int, crop_type: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Draw a natural leaf silhouette with venation and color gradient."""
    cx, cy = width // 2, height // 2
    
    # Base leaf color palette (Hue variations depending on crop)
    if "Tomato" in crop_type or "Potato" in crop_type:
        base_color = (42 + random.randint(-6, 6), 118 + random.randint(-10, 10), 38 + random.randint(-6, 6))
    elif "Corn" in crop_type:
        base_color = (65 + random.randint(-8, 8), 135 + random.randint(-10, 10), 40 + random.randint(-6, 6))
    elif "Grape" in crop_type:
        base_color = (35 + random.randint(-5, 5), 105 + random.randint(-8, 8), 30 + random.randint(-5, 5))
    else:
        base_color = (50 + random.randint(-8, 8), 125 + random.randint(-12, 12), 45 + random.randint(-8, 8))

    # Outer leaf boundary (elliptic/lanceolate)
    rx = int(width * 0.38 * random.uniform(0.9, 1.05))
    ry = int(height * 0.44 * random.uniform(0.9, 1.05))
    
    # Draw leaf body polygon
    points = []
    num_pts = 36
    for i in range(num_pts):
        angle = 2 * math.pi * i / num_pts
        # Modulate radius to simulate natural leaf shape
        r_mod = 1.0 - 0.25 * math.sin(angle) + 0.08 * math.sin(3 * angle)
        px = cx + rx * r_mod * math.cos(angle)
        py = cy + ry * r_mod * math.sin(angle)
        points.append((px, py))
        
    draw.polygon(points, fill=base_color, outline=(base_color[0] - 15, base_color[1] - 15, base_color[2] - 10))
    
    # Central midrib vein
    vein_color = (base_color[0] + 25, base_color[1] + 30, base_color[2] + 15)
    draw.line([(cx, cy - ry + 15), (cx, cy + ry - 15)], fill=vein_color, width=3)
    
    # Lateral secondary veins
    for y_offset in range(-ry + 35, ry - 35, 25):
        vy = cy + y_offset
        draw.line([(cx, vy), (cx - rx * 0.7, vy - 20)], fill=vein_color, width=1)
        draw.line([(cx, vy), (cx + rx * 0.7, vy - 20)], fill=vein_color, width=1)
        
    return base_color, (cx, cy)

def _inject_disease_symptoms(image: Image.Image, class_name: str):
    """Draw distinctive pathological disease lesions based on botanical symptoms."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    cx, cy = width // 2, height // 2
    
    if "healthy" in class_name.lower():
        # Clean leaf, occasional harmless chlorophyll shade variation
        return

    # 1. Early Blight (Concentric dark brown rings with chlorotic yellow halo)
    if "early_blight" in class_name.lower():
        num_spots = random.randint(3, 7)
        for _ in range(num_spots):
            sx = cx + random.randint(-int(width * 0.22), int(width * 0.22))
            sy = cy + random.randint(-int(height * 0.25), int(height * 0.25))
            r = random.randint(14, 28)
            # Yellow chlorotic halo
            draw.ellipse([sx - r - 4, sy - r - 4, sx + r + 4, sy + r + 4], fill=(185, 175, 45))
            # Dark brown lesion body
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(70, 42, 20))
            # Inner concentric ring
            draw.ellipse([sx - r // 2, sy - r // 2, sx + r // 2, sy + r // 2], fill=(45, 25, 12))

    # 2. Late Blight (Large dark water-soaked necrotized margins)
    elif "late_blight" in class_name.lower():
        num_patches = random.randint(2, 4)
        for _ in range(num_patches):
            sx = cx + random.randint(-int(width * 0.25), int(width * 0.25))
            sy = cy + random.randint(-int(height * 0.28), int(height * 0.28))
            r = random.randint(25, 45)
            # Dark olive-brown necrotic blotch
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(55, 50, 32))
            # Translucent whitish mold margin
            draw.arc([sx - r - 2, sy - r - 2, sx + r + 2, sy + r + 2], start=0, end=360, fill=(210, 215, 200), width=2)

    # 3. Rust (Elongated reddish-brown/orange pustules)
    elif "rust" in class_name.lower():
        num_pustules = random.randint(25, 45)
        for _ in range(num_pustules):
            sx = cx + random.randint(-int(width * 0.26), int(width * 0.26))
            sy = cy + random.randint(-int(height * 0.30), int(height * 0.30))
            rw = random.randint(3, 6)
            rh = random.randint(5, 10)
            draw.ellipse([sx - rw, sy - rh, sx + rw, sy + rh], fill=(195, 75, 20), outline=(135, 40, 10))

    # 4. Bacterial Spot / Septoria (Multiple small circular dark brown spots with yellow halos)
    elif "bacterial_spot" in class_name.lower() or "septoria" in class_name.lower():
        num_specks = random.randint(20, 40)
        for _ in range(num_specks):
            sx = cx + random.randint(-int(width * 0.25), int(width * 0.25))
            sy = cy + random.randint(-int(height * 0.30), int(height * 0.30))
            r = random.randint(3, 7)
            draw.ellipse([sx - r - 2, sy - r - 2, sx + r + 2, sy + r + 2], fill=(175, 160, 40))
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(40, 28, 18))

    # 5. Powdery Mildew (White-gray powdery fungal patches)
    elif "powdery_mildew" in class_name.lower():
        num_patches = random.randint(5, 10)
        for _ in range(num_patches):
            sx = cx + random.randint(-int(width * 0.24), int(width * 0.24))
            sy = cy + random.randint(-int(height * 0.26), int(height * 0.26))
            r = random.randint(15, 30)
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(225, 228, 225, 180))

    # 6. Viral Mosaic / Leaf Curl (Mottled yellow streaks or mosaic patches)
    elif "mosaic" in class_name.lower() or "curl" in class_name.lower():
        num_streaks = random.randint(12, 22)
        for _ in range(num_streaks):
            sx = cx + random.randint(-int(width * 0.25), int(width * 0.25))
            sy = cy + random.randint(-int(height * 0.28), int(height * 0.28))
            rw = random.randint(8, 18)
            rh = random.randint(12, 26)
            draw.ellipse([sx - rw, sy - rh, sx + rw, sy + rh], fill=(170, 180, 45))

    # 7. Scab / Black Rot (Dark black-brown circular sunken lesions)
    elif "scab" in class_name.lower() or "black_rot" in class_name.lower() or "esca" in class_name.lower():
        num_cankers = random.randint(5, 12)
        for _ in range(num_cankers):
            sx = cx + random.randint(-int(width * 0.24), int(width * 0.24))
            sy = cy + random.randint(-int(height * 0.26), int(height * 0.26))
            r = random.randint(8, 18)
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(28, 22, 18), outline=(65, 45, 30))

    # 8. General fallback spot
    else:
        num_spots = random.randint(6, 14)
        for _ in range(num_spots):
            sx = cx + random.randint(-int(width * 0.22), int(width * 0.22))
            sy = cy + random.randint(-int(height * 0.25), int(height * 0.25))
            r = random.randint(6, 14)
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(75, 45, 25))

def generate_synthetic_leaf(
    class_name: str,
    size: Tuple[int, int] = (224, 224),
    background_type: str = "field"
) -> Image.Image:
    """
    Generate a realistic synthetic leaf image for the specified disease class.
    
    Args:
        class_name: PlantVillage class key, e.g. "Tomato___Early_blight"
        size: (width, height)
        background_type: "field" (soil/vegetation texture) or "lab" (neutral gray)
    """
    width, height = size
    
    # 1. Background generation
    if background_type == "lab":
        bg_val = random.randint(55, 75)
        img = Image.new("RGB", size, (bg_val, bg_val, bg_val))
    else:
        # Realistic outdoor field soil / gravel background
        soil_arr = np.random.randint(65, 95, (height, width, 3), dtype=np.uint8)
        soil_arr[:, :, 0] += 15  # Slight brownish tint
        soil_arr[:, :, 2] = np.clip(soil_arr[:, :, 2] - 15, 0, 255)
        img = Image.fromarray(soil_arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))

    draw = ImageDraw.Draw(img)
    crop = class_name.split("___")[0]
    
    # 2. Draw leaf silhouette & veins
    _draw_leaf_base(draw, width, height, crop)
    
    # 3. Inject disease symptoms
    _inject_disease_symptoms(img, class_name)
    
    # 4. Subtle post-process smoothing for natural texture integration
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img

def create_synthetic_dataset(
    output_dir: Union[str, Path],
    classes: Optional[List[str]] = None,
    samples_per_class: int = 15,
    img_size: Tuple[int, int] = (224, 224)
) -> Dict[str, int]:
    """
    Generate a full directory structure of synthetic images for testing and rapid training.
    """
    output_path = Path(output_dir)
    classes = classes or CLASS_NAMES
    counts = {}
    
    for cls in classes:
        cls_dir = output_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = generate_synthetic_leaf(cls, size=img_size, background_type="field")
            img_file = cls_dir / f"sample_{i+1:03d}.jpg"
            img.save(img_file, quality=95)
        counts[cls] = samples_per_class
        
    return counts
