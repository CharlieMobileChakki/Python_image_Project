"""Utilities for inpainting tensor preparation and mask padding."""
from __future__ import annotations

import numpy as np
from PIL import Image


def ensure_patch_dimension(image_rgb: Image.Image, target_size: int = 512) -> Image.Image:
    """Ensure RGB crop matches fixed target square resolution."""
    if image_rgb.size != (target_size, target_size):
        return image_rgb.resize((target_size, target_size), Image.Resampling.BICUBIC)
    return image_rgb


def ensure_mask_dimension(mask_bin: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Ensure binary mask matches fixed target square resolution."""
    if mask_bin.shape != (target_size, target_size):
        mask_im = Image.fromarray(mask_bin.astype(np.uint8), mode="L").resize(
            (target_size, target_size), Image.Resampling.NEAREST
        )
        return np.array(mask_im)
    return mask_bin

