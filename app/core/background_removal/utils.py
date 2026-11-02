"""Background removal utility functions."""
from __future__ import annotations

import numpy as np
from PIL import Image


def estimate_background_color(image: Image.Image | np.ndarray) -> np.ndarray:
    """Estimate median background color from canvas outer border margins."""
    arr = np.array(image)
    H, W = arr.shape[:2]
    border_mask = np.ones((H, W), dtype=bool)
    border_mask[int(H * 0.04) : int(H * 0.96), int(W * 0.04) : int(W * 0.96)] = False
    border_pixels = arr[border_mask]
    if len(border_pixels) == 0:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return np.median(border_pixels[:, :3], axis=0).astype(np.float32)

