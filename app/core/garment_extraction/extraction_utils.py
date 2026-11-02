"""Extraction utility functions for garment bounding boxes and mask arithmetic."""
from __future__ import annotations

from typing import Any
import numpy as np


def get_garment_bounding_box(binary_mask: np.ndarray) -> list[int] | None:
    """Return [x_min, y_min, x_max, y_max] bounding box for a binary mask."""
    ys, xs = np.where(binary_mask)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

