"""Exposed skin and facial region mask extraction."""
from __future__ import annotations

import numpy as np


def extract_exposed_skin_mask(pred_map: np.ndarray) -> np.ndarray:
    """Return binary mask of exposed skin pixels (face: 11, arms: 14, 15, legs: 12, 13)."""
    return np.isin(pred_map, [11, 12, 13, 14, 15])

