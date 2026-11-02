"""Utility functions for semantic parsing map processing."""
from __future__ import annotations

import numpy as np


def get_class_pixel_counts(pred_map: np.ndarray) -> dict[int, int]:
    """Return dictionary mapping class IDs to total pixel counts."""
    unique, counts = np.unique(pred_map, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))

