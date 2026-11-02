"""Hair region mask extraction."""
from __future__ import annotations

import numpy as np


def extract_hair_mask(pred_map: np.ndarray) -> np.ndarray:
    """Return binary mask of hair pixels (class 2)."""
    return pred_map == 2

