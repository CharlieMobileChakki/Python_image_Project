"""Hand and arm region mask extraction."""
from __future__ import annotations

import numpy as np


def extract_arms_mask(pred_map: np.ndarray) -> np.ndarray:
    """Return binary mask of arm/hand pixels (classes 14 and 15)."""
    return np.isin(pred_map, [14, 15])

