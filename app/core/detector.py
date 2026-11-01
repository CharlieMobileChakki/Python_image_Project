"""Detector module: Identifies human presence and classifies garment items."""
from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image

# ATR 18 Semantic Classes
ATR_CLASSES = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

# Body classes indicating human presence
HUMAN_BODY_CLASSES = {2, 9, 10, 11, 12, 13, 14, 15}

# Primary garment classes
GARMENT_CLASSES = {
    4: {"key": "upper_clothes", "label": "T-Shirt / Top / Shirt"},
    5: {"key": "skirt", "label": "Skirt / Lehenga"},
    6: {"key": "pants", "label": "Pants / Jeans / Trousers"},
    7: {"key": "dress", "label": "Dress / Gown / Anarkali"},
}


def analyze_semantic_map(pred_map: np.ndarray) -> dict[str, Any]:
    """Analyze SegFormer prediction map to determine human presence and garments.

    Args:
        pred_map: 2D numpy array of class IDs (0..17), matching image dimensions.

    Returns:
        dict containing has_human, human_pixel_count, detected_garments list.
    """
    total_pixels = pred_map.size
    unique_classes, counts = np.unique(pred_map, return_counts=True)
    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))

    # Calculate human body presence
    human_pixels = sum(class_counts.get(cid, 0) for cid in HUMAN_BODY_CLASSES)
    human_pct = (human_pixels / total_pixels) * 100.0

    # Human is present if face/arms/legs exceed threshold (min 0.35% of image or 800 px)
    has_human = (human_pixels > 800) and (human_pct > 0.35)

    # Detect garments
    detected_garments = []
    for cid, info in GARMENT_CLASSES.items():
        cnt = class_counts.get(cid, 0)
        pct = (cnt / total_pixels) * 100.0
        # Valid garment if it covers at least 0.8% of image and > 1500 pixels
        if cnt > 1500 and pct > 0.8:
            y_idxs, x_idxs = np.where(pred_map == cid)
            bbox = [int(x_idxs.min()), int(y_idxs.min()), int(x_idxs.max()), int(y_idxs.max())]
            detected_garments.append({
                "class_id": cid,
                "category": info["key"],
                "label": info["label"],
                "pixel_count": int(cnt),
                "coverage_pct": round(pct, 2),
                "bounding_box": bbox,
            })

    return {
        "has_human": has_human,
        "human_pixels": int(human_pixels),
        "human_coverage_pct": round(human_pct, 2),
        "detected_garments": detected_garments,
    }

