"""Utilities for human body part metrics and pixel counting."""
from __future__ import annotations

import numpy as np

# SegFormer human body classes:
# 1: hat, 2: hair, 3: sunglasses, 9: left-shoe, 10: right-shoe, 11: face, 12: left-leg, 13: right-leg, 14: left-arm, 15: right-arm
HUMAN_BODY_CLASSES = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15]
HEAD_FACE_CLASSES = [11]
HAIR_CLASSES = [2]
ARM_CLASSES = [14, 15]
LEG_CLASSES = [12, 13]


def compute_human_pixel_metrics(pred_map: np.ndarray) -> dict[str, float]:
    """Calculate pixel counts and percentages for all human body segments."""
    total_pixels = max(1, pred_map.size)
    unique, counts = np.unique(pred_map, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))

    face_pixels = class_counts.get(11, 0)
    hair_pixels = class_counts.get(2, 0)
    arm_pixels = class_counts.get(14, 0) + class_counts.get(15, 0)
    leg_pixels = class_counts.get(12, 0) + class_counts.get(13, 0)
    human_pixels = sum(class_counts.get(cid, 0) for cid in HUMAN_BODY_CLASSES)

    return {
        "total_pixels": float(total_pixels),
        "human_pixels": float(human_pixels),
        "human_pct": (human_pixels / total_pixels) * 100.0,
        "face_pixels": float(face_pixels),
        "face_pct": (face_pixels / total_pixels) * 100.0,
        "hair_pixels": float(hair_pixels),
        "hair_pct": (hair_pixels / total_pixels) * 100.0,
        "arm_pixels": float(arm_pixels),
        "arm_pct": (arm_pixels / total_pixels) * 100.0,
        "leg_pixels": float(leg_pixels),
        "leg_pct": (leg_pixels / total_pixels) * 100.0,
    }

