"""Human detector: Differentiates living human models from flat-lay products and mannequins."""
from __future__ import annotations

import numpy as np

from app.core.human_detection.detection_utils import (
    HUMAN_BODY_CLASSES,
    compute_human_pixel_metrics,
)


def detect_human_presence(pred_map: np.ndarray) -> tuple[bool, str, dict[str, float]]:
    """Determine whether an image contains a real human model, flat-lay product, or mannequin.

    Returns:
        tuple(has_human, scene_type, metrics_dict)
        where scene_type in {"person", "flat_lay", "mannequin"}
    """
    metrics = compute_human_pixel_metrics(pred_map)
    face_pixels = metrics["face_pixels"]
    hair_pixels = metrics["hair_pixels"]
    arm_pixels = metrics["arm_pixels"]
    leg_pixels = metrics["leg_pixels"]
    human_pixels = metrics["human_pixels"]
    face_pct = metrics["face_pct"]
    hair_pct = metrics["hair_pct"]
    arm_pct = metrics["arm_pct"]
    leg_pct = metrics["leg_pct"]
    human_pct = metrics["human_pct"]

    # Distinguish Person vs Mannequin vs Flat-lay
    # A genuine human model has coherent facial features, substantial hair, or active arm/leg skin
    has_real_face = (face_pixels >= 250 and face_pct >= 0.25)
    has_face_and_hair = ((face_pixels + hair_pixels) >= 300 and (face_pct + hair_pct) >= 0.40)
    has_real_limbs = ((arm_pixels + leg_pixels) >= 350 and (arm_pct + leg_pct) >= 0.45)
    has_large_human = (human_pixels >= 1000 and human_pct >= 1.5)

    is_person = (has_real_face or has_face_and_hair or has_real_limbs or has_large_human)

    if is_person:
        scene_type = "person"
        has_human = True
    elif human_pixels == 0:
        # Pure flat lay: zero human body pixels
        scene_type = "flat_lay"
        has_human = False
    elif (face_pixels == 0 and hair_pixels == 0 and arm_pixels == 0 and leg_pixels == 0) and human_pixels <= 500:
        # Isolated mannequin stand, wooden base, or shoe display fixture
        scene_type = "mannequin"
        has_human = False
    elif human_pixels <= 500 or (face_pixels == 0 and hair_pixels < 300 and arm_pixels < 300 and leg_pixels == 0):
        # Product shot laid flat, table, hanger, or clean product photo
        scene_type = "flat_lay"
        has_human = False
    else:
        # Isolated mannequin stand, head-less torso form, or display fixture
        scene_type = "mannequin"
        has_human = False

    return has_human, scene_type, metrics
