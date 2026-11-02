"""Human detection subpackage: Person presence verification and scene categorization."""
from app.core.human_detection.detection_utils import (
    ARM_CLASSES,
    HAIR_CLASSES,
    HEAD_FACE_CLASSES,
    HUMAN_BODY_CLASSES,
    LEG_CLASSES,
    compute_human_pixel_metrics,
)
from app.core.human_detection.human_detector import detect_human_presence

__all__ = [
    "ARM_CLASSES",
    "HAIR_CLASSES",
    "HEAD_FACE_CLASSES",
    "HUMAN_BODY_CLASSES",
    "LEG_CLASSES",
    "compute_human_pixel_metrics",
    "detect_human_presence",
]

