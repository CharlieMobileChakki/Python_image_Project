"""Image processing package: Loading, color calibration, validation, and encoding."""
from __future__ import annotations

from app.core.image_processing.face_analyzer import (
    compute_face_similarity,
    detect_face_and_landmarks,
    extract_face_embedding,
    refine_face_identity,
)
from app.core.image_processing.image_loader import load_image_bytes
from app.core.image_processing.image_utils import image_to_base64_data_uri
from app.core.image_processing.studio_composer import (
    calibrate_product_colors,
    validate_product_output,
)

__all__ = [
    "load_image_bytes",
    "image_to_base64_data_uri",
    "calibrate_product_colors",
    "validate_product_output",
    "detect_face_and_landmarks",
    "extract_face_embedding",
    "compute_face_similarity",
    "refine_face_identity",
]

