"""Image processing package: Loading, normalizing, rotating, and converting."""
from __future__ import annotations

from app.core.image_processing.artwork_preserver import (
    detect_garment_landmarks,
    ensure_garment_fabric_opacity,
    map_and_preserve_artwork,
)
from app.core.image_processing.image_loader import load_image_bytes
from app.core.image_processing.image_normalizer import (
    detect_rotation_angle,
    normalize_to_1024,
    straighten_garment,
)
from app.core.image_processing.image_utils import image_to_base64_data_uri
from app.core.image_processing.studio_composer import (
    calibrate_product_colors,
    create_mode_a_product_only,
    create_mode_b_studio_presentation,
    validate_product_output,
)

__all__ = [
    "load_image_bytes",
    "detect_rotation_angle",
    "straighten_garment",
    "normalize_to_1024",
    "image_to_base64_data_uri",
    "create_mode_a_product_only",
    "create_mode_b_studio_presentation",
    "calibrate_product_colors",
    "validate_product_output",
    "map_and_preserve_artwork",
    "ensure_garment_fabric_opacity",
    "detect_garment_landmarks",
]

