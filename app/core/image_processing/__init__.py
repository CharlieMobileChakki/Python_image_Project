"""Image processing package: Loading, normalizing, rotating, and converting."""
from __future__ import annotations

from app.core.image_processing.image_loader import load_image_bytes
from app.core.image_processing.image_normalizer import (
    detect_rotation_angle,
    normalize_to_1024,
    straighten_garment,
)
from app.core.image_processing.image_utils import image_to_base64_data_uri
from app.core.image_processing.studio_composer import (
    calibrate_product_colors,
    composite_protected_product_over_stand,
    create_mode_a_product_only,
    create_mode_b_studio_presentation,
    create_mode_c_display_stand,
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
    "create_mode_c_display_stand",
    "composite_protected_product_over_stand",
    "calibrate_product_colors",
    "validate_product_output",
]
