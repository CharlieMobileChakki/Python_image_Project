"""Background removal subpackage: BiRefNet Lite salient extraction and quality validation."""
from app.core.background_removal.birefnet_removal import extract_isolated_flow_a
from app.core.background_removal.postprocessing import (
    refine_garment_halo_non_destructive,
    validate_garment_background,
)
from app.core.background_removal.preprocessing import prepare_image_for_background_removal
from app.core.background_removal.utils import estimate_background_color

__all__ = [
    "estimate_background_color",
    "extract_isolated_flow_a",
    "prepare_image_for_background_removal",
    "refine_garment_halo_non_destructive",
    "validate_garment_background",
]

