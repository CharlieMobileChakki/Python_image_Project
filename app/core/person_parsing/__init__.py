"""Person parsing subpackage: SegFormer forward pass and anatomical region segmenters."""
from app.core.person_parsing.body_parts import (
    BODY_PART_CLASSES,
    NON_GARMENT_BODY_CLASSES,
)
from app.core.person_parsing.hair_detection import extract_hair_mask
from app.core.person_parsing.hand_detection import extract_arms_mask
from app.core.person_parsing.parsing_utils import get_class_pixel_counts
from app.core.person_parsing.segformer_parser import run_segformer_inference
from app.core.person_parsing.skin_region import extract_exposed_skin_mask

__all__ = [
    "BODY_PART_CLASSES",
    "NON_GARMENT_BODY_CLASSES",
    "extract_arms_mask",
    "extract_exposed_skin_mask",
    "extract_hair_mask",
    "get_class_pixel_counts",
    "run_segformer_inference",
]

