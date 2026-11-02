"""Garment extraction subpackage: Outfit classification, mask fusion, and multi-piece separation."""
from app.core.garment_extraction.coverage_rules import (
    GARMENT_PRIORITY_THRESHOLDS,
    MALE_GARMENT_THRESHOLDS,
)
from app.core.garment_extraction.extraction_utils import get_garment_bounding_box
from app.core.garment_extraction.garment_aliases import (
    ATR_CLASSES,
    GARMENT_ALIASES,
    GARMENT_CLASSES,
)
from app.core.garment_extraction.garment_classifier import (
    analyze_semantic_map,
    check_garment_physical_continuity,
    classify_outfit_structure,
)
from app.core.garment_extraction.garment_masks import (
    extract_garment_cutout_flow_b,
    extract_garment_cutout_guided,
)
from app.core.garment_extraction.garment_splitter import (
    detect_multi_garment_layout,
)

__all__ = [
    "ATR_CLASSES",
    "GARMENT_ALIASES",
    "GARMENT_CLASSES",
    "GARMENT_PRIORITY_THRESHOLDS",
    "MALE_GARMENT_THRESHOLDS",
    "analyze_semantic_map",
    "check_garment_physical_continuity",
    "classify_outfit_structure",
    "detect_multi_garment_layout",
    "extract_garment_cutout_flow_b",
    "extract_garment_cutout_guided",
    "get_garment_bounding_box",
]

