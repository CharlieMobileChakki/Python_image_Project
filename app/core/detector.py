"""Compatibility shim for detector module. Re-exports from human_detection and garment_extraction packages."""
from __future__ import annotations

from app.core.human_detection import (
    HEAD_FACE_CLASSES,
    HUMAN_BODY_CLASSES,
    compute_human_pixel_metrics,
    detect_human_presence,
)
from app.core.garment_extraction import (
    ATR_CLASSES,
    GARMENT_ALIASES,
    GARMENT_CLASSES,
    GARMENT_PRIORITY_THRESHOLDS,
    MALE_GARMENT_THRESHOLDS,
    analyze_semantic_map,
    check_garment_physical_continuity,
    classify_outfit_structure,
    detect_multi_garment_layout,
)

__all__ = [
    "ATR_CLASSES",
    "GARMENT_CLASSES",
    "GARMENT_ALIASES",
    "HUMAN_BODY_CLASSES",
    "HEAD_FACE_CLASSES",
    "GARMENT_PRIORITY_THRESHOLDS",
    "MALE_GARMENT_THRESHOLDS",
    "compute_human_pixel_metrics",
    "detect_human_presence",
    "check_garment_physical_continuity",
    "analyze_semantic_map",
    "classify_outfit_structure",
    "detect_multi_garment_layout",
]
