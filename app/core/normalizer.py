"""Compatibility shim for normalizer module. Re-exports from image_processing package."""
from __future__ import annotations

from app.core.image_processing import (
    detect_rotation_angle,
    normalize_to_1024,
    straighten_garment,
)

__all__ = [
    "detect_rotation_angle",
    "straighten_garment",
    "normalize_to_1024",
]
