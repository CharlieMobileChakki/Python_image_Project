"""Compatibility shim for validator module. Re-exports from background_removal package."""
from __future__ import annotations

from app.core.background_removal import (
    estimate_background_color,
    refine_garment_halo_non_destructive,
    validate_garment_background,
)

__all__ = [
    "estimate_background_color",
    "validate_garment_background",
    "refine_garment_halo_non_destructive",
]
