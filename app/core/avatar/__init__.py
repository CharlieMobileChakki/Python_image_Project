"""AI Stylized Avatar Profile Picture Generator package."""
from __future__ import annotations

from app.core.avatar.avatar_cropper import (
    crop_bust_up_portrait,
    detect_face_bbox_yunet,
)
from app.core.avatar.avatar_pipeline import process_avatar_pipeline
from app.core.avatar.avatar_stylist import (
    analyze_avatar_identity,
    build_avatar_stylized_prompt,
    resolve_halo_and_doodle,
    HALO_PALETTE,
    DOODLE_STYLES,
)

__all__ = [
    "DOODLE_STYLES",
    "HALO_PALETTE",
    "analyze_avatar_identity",
    "build_avatar_stylized_prompt",
    "crop_bust_up_portrait",
    "detect_face_bbox_yunet",
    "process_avatar_pipeline",
    "resolve_halo_and_doodle",
]
