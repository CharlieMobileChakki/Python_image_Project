"""Intelligent bust-up portrait cropper for AI Avatar Generation.

Extracts a perfectly framed bust-up portrait (hair, head, neck, shoulders, upper chest)
using the YuNet neural network face detector with robust portrait fallback.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageStat

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
YUNET_ONNX_PATH = ASSETS_DIR / "face_detection_yunet_2023mar.onnx"


def detect_face_bbox_yunet(img_rgb: Image.Image) -> tuple[int, int, int, int] | None:
    """Detect primary face box ``(x1, y1, x2, y2)`` using OpenCV YuNet ONNX model."""
    w, h = img_rgb.size
    try:
        import cv2

        if YUNET_ONNX_PATH.exists() and hasattr(cv2, "FaceDetectorYN_create"):
            detector = cv2.FaceDetectorYN_create(
                str(YUNET_ONNX_PATH), "", (w, h), 0.50, 0.30, 5000
            )
            bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            _, faces = detector.detect(bgr)
            if faces is not None and len(faces) > 0:
                best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
                x, y, fw, fh = (float(v) for v in best[:4])
                return (
                    max(0, int(x)),
                    max(0, int(y)),
                    min(w, int(x + fw)),
                    min(h, int(y + fh)),
                )
    except Exception as exc:
        logger.debug("YuNet face detection skipped: %s", exc)
    return None


def crop_bust_up_portrait(
    avatar_rgb: Image.Image,
    face_box_2d: list[int] | None = None,
    target_size: int = 1024,
) -> Image.Image:
    """Extract a balanced, centered bust-up portrait suitable for avatar stylization.
    
    Includes hair volume, full head, neck, shoulders, and upper chest/garment.
    Ensures the character is ideally positioned to be framed against a circular halo badge.
    """
    w, h = avatar_rgb.size
    if w < 64 or h < 64:
        return avatar_rgb

    fx1, fy1, fx2, fy2 = (0, 0, 0, 0)
    has_face = False

    # 1. Check Vision face_box_2d [ymin, xmin, ymax, xmax] (0..1000 scale)
    if face_box_2d and len(face_box_2d) == 4:
        ymin, xmin, ymax, xmax = face_box_2d
        fx1 = max(0, int(xmin * w / 1000.0))
        fy1 = max(0, int(ymin * h / 1000.0))
        fx2 = min(w, int(xmax * w / 1000.0))
        fy2 = min(h, int(ymax * h / 1000.0))
        if fx2 - fx1 >= 32 and fy2 - fy1 >= 32:
            has_face = True

    # 2. Check YuNet neural face detector
    if not has_face:
        yunet_box = detect_face_bbox_yunet(avatar_rgb)
        if yunet_box is not None:
            fx1, fy1, fx2, fy2 = yunet_box
            if fx2 - fx1 >= 32 and fy2 - fy1 >= 32:
                has_face = True

    # 3. Fallback: Upper-center portrait crop
    if not has_face:
        fx1 = int(w * 0.25)
        fy1 = int(h * 0.10)
        fx2 = int(w * 0.75)
        fy2 = int(h * 0.55)

    fw = max(1.0, float(fx2 - fx1))
    fh = max(1.0, float(fy2 - fy1))
    cx = (fx1 + fx2) / 2.0
    cy = (fy1 + fy2) / 2.0

    # Expand to bust-up framing:
    # - Top: include full hair volume + comfortable headroom (0.65 * fh above face top)
    # - Bottom: include neck, collar, shoulders, and upper chest (1.80 * fh below face bottom)
    # - Sides: include shoulders and hair sides (1.40 * fw on each side of face center)
    crop_top = max(0.0, fy1 - 0.65 * fh)
    crop_bottom = min(float(h), fy2 + 1.80 * fh)
    crop_height = crop_bottom - crop_top

    # Make width roughly proportional for bust-up avatar (near square 1:1 or 4:5)
    desired_width = crop_height * 0.95
    crop_left = max(0.0, cx - desired_width / 2.0)
    crop_right = min(float(w), cx + desired_width / 2.0)

    # Adjust if hitting image borders to keep subject centered
    actual_w = crop_right - crop_left
    if actual_w < desired_width and crop_left <= 0.0:
        crop_right = min(float(w), crop_left + desired_width)
    elif actual_w < desired_width and crop_right >= float(w):
        crop_left = max(0.0, crop_right - desired_width)

    ix1 = int(round(crop_left))
    iy1 = int(round(crop_top))
    ix2 = int(round(crop_right))
    iy2 = int(round(crop_bottom))

    if ix2 - ix1 < 48 or iy2 - iy1 < 48:
        cropped = avatar_rgb
    else:
        cropped = avatar_rgb.crop((ix1, iy1, ix2, iy2))

    # Gentle lighting lift for underexposed / dark images
    try:
        luma = float(ImageStat.Stat(cropped.convert("L")).mean[0])
        if 15.0 < luma < 80.0:
            lift = min(1.22, 92.0 / luma)
            cropped = ImageEnhance.Brightness(cropped).enhance(lift)
        if min(cropped.size) < 500:
            cropped = ImageEnhance.Sharpness(cropped).enhance(1.15)
    except Exception:
        pass

    # Resize cleanly to target dimension
    cropped_resized = cropped.copy()
    cropped_resized.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    return cropped_resized
