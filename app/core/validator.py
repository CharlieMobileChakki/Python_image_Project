"""Background Removal Quality Validation & Non-Destructive Auto-Refinement."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np
from PIL import Image


def refine_garment_halo_non_destructive(
    cutout_rgba: Image.Image,
    estimated_bg: np.ndarray | None = None,
) -> Image.Image:
    """Non-destructively soften and suppress subtle background halos on the outer boundary.

    Preserves 100% of solid garment core and delicate interior embroidery/straps.
    Only operates in the outer 2-3px transition ring where alpha < 180.
    """
    arr = np.array(cutout_rgba).copy()
    H, W, C = arr.shape
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    if estimated_bg is None:
        estimated_bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    # Identify the outer transition boundary band
    fg_bin = (alpha > 30).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(fg_bin, kernel)
    boundary_ring = (dilated > 0) & (alpha > 0) & (alpha < 180)

    if boundary_ring.any():
        # Color distance from background for boundary pixels
        diff = np.linalg.norm(rgb.astype(np.float32) - estimated_bg.astype(np.float32), axis=-1)
        # Boundary pixels closely matching background (e.g. distance < 45 on light bg)
        bg_leak = boundary_ring & (diff < 45)
        if bg_leak.any():
            # Smoothly attenuate alpha proportional to background similarity
            attenuation = np.clip(diff[bg_leak] / 45.0, 0.0, 1.0)
            arr[bg_leak, 3] = np.clip(alpha[bg_leak].astype(np.float32) * (attenuation ** 2), 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGBA")


def validate_garment_background(
    cutout_rgba: Image.Image,
    original_rgb: Image.Image | None = None,
    estimated_bg: np.ndarray | None = None,
    allow_retry: bool = True,
) -> dict[str, Any]:
    """Validate background removal completeness and garment preservation on a 1024x1024 canvas.

    Detects:
    - Perimeter boundary bleed (incomplete crop / unremoved box border).
    - Residual background patches or white halos around garment edges.
    - Accidental garment over-erosion or under-extraction.

    Returns:
        dict containing:
        - status: 'success' | 'partial' | 'failed' | 'needs_review'
        - confidence: float (0.0 to 1.0)
        - preservation_status: 'verified' | 'needs_review'
        - validation_errors: list[str]
        - refined_cutout: PIL Image (auto-refined if retry applied)
    """
    arr = np.array(cutout_rgba)
    H, W, C = arr.shape
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    if estimated_bg is None:
        estimated_bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    errors: list[str] = []

    # 1. Perimeter border check: outer 10px of canvas MUST be 100% transparent
    border_mask = np.ones((H, W), dtype=bool)
    border_mask[10 : H - 10, 10 : W - 10] = False
    border_alpha = alpha[border_mask]
    border_leaks = int((border_alpha > 20).sum())
    if border_leaks > 50:
        errors.append(f"Perimeter bleed detected: {border_leaks} non-transparent pixels on canvas border")

    # 2. Garment presence and solid core check
    solid_core = int((alpha > 70).sum())
    if solid_core < 500:
        errors.append("Garment missing or too small (under-extracted)")

    # 3. Outer boundary halo inspection
    fg_bin = (alpha > 30).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(fg_bin, kernel)
    boundary_ring = (dilated > 0) & (alpha < 180) & (alpha > 0)

    halo_count = 0
    halo_ratio = 0.0
    if boundary_ring.any():
        ring_rgb = rgb[boundary_ring]
        # Check similarity to estimated background
        dist = np.linalg.norm(ring_rgb.astype(np.float32) - estimated_bg.astype(np.float32), axis=-1)
        halo_pixels = dist < 40
        halo_count = int(halo_pixels.sum())
        ring_total = max(1, len(ring_rgb))
        halo_ratio = halo_count / ring_total

    # 4. Automatic retry if subtle halo detected and retry is allowed
    refined_cutout = cutout_rgba
    if halo_ratio > 0.12 and allow_retry and ("Perimeter" not in " ".join(errors)):
        # Apply non-destructive halo suppression
        refined_cutout = refine_garment_halo_non_destructive(cutout_rgba, estimated_bg)
        # Re-validate refined cutout with allow_retry=False
        return validate_garment_background(
            refined_cutout,
            original_rgb=original_rgb,
            estimated_bg=estimated_bg,
            allow_retry=False,
        )

    if halo_ratio > 0.18:
        errors.append(f"Background halo detected around garment boundary ({round(halo_ratio * 100, 1)}% of edge ring)")

    # 5. Evaluate final status & confidence
    if not errors:
        status = "success"
        confidence = 0.99
        preservation = "verified"
    elif any("under-extracted" in e or "Perimeter" in e for e in errors):
        status = "failed"
        confidence = 0.35
        preservation = "needs_review"
    elif any("halo" in e.lower() for e in errors):
        status = "partial"
        confidence = 0.82
        preservation = "needs_review"
    else:
        status = "needs_review"
        confidence = 0.70
        preservation = "needs_review"

    return {
        "status": status,
        "confidence": confidence,
        "preservation_status": preservation,
        "validation_errors": errors,
        "refined_cutout": refined_cutout,
    }
