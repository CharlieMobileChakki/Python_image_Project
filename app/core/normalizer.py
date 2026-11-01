"""Normalizer module: Auto-angle detection, straightening, and 1024x1024 transparent canvas normalization."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def detect_rotation_angle(cutout_rgba: Image.Image) -> float:
    """Detect tilt angle of the garment using minimum area bounding box on alpha mask.

    Returns:
        Angle in degrees to rotate the garment to make it upright (-45.0 to 45.0).
        0.0 if garment is already straight or angle is negligible.
    """
    alpha = np.array(cutout_rgba.split()[3])
    # Threshold alpha to get clean binary silhouette
    _, thresh = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Pick largest contour (primary garment body)
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 500:
        return 0.0

    # Minimum area rectangle around the garment
    rect = cv2.minAreaRect(main_contour)
    # rect = ((center_x, center_y), (width, height), angle)
    (cx, cy), (w, h), angle = rect

    # Normalize angle based on orientation
    if w < h:
        # Standard vertical garment (height > width)
        deskew_angle = angle
    else:
        # Horizontal aspect or rotated 90 deg
        deskew_angle = angle + 90.0 if angle < -45 else angle - 90.0

    # If minor tilt (less than 0.7 degrees), treat as already straight
    if abs(deskew_angle) < 0.7 or abs(deskew_angle) > 40.0:
        return 0.0

    return round(float(deskew_angle), 2)


def straighten_garment(cutout_rgba: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate garment to align vertically upright without clipping or distortion."""
    if abs(angle_deg) < 0.5:
        return cutout_rgba

    # PIL rotate with expand=True to preserve all fabric pixels without clipping
    # Negative angle to deskew back to upright
    return cutout_rgba.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=True)


def normalize_to_1024(cutout_rgba: Image.Image, target_size: int = 1024, fill_ratio: float = 0.82) -> Image.Image:
    """Tight-crop the alpha subject and center it on a 1024x1024 transparent square canvas.

    Args:
        cutout_rgba: Input RGBA image.
        target_size: Dimensions of the output square canvas (default 1024).
        fill_ratio: Percentage of canvas occupied by longest edge (default 0.82 = 840 px).

    Returns:
        1024x1024 transparent RGBA image.
    """
    alpha = cutout_rgba.split()[3]
    # Filter faint alpha noise when calculating tight bounding box
    thresholded_alpha = alpha.point(lambda v: 255 if v > 15 else 0)
    bbox = thresholded_alpha.getbbox()

    if bbox is None:
        # Empty alpha fallback: return blank 1024x1024 canvas
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    # Crop tight to garment silhouette
    cropped = cutout_rgba.crop(bbox)
    crop_w, crop_h = cropped.size

    # Scale proportionally so longest dimension fits target_size * fill_ratio
    max_dimension = max(crop_w, crop_h)
    target_dimension = max(1, round(target_size * fill_ratio))
    scale = target_dimension / max_dimension

    new_w = max(1, round(crop_w * scale))
    new_h = max(1, round(crop_h * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center on transparent square canvas
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y), mask=resized.split()[3])

    return canvas

