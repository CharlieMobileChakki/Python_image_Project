"""Normalizer module: Auto-angle detection, straightening, and 1024x1024 transparent canvas normalization."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def detect_rotation_angle(cutout_rgba: Image.Image) -> float:
    """Detect tilt angle of the garment using moments of the primary garment silhouette.

    Calculates the exact orientation of the vertical symmetry spine:
    - Isolates the main garment body to prevent stands, hangers, or props from distorting angle.
    - Uses central moments to find the principal axis of inertia.
    - Determines vertical tilt relative to the upright Y axis.

    Returns:
        Angle in degrees to rotate the garment to make it upright (-45.0 to 45.0).
        0.0 if garment is already straight or angle is negligible.
    """
    alpha = np.array(cutout_rgba.split()[3])
    # Threshold alpha to get clean binary silhouette
    binary = (alpha > 25).astype(np.uint8)
    if binary.sum() < 500:
        return 0.0

    # Isolate primary garment component to ignore stands or accessories
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return 0.0

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_contour_mask = (labels == largest_idx).astype(np.uint8)
    if stats[largest_idx, cv2.CC_STAT_AREA] < 500:
        return 0.0

    # Check horizontal aspect ratio: if image was taken sideways (w > 1.3 * h)
    w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    if w > 1.35 * h:
        return 90.0

    # Calculate image moments of the main garment body
    M = cv2.moments(main_contour_mask)
    if M["m00"] < 100:
        return 0.0

    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]

    # Principal inertia axis orientation (radians)
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = float(np.degrees(theta))

    # A vertical garment has principal axis along Y (+/- 90 degrees)
    if angle_deg > 0:
        tilt = 90.0 - angle_deg
    else:
        tilt = -90.0 - angle_deg

    # If already straight within 0.6 degrees, return 0.0
    if abs(tilt) < 0.6:
        return 0.0

    # Limit standard deskew to reasonable +/- 45 degrees
    if abs(tilt) > 45.0:
        return 0.0

    return round(float(tilt), 2)


def straighten_garment(cutout_rgba: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate garment to align vertically upright without clipping or distortion."""
    if abs(angle_deg) < 0.5:
        return cutout_rgba

    # PIL rotate with expand=True preserves all fabric pixels without clipping
    # Negative angle rotates back to vertical upright
    rotated = cutout_rgba.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=True)

    # Clean zero-alpha RGB after rotation to eliminate background color smearing
    arr = np.array(rotated)
    alpha = arr[:, :, 3]
    arr[:, :, :3] = np.where(alpha[:, :, None] == 0, 0, arr[:, :, :3])
    return Image.fromarray(arr, mode="RGBA")


def normalize_to_1024(cutout_rgba: Image.Image, target_size: int = 1024, fill_ratio: float = 0.82) -> Image.Image:
    """Tight-crop the garment and center it on a pristine 1024x1024 transparent square canvas.

    Enforces:
    - Zero background RGB leakage (transparent pixels have RGB 0, 0, 0).
    - Elimination of faint alpha noise, ghost boxes, and boundary lines.
    - Proportional scaling with LANCZOS resampling.
    """
    arr = np.array(cutout_rgba).copy()
    alpha = arr[:, :, 3]

    # 1. Eliminate faint alpha noise and ghost box artifacts (< 20 is background noise)
    alpha[alpha < 20] = 0

    # 2. Strict RGB decontamination: zero out RGB where alpha is 0
    arr[:, :, :3] = np.where(alpha[:, :, None] == 0, 0, arr[:, :, :3])
    arr[:, :, 3] = alpha

    # 2B. Strip white studio bounding box frame lines near the outer boundary
    is_white = (alpha > 0) & (arr[:, :, 0] > 225) & (arr[:, :, 1] > 225) & (arr[:, :, 2] > 225)
    if is_white.sum() > 0:
        ys_nz, xs_nz = np.where(alpha > 0)
        if len(xs_nz) > 0:
            xmin, xmax = xs_nz.min(), xs_nz.max()
            outer_white = is_white & (
                (np.abs(np.arange(arr.shape[1])[None, :] - xmin) < 8) |
                (np.abs(np.arange(arr.shape[1])[None, :] - xmax) < 8)
            )
            alpha[outer_white] = 0
            arr[:, :, :3] = np.where(alpha[:, :, None] == 0, 0, arr[:, :, :3])
            arr[:, :, 3] = alpha

    # 3. Component filtering: retain main garment body & connected elements
    binary = (alpha > 0).astype(np.uint8)
    if binary.sum() == 0:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels > 2:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        main_mask = (labels == largest_idx)
        # Dilate main mask by strict 2px to retain adjacent sleeves/drapes while dropping detached props
        dilated_main = cv2.dilate(main_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        keep_mask = np.zeros_like(alpha, dtype=bool)
        for i in range(1, num_labels):
            # Keep if component touches dilated main garment
            if (i == largest_idx) or (labels == i)[dilated_main].any():
                keep_mask |= (labels == i)
        alpha = np.where(keep_mask, alpha, 0).astype(np.uint8)
        arr[:, :, :3] = np.where(alpha[:, :, None] == 0, 0, arr[:, :, :3])
        arr[:, :, 3] = alpha

    # 3B. Smooth outer boundary teeth without dilating outward
    opened = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    smoothed = cv2.GaussianBlur(opened, (5, 5), 0.8)
    alpha = np.where((opened > 0) & (smoothed >= 18), smoothed, 0).astype(np.uint8)
    arr[:, :, :3] = np.where(alpha[:, :, None] == 0, 0, arr[:, :, :3])
    arr[:, :, 3] = alpha

    clean_rgba = Image.fromarray(arr, mode="RGBA")

    # 4. Tight-crop to clean garment silhouette
    thresh_alpha = Image.fromarray((alpha > 15).astype(np.uint8) * 255, mode="L")
    bbox = thresh_alpha.getbbox()
    if bbox is None:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    cropped = clean_rgba.crop(bbox)
    crop_w, crop_h = cropped.size

    # 5. Scale proportionally so longest dimension fits target_size * fill_ratio
    max_dimension = max(crop_w, crop_h)
    target_dimension = max(1, round(target_size * fill_ratio))
    scale = target_dimension / max_dimension

    new_w = max(1, round(crop_w * scale))
    new_h = max(1, round(crop_h * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6. Post-resize anti-ringing & alpha clamping
    res_arr = np.array(resized)
    res_alpha = res_arr[:, :, 3]
    res_alpha[res_alpha < 15] = 0
    res_arr[:, :, :3] = np.where(res_alpha[:, :, None] == 0, 0, res_arr[:, :, :3])
    res_arr[:, :, 3] = res_alpha
    final_resized = Image.fromarray(res_arr, mode="RGBA")

    # 7. Center on transparent square canvas
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(final_resized, (paste_x, paste_y), mask=final_resized.split()[3])

    return canvas
