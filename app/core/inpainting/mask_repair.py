"""Arm, hand, hair, and prop occlusion mask detection for garment inpainting."""
from __future__ import annotations

import cv2
import numpy as np


def compute_arm_occlusion_mask(
    pred_map: np.ndarray,
    garment_class_id: int | list[int] | tuple[int, ...],
    cutout_alpha: np.ndarray | None = None,
    arm_classes: tuple[int, ...] = (14, 15),
    dilation_radius: int = 11,
) -> np.ndarray:
    """Detect regions where model's arms, hands, hair, or handheld props occlude garment fabric.

    Reconstructs:
    - Fabric hidden by crossed arms, resting hands, or fingers on the torso/waist/hemline.
    - Notches, voids, and missing fabric occluded by handheld props (microphones, phones, bags).
    - Fabric covered by long hair falling across the chest or shoulders.

    Strictly protects:
    - Natural necklines (top open collar/neckline).
    - Outer background (never spills into empty background beyond the garment silhouette).
    """
    if isinstance(garment_class_id, (list, tuple)):
        garment_mask = np.isin(pred_map, garment_class_id)
    else:
        garment_mask = (pred_map == garment_class_id)

    # Use cutout alpha if available as the most accurate depiction of extracted fabric
    if cutout_alpha is not None:
        effective_garment = (cutout_alpha > 20)
    else:
        effective_garment = garment_mask

    if effective_garment.sum() < 60:
        return np.zeros_like(pred_map, dtype=np.uint8)

    H, W = pred_map.shape
    garment_rows = np.any(effective_garment, axis=1)
    if not np.any(garment_rows):
        return np.zeros_like(pred_map, dtype=np.uint8)

    y_indices = np.where(garment_rows)[0]
    y_min, y_max = y_indices[0], y_indices[-1]
    garment_height = max(1, y_max - y_min)

    # Protect natural neckline (top 8% where collar/neck/face resides)
    y_top_limit = y_min + int(0.08 * garment_height)

    # Occluding body and prop classes in SegFormer
    # arms: 14, 15; hair: 2; accessories/bag/belt: 1, 3, 8, 16
    arms_mask = np.isin(pred_map, arm_classes)
    hair_mask = (pred_map == 2)
    acc_mask = np.isin(pred_map, [1, 3, 8, 16])
    bg_mask = (pred_map == 0)

    # 1. Natural garment envelope / convex torso hull
    contours, _ = cv2.findContours(
        effective_garment.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(pred_map, dtype=np.uint8)

    # Build silhouette hull
    hull_mask = np.zeros((H, W), dtype=np.uint8)
    for c in contours:
        if cv2.contourArea(c) > 200:
            h = cv2.convexHull(c)
            cv2.drawContours(hull_mask, [h], -1, 255, -1)

    # 2. Identify crossed arms / hands across the garment torso
    crossed_limbs = np.zeros_like(effective_garment, dtype=bool)
    if arms_mask.sum() > 0:
        for y in range(y_top_limit, min(H, y_max + 1)):
            row_g = np.where(effective_garment[y])[0]
            row_a = np.where(arms_mask[y])[0]
            if len(row_a) == 0:
                continue
            if len(row_g) > 0:
                g_left, g_right = row_g[0], row_g[-1]
                # Limb inside garment horizontal span or within 12px of edge
                in_span = (row_a >= max(0, g_left - 12)) & (row_a <= min(W - 1, g_right + 12))
                crossed_limbs[y, row_a[in_span]] = True
            elif hull_mask[y].any():
                h_left, h_right = np.where(hull_mask[y] == 255)[0][[0, -1]]
                in_hull = (row_a >= h_left) & (row_a <= h_right)
                crossed_limbs[y, row_a[in_hull]] = True

    # 3. Identify missing notches / voids within the natural garment envelope
    hull_void = (hull_mask == 255) & (~effective_garment)
    # Restrict to areas touched by crossed limbs, arms, accessories, or hair
    occlusion_sources = crossed_limbs | arms_mask | acc_mask | (hair_mask & (hull_mask == 255))
    dilated_sources = cv2.dilate(occlusion_sources.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    enclosed_occlusions = hull_void & dilated_sources

    # Union of crossed limbs and enclosed occlusions
    repair_candidate = (crossed_limbs & (hull_mask == 255)) | enclosed_occlusions

    # Exclude natural neckline at the top
    repair_candidate[:y_top_limit, :] = False

    # Filter out tiny isolated noise (< 25px)
    if repair_candidate.sum() < 30:
        return np.zeros_like(pred_map, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(repair_candidate.astype(np.uint8))
    filtered_repair = np.zeros_like(repair_candidate, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 25:
            filtered_repair |= (labels == i)

    if not filtered_repair.any():
        return np.zeros_like(pred_map, dtype=np.uint8)

    # 4. Smooth seam dilation to cover boundary contact seams and shadows
    seam_rad = max(5, min(dilation_radius, 13))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seam_rad, seam_rad))
    dilated_repair = cv2.dilate(filtered_repair.astype(np.uint8) * 255, kernel)

    # Restrict strictly inside the garment envelope (never spill into outer background)
    dilated_repair[hull_mask == 0] = 0
    dilated_repair[:y_top_limit, :] = 0

    return dilated_repair
