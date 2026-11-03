"""Arm and hand occlusion mask detection for human model inpainting."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter


def compute_arm_occlusion_mask(
    pred_map: np.ndarray,
    garment_class_id: int | list[int] | tuple[int, ...],
    cutout_alpha: np.ndarray | None = None,
    arm_classes: tuple[int, ...] = (14, 15),
    dilation_radius: int = 17,
) -> np.ndarray:
    """Detect regions where model's arms cut into or cross over garment fabric.

    Strictly triggers ONLY for genuine human occlusions:
    - Reconstructs fabric hidden by hands/arms crossing over the torso/garment body.
    - Excludes arms resting outside on walls or hanging down in the background.
    - Excludes background regions (Class 0), natural necklines, and bottom hemlines.
    """
    if isinstance(garment_class_id, (list, tuple)):
        garment_mask = np.isin(pred_map, garment_class_id)
    else:
        garment_mask = (pred_map == garment_class_id)

    arms_mask = np.isin(pred_map, arm_classes)
    bg_mask = (pred_map == 0)

    if garment_mask.sum() == 0 or (arms_mask.sum() < 100 and cutout_alpha is None):
        return np.zeros_like(garment_mask, dtype=np.uint8)

    H, W = pred_map.shape
    garment_rows = np.any(garment_mask, axis=1)
    if not np.any(garment_rows):
        return np.zeros_like(garment_mask, dtype=np.uint8)

    y_indices = np.where(garment_rows)[0]
    y_min, y_max = y_indices[0], y_indices[-1]
    garment_height = max(1, y_max - y_min)

    # Protect natural neckline (top 10%) and natural hemline (bottom 5%)
    y_top_limit = y_min + int(0.10 * garment_height)
    y_bot_limit = y_max - int(0.05 * garment_height)

    # 1. Arm regions genuinely crossing the garment torso/body
    crossed_arms = np.zeros_like(garment_mask, dtype=bool)
    if arms_mask.sum() > 0:
        for y in range(y_top_limit, y_bot_limit):
            row_g = np.where(garment_mask[y])[0]
            row_a = np.where(arms_mask[y])[0]
            if len(row_g) < 25 or len(row_a) == 0:
                continue
            g_left, g_right = row_g[0], row_g[-1]
            # Arm pixel must be embedded horizontally inside the garment silhouette
            in_torso = (row_a >= g_left) & (row_a <= g_right)
            valid_x = row_a[in_torso]
            if len(valid_x) > 0:
                crossed_arms[y, valid_x] = True

        # Filter out arm components that are purely external limbs in the background
        if crossed_arms.any():
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crossed_arms.astype(np.uint8))
            kernel = np.ones((7, 7), np.uint8)
            filtered_arms = np.zeros_like(crossed_arms, dtype=bool)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 40:
                    continue
                comp = (labels == i)
                dilated = cv2.dilate(comp.astype(np.uint8), kernel) > 0
                ring = dilated & ~comp
                bg_touch = (ring & bg_mask).sum()
                g_touch = (ring & garment_mask).sum()
                total_ring = max(1, ring.sum())
                # Keep if embedded in torso OR substantially resting against/overlapping garment
                if (bg_touch / total_ring) < 0.45 or g_touch > 40:
                    filtered_arms |= comp
            crossed_arms = filtered_arms

    # If there are no confirmed human arm/hand pixels crossing the garment, do not repair!
    # Prevents artificial holes being carved into plackets, necklines, or embroidery on flat-lays or mannequins.
    if not crossed_arms.any():
        return np.zeros_like(garment_mask, dtype=np.uint8)

    # 2. Internal transparent holes enclosed within the garment silhouette that touch crossed arms
    internal_holes = np.zeros_like(garment_mask, dtype=bool)
    if cutout_alpha is not None:
        contours, _ = cv2.findContours(
            (cutout_alpha > 20).astype(np.uint8) * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        filled_silhouette = np.zeros_like(cutout_alpha)
        for c in contours:
            cv2.drawContours(filled_silhouette, [c], -1, 255, -1)
        # Internal void: strictly enclosed, NOT background, NOT in neckline or hemline
        candidate_holes = (filled_silhouette == 255) & (cutout_alpha < 30) & (~bg_mask)
        candidate_holes[:y_top_limit, :] = False
        candidate_holes[y_bot_limit:, :] = False
        # Must touch or be near crossed arm pixels
        dilated_arms = cv2.dilate(crossed_arms.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        internal_holes = candidate_holes & dilated_arms

    # Union of genuine crossed arms and enclosed arm holes
    repair_mask = crossed_arms | internal_holes
    if repair_mask.sum() < 60:
        return np.zeros_like(garment_mask, dtype=np.uint8)

    # Dilate slightly so LaMa seamlessly blends the boundary into surrounding intact fabric
    mask_im = Image.fromarray((repair_mask.astype(np.uint8) * 255), mode="L")
    dilated_mask = np.array(mask_im.filter(ImageFilter.MaxFilter(dilation_radius)))
    return dilated_mask

