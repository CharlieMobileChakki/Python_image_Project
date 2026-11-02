"""Multi-garment layout detection and spatial component splitting for catalog grids."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np
from PIL import Image


def detect_multi_garment_layout(
    image_rgb: Image.Image,
    fg_alpha: np.ndarray | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Detect if an image contains multiple independently arranged garments on flat-lay or catalog grid.

    Returns:
        (is_multi_garment, list of detected garment components with masks and bounding boxes)
    """
    arr = np.array(image_rgb)
    H, W, _ = arr.shape
    total_px = H * W

    # 1. Estimate background color from outer border margins (top, bottom, left, right 4%)
    border_mask = np.ones((H, W), dtype=bool)
    border_mask[int(H * 0.04) : int(H * 0.96), int(W * 0.04) : int(W * 0.96)] = False
    border_pixels = arr[border_mask]
    bg_median = np.median(border_pixels, axis=0)
    dist = np.linalg.norm(arr.astype(np.float32) - bg_median, axis=-1)

    # 2. Foreground color difference from background
    color_fg = (dist > 25).astype(np.uint8)

    # If BiRefNet alpha is available, combine it while ensuring background gaps remain zero
    if fg_alpha is not None:
        fg_mask = np.where((color_fg > 0) | (fg_alpha > 50), 1, 0).astype(np.uint8)
        # Suppress where background color is clearly present
        fg_mask[dist < 22] = 0
    else:
        fg_mask = color_fg

    # Clean tiny noise specks
    kernel = np.ones((3, 3), np.uint8)
    fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_clean, connectivity=8)
    min_area = max(50, int(0.002 * total_px))  # At least 0.2% of canvas

    significant_comps: list[dict[str, Any]] = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            # Filter out entire canvas border or 1-2px border lines
            if w > 0.98 * W and h > 0.98 * H:
                continue
            if (x <= 1 and w <= 2) or (y <= 1 and h <= 2) or (x + w >= W - 1 and w <= 2) or (y + h >= H - 1 and h <= 2):
                continue

            # Classify category based on aspect ratio & position
            aspect = h / max(1, w)
            if aspect > 1.45:
                cat = "dress"
                label = "Dress / Gown / One-piece"
            elif aspect > 1.1:
                if y > H * 0.45:
                    cat = "skirt"
                    label = "Skirt"
                else:
                    cat = "dress"
                    label = "Dress / Top"
            elif aspect < 0.75:
                cat = "skirt"
                label = "Skirt"
            else:
                cat = "upper_clothes"
                label = "T-Shirt / Top / Shirt"

            significant_comps.append({
                "label": i,
                "area": int(area),
                "bbox": [x, y, x + w, y + h],
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "category": cat,
                "label_name": label,
                "aspect_ratio": round(aspect, 2),
                "centroid": centroids[i].tolist(),
                "comp_mask": (labels == i),
            })

    total_fg_area = sum(c["area"] for c in significant_comps)

    # Multi-garment condition:
    # At least 2 significant garments, and the largest does not dominate >= 80% of total area
    is_multi = False
    if len(significant_comps) >= 2 and total_fg_area > 0:
        largest_area = max(c["area"] for c in significant_comps)
        largest_share = largest_area / total_fg_area
        if largest_share < 0.80:
            is_multi = True
            # Sort garments in reading order: row-by-row, left-to-right
            row_height = max(30, int(H / 5))
            significant_comps.sort(key=lambda c: (c["y"] // row_height, c["x"]))

    return is_multi, significant_comps

