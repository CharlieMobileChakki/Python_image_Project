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

    Relies strictly on true salient foreground alpha (BiRefNet Lite) rather than fragile
    outer border color metrics. Prevents textured carpets, complex studio walls, shadows,
    and mannequins from being erroneously classified as separate garments.

    Incorporates:
    1. Multi-point perimeter background sampling & studio drop-shadow suppression.
    2. Edge-seeded floodfill to clear contiguous background channels between garments.
    3. Morphological opening to sever faint diagonal bridges.
    4. Euclidean Distance Transform + Watershed bottleneck severing to cleanly separate
       touching or clustered garments in dense catalog collection collages.

    Args:
        image_rgb: PIL RGB image.
        fg_alpha: Optional uint8 alpha mask from salient foreground extraction.

    Returns:
        (is_multi_garment, list of detected garment components with masks and bounding boxes)
    """
    arr = np.array(image_rgb)
    H, W, _ = arr.shape
    total_px = H * W

    # Use salient foreground alpha from BiRefNet Lite
    if fg_alpha is None:
        alpha = np.ones((H, W), dtype=np.uint8) * 255
    else:
        alpha = fg_alpha.copy()

    # 1. Background color detection from border perimeter band (not just 4 single corners)
    band_h = max(2, min(12, H // 25))
    band_w = max(2, min(12, W // 25))
    border_pixels = np.concatenate([
        arr[:band_h, :].reshape(-1, 3),
        arr[-band_h:, :].reshape(-1, 3),
        arr[:, :band_w].reshape(-1, 3),
        arr[:, -band_w:].reshape(-1, 3),
    ], axis=0).astype(np.float32)

    bg_color = np.median(border_pixels, axis=0)
    bg_std = float(np.std(border_pixels, axis=0).mean())
    is_light_bg = float(np.mean(bg_color)) > 170.0

    cleaned_alpha = alpha.copy()

    # 2. Studio drop-shadow suppression & perimeter-seeded floodfill
    if bg_std < 45.0 or is_light_bg:
        color_diff = np.linalg.norm(arr.astype(np.float32) - bg_color, axis=2)
        is_bg_color = (color_diff < max(24.0, min(50.0, bg_std * 2.5)))

        if is_light_bg:
            r = arr[:, :, 0].astype(np.int32)
            g = arr[:, :, 1].astype(np.int32)
            b = arr[:, :, 2].astype(np.int32)
            saturation = np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))
            brightness = (r + g + b) / 3.0
            is_shadow = (brightness > 180.0) & (saturation < 22)
            is_bg_candidate = is_bg_color | is_shadow
        else:
            is_bg_candidate = is_bg_color

        flood_map = np.zeros((H + 2, W + 2), dtype=np.uint8)
        bg_seed_mask = is_bg_candidate.astype(np.uint8) * 255

        step_x = max(1, W // 30)
        step_y = max(1, H // 30)
        for y in [0, H - 1]:
            for x in range(0, W, step_x):
                if bg_seed_mask[y, x] == 255:
                    cv2.floodFill(bg_seed_mask, flood_map, (x, y), 128, flags=4 | (255 << 8))

        for x in [0, W - 1]:
            for y in range(0, H, step_y):
                if bg_seed_mask[y, x] == 255:
                    cv2.floodFill(bg_seed_mask, flood_map, (x, y), 128, flags=4 | (255 << 8))

        connected_bg = (bg_seed_mask == 128)
        cleaned_alpha[connected_bg] = 0

    # 3. Binary threshold & morphological opening to break fine bridges
    bin_mask = (cleaned_alpha > 45).astype(np.uint8)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_open)

    # 4. Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    min_area = max(150, int(0.003 * total_px))
    min_w = max(10, int(0.02 * W))
    min_h = max(10, int(0.02 * H))

    raw_comps: list[dict[str, Any]] = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Filter out canvas borders and edge artifacts
        if w > 0.98 * W and h > 0.98 * H:
            continue
        if (x <= 1 and w <= 2) or (y <= 1 and h <= 2) or (x + w >= W - 1 and w <= 2) or (y + h >= H - 1 and h <= 2):
            continue

        if area >= min_area and (w >= min_w or h >= min_h):
            comp_mask = (labels == i)
            raw_comps.append({
                "area": int(area),
                "bbox": [x, y, x + w, y + h],
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "comp_mask": comp_mask,
            })

    # 5. Bottleneck severing via Distance Transform & Watershed on merged clusters
    resolved_comps: list[dict[str, Any]] = []
    for comp in raw_comps:
        w, h = comp["w"], comp["h"]
        c_mask = comp["comp_mask"].astype(np.uint8) * 255

        # Check if component is a multi-garment cluster spanning multiple rows/cols
        is_large = (w > 0.32 * W and h > 0.32 * H) or (comp["area"] > 0.15 * total_px and len(raw_comps) <= 4)
        dist = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
        max_d = float(dist.max())

        split_occurred = False
        if is_large and max_d > 10.0:
            sure_fg = (dist > 0.38 * max_d).astype(np.uint8)
            num_seeds, seed_markers = cv2.connectedComponents(sure_fg)

            valid_seeds = 0
            filtered_seeds = np.zeros_like(sure_fg)
            for s in range(1, num_seeds):
                s_area = int((seed_markers == s).sum())
                if s_area >= max(40, int(0.001 * total_px)):
                    valid_seeds += 1
                    filtered_seeds[seed_markers == s] = valid_seeds

            if valid_seeds >= 2:
                unknown = cv2.subtract(c_mask, (filtered_seeds > 0).astype(np.uint8) * 255)
                markers = filtered_seeds.copy() + 1
                markers[unknown == 255] = 0

                c_mask_3c = cv2.cvtColor(c_mask, cv2.COLOR_GRAY2BGR)
                cv2.watershed(c_mask_3c, markers)

                for s in range(1, valid_seeds + 1):
                    sub_mask = (markers == s + 1)
                    s_area = int(sub_mask.sum())
                    if s_area >= min_area:
                        ys, xs = np.where(sub_mask)
                        sx0, sx1 = int(xs.min()), int(xs.max())
                        sy0, sy1 = int(ys.min()), int(ys.max())
                        sw = sx1 - sx0
                        sh = sy1 - sy0
                        if sw >= min_w or sh >= min_h:
                            resolved_comps.append({
                                "area": s_area,
                                "bbox": [sx0, sy0, sx1, sy1],
                                "x": sx0,
                                "y": sy0,
                                "w": sw,
                                "h": sh,
                                "comp_mask": sub_mask,
                            })
                            split_occurred = True

        if not split_occurred:
            resolved_comps.append(comp)

    # 6. Categorize, label, and compute centroids for all resolved garments
    significant_comps: list[dict[str, Any]] = []
    for idx, comp in enumerate(resolved_comps):
        x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
        area = comp["area"]
        aspect = h / max(1, w)

        if aspect > 1.40:
            cat = "dress"
            label = "Dress / Gown / Full Outfit"
        elif aspect > 1.15 and y > H * 0.38:
            cat = "pants"
            label = "Pants / Lower Garment"
        elif aspect < 0.85:
            cat = "upper_clothes"
            label = "Top / Shirt / Upper Garment"
        else:
            cat = "upper_clothes"
            label = "Garment Cutout"

        ys, xs = np.where(comp["comp_mask"])
        cx = float(np.mean(xs)) if len(xs) > 0 else float(x + w / 2)
        cy = float(np.mean(ys)) if len(ys) > 0 else float(y + h / 2)

        significant_comps.append({
            "label": idx + 1,
            "area": area,
            "bbox": comp["bbox"],
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "category": cat,
            "label_name": label,
            "aspect_ratio": round(aspect, 2),
            "centroid": [cx, cy],
            "comp_mask": comp["comp_mask"],
        })

    total_fg_area = sum(c["area"] for c in significant_comps)

    # Multi-garment condition:
    # Requires at least 2 distinct significant garments where the largest item
    # does NOT dominate >= 88% of total foreground area, or >= 3 garments.
    is_multi = False
    if len(significant_comps) >= 2 and total_fg_area > 0:
        largest_area = max(c["area"] for c in significant_comps)
        largest_share = largest_area / total_fg_area
        if len(significant_comps) >= 3 or largest_share < 0.88:
            is_multi = True
            row_height = max(30, int(H / 5))
            significant_comps.sort(key=lambda c: (c["y"] // row_height, c["x"]))

    return is_multi, significant_comps
