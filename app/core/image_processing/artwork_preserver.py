"""Artwork & Landmark Preservation Engine for Apparel Ghost-Mannequin Pipeline.

Preserves authentic visible logos, crests, manufacturer marks, and sleeve patches
by extracting high-resolution source pixels and mapping them onto reconstructed
ghost-mannequin landmarks. Prevents hallucinated or missing logos, and repairs
interior alpha holes in sleeve/torso fabric.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def crop_mark_from_source(
    source_rgb: Image.Image,
    box_2d: list[int | float] | tuple[int | float, ...],
    pad_ratio: float = 0.06,
    rotation_deg: float = 0.0,
) -> tuple[Image.Image | None, list[int]]:
    """Crop an authentic mark or patch from the source photo using normalized box_2d coordinates.

    Args:
        source_rgb: Original uploaded PIL RGB image.
        box_2d: [ymin, xmin, ymax, xmax] normalized on 0-1000 scale.
        pad_ratio: Padding around the box.
        rotation_deg: Angle in degrees to rotate the cropped mark upright (e.g. 90, 180, 270).

    Returns:
        (cropped_image, [x0, y0, x1, y1]) or (None, [])
    """
    try:
        w, h = source_rgb.size
        if not box_2d or len(box_2d) != 4:
            return None, []

        ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
        is_norm_1 = all(0.0 <= v <= 1.0 for v in (ymin, xmin, ymax, xmax))
        sx = w if is_norm_1 else (w / 1000.0)
        sy = h if is_norm_1 else (h / 1000.0)

        x0 = int(xmin * sx)
        y0 = int(ymin * sy)
        x1 = int(xmax * sx)
        y1 = int(ymax * sy)

        # Apply padding
        bw = x1 - x0
        bh = y1 - y0
        if bw <= 5 or bh <= 5:
            return None, []

        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)

        cx0 = max(0, x0 - pad_x)
        cy0 = max(0, y0 - pad_y)
        cx1 = min(w, x1 + pad_x)
        cy1 = min(h, y1 + pad_y)

        cropped = source_rgb.crop((cx0, cy0, cx1, cy1))

        # Rotate upright if garment was rotated in source photo
        if abs(rotation_deg) >= 1.0:
            cropped = cropped.rotate(-rotation_deg, resample=Image.Resampling.BICUBIC, expand=True)

        return cropped, [cx0, cy0, cx1, cy1]
    except Exception as exc:
        logger.warning("Failed to crop mark from source (%s)", exc)
        return None, []


def detect_garment_landmarks(cutout_img: Image.Image) -> dict[str, Any]:
    """Detect anatomical landmarks on a 3D ghost-mannequin garment.

    Returns dictionary of landmarks:
    - bounding_box: [x0, y0, x1, y1]
    - center_x, center_y
    - neckline_top, neckline_bottom
    - chest_y: approximate chest line (aligned with existing marks if present)
    - viewer_left_chest: (x, y) target center for viewer-left mark
    - viewer_right_chest: (x, y) target center for viewer-right mark
    """
    img_rgba = cutout_img.convert("RGBA")
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3]
    h, w, _ = arr.shape

    # Check if alpha is genuinely transparent or all opaque (e.g. RGB screenshot)
    if np.all(alpha > 250):
        rgb = arr[:, :, :3]
        corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
        bg_col = np.median(corners, axis=0)
        fg_mask = (np.linalg.norm(rgb.astype(np.float32) - bg_col, axis=2) > 30).astype(np.uint8)
    else:
        fg_mask = (alpha > 40).astype(np.uint8)

    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return {}

    x0, x1 = int(np.min(xs)), int(np.max(xs))
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    gw = max(1, x1 - x0)
    gh = max(1, y1 - y0)
    cx = x0 + gw // 2

    # Neckline detection: center 25% of width in upper 35% of garment height
    neck_slice = fg_mask[y0 : y0 + int(gh * 0.35), cx - int(gw * 0.12) : cx + int(gw * 0.12)]
    neck_ys, _ = np.where(neck_slice > 0)
    if len(neck_ys) > 0:
        neck_top = y0 + int(np.min(neck_ys))
        neck_bottom = y0 + int(np.max(neck_ys))
    else:
        neck_top = y0
        neck_bottom = y0 + int(gh * 0.15)

    default_chest_y = int(neck_bottom + gh * 0.08)

    # Check if an existing mark is visible on viewer-right or viewer-left to align chest_y
    rgb = arr[:, :, :3]
    band_y0 = max(0, neck_bottom)
    band_y1 = min(h, neck_bottom + int(gh * 0.22))

    vr_x0 = cx + int(gw * 0.08)
    vr_x1 = min(w, cx + int(gw * 0.35))
    patch_vr = rgb[band_y0:band_y1, vr_x0:vr_x1]

    chest_y = default_chest_y
    detected_existing_side = None
    if patch_vr.size > 0:
        patch_gray = cv2.cvtColor(patch_vr, cv2.COLOR_RGB2GRAY)
        base_val = np.median(patch_gray)
        mark_mask = np.abs(patch_gray.astype(np.float32) - base_val) > 28.0
        m_ys, m_xs = np.where(mark_mask)
        if len(m_ys) > 15:
            chest_y = band_y0 + int(np.mean(m_ys))
            detected_existing_side = "viewer_right"

    viewer_left_x = int(cx - gw * 0.19)
    viewer_right_x = int(cx + gw * 0.19)

    return {
        "bounding_box": [x0, y0, x1, y1],
        "width": gw,
        "height": gh,
        "center_x": cx,
        "neck_top": neck_top,
        "neck_bottom": neck_bottom,
        "chest_y": chest_y,
        "detected_existing_side": detected_existing_side,
        "viewer_left_chest": (viewer_left_x, chest_y),
        "viewer_right_chest": (viewer_right_x, chest_y),
    }


def ensure_garment_fabric_opacity(cutout_rgba: Image.Image) -> Image.Image:
    """Detect and heal unnatural transparent holes/voids inside the sleeve or torso fabric.

    In a valid 3D ghost mannequin:
    - The outer background is transparent.
    - Hollow collar aperture and hollow cuff apertures are open.
    - BUT interior sleeve and torso fabric MUST BE 100% OPAQUE.
    If a dark shadow or diffusion defect causes BiRefNet to carve a transparent hole
    in the sleeve, this function restores alpha to 255 and cleans the fabric color.
    """
    arr = np.array(cutout_rgba.convert("RGBA")).copy()
    h, w, _ = arr.shape
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # Find solid core of garment
    fg_mask = (alpha > 40).astype(np.uint8)
    if fg_mask.sum() < 1000:
        return cutout_rgba

    # 1. Exterior background mask via floodfill from border
    padded = np.pad(fg_mask, 2, mode="constant", constant_values=0)
    flood = np.zeros((h + 6, w + 6), dtype=np.uint8)
    bg_seed = (padded == 0).astype(np.uint8) * 255
    cv2.floodFill(bg_seed, flood, (0, 0), 128)
    exterior_bg = (bg_seed[2 : h + 2, 2 : w + 2] == 128)

    # 2. An internal void / hole is inside the garment silhouette, has low alpha, and is NOT exterior background
    holes = (alpha < 180) & (~exterior_bg)

    # Exclude hollow neck aperture at top collar center
    ys_fg, xs_fg = np.where(fg_mask > 0)
    if len(ys_fg) > 0:
        min_y = np.min(ys_fg)
        cx = int(np.mean(xs_fg))
        y_grid, x_grid = np.indices((h, w))
        neck_aperture = (y_grid < min_y + int(h * 0.12)) & (np.abs(x_grid - cx) < int(w * 0.12))
        holes[neck_aperture] = False

    hole_pixels = int(holes.sum())
    if hole_pixels > 0:
        logger.info("Healing %d transparent sleeve/torso hole pixels", hole_pixels)
        # Restore alpha to 255
        arr[holes, 3] = 255

        # Check if the RGB inside the hole was a dark shadow / smudge artifact (< 140)
        hole_rgb = rgb[holes]
        if len(hole_rgb) > 0 and np.mean(hole_rgb) < 140:
            dilated_hole = cv2.dilate(holes.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            surrounding = (dilated_hole > 0) & (~holes) & (alpha > 200)
            if surrounding.any():
                clean_color = np.median(rgb[surrounding], axis=0).astype(np.uint8)
                arr[holes, :3] = clean_color

    return Image.fromarray(arr, mode="RGBA")


def map_and_preserve_artwork(
    source_rgb: Image.Image,
    reconstructed_rgba: Image.Image,
    chest_marks: list[dict[str, Any]],
    tolerance_px: int = 40,
    source_rotation_deg: float = 0.0,
) -> tuple[Image.Image, dict[str, Any]]:
    """Preserve authentic visible logos/marks on reconstructed 3D ghost mannequin.

    Features:
    - Crops authentic marks directly from source photo using box_2d.
    - Rotates mark upright if source image was oriented sideways/tilted.
    - Inspects reconstructed garment chest for missing or dropped marks.
    - Maps authentic pixels onto the corresponding anatomical landmarks.
    - Adapts to any fabric color (white, dark, colored) by sampling local border fabric.
    - NEVER inserts stock or synthetic logos from guessed brand names.
    - Restores sleeve/torso opacity.

    Returns:
        (updated_rgba_image, metadata_dict)
    """
    landmarks = detect_garment_landmarks(reconstructed_rgba)
    if not landmarks or not chest_marks:
        return reconstructed_rgba, {"preserved_marks": [], "review_flagged": False}

    output_arr = np.array(reconstructed_rgba.convert("RGBA"), dtype=np.uint8).copy()
    gw = landmarks["width"]
    gh = landmarks["height"]
    preserved_records = []
    review_flagged = False

    for mark in chest_marks:
        side = mark.get("side", "").strip().lower()
        m_type = mark.get("type", "unidentified_mark")
        box_2d = mark.get("box_2d")

        if not side or not box_2d:
            continue

        target_pt = landmarks.get("viewer_left_chest") if side == "viewer_left" else landmarks.get("viewer_right_chest")
        if not target_pt:
            continue

        tx, ty = target_pt

        # Crop authentic mark from source photo
        crop_pil, crop_box = crop_mark_from_source(
            source_rgb,
            box_2d,
            pad_ratio=0.08,
            rotation_deg=source_rotation_deg,
        )
        if crop_pil is None:
            continue

        cw, ch = crop_pil.size
        # Target scale on reconstructed canvas relative to garment width
        # A typical chest logo is ~8% to 11% of garment width
        target_w = max(20, int(gw * 0.095))
        scale = target_w / float(cw)
        target_h = max(16, int(ch * scale))
        resized_crop = crop_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        crop_arr = np.array(resized_crop)

        # Check if reconstructed garment already has a clear mark in that region
        rx0 = max(0, tx - target_w // 2)
        ry0 = max(0, ty - target_h // 2)
        rx1 = min(reconstructed_rgba.width, rx0 + target_w)
        ry1 = min(reconstructed_rgba.height, ry0 + target_h)

        rec_patch = output_arr[ry0:ry1, rx0:rx1, :3]
        rec_gray = cv2.cvtColor(rec_patch, cv2.COLOR_RGB2GRAY)
        patch_std = float(np.std(rec_gray))
        has_existing_mark = patch_std > 22.0 and np.mean(rec_gray) < 220.0

        # If mark is missing or dropped on either side
        if not has_existing_mark or (side == "viewer_left" and patch_std < 18.0):
            logger.info("Restoring authentic %s on %s at (%d, %d)", m_type, side, tx, ty)

            # Clean color-adaptive alpha mask for cropped logo:
            # Sample border pixels to detect local base fabric color
            borders = np.concatenate(
                [crop_arr[0, :, :3], crop_arr[-1, :, :3], crop_arr[:, 0, :3], crop_arr[:, -1, :3]],
                axis=0,
            )
            base_col = np.median(borders, axis=0)
            dist = np.linalg.norm(crop_arr[:, :, :3].astype(np.float32) - base_col, axis=2)
            logo_mask = np.clip((dist - 16.0) / 25.0, 0.0, 1.0)
            logo_mask = cv2.GaussianBlur(logo_mask, (3, 3), 0)

            # Composite onto output_arr
            pw = rx1 - rx0
            ph = ry1 - ry0
            if pw > 0 and ph > 0:
                sub_crop = crop_arr[:ph, :pw]
                sub_mask = logo_mask[:ph, :pw, np.newaxis]
                bg_slice = output_arr[ry0:ry1, rx0:rx1, :3].astype(np.float32)
                fg_slice = sub_crop.astype(np.float32)
                blended = (fg_slice * sub_mask + bg_slice * (1.0 - sub_mask)).astype(np.uint8)
                output_arr[ry0:ry1, rx0:rx1, :3] = blended
                output_arr[ry0:ry1, rx0:rx1, 3] = 255

            preserved_records.append({
                "type": m_type,
                "side": side,
                "name": mark.get("name", "unidentified"),
                "status": "restored_from_source_pixels",
                "target_coordinates": [tx, ty],
            })
        else:
            preserved_records.append({
                "type": m_type,
                "side": side,
                "name": mark.get("name", "unidentified"),
                "status": "already_present_in_generation",
                "target_coordinates": [tx, ty],
            })

    result_img = Image.fromarray(output_arr, mode="RGBA")
    # Always heal interior alpha holes in sleeve fabric
    result_img = ensure_garment_fabric_opacity(result_img)

    return result_img, {
        "preserved_marks": preserved_records,
        "review_flagged": review_flagged,
    }
