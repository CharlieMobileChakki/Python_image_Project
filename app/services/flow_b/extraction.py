"""Flow B: Semantic Garment Extraction Service Module.

Provides model loading, semantic segmentation via SegFormer, multi-class union,
confidence-based soft alpha, and 1024x1024 canvas normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = os.getenv("FLOWB_MODEL", "mattmdjaga/segformer_b2_clothes").strip()
OUTPUT_SIZE = int(os.getenv("OUTPUT_SIZE", "1024"))
CANVAS_FILL_RATIO = float(os.getenv("CANVAS_FILL_RATIO", "0.82"))
MAX_INPUT_EDGE = int(os.getenv("MAX_INPUT_EDGE", "2400"))

# Color palette for SegFormer ATR 18 classes (R, G, B)
CLASS_PALETTE = [
    (0, 0, 0),        # 0: Background
    (128, 0, 0),      # 1: Hat
    (0, 128, 0),      # 2: Hair
    (128, 128, 0),    # 3: Sunglasses
    (0, 0, 128),      # 4: Upper-clothes
    (128, 0, 128),    # 5: Skirt
    (0, 128, 128),    # 6: Pants
    (128, 128, 128),  # 7: Dress
    (64, 0, 0),       # 8: Belt
    (192, 0, 0),      # 9: Left-shoe
    (64, 128, 0),     # 10: Right-shoe
    (192, 128, 0),    # 11: Face
    (64, 0, 128),     # 12: Left-leg
    (192, 0, 128),    # 13: Right-leg
    (64, 128, 128),   # 14: Left-arm
    (192, 128, 128),  # 15: Right-arm
    (0, 64, 0),       # 16: Bag
    (128, 64, 0),     # 17: Scarf
]

# Explicit alias mapping grounded in empirical fashion parsing observation
EXPLICIT_ALIASES: dict[str, list[str]] = {
    "anarkali": ["upper-clothes", "skirt"],
}

_model: SegformerForSemanticSegmentation | None = None
_processor: AutoImageProcessor | None = None
_id2label: dict[int, str] = {}


@dataclass
class FlowBResult:
    png_bytes: bytes
    original_size: tuple[int, int]
    target_classes: list[str]
    target_class_ids: list[int]
    applied_alias: str | None
    mask_coverage_pct: float
    alpha_mode: str
    processing_time_ms: int
    bounding_box: list[int]


def load_flowb_model(model_id: str = MODEL_ID) -> None:
    """Load SegFormer processor and model weights once for service lifetime."""
    global _model, _processor, _id2label
    if _model is not None and _processor is not None:
        return

    print(f"[Flow B] Loading model '{model_id}' on CPU...")
    start_t = time.perf_counter()
    _processor = AutoImageProcessor.from_pretrained(model_id)
    _model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    _model.eval()
    _id2label = {int(k): v for k, v in _model.config.id2label.items()}
    load_ms = round((time.perf_counter() - start_t) * 1000)
    print(f"[Flow B] Model successfully initialized in {load_ms} ms ({len(_id2label)} classes).")


def is_model_loaded() -> bool:
    """Check if model and processor are initialized and ready."""
    return _model is not None and _processor is not None and len(_id2label) > 0


def get_id2label() -> dict[int, str]:
    """Get the active class mapping."""
    return dict(_id2label)


def resolve_single_class(id2label: dict[int, str], token: str) -> tuple[int, str]:
    """Resolve a single garment token to (class_id, canonical_label)."""
    target_clean = token.strip().lower()
    if not target_clean:
        raise ValueError("Target class token cannot be empty.")

    # 1. Exact match (case-insensitive)
    for cid, label in id2label.items():
        if label.lower() == target_clean:
            return cid, label

    # 2. Normalized match (ignoring hyphens, underscores, spaces)
    target_norm = re.sub(r"[-_\s]+", "", target_clean)
    for cid, label in id2label.items():
        label_norm = re.sub(r"[-_\s]+", "", label.lower())
        if label_norm == target_norm:
            return cid, label

    # 3. Substring match
    for cid, label in id2label.items():
        if target_clean in label.lower():
            return cid, label

    available = [name for _, name in sorted(id2label.items())]
    raise ValueError(
        f"Unknown garment class '{token}'. Available classes: {available}. "
        f"Available aliases: {list(EXPLICIT_ALIASES.keys())}"
    )


def resolve_target_classes(
    id2label: dict[int, str],
    query: str,
) -> tuple[list[tuple[int, str]], str | None]:
    """Resolve class query string into a list of (class_id, canonical_label) tuples."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Target class query cannot be empty.")

    lower_query = clean_query.lower()
    alias_applied: str | None = None

    if lower_query in EXPLICIT_ALIASES:
        class_tokens = EXPLICIT_ALIASES[lower_query]
        alias_applied = lower_query
    else:
        class_tokens = [tok.strip() for tok in clean_query.split(",") if tok.strip()]
        if not class_tokens:
            raise ValueError("Target class query yielded no valid class tokens.")

    resolved_classes: list[tuple[int, str]] = []
    seen_ids: set[int] = set()

    for token in class_tokens:
        cid, label = resolve_single_class(id2label, token)
        if cid not in seen_ids:
            seen_ids.add(cid)
            resolved_classes.append((cid, label))

    return resolved_classes, alias_applied


def validate_alpha_thresholds(alpha_low: float, alpha_high: float) -> None:
    """Validate probability-to-alpha threshold parameters."""
    if not (0.0 <= alpha_low <= 1.0):
        raise ValueError(f"alpha_low ({alpha_low}) must be between 0.0 and 1.0")
    if not (0.0 <= alpha_high <= 1.0):
        raise ValueError(f"alpha_high ({alpha_high}) must be between 0.0 and 1.0")
    if alpha_low >= alpha_high:
        raise ValueError(
            f"alpha_low ({alpha_low}) must be strictly less than alpha_high ({alpha_high})"
        )


def compute_soft_alpha(
    prob_map: np.ndarray,
    alpha_low: float = 0.20,
    alpha_high: float = 0.80,
) -> np.ndarray:
    """Map composite garment probability [0.0, 1.0] to 8-bit alpha [0, 255] via linear ramp."""
    validate_alpha_thresholds(alpha_low, alpha_high)
    clipped = np.clip(prob_map, alpha_low, alpha_high)
    normalized = (clipped - alpha_low) / (alpha_high - alpha_low)
    alpha = np.round(normalized * 255.0).astype(np.uint8)
    return alpha


def normalize_canvas(
    cutout: Image.Image,
    output_size: int = OUTPUT_SIZE,
    fill_ratio: float = CANVAS_FILL_RATIO,
) -> tuple[bytes, float, list[int]]:
    """Tight-crop non-empty garment alpha bounds and center on transparent square canvas."""
    alpha = cutout.getchannel("A")
    # Discard faint alpha noise for bounding box estimation
    threshold = alpha.point(lambda value: 255 if value > 8 else 0)
    bbox = threshold.getbbox()
    if bbox is None:
        raise ValueError(
            "No foreground garment was detected for the requested class(es). "
            "Try a clearer photo with the requested garment visible."
        )

    # Compute coverage percentage
    total_pixels = cutout.width * cutout.height
    garment_pixels = int((np.array(threshold) == 255).sum())
    coverage_pct = round((garment_pixels / total_pixels) * 100, 2)

    cropped = cutout.crop(bbox)
    if cropped.width < 2 or cropped.height < 2:
        raise ValueError("The detected garment mask is too small to process.")

    max_item_size = max(64, int(output_size * fill_ratio))
    scale = min(max_item_size / cropped.width, max_item_size / cropped.height)
    new_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 0))
    x = (output_size - resized.width) // 2
    y = (output_size - resized.height) // 2
    canvas.alpha_composite(resized, (x, y))

    output = BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue(), coverage_pct, list(bbox)


def extract_garment_worn(
    image_bytes: bytes,
    class_name: str,
    alpha_mode: str = "soft",
    alpha_low: float = 0.20,
    alpha_high: float = 0.80,
) -> FlowBResult:
    """Execute Flow B extraction pipeline on raw image bytes."""
    if not is_model_loaded():
        load_flowb_model()

    mode_clean = alpha_mode.strip().lower()
    if mode_clean not in {"soft", "hard"}:
        raise ValueError("alpha_mode must be either 'soft' or 'hard'.")

    validate_alpha_thresholds(alpha_low, alpha_high)

    # Resolve garment classes
    id2label = get_id2label()
    resolved_classes, alias_applied = resolve_target_classes(id2label, class_name)
    target_ids = [cid for cid, _ in resolved_classes]
    target_labels = [label for _, label in resolved_classes]

    # Validate and load image
    try:
        raw_img = Image.open(BytesIO(image_bytes))
        fmt = (raw_img.format or "").upper()
        if fmt not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Supported image types are JPG, PNG, and WEBP.")
        image = ImageOps.exif_transpose(raw_img).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid/safe image.") from exc

    orig_w, orig_h = image.size
    start_infer = time.perf_counter()

    # Preprocess
    assert _processor is not None and _model is not None
    inputs = _processor(images=image, return_tensors="pt")

    # Inference (CPU)
    with torch.no_grad():
        outputs = _model(**inputs)

    # Upsample logits to original image dimensions
    upsampled_logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )

    if mode_clean == "hard":
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        binary_mask = np.isin(pred_seg, target_ids).astype(np.uint8) * 255
        mask_img = Image.fromarray(binary_mask, mode="L")
    else:
        probs = torch.softmax(upsampled_logits, dim=1)[0]
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=probs.device)
        composite_prob = torch.index_select(probs, 0, target_tensor).sum(dim=0).cpu().numpy()
        soft_alpha = compute_soft_alpha(composite_prob, alpha_low=alpha_low, alpha_high=alpha_high)
        mask_img = Image.fromarray(soft_alpha, mode="L")

    # Form transparent cutout
    cutout = image.convert("RGBA")
    cutout.putalpha(mask_img)

    # Normalize to 1024x1024 canvas
    png_bytes, coverage_pct, bbox = normalize_canvas(cutout, OUTPUT_SIZE, CANVAS_FILL_RATIO)
    elapsed_ms = round((time.perf_counter() - start_infer) * 1000)

    return FlowBResult(
        png_bytes=png_bytes,
        original_size=(orig_w, orig_h),
        target_classes=target_labels,
        target_class_ids=target_ids,
        applied_alias=alias_applied,
        mask_coverage_pct=coverage_pct,
        alpha_mode=mode_clean,
        processing_time_ms=elapsed_ms,
        bounding_box=bbox,
    )

