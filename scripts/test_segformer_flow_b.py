"""Flow B: Semantic Garment Extraction using SegFormer with Multi-Class Union and Soft Alpha.

Isolates specific garment classes (e.g. pants, upper-clothes, skirt, dress) from
images of people wearing clothes, excluding hands, skin, shoes, and background.
Supports multi-class mask unions (e.g. "upper-clothes,skirt"), explicit aliases
(e.g. "anarkali"), and confidence-based soft-alpha edge refinement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

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
# Traditional flared Indian ethnic dresses (e.g. anarkali) are segmented into
# upper-clothes (bodice/sleeves) and skirt (flared maxi skirt) by the ATR-trained model.
EXPLICIT_ALIASES: dict[str, list[str]] = {
    "anarkali": ["upper-clothes", "skirt"],
}


def resolve_single_class(id2label: dict[int, str], token: str) -> tuple[int, str]:
    """Find target class ID from a single name token (case-insensitive, exact, normalized, or substring)."""
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
    """Resolve class query string into a list of (class_id, canonical_label) tuples.
    
    Supports:
      - Single class: 'pants'
      - Comma-separated classes: 'upper-clothes,skirt'
      - Explicit alias: 'anarkali' -> ['upper-clothes', 'skirt']
    """
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


def run_flow_b_extraction(
    image_path: Path,
    target_class_query: str,
    output_dir: Path,
    model_id: str = "mattmdjaga/segformer_b2_clothes",
    alpha_low: float = 0.20,
    alpha_high: float = 0.80,
) -> dict[str, Any]:
    """Execute Flow B extraction with multi-class union and confidence-based soft alpha."""
    validate_alpha_thresholds(alpha_low, alpha_high)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{model_id}' on CPU...")
    start_load = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model.eval()
    load_time_ms = round((time.perf_counter() - start_load) * 1000)
    print(f"Model loaded in {load_time_ms} ms ({load_time_ms / 1000:.2f} s).")

    # Read model id2label
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    resolved_classes, alias_applied = resolve_target_classes(id2label, target_class_query)
    target_ids = [cid for cid, _ in resolved_classes]
    target_labels = [label for _, label in resolved_classes]

    print(f"\nTarget Query: '{target_class_query}'")
    if alias_applied:
        print(f"  Applied Alias: '{alias_applied}' -> {target_labels}")
    print(f"  Resolved Classes ({len(resolved_classes)}):")
    for cid, label in resolved_classes:
        print(f"    [{cid:2d}] {label}")

    # Load input image
    print(f"\nReading input image: {image_path}")
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    orig_w, orig_h = image.size
    print(f"Original image dimensions: {orig_w} x {orig_h}")

    # Prepare inputs and run inference (single forward pass)
    inputs = processor(images=image, return_tensors="pt")
    print("Running SegFormer semantic segmentation inference...")
    start_infer = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    infer_time_ms = round((time.perf_counter() - start_infer) * 1000)
    print(f"Inference completed in {infer_time_ms} ms ({infer_time_ms / 1000:.2f} s).")

    # Bilinear upsample logits to original image resolution (orig_h, orig_w)
    upsampled_logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )

    # 1. Hard baseline via argmax
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    binary_mask = np.isin(pred_seg, target_ids).astype(np.uint8) * 255
    binary_mask_img = Image.fromarray(binary_mask, mode="L")
    binary_mask_path = output_dir / "binary_mask.png"
    binary_mask_img.save(binary_mask_path)
    print(f"Saved binary mask: {binary_mask_path}")

    # 2. Hard transparent RGBA cutout
    hard_cutout = image.convert("RGBA")
    hard_cutout.putalpha(binary_mask_img)
    hard_cutout_path = output_dir / "hard_cutout.png"
    hard_cutout.save(hard_cutout_path)
    print(f"Saved hard cutout: {hard_cutout_path}")

    # 3. Soft alpha from softmax class probabilities
    probs = torch.softmax(upsampled_logits, dim=1)[0]
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=probs.device)
    composite_prob = torch.index_select(probs, 0, target_tensor).sum(dim=0).cpu().numpy()

    soft_alpha = compute_soft_alpha(composite_prob, alpha_low=alpha_low, alpha_high=alpha_high)
    soft_alpha_img = Image.fromarray(soft_alpha, mode="L")
    soft_alpha_path = output_dir / "soft_alpha.png"
    soft_alpha_img.save(soft_alpha_path)
    print(f"Saved soft alpha: {soft_alpha_path}")

    # 4. Soft transparent RGBA cutout
    soft_cutout = image.convert("RGBA")
    soft_cutout.putalpha(soft_alpha_img)
    soft_cutout_path = output_dir / "soft_cutout.png"
    soft_cutout.save(soft_cutout_path)
    print(f"Saved soft cutout: {soft_cutout_path}")

    # 5. Semantic class map (color-coded visualization)
    class_map_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    unique_classes = np.unique(pred_seg)
    for cid in unique_classes:
        color = CLASS_PALETTE[cid % len(CLASS_PALETTE)]
        class_map_rgb[pred_seg == cid] = color
    class_map_img = Image.fromarray(class_map_rgb, mode="RGB")
    class_map_path = output_dir / "class_map.png"
    class_map_img.save(class_map_path)
    print(f"Saved class map: {class_map_path}")

    # 6. Composite overlay on original image
    overlay = image.convert("RGBA")
    highlight = Image.new("RGBA", (orig_w, orig_h), (0, 220, 100, 110))
    overlay_cutout = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
    overlay_cutout.paste(highlight, (0, 0), binary_mask_img)
    combined_overlay = Image.alpha_composite(overlay, overlay_cutout)
    overlay_path = output_dir / "overlay.png"
    combined_overlay.convert("RGB").save(overlay_path)
    print(f"Saved overlay: {overlay_path}")

    # Metrics computation
    total_pixels = orig_w * orig_h
    hard_pixels = int((binary_mask == 255).sum())
    hard_coverage_pct = round((hard_pixels / total_pixels) * 100, 2)

    soft_solid_pixels = int((soft_alpha == 255).sum())
    soft_transition_pixels = int(((soft_alpha > 0) & (soft_alpha < 255)).sum())
    soft_mean_alpha = float(soft_alpha.mean())

    hard_bbox = binary_mask_img.getbbox()
    soft_bbox = soft_alpha_img.getbbox()

    metrics = {
        "input_filename": image_path.name,
        "input_dimensions": [orig_w, orig_h],
        "model_id": model_id,
        "target_class_query": target_class_query,
        "applied_alias": alias_applied,
        "resolved_classes": [
            {"id": cid, "canonical_label": label} for cid, label in resolved_classes
        ],
        "target_class_ids": target_ids,
        "target_canonical_labels": target_labels,
        "alpha_thresholds": {
            "alpha_low": alpha_low,
            "alpha_high": alpha_high,
        },
        "model_load_time_ms": load_time_ms,
        "inference_time_ms": infer_time_ms,
        "total_pixels": total_pixels,
        "hard_mask_pixels": hard_pixels,
        "hard_coverage_pct": hard_coverage_pct,
        "soft_alpha_solid_pixels": soft_solid_pixels,
        "soft_alpha_transition_pixels": soft_transition_pixels,
        "soft_mean_alpha": round(soft_mean_alpha, 2),
        "hard_bounding_box": list(hard_bbox) if hard_bbox else None,
        "soft_bounding_box": list(soft_bbox) if soft_bbox else None,
        "detected_classes": {int(c): id2label.get(int(c), "unknown") for c in unique_classes},
        "output_files": {
            "binary_mask": str(binary_mask_path),
            "soft_alpha": str(soft_alpha_path),
            "hard_cutout": str(hard_cutout_path),
            "soft_cutout": str(soft_cutout_path),
            "class_map": str(class_map_path),
            "overlay": str(overlay_path),
        },
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics JSON: {metrics_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flow B: Multi-Class Union and Confidence-Based Soft Alpha Garment Extraction"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=r"D:\my codes\Arangtik\backend\uploads\img_mv3nprm1_b7901a.jpg",
        help="Path to garment-worn image",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="pants",
        help="Garment class name(s) to extract (e.g. 'pants', 'upper-clothes,skirt', 'anarkali')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_output/phase10/test",
        help="Output directory for Flow B artifacts",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mattmdjaga/segformer_b2_clothes",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--alpha-low",
        type=float,
        default=0.20,
        help="Low probability threshold below which alpha is 0 (default: 0.20)",
    )
    parser.add_argument(
        "--alpha-high",
        type=float,
        default=0.80,
        help="High probability threshold above which alpha is 255 (default: 0.80)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Input image not found: {image_path}")

    metrics = run_flow_b_extraction(
        image_path=image_path,
        target_class_query=args.class_name,
        output_dir=Path(args.output_dir),
        model_id=args.model_id,
        alpha_low=args.alpha_low,
        alpha_high=args.alpha_high,
    )

    print("\n=== Execution Summary ===")
    print(f"Image:            {metrics['input_filename']} ({metrics['input_dimensions'][0]}x{metrics['input_dimensions'][1]})")
    print(f"Target Query:     '{metrics['target_class_query']}'")
    if metrics["applied_alias"]:
        print(f"Applied Alias:    '{metrics['applied_alias']}'")
    print(f"Target Classes:   {metrics['target_canonical_labels']} (IDs: {metrics['target_class_ids']})")
    print(f"Inference Time:   {metrics['inference_time_ms']} ms ({metrics['inference_time_ms']/1000:.2f} s)")
    print(f"Hard Mask Pixels: {metrics['hard_mask_pixels']} ({metrics['hard_coverage_pct']}%)")
    print(f"Soft Transition:  {metrics['soft_alpha_transition_pixels']} px")
    print(f"Bounding Box:     {metrics['hard_bounding_box']}")


if __name__ == "__main__":
    main()
