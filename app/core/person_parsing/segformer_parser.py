"""SegFormer B2 Clothes forward pass and probability upsampling."""
from __future__ import annotations

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from app.core.model_manager.model_loader import get_flow_b_models


def run_segformer_inference(
    image_rgb: Image.Image,
    fg_alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    """Execute Flow B SegFormer forward pass on person/foreground part.

    Args:
        image_rgb: PIL RGB image of the extracted person / foreground.
        fg_alpha: Optional 2D uint8 mask of the foreground from BiRefNet Lite.
                  When provided, background pixels (fg_alpha < 20) are strictly zeroed out
                  so background clutter cannot contaminate person parsing.

    Returns:
        tuple(pred_map_2d_uint8, prob_tensor_18_classes_float32)
    """
    processor, model = get_flow_b_models()
    orig_w, orig_h = image_rgb.size
    inputs = processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Upsample logits to original image resolution
        upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        probs = F.softmax(upsampled, dim=1).squeeze(0)
        pred_map = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

    probs_cpu = probs.cpu()

    # Enforce background boundary if BiRefNet Lite foreground alpha is provided
    if fg_alpha is not None:
        bg_pixels = (fg_alpha < 20)
        if bg_pixels.any():
            pred_map[bg_pixels] = 0
            # Set background prob to 1.0 and all other classes to 0.0 for true background
            probs_cpu[0, bg_pixels] = 1.0
            probs_cpu[1:, bg_pixels] = 0.0

    return pred_map, probs_cpu

