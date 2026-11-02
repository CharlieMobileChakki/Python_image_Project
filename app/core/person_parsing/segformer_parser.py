"""SegFormer B2 Clothes forward pass and probability upsampling."""
from __future__ import annotations

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from app.core.model_manager.model_loader import get_flow_b_models


def run_segformer_inference(image_rgb: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    """Execute Flow B SegFormer forward pass.

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

    return pred_map, probs.cpu()

