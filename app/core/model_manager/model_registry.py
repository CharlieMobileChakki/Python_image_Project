"""Model registry defining identifiers, checkpoint paths, and status probes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# HuggingFace cache config
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

FLOW_A_MODEL_NAME = os.getenv("REMBG_MODEL", "birefnet-general-lite").strip()
FLOW_B_MODEL_ID = "mattmdjaga/segformer_b2_clothes"

LAMA_MODEL_PATH = Path(
    r"D:\huggingface_cache\hub\models--opencv--inpainting_lama\snapshots\aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f\inpainting_lama_2025jan.onnx"
)

