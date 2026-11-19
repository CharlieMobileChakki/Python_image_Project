"""Model registry defining identifiers and cache paths for Flow A (BiRefNet Lite)."""
from __future__ import annotations

import os

FLOW_A_MODEL_NAME = os.getenv("REMBG_MODEL", "birefnet-general-lite").strip()
