"""Model registry defining identifiers and cache paths for Flow A."""
from __future__ import annotations

import os
from pathlib import Path

# Rembg U2NET cache directory (local workspace)
_default_u2net = Path(__file__).resolve().parent.parent.parent.parent / ".u2net"
if _default_u2net.exists():
    os.environ["U2NET_HOME"] = os.getenv("U2NET_HOME", str(_default_u2net))

FLOW_A_MODEL_NAME = os.getenv("REMBG_MODEL", "birefnet-general-lite").strip()
