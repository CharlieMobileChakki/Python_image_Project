"""Pipeline configuration and paths."""
from __future__ import annotations

from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

