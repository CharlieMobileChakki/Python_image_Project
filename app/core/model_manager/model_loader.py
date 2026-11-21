"""Centralized model loader for Flow A (BiRefNet Lite)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from rembg import new_session

from app.core.model_manager.model_registry import FLOW_A_MODEL_NAME

_flow_a_session: Any | None = None


def _ensure_rembg_home() -> None:
    """Ensure rembg uses workspace .models if available and REMBG_HOME is unset."""
    if not os.getenv("REMBG_HOME"):
        workspace_models = Path(__file__).resolve().parent.parent.parent.parent / ".models"
        if workspace_models.exists():
            os.environ["REMBG_HOME"] = str(workspace_models)


def load_flow_a() -> None:
    """Preload Flow A (BiRefNet Lite) session."""
    global _flow_a_session
    if _flow_a_session is None:
        _ensure_rembg_home()
        _flow_a_session = new_session(FLOW_A_MODEL_NAME)


def get_flow_a_session() -> Any:
    """Retrieve active Flow A session, loading if necessary."""
    global _flow_a_session
    if _flow_a_session is None:
        load_flow_a()
    return _flow_a_session


def are_models_loaded() -> dict[str, bool]:
    """Check readiness status of local models."""
    return {
        "flow_a_birefnet": _flow_a_session is not None,
    }
