"""Model manager subpackage: Centralized model lifecycle, loading, and registry."""
from app.core.model_manager.model_registry import (
    FLOW_A_MODEL_NAME,
    FLOW_B_MODEL_ID,
    LAMA_MODEL_PATH,
)
from app.core.model_manager.model_loader import (
    are_models_loaded,
    get_flow_a_session,
    get_flow_b_models,
    get_inpainter_session,
    is_inpainter_loaded,
    load_flow_a,
    load_flow_b,
    load_inpainter_model,
    load_segmenter_models,
)

__all__ = [
    "FLOW_A_MODEL_NAME",
    "FLOW_B_MODEL_ID",
    "LAMA_MODEL_PATH",
    "are_models_loaded",
    "get_flow_a_session",
    "get_flow_b_models",
    "get_inpainter_session",
    "is_inpainter_loaded",
    "load_flow_a",
    "load_flow_b",
    "load_inpainter_model",
    "load_segmenter_models",
]

