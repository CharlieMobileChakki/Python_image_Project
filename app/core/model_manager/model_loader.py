"""Centralized model loader for BiRefNet Lite, SegFormer B2, and LaMa ONNX."""
from __future__ import annotations

import os
from typing import Any

from rembg import new_session
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import onnxruntime as ort

from app.core.model_manager.model_registry import (
    FLOW_A_MODEL_NAME,
    FLOW_B_MODEL_ID,
    LAMA_MODEL_PATH,
)

_flow_a_session: Any | None = None
_flow_b_processor: AutoImageProcessor | None = None
_flow_b_model: SegformerForSemanticSegmentation | None = None
_lama_session: ort.InferenceSession | None = None


def load_flow_a() -> None:
    """Preload Flow A (BiRefNet Lite) session."""
    global _flow_a_session
    if _flow_a_session is None:
        _flow_a_session = new_session(FLOW_A_MODEL_NAME)


def get_flow_a_session() -> Any:
    """Retrieve active Flow A session, loading if necessary."""
    global _flow_a_session
    if _flow_a_session is None:
        load_flow_a()
    return _flow_a_session


def load_flow_b() -> None:
    """Preload Flow B (SegFormer B2) model."""
    global _flow_b_processor, _flow_b_model
    if _flow_b_model is None or _flow_b_processor is None:
        try:
            _flow_b_processor = AutoImageProcessor.from_pretrained(FLOW_B_MODEL_ID, local_files_only=True)
            _flow_b_model = SegformerForSemanticSegmentation.from_pretrained(FLOW_B_MODEL_ID, local_files_only=True)
        except Exception:
            _flow_b_processor = AutoImageProcessor.from_pretrained(FLOW_B_MODEL_ID)
            _flow_b_model = SegformerForSemanticSegmentation.from_pretrained(FLOW_B_MODEL_ID)
        _flow_b_model.eval()


def get_flow_b_models() -> tuple[AutoImageProcessor, SegformerForSemanticSegmentation]:
    """Retrieve SegFormer processor and model, loading if necessary."""
    global _flow_b_processor, _flow_b_model
    if _flow_b_model is None or _flow_b_processor is None:
        load_flow_b()
    assert _flow_b_processor is not None and _flow_b_model is not None
    return _flow_b_processor, _flow_b_model


def load_segmenter_models() -> None:
    """Preload both segmentation models into memory."""
    load_flow_a()
    load_flow_b()


def are_models_loaded() -> dict[str, bool]:
    """Check readiness status of segmentation models."""
    return {
        "flow_a_birefnet": _flow_a_session is not None,
        "flow_b_segformer": _flow_b_model is not None,
    }


def load_inpainter_model() -> None:
    """Load LaMa ONNX session once into memory with optimized CPU threads."""
    global _lama_session
    if _lama_session is None:
        if not LAMA_MODEL_PATH.exists():
            raise FileNotFoundError(f"LaMa ONNX checkpoint not found at: {LAMA_MODEL_PATH}")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = os.cpu_count() or 4
        _lama_session = ort.InferenceSession(
            str(LAMA_MODEL_PATH), sess_opts, providers=["CPUExecutionProvider"]
        )


def get_inpainter_session() -> ort.InferenceSession:
    """Retrieve active LaMa ONNX session, loading if necessary."""
    global _lama_session
    if _lama_session is None:
        load_inpainter_model()
    assert _lama_session is not None
    return _lama_session


def is_inpainter_loaded() -> bool:
    """Return True if LaMa ONNX session is active."""
    return _lama_session is not None

