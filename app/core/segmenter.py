"""Compatibility shim for segmenter module. Re-exports from model_manager, background_removal, person_parsing, and garment_extraction."""
from __future__ import annotations

from app.core.background_removal import extract_isolated_flow_a
from app.core.garment_extraction import (
    extract_garment_cutout_flow_b,
    extract_garment_cutout_guided,
)
from app.core.model_manager import (
    FLOW_A_MODEL_NAME,
    FLOW_B_MODEL_ID,
    are_models_loaded,
    get_flow_a_session,
    get_flow_b_models,
    load_flow_a,
    load_flow_b,
    load_segmenter_models,
)
from app.core.person_parsing import (
    extract_arms_mask,
    extract_exposed_skin_mask,
    extract_hair_mask,
    run_segformer_inference,
)

__all__ = [
    "FLOW_A_MODEL_NAME",
    "FLOW_B_MODEL_ID",
    "load_flow_a",
    "load_flow_b",
    "load_segmenter_models",
    "get_flow_a_session",
    "get_flow_b_models",
    "are_models_loaded",
    "extract_isolated_flow_a",
    "run_segformer_inference",
    "extract_arms_mask",
    "extract_hair_mask",
    "extract_exposed_skin_mask",
    "extract_garment_cutout_flow_b",
    "extract_garment_cutout_guided",
]
