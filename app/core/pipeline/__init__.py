"""Pipeline package: End-to-end processing and routing."""
from __future__ import annotations

from app.core.image_processing import image_to_base64_data_uri
from app.core.inpainting import compute_arm_occlusion_mask
from app.core.pipeline.pipeline_config import OUTPUT_DIR
from app.core.pipeline.pipeline_router import (
    ExecutionFlow,
    determine_pipeline_execution_mode,
    resolve_execution_flow,
)
from app.core.pipeline.unified_pipeline import (
    execute_flow_a_without_person,
    execute_flow_b_with_person,
    process_garment_pipeline,
)

__all__ = [
    "OUTPUT_DIR",
    "ExecutionFlow",
    "image_to_base64_data_uri",
    "compute_arm_occlusion_mask",
    "determine_pipeline_execution_mode",
    "resolve_execution_flow",
    "execute_flow_a_without_person",
    "execute_flow_b_with_person",
    "process_garment_pipeline",
]
