"""Pipeline router: Determines execution flow (Flow A vs Flow B, Multi-garment vs Single)."""
from __future__ import annotations

from typing import Any


def determine_pipeline_execution_mode(
    has_human: bool,
    scene_type: str,
    is_multi: bool,
    num_multi_comps: int,
) -> str:
    """Determine the high-level pipeline execution strategy.

    Returns:
        "multi_garment" | "human_worn" | "mannequin" | "isolated_product"
    """
    if is_multi and num_multi_comps >= 2 and not has_human:
        return "multi_garment"
    if scene_type == "person" or has_human:
        return "human_worn"
    if scene_type == "mannequin":
        return "mannequin"
    return "isolated_product"

