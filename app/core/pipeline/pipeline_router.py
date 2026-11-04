"""Pipeline router: Resolves execution flow between Flow A (Without Person) and Flow B (With Person)."""
from __future__ import annotations

from enum import Enum
from typing import Any


class ExecutionFlow(str, Enum):
    """Supported pipeline execution flows."""
    FLOW_A_WITHOUT_PERSON = "without_person"
    FLOW_B_WITH_PERSON = "with_person"


def resolve_execution_flow(
    person_mode: str | None = None,
    has_human: bool | None = None,
    mode_hint: str | None = None,
) -> str:
    """Determine whether incoming data routes to Flow A (Without Person) or Flow B (With Person).

    Flow Routing Rules:
    - Flow A (Without Person): Flat-lay, ghost mannequin, hanger, or catalog product shot.
      Direct background removal via BiRefNet Lite with zero human-parsing overhead,
      normalized to a clean 1024x1024 transparent canvas.
    - Flow B (With Person): Garment worn on a human model.
      Background removal -> SegFormer B2 clothes parsing -> Garment extraction & boundary
      snapping -> LaMa FFC inpainting for fabric occlusions -> Clean 1024x1024 transparent canvas.

    Args:
        person_mode: Explicit mode string: 'with_person' or 'without_person'.
        has_human: Optional explicit boolean flag from incoming payload.
        mode_hint: Optional legacy parameter ('worn', 'isolated', 'auto').

    Returns:
        'without_person' (Flow A) or 'with_person' (Flow B).
    """
    # 1. Direct boolean flag if explicitly provided
    if has_human is False:
        return ExecutionFlow.FLOW_A_WITHOUT_PERSON.value
    if has_human is True:
        return ExecutionFlow.FLOW_B_WITH_PERSON.value

    # 2. String representation inspection
    raw_mode = (person_mode or mode_hint or "").strip().lower()

    # Flow A keywords (Without Person)
    if raw_mode in {
        "without_person",
        "without",
        "no_person",
        "no_human",
        "without_human",
        "isolated",
        "flat_lay",
        "product",
        "flow_a",
        "false",
        "0",
    }:
        return ExecutionFlow.FLOW_A_WITHOUT_PERSON.value

    # Flow B keywords (With Person)
    if raw_mode in {
        "with_person",
        "with",
        "person",
        "human",
        "worn",
        "model",
        "flow_b",
        "true",
        "1",
    }:
        return ExecutionFlow.FLOW_B_WITH_PERSON.value

    # Default fallback when unspecified
    return ExecutionFlow.FLOW_B_WITH_PERSON.value


def determine_pipeline_execution_mode(
    has_human: bool,
    scene_type: str,
    is_multi: bool,
    num_multi_comps: int,
) -> str:
    """Legacy helper for granular downstream reporting.

    Returns:
        'multi_garment' | 'human_worn' | 'mannequin' | 'isolated_product'
    """
    if is_multi and num_multi_comps >= 2 and not has_human:
        return "multi_garment"
    if scene_type == "person" or has_human:
        return "human_worn"
    if scene_type == "mannequin":
        return "mannequin"
    return "isolated_product"


