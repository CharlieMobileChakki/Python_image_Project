"""Flow B subpackage: ChatGPT Vision prompt generation + Remote RTX 3090 Qwen GPU client."""
from __future__ import annotations

from app.core.flow_b.prompt_generator import (
    COMMON_STUDIO_RULES,
    analyze_garment_and_build_prompts,
    analyze_jewelry_and_build_prompts,
    create_multi_angle_composite,
    is_jewelry_domain,
)
from app.core.flow_b.qwen_client import (
    check_gpu_worker_health,
    generate_garment_via_gpu,
    is_gpu_worker_configured,
    remove_background_via_gpu,
)

__all__ = [
    "COMMON_STUDIO_RULES",
    "analyze_garment_and_build_prompts",
    "analyze_jewelry_and_build_prompts",
    "check_gpu_worker_health",
    "create_multi_angle_composite",
    "generate_garment_via_gpu",
    "is_gpu_worker_configured",
    "is_jewelry_domain",
    "remove_background_via_gpu",
]

