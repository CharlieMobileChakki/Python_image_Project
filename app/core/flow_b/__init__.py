"""Flow B subpackage: ChatGPT Vision prompt generation + Remote RTX 3090 Qwen GPU client."""
from __future__ import annotations

from app.core.flow_b.prompt_generator import (
    COMMON_STUDIO_RULES,
    analyze_garment_and_build_prompts,
)
from app.core.flow_b.qwen_client import (
    check_gpu_worker_health,
    generate_garment_via_gpu,
    is_gpu_worker_configured,
)

__all__ = [
    "COMMON_STUDIO_RULES",
    "analyze_garment_and_build_prompts",
    "check_gpu_worker_health",
    "generate_garment_via_gpu",
    "is_gpu_worker_configured",
]

