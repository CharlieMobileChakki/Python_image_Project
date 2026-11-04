"""Pydantic schemas for the Qwen-Image-2.1 Garment Generation API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class QwenGarmentGenerationRequest(BaseModel):
    """Configuration options for Qwen-Image-2.1 Garment Generation."""
    prompt: Optional[str] = Field(
        None,
        description="Optional guiding instruction. If omitted, the default garment isolation prompt is used."
    )
    resolution: int = Field(
        1024,
        ge=512,
        le=2048,
        description="Target output resolution side length in pixels (default: 1024)."
    )
    inference_steps: int = Field(
        28,
        ge=8,
        le=50,
        description="Number of denoising inference steps (default: 28, matching official demo)."
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible generation."
    )
    transparent_background: bool = Field(
        True,
        description="Whether to generate an isolated garment with native transparent RGBA background."
    )


class QwenGarmentGenerationResponse(BaseModel):
    """Response payload for Qwen-Image-2.1 Garment Generation."""
    success: bool = Field(..., description="Whether generation succeeded.")
    status: str = Field(..., description="Status summary ('success', 'pending_gpu_testing', 'failed').")
    model_name: str = Field("Qwen/Qwen-Image-2.1", description="Model checkpoint used.")
    image_format: str = Field("PNG", description="Output image encoding format.")
    image_width: int = Field(..., description="Width of generated image in pixels.")
    image_height: int = Field(..., description="Height of generated image in pixels.")
    has_alpha_channel: bool = Field(..., description="Whether output has an alpha transparency channel (RGBA).")
    inference_time_seconds: float = Field(..., description="Model inference duration in seconds.")
    gpu_memory_peak_mb: Optional[float] = Field(None, description="Peak VRAM allocated during inference in MB.")
    output_png_url: Optional[str] = Field(None, description="Direct download URL for generated PNG.")
    output_png_base64: Optional[str] = Field(None, description="Base64 data URI of the generated PNG.")
    error_detail: Optional[str] = Field(None, description="Error or environment blocker explanation.")

