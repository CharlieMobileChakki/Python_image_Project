"""Pydantic schemas for Arangtik Garment AI API request and response validation."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class GarmentItem(BaseModel):
    """Metadata for a single extracted and normalized 2D garment."""
    garment_id: str = Field(..., description="Unique identifier for this garment in the request")
    category: str = Field(..., description="Canonical category (e.g. upper_clothes, pants, dress, skirt)")
    recognized_label: str = Field(..., description="Human-friendly label (e.g. T-Shirt, Jeans, Anarkali)")
    confidence: float = Field(default=1.0, description="Segmentation confidence score (0.0 to 1.0)")
    rotation_angle_deg: float = Field(default=0.0, description="Angle in degrees garment was straightened by")
    repaired: bool = Field(default=False, description="Whether fabric inpainting was applied to fix occlusions")
    repair_method: str | None = Field(default=None, description="Inpainting model used (e.g. lama_ffc_onnx)")
    image_dimensions: list[int] = Field(default_factory=lambda: [1024, 1024], description="Width and Height [1024, 1024]")
    output_png_url: str = Field(..., description="Download/preview URL for the transparent 1024x1024 PNG")
    output_png_base64: str | None = Field(default=None, description="Optional Base64 data URI for direct client embed")


class ProcessGarmentResponse(BaseModel):
    """Master Unified Pipeline response schema."""
    status: str = Field(default="success", description="Status string: success or error")
    mode: str = Field(..., description="Processing mode: 'human_worn' or 'isolated_product'")
    has_human: bool = Field(..., description="True if a human model was detected in the photo")
    execution_time_ms: float = Field(..., description="Total processing time in milliseconds")
    garments_found: int = Field(..., description="Count of garments extracted and normalized")
    garments: list[GarmentItem] = Field(default_factory=list, description="List of extracted clean 2D garments")
    diagnostics: dict[str, Any] = Field(default_factory=dict, description="Diagnostic telemetry and timings")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="ok")
    service: str = Field(default="arangtik-garment-ai")
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)

