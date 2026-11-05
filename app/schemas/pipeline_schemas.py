"""Pydantic schemas for Arangtik Garment AI API request and response validation."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class GarmentItem(BaseModel):
    """Clean production metadata for a single extracted and normalized 2D garment."""
    output_png_url: str = Field(..., description="Download/preview URL for the transparent 1024x1024 PNG")
    width: int = Field(default=1024, description="Canvas width in pixels")
    height: int = Field(default=1024, description="Canvas height in pixels")
    category: str = Field(default="product", description="Garment category (e.g. product, upper_clothes, pants, dress)")
    recognized_label: str = Field(default="Garment Cutout", description="Human-friendly label")
    output_png_base64: str | None = Field(default=None, description="Optional Base64 data URI for direct client embed")


class ProcessGarmentResponse(BaseModel):
    """Clean production response schema with only essential fields."""
    success: bool = Field(default=True, description="Overall pipeline status boolean")
    job_id: str = Field(..., description="Unique job identifier")
    execution_time_ms: float = Field(..., description="Total processing time in milliseconds")
    total_garments: int = Field(default=1, description="Count of extracted clean garments")
    garments: list[GarmentItem] = Field(default_factory=list, description="List of extracted clean 2D garments")


class DownstreamPackageRequest(BaseModel):
    """Request schema for packaging extracted garments for downstream forward."""
    job_id: str | None = None
    garment_ids: list[str] = Field(default_factory=list)
    target_api_url: str | None = None


class DownstreamPackageResponse(BaseModel):
    """Response schema for downstream export package."""
    success: bool = True
    total_garments: int
    package_type: str = "batch_array"
    garments: list[GarmentItem]


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="ok")
    service: str = Field(default="arangtik-garment-ai")
    flow: str = Field(default="Unified Pipeline")
    version: str = Field(default="1.0.0")
    model: str = Field(default="birefnet-lite + segformer-b2 + lama-ffc")
    docs_url: str = Field(default="/docs")
    ui_url: str = Field(default="/ui")
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)

