"""Pydantic schemas for Arangtik Garment AI API request and response validation."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class GarmentItem(BaseModel):
    """Clean production metadata for a single extracted and normalized 2D garment."""
    category: str = Field(default="product", description="Garment category (e.g. upper_clothes, pants, dress)")
    title: str = Field(default="Garment Cutout", description="Human-friendly garment title/label")
    url: str = Field(..., description="Direct URL to the 1024x1024 transparent PNG")
    resolution: str = Field(default="1024x1024", description="Output canvas resolution")
    # Optional aliases for backwards compatibility
    output_png_url: str | None = Field(default=None, description="Alias for url")
    recognized_label: str | None = Field(default=None, description="Alias for title")
    garment_id: str | None = Field(default=None, description="Garment identifier")
    output_png_base64: str | None = Field(default=None, description="Optional Base64 data URI")
    stand_type: str | None = Field(default=None, description="Display stand type (e.g. white_neck_bust, slotted_ring_block, t_bar_earring_stand)")
    views_analyzed: int | None = Field(default=1, description="Number of multi-angle reference photos analyzed")


class ProcessGarmentResponse(BaseModel):
    """Clean, lightweight production response schema for Node.js and client integration."""
    status: str = Field(default="success", description="Status string: 'success'")
    total_garments: int = Field(default=1, description="Count of extracted clean garments")
    garments: list[GarmentItem] = Field(default_factory=list, description="List of extracted garments")
    processing_time_sec: float = Field(default=0.0, description="Total end-to-end processing time in seconds")
    # Optional fields for backwards compatibility
    job_id: str | None = Field(default=None, description="Job identifier")
    success: bool | None = Field(default=True, description="Success flag")
    message: str | None = Field(default=None, description="Status message")
    image: str | None = Field(default=None, description="Base64 data URI if requested")


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

