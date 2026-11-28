"""Pydantic schemas for Arangtik Garment AI API request and response validation."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class GarmentItem(BaseModel):
    """Clean production metadata for a generated 2D studio product image."""
    category: str = Field(default="product", description="Product category (e.g. upper_clothes, pants, dress, necklace)")
    title: str = Field(default="Studio Garment", description="Human-friendly product title/label")
    url: str = Field(..., description="Direct URL to the generated 1024x1024 white-background PNG")
    resolution: str = Field(default="1024x1024", description="Output canvas resolution")
    output_png_url: str | None = Field(default=None, description="Alias for url")
    recognized_label: str | None = Field(default=None, description="Alias for title")
    garment_id: str | None = Field(default=None, description="Garment identifier")
    output_png_base64: str | None = Field(default=None, description="Optional Base64 data URI")
    stand_type: str | None = Field(default=None, description="Display stand type for jewelry outputs")
    presentation_mode: str | None = Field(default="ghost_mannequin", description="Presentation style")
    detected_item_count: int | None = Field(default=1, description="Number of physical product items verified")
    validation_metrics: dict[str, Any] | None = Field(default=None, description="Output quality metrics (Delta-E, count match, sharpness)")
    engine: str | None = Field(default=None, description="GPU Engine used for synthesis")
    gpu_name: str | None = Field(default=None, description="Name of the GPU used")
    views_analyzed: int | None = Field(default=1, description="Number of reference photos analyzed")


class ProcessGarmentResponse(BaseModel):
    """Clean production response schema for API callers and UI."""
    status: str = Field(default="success", description="Status string: 'success'")
    total_garments: int = Field(default=1, description="Count of generated studio images")
    garments: list[GarmentItem] = Field(default_factory=list, description="List of generated product images")
    processing_time_sec: float = Field(default=0.0, description="Total end-to-end processing time in seconds")
    job_id: str | None = Field(default=None, description="Job identifier")
    success: bool | None = Field(default=True, description="Success flag")
    message: str | None = Field(default=None, description="Status message")
    image: str | None = Field(default=None, description="Base64 data URI if requested")
    execution_time_ms: float | None = Field(default=None, description="Execution time in milliseconds")
    mode: str | None = Field(default=None, description="Pipeline execution mode")
    diagnostics: dict[str, Any] | None = Field(default=None, description="Pipeline diagnostics")


class DownstreamPackageRequest(BaseModel):
    """Request schema for packaging generated items for downstream forward."""
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
    flow: str = Field(default="Direct 1024x1024 Studio Pipeline")
    version: str = Field(default="4.1.0")
    model: str = Field(default="Qwen/Qwen-Image-2.1-Turbo")
    docs_url: str = Field(default="/docs")
    ui_url: str = Field(default="/ui")
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)


class TryOnSelectedItem(BaseModel):
    """Forensic styling description for a garment or accessory selected in an outfit combination."""
    role: str = Field(..., description="Item role: top, bottom, full_outfit, outerwear, or accessory")
    index: int = Field(..., description="0-based upload index within its input group")
    label: str = Field(..., description="Human-readable item title")
    details: str = Field(..., description="Forensic design, pattern, fabric, or material breakdown")
    body_placement: str | None = Field(default=None, description="Wearing placement for accessories")
    is_complementary: bool = Field(default=False, description="True if item was AI-added as a complementary modest piece")


class TryOnCombination(BaseModel):
    """One curated and generated virtual try-on outfit look."""
    combination_index: int
    title: str
    occasion: str = "smart_casual"
    style_reason: str
    url: str
    resolution: str = "768x1024"
    output_png_url: str | None = None
    output_png_base64: str | None = None
    selected_garment_indices: list[int] = Field(default_factory=list)
    selected_accessory_indices: list[int] = Field(default_factory=list)
    garment_details: list[TryOnSelectedItem] = Field(default_factory=list)
    accessory_details: list[TryOnSelectedItem] = Field(default_factory=list)
    avatar_description: str | None = None
    prompt_used: str | None = None
    engine: str | None = None
    gpu_name: str | None = None
    photography_style: str | None = None
    pose_preference: str | None = None


class VirtualTryOnResponse(BaseModel):
    """Response schema for multi-garment and accessory virtual try-on generation."""
    status: str = Field(default="success")
    success: bool = Field(default=True)
    job_id: str
    image_url: str | None = Field(default=None, description="Direct URL of the generated virtual try-on look")
    output_png_url: str | None = Field(default=None, description="Direct URL alias for image_url")
    total_combinations: int = 1
    combinations: list[TryOnCombination] = Field(default_factory=list)
    garment_catalog: list[dict[str, Any]] = Field(default_factory=list)
    accessory_catalog: list[dict[str, Any]] = Field(default_factory=list)
    processing_time_sec: float = 0.0
    execution_time_ms: float = 0.0
    mode: str = "virtual_tryon_stylist"
    diagnostics: dict[str, Any] = Field(default_factory=dict)

