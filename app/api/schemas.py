"""Pydantic schemas for Arangtik Garment AI API request and response validation."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class GarmentItem(BaseModel):
    """Metadata for a single extracted and normalized 2D garment."""
    garment_id: str = Field(..., description="Unique identifier for this garment in the request (e.g. garment_001)")
    category: str = Field(..., description="Canonical category (e.g. upper_clothes, pants, dress, skirt, saree, lehenga)")
    recognized_label: str = Field(..., description="Human-friendly label (e.g. T-Shirt, Jeans, Saree, Anarkali)")
    image: str | None = Field(default=None, description="Filename of extracted garment (e.g. garment_001.png)")
    confidence: float = Field(default=1.0, description="Segmentation confidence score (0.0 to 1.0)")
    coverage_percentage: float | None = Field(default=None, description="Percentage of outfit area occupied by this item")
    priority: str | None = Field(default=None, description="Garment priority: first_priority, second_priority, low_priority, ignore_below")
    is_fused: bool = Field(default=False, description="Whether multiple semantic classes were fused into this complete ensemble")
    rotation_angle_deg: float = Field(default=0.0, description="Angle in degrees garment was straightened by")
    repaired: bool = Field(default=False, description="Whether fabric inpainting was applied to fix occlusions")
    repair_method: str | None = Field(default=None, description="Inpainting model used (e.g. lama_ffc_onnx)")
    width: int = Field(default=1024, description="Canvas width in pixels")
    height: int = Field(default=1024, description="Canvas height in pixels")
    image_dimensions: list[int] = Field(default_factory=lambda: [1024, 1024], description="Width and Height [1024, 1024]")
    output_png_url: str = Field(..., description="Download/preview URL for the transparent 1024x1024 PNG")
    output_png_base64: str | None = Field(default=None, description="Optional Base64 data URI for direct client embed")
    # Quality validation fields
    background_removal_status: str = Field(default="success", description="Status of background removal: success, partial, failed, needs_review")
    background_removal_confidence: float = Field(default=0.98, description="Confidence of background removal (0.0 to 1.0)")
    garment_preservation_status: str = Field(default="verified", description="Garment preservation status: verified, needs_review")
    validation_errors: list[str] = Field(default_factory=list, description="Validation issues or warnings detected")


class ProcessGarmentResponse(BaseModel):
    """Master Unified Pipeline response schema."""
    success: bool = Field(default=True, description="Overall pipeline status boolean")
    status: str = Field(default="success", description="Status string: success or error")
    mode: str = Field(..., description="Processing mode: 'multi_garment', 'human_worn', 'mannequin', or 'isolated_product'")
    has_human: bool = Field(..., description="True if a human model was detected in the photo")
    scene_type: str | None = Field(default=None, description="Scene category: 'multi_garment', 'person', 'mannequin', or 'flat_lay'")
    outfit_mode: str | None = Field(default=None, description="Classification mode: 'mode_a_independent', 'mode_b_single_piece', or 'mode_c_traditional_ensemble'")
    execution_time_ms: float = Field(..., description="Total processing time in milliseconds")
    total_detected: int = Field(default=1, description="Count of detected garments in source image")
    total_extracted: int = Field(default=1, description="Count of successfully extracted garments")
    total_background_removed: int = Field(default=1, description="Count of garments with background removed cleanly")
    total_partial: int = Field(default=0, description="Count of garments with partial background removal")
    total_failed: int = Field(default=0, description="Count of garments with failed background removal")
    garments_found: int = Field(..., description="Count of garments extracted and normalized")
    garments: list[GarmentItem] = Field(default_factory=list, description="List of extracted clean 2D garments")
    diagnostics: dict[str, Any] = Field(default_factory=dict, description="Diagnostic telemetry and timings")


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
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)

