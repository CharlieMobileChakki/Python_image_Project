"""Pydantic schemas for AI Stylized Avatar Studio API."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class AvatarIdentityFeatures(BaseModel):
    """Forensic identity and visual descriptors extracted from user photo."""
    gender: str = Field(default="unspecified", description="Gender / presentation (female, male, etc.)")
    expression: str = Field(default="friendly smile", description="Facial expression and smile description")
    teeth_visible: bool = Field(default=False, description="Whether teeth are visibly exposed in the photo")
    hair_details: str = Field(default="", description="Hairstyle, length, color, volume, and texture")
    eyewear: str | None = Field(default=None, description="Eyeglasses details if present or None")
    facial_hair: str | None = Field(default=None, description="Facial hair/mustache/stubble if present or None")
    skin_tone_and_complexion: str = Field(default="", description="Skin tone and complexion coordinates")
    upper_garment: str = Field(default="", description="Stylized upper clothing description")
    distinctive_features: str | None = Field(default=None, description="Bindi, jewelry, or unique facial marks")


class AvatarGenerationResponse(BaseModel):
    """Production response schema for AI Stylized Avatar Profile Picture Generator."""
    status: str = Field(default="success", description="Status string: 'success'")
    success: bool = Field(default=True, description="Success boolean flag")
    job_id: str = Field(..., description="Unique job identifier")
    avatar_url: str = Field(..., description="Direct URL to generated 1024x1024 stylized avatar PNG")
    output_png_url: str | None = Field(default=None, description="Alias for avatar_url")
    avatar_base64: str | None = Field(default=None, description="Base64 data URI if requested")
    style: str = Field(default="vector_art", description="Artistic style used (vector_art, digital_illustration, etc.)")
    halo_color: str = Field(default="soft_blue", description="Pastel circular backdrop halo color")
    doodle_accent: str = Field(default="crown", description="Minimalist doodle accent (crown, heart, sparks, none)")
    identity_features: AvatarIdentityFeatures | None = Field(default=None, description="Extracted visual and outfit identity features")
    resolution: str = Field(default="1024x1024", description="Output resolution")
    processing_time_sec: float = Field(default=0.0, description="Total pipeline latency in seconds")
    execution_time_ms: float = Field(default=0.0, description="Total pipeline latency in milliseconds")
    prompt_used: str | None = Field(default=None, description="Prompt passed to generative engine")
    engine: str | None = Field(default=None, description="GPU engine used")
    gpu_name: str | None = Field(default=None, description="GPU hardware identifier")
    diagnostics: dict[str, Any] = Field(default_factory=dict, description="Pipeline diagnostic metadata")
