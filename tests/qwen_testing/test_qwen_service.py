"""Tests for Qwen-Image-2.1 Garment Generation Service and API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from io import BytesIO
from pathlib import Path
import pytest
from PIL import Image

from app.experimental.qwen_generation import (
    DEFAULT_GARMENT_SOLID_PROMPT,
    DEFAULT_GARMENT_TRANSPARENT_PROMPT,
    OFFICIAL_MODEL_ID,
    QwenGarmentGenerationEngine,
    get_qwen_garment_engine,
)
from app.schemas.qwen_garment_generation import (
    QwenGarmentGenerationRequest,
    QwenGarmentGenerationResponse,
)


def test_qwen_engine_defaults():
    """Verify official model ID, default parameters, and lazy loading."""
    engine = QwenGarmentGenerationEngine()
    assert engine.model_id == OFFICIAL_MODEL_ID
    assert engine.configured_device == "cuda"
    assert engine.dtype_str == "bfloat16"
    assert engine.is_loaded() is False  # Lazy loaded


def test_qwen_prompts_transparency_keywords():
    """Verify that the transparency prompt explicitly contains official trigger phrases."""
    assert "This is an RGBA image with transparency" in DEFAULT_GARMENT_TRANSPARENT_PROMPT
    assert "The image has an alpha channel and the background is transparent" in DEFAULT_GARMENT_TRANSPARENT_PROMPT
    assert "Remove the person" in DEFAULT_GARMENT_TRANSPARENT_PROMPT
    assert "reconstruct" in DEFAULT_GARMENT_TRANSPARENT_PROMPT.lower()


def test_qwen_engine_availability_on_current_host():
    """Verify that is_available cleanly flags CUDA absence without throwing unhandled exceptions."""
    engine = get_qwen_garment_engine()
    is_avail, reason = engine.is_available()
    # On a dev CPU machine, this should return False with a descriptive message
    assert isinstance(is_avail, bool)
    assert isinstance(reason, str)
    if not is_avail:
        assert "CUDA" in reason or "diffusers" in reason or "GPU" in reason


def test_qwen_generate_safe_fallback_on_unsupported_host():
    """Verify that generate() returns 'pending_gpu_testing' when GPU/CUDA is absent without hanging."""
    engine = get_qwen_garment_engine()
    test_img = Image.new("RGB", (512, 512), (150, 50, 100))
    result = engine.generate(
        reference_image=test_img,
        resolution=512,
        inference_steps=8,
        seed=123,
        transparent_background=True,
    )
    assert result.status == "pending_gpu_testing"
    assert result.has_alpha_channel is True
    assert result.width == 512
    assert result.height == 512
    assert result.error_detail is not None


def test_qwen_schemas_validation():
    """Verify request and response Pydantic schemas."""
    req = QwenGarmentGenerationRequest(
        resolution=1024,
        inference_steps=28,
        seed=42,
        transparent_background=True,
    )
    assert req.resolution == 1024
    assert req.inference_steps == 28
    assert req.seed == 42
    assert req.transparent_background is True

    resp = QwenGarmentGenerationResponse(
        success=True,
        status="success",
        model_name="Qwen/Qwen-Image-2.1",
        image_format="PNG",
        image_width=1024,
        image_height=1024,
        has_alpha_channel=True,
        inference_time_seconds=12.4,
        gpu_memory_peak_mb=14500.0,
        output_png_url="/api/v1/outputs/test_garment.png",
    )
    assert resp.success is True
    assert resp.model_name == "Qwen/Qwen-Image-2.1"
    assert resp.has_alpha_channel is True
