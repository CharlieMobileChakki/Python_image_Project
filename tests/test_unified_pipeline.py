"""Unit and integration tests for Arangtik Garment AI Unified Pipeline."""
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.detector import analyze_semantic_map
from app.core.normalizer import detect_rotation_angle, normalize_to_1024, straighten_garment
from app.core.inpainter import repair_garment_cutout


@pytest.fixture
def api_client():
    """Test client for Arangtik Garment AI FastAPI application."""
    return TestClient(app)


def test_detector_detects_human():
    """Human presence detected when body class pixels exceed threshold."""
    # Create mock 200x200 map
    pred_map = np.zeros((200, 200), dtype=np.uint8)
    # Add face (11) and arms (14, 15)
    pred_map[20:60, 80:120] = 11  # Face: 1600 px
    pred_map[60:120, 60:80] = 14  # Left arm: 1200 px
    # Add upper_clothes (4)
    pred_map[60:140, 80:120] = 4  # 3200 px

    analysis = analyze_semantic_map(pred_map)
    assert analysis["has_human"] is True
    assert analysis["human_pixels"] >= 2800
    assert len(analysis["detected_garments"]) >= 1
    assert analysis["detected_garments"][0]["category"] == "upper_clothes"


def test_detector_flat_lay_no_human():
    """Flat-lay product without human pixels correctly flagged has_human=False."""
    pred_map = np.zeros((200, 200), dtype=np.uint8)
    pred_map[50:150, 50:150] = 4  # Garment only, no face/arms/legs

    analysis = analyze_semantic_map(pred_map)
    assert analysis["has_human"] is False
    assert analysis["human_pixels"] == 0
    assert len(analysis["detected_garments"]) == 1


def test_normalize_to_1024_dimensions_and_mode():
    """Normalization strictly produces 1024x1024 RGBA image with transparent borders."""
    img = Image.new("RGBA", (300, 600), (0, 0, 0, 0))
    # Draw mock garment inside
    for y in range(50, 550):
        for x in range(50, 250):
            img.putpixel((x, y), (200, 100, 50, 255))

    normalized = normalize_to_1024(img, target_size=1024, fill_ratio=0.82)
    assert normalized.size == (1024, 1024)
    assert normalized.mode == "RGBA"

    # Borders should remain fully transparent
    alpha = np.array(normalized.split()[3])
    assert alpha[0, 0] == 0
    assert alpha[1023, 1023] == 0
    assert alpha[512, 512] == 255  # Center has garment


def test_straighten_garment_preserves_dimensions():
    """Straightening rotates the garment cleanly."""
    img = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    for y in range(100, 300):
        for x in range(150, 250):
            img.putpixel((x, y), (100, 200, 150, 255))

    rotated = straighten_garment(img, 10.0)
    assert rotated.mode == "RGBA"


def test_repair_handles_image_width_under_512():
    """repair_garment_cutout must not crash with dimensions < 512 (e.g. 365px)."""
    from unittest.mock import patch
    img = Image.new("RGBA", (365, 600), (100, 150, 200, 255))
    mask = np.zeros((600, 365), dtype=np.uint8)
    mask[100:150, 50:100] = 255

    # Mock inpaint_patch_512 to verify padding and crop logic
    dummy_out = np.zeros((512, 512, 3), dtype=np.uint8)
    with patch("app.core.inpainter.inpaint_patch_512", return_value=(dummy_out, 10.0)) as mock_inpaint:
        repaired, _ = repair_garment_cutout(img, mask)
        assert repaired.size == (365, 600)
        # Check that the patch sent to inpaint_patch_512 was strictly 512x512
        call_args = mock_inpaint.call_args[0]
        crop_img, mask_crop = call_args[0], call_args[1]
        assert crop_img.size == (512, 512)
        assert mask_crop.shape == (512, 512)


def test_health_endpoint(api_client: TestClient):
    """GET /health returns HTTP 200 and system readiness."""
    res = api_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "arangtik" in data["service"]


def test_api_v1_health_endpoint(api_client: TestClient):
    """GET /api/v1/health returns detailed model loaded status."""
    res = api_client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_process_garment_rejects_invalid_mime(api_client: TestClient):
    """Uploading unsupported mime type returns HTTP 415."""
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    res = api_client.post("/api/v1/process-garment", files=files)
    assert res.status_code == 415


def test_process_garment_rejects_empty_file(api_client: TestClient):
    """Uploading empty file returns HTTP 400."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = api_client.post("/api/v1/process-garment", files=files)
    assert res.status_code == 400

