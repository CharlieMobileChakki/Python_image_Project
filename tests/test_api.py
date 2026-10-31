from io import BytesIO
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, MAX_UPLOAD_BYTES, MODEL_NAME


def test_health_endpoint(client: TestClient):
    """GET /health returns HTTP 200 with service, flow, and configured model name."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "arangtik-garment-extraction"
    assert data["flow"] == "A"
    assert data["model"] == MODEL_NAME


def test_root_endpoint(client: TestClient):
    """GET / returns HTTP 200 with service and discovery links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "health" in data
    assert "docs" in data


def test_extract_invalid_mime_type_rejected(client: TestClient, sample_valid_jpeg_bytes: bytes):
    """Files with unsupported content types must return HTTP 415."""
    files = {"file": ("test.gif", sample_valid_jpeg_bytes, "image/gif")}
    response = client.post("/api/v1/extract-garment", files=files)
    assert response.status_code == 415
    assert "Supported image types" in response.json()["detail"]


def test_extract_empty_file_rejected(client: TestClient):
    """Empty uploads must return HTTP 400."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/v1/extract-garment", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_extract_oversized_file_rejected(client: TestClient):
    """Files exceeding MAX_UPLOAD_BYTES must return HTTP 413."""
    # Send MAX_UPLOAD_BYTES + 1 bytes of dummy data
    oversized_data = b"X" * (MAX_UPLOAD_BYTES + 1)
    files = {"file": ("huge.jpg", oversized_data, "image/jpeg")}
    response = client.post("/api/v1/extract-garment", files=files)
    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]


def test_extract_corrupt_image_rejected(client: TestClient):
    """Corrupt image bytes must be rejected with HTTP 422."""
    corrupt_data = b"This is not a real image format data block."
    files = {"file": ("corrupt.jpg", corrupt_data, "image/jpeg")}
    response = client.post("/api/v1/extract-garment", files=files)
    assert response.status_code == 422
    assert "not a valid/safe image" in response.json()["detail"]


def test_extract_successful_response_and_headers(
    client: TestClient,
    sample_valid_jpeg_bytes: bytes,
    sample_valid_rgba_png_bytes: bytes,
):
    """Successful extraction returns image/png and expected diagnostic headers."""
    mock_dims = (800, 600)
    with patch("app.main.extract_garment", return_value=(sample_valid_rgba_png_bytes, mock_dims)):
        files = {"file": ("garment.jpg", sample_valid_jpeg_bytes, "image/jpeg")}
        response = client.post("/api/v1/extract-garment", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-model-name"] == MODEL_NAME
    assert response.headers["x-original-width"] == "800"
    assert response.headers["x-original-height"] == "600"
    assert "x-processing-time-ms" in response.headers
    assert int(response.headers["x-processing-time-ms"]) >= 0


def test_extract_returns_valid_1024_rgba_png(
    client: TestClient,
    sample_valid_jpeg_bytes: bytes,
    sample_valid_rgba_png_bytes: bytes,
):
    """The returned binary content must be an RGBA 1024x1024 PNG with an alpha channel."""
    with patch("app.main.extract_garment", return_value=(sample_valid_rgba_png_bytes, (500, 500))):
        files = {"file": ("shirt.png", sample_valid_jpeg_bytes, "image/png")}
        response = client.post("/api/v1/extract-garment", files=files)

    assert response.status_code == 200
    img = Image.open(BytesIO(response.content))
    assert img.format == "PNG"
    assert img.size == (1024, 1024)
    assert img.mode == "RGBA"

    alpha = img.getchannel("A")
    extrema = alpha.getextrema()
    assert extrema[0] == 0  # has fully transparent pixels
    assert extrema[1] > 0   # has visible foreground pixels


def test_extract_model_runtime_error_handled(client: TestClient, sample_valid_jpeg_bytes: bytes):
    """Inference or detection failures raised in extract_garment return HTTP 422."""
    with patch("app.main.extract_garment", side_effect=RuntimeError("No foreground garment was detected.")):
        files = {"file": ("empty_fg.jpg", sample_valid_jpeg_bytes, "image/jpeg")}
        response = client.post("/api/v1/extract-garment", files=files)

    assert response.status_code == 422
    assert "No foreground garment was detected." in response.json()["detail"]


def test_lifespan_startup_failure_propagates():
    """If load_model fails during application startup, the lifespan raises the exception."""
    with patch("app.main.load_model", side_effect=RuntimeError("Failed to load ONNX model weights")):
        with pytest.raises(RuntimeError, match="Failed to load ONNX model weights"):
            with TestClient(app):
                pass

