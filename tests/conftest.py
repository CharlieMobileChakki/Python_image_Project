from io import BytesIO
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image

try:
    from app.main import app as flowa_app
except ImportError:
    flowa_app = None


@pytest.fixture
def mock_load_model():
    """Prevent loading real heavy ONNX weights during unit tests."""
    if flowa_app is None:
        yield None
        return
    with patch("app.main.load_model", return_value=None) as m1, \
         patch("app.main.load_segmenter_models", return_value=None) as m2, \
         patch("app.main.load_inpainter_model", return_value=None) as m3:
        yield (m1, m2, m3)


@pytest.fixture
def client(mock_load_model):
    """FastAPI test client with mocked lifespan model loading."""
    if flowa_app is None:
        pytest.skip("Flow A dependencies not available in this environment.")
    with TestClient(flowa_app) as test_client:
        yield test_client


@pytest.fixture
def sample_valid_jpeg_bytes() -> bytes:
    """Generate a small valid JPEG in memory."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_valid_rgba_png_bytes() -> bytes:
    """Generate a valid 1024x1024 transparent RGBA PNG with non-zero alpha in memory."""
    img = Image.new("RGBA", (1024, 1024), color=(0, 0, 0, 0))
    # Add a mock foreground garment box in the center
    for y in range(200, 800):
        for x in range(200, 800):
            img.putpixel((x, y), (50, 100, 200, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
