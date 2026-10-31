from pathlib import Path
from io import BytesIO
import pytest
from PIL import Image

from app.services.extraction import extract_garment, MODEL_NAME

SAMPLE_IMAGE_PATH = Path(r"D:\my codes\Arangtik\backend\uploads\img_mv51arcf_16febe.jpg")


@pytest.mark.integration
def test_real_garment_extraction():
    """Optional end-to-end extraction test with real pretrained ONNX model weights.

    Run with: pytest -m integration
    """
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip(f"Sample test image not found at {SAMPLE_IMAGE_PATH}")

    image_bytes = SAMPLE_IMAGE_PATH.read_bytes()
    png_bytes, original_size = extract_garment(image_bytes)

    assert len(png_bytes) > 0
    assert original_size == (900, 1100)

    # Inspect the generated PNG
    img = Image.open(BytesIO(png_bytes))
    assert img.format == "PNG"
    assert img.size == (1024, 1024)
    assert img.mode == "RGBA"

    alpha = img.getchannel("A")
    extrema = alpha.getextrema()
    assert extrema[0] == 0    # Must have transparent background
    assert extrema[1] >= 250  # Must have solid garment pixels

