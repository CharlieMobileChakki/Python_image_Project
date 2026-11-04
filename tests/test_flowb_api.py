"""Unit and API contract tests for Flow B FastAPI service.

Mocked tests verify all status codes, error conditions, headers, and 1024x1024 normalization.
Integration tests verify end-to-end SegFormer inference with real images.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from app.services.flow_b.extraction import (
    EXPLICIT_ALIASES,
    FlowBResult,
    get_id2label,
    is_model_loaded,
    resolve_target_classes,
)
from app.services.flow_b.main import app

client = TestClient(app)

MOCK_ID2LABEL = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}


def make_dummy_image_bytes(width: int = 100, height: int = 100, format: str = "JPEG") -> bytes:
    buf = BytesIO()
    img = Image.new("RGB", (width, height), (120, 140, 200))
    img.save(buf, format=format)
    return buf.getvalue()


def dummy_flowb_result(
    class_name: str = "pants",
    alpha_mode: str = "soft",
    orig_w: int = 600,
    orig_h: int = 800,
) -> FlowBResult:
    # 1024x1024 RGBA canvas with centered opaque item
    canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    garment = Image.new("RGBA", (500, 600), (30, 80, 180, 255))
    canvas.alpha_composite(garment, (262, 212))
    buf = BytesIO()
    canvas.save(buf, format="PNG")

    resolved, alias = resolve_target_classes(MOCK_ID2LABEL, class_name)
    target_labels = [l for _, l in resolved]
    target_ids = [c for c, _ in resolved]

    return FlowBResult(
        png_bytes=buf.getvalue(),
        original_size=(orig_w, orig_h),
        target_classes=target_labels,
        target_class_ids=target_ids,
        applied_alias=alias,
        mask_coverage_pct=18.5,
        alpha_mode=alpha_mode,
        processing_time_ms=145,
        bounding_box=[100, 150, 500, 750],
    )


class TestFlowBAPI(unittest.TestCase):
    """Test suite covering Flow B FastAPI endpoints, error handling, and response contracts."""

    # 1. Root and Health Endpoints
    def test_root_endpoint(self) -> None:
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "arangtik-garment-flow-b")
        self.assertIn("/health", data["health"])
        self.assertIn("/docs", data["docs"])

    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    def test_health_when_loaded(self, mock_loaded) -> None:
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["flow"], "B")
        self.assertTrue(data["model_loaded"])
        self.assertIn("anarkali", data["supported_aliases"])

    @patch("app.services.flow_b.main.is_model_loaded", return_value=False)
    def test_health_when_initializing(self, mock_loaded) -> None:
        response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "initializing")
        self.assertFalse(data["model_loaded"])

    # 2. Upload and Format Validations
    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    def test_unsupported_media_type(self, mock_loaded) -> None:
        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"class_name": "pants"},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("Supported image types", response.json()["detail"])

    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    def test_empty_file_upload(self, mock_loaded) -> None:
        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"class_name": "pants"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("uploaded file is empty", response.json()["detail"])

    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    def test_oversized_file_upload(self, mock_loaded) -> None:
        oversized = b"0" * (15 * 1024 * 1024 + 10)
        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("big.jpg", oversized, "image/jpeg")},
            data={"class_name": "pants"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds the 15 MB upload limit", response.json()["detail"])

    @patch("app.services.flow_b.main.is_model_loaded", return_value=False)
    def test_endpoint_when_model_unready(self, mock_loaded) -> None:
        img_bytes = make_dummy_image_bytes()
        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"class_name": "pants"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("still initializing", response.json()["detail"])

    # 3. Parameter Validations
    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    def test_invalid_alpha_mode(self, mock_loaded) -> None:
        img_bytes = make_dummy_image_bytes()
        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"class_name": "pants", "alpha_mode": "super_soft"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("alpha_mode must be either 'soft' or 'hard'", response.json()["detail"])

    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    @patch("app.services.flow_b.extraction.get_id2label", return_value=MOCK_ID2LABEL)
    def test_invalid_class_name(self, mock_id2label, mock_loaded) -> None:
        img_bytes = make_dummy_image_bytes()
        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"class_name": "astronault_spacesuit"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("Unknown garment class", response.json()["detail"])

    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    @patch("app.services.flow_b.extraction.get_id2label", return_value=MOCK_ID2LABEL)
    def test_invalid_alpha_thresholds(self, mock_id2label, mock_loaded) -> None:
        img_bytes = make_dummy_image_bytes()
        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"class_name": "pants", "alpha_low": 0.85, "alpha_high": 0.20},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("strictly less than", response.json()["detail"])

    # 4. Error Handling: Empty Mask & Model Failures
    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    @patch("app.services.flow_b.main.extract_garment_worn", side_effect=ValueError("No foreground garment was detected"))
    def test_empty_mask_returns_422(self, mock_extract, mock_loaded) -> None:
        img_bytes = make_dummy_image_bytes()
        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"class_name": "pants"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("No foreground garment was detected", response.json()["detail"])

    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    @patch("app.services.flow_b.main.extract_garment_worn", side_effect=RuntimeError("PyTorch execution failed"))
    def test_model_runtime_error_returns_422(self, mock_extract, mock_loaded) -> None:
        img_bytes = make_dummy_image_bytes()
        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"class_name": "pants"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("PyTorch execution failed", response.json()["detail"])

    # 5. Successful Single-Class, Multi-Class, and Alias Extractions
    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    @patch("app.services.flow_b.main.extract_garment_worn")
    def test_successful_single_class_extraction(self, mock_extract, mock_loaded) -> None:
        mock_extract.return_value = dummy_flowb_result(class_name="pants", orig_w=1200, orig_h=800)
        img_bytes = make_dummy_image_bytes()

        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("trousers.jpg", img_bytes, "image/jpeg")},
            data={"class_name": "pants", "alpha_mode": "soft"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertIn("Pants", response.headers["x-target-classes"])
        self.assertEqual(response.headers["x-original-width"], "1200")
        self.assertEqual(response.headers["x-original-height"], "800")
        self.assertEqual(response.headers["x-mask-coverage"], "18.5%")
        self.assertIn("x-processing-time-ms", response.headers)

        # Verify output image is 1024x1024 RGBA PNG with transparent pixels
        out_img = Image.open(BytesIO(response.content))
        self.assertEqual(out_img.size, (1024, 1024))
        self.assertEqual(out_img.mode, "RGBA")
        alpha_arr = np.array(out_img.getchannel("A"))
        self.assertTrue((alpha_arr == 0).any(), "Expected transparent background pixels")
        self.assertTrue((alpha_arr > 0).any(), "Expected foreground garment pixels")

    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    @patch("app.services.flow_b.main.extract_garment_worn")
    def test_successful_multi_class_extraction(self, mock_extract, mock_loaded) -> None:
        mock_extract.return_value = dummy_flowb_result(class_name="upper-clothes,skirt")
        img_bytes = make_dummy_image_bytes()

        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("suit.png", img_bytes, "image/png")},
            data={"class_name": "upper-clothes,skirt"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Upper-clothes", response.headers["x-target-classes"])
        self.assertIn("Skirt", response.headers["x-target-classes"])

    @patch("app.services.flow_b.main.is_model_loaded", return_value=True)
    @patch("app.services.flow_b.main.extract_garment_worn")
    def test_successful_anarkali_alias_extraction(self, mock_extract, mock_loaded) -> None:
        mock_extract.return_value = dummy_flowb_result(class_name="anarkali")
        img_bytes = make_dummy_image_bytes()

        response = client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("anarkali.jpg", img_bytes, "image/jpeg")},
            data={"class_name": "anarkali"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-target-classes"], "Upper-clothes,Skirt")


# Integration test with real inference (isolated under marker)
@pytest.mark.integration
def test_real_flowb_inference() -> None:
    test_image_path = Path(r"D:\my codes\Arangtik\backend\uploads\img_mv3nprm1_b7901a.jpg")
    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    with open(test_image_path, "rb") as f:
        file_bytes = f.read()

    with TestClient(app) as lifespan_client:
        response = lifespan_client.post(
            "/api/v1/extract-garment/worn",
            files={"file": ("pants.jpg", file_bytes, "image/jpeg")},
            data={"class_name": "pants", "alpha_mode": "soft"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-target-classes"] == "Pants"

        out_img = Image.open(BytesIO(response.content))
        assert out_img.size == (1024, 1024)
        assert out_img.mode == "RGBA"
        alpha = np.array(out_img.getchannel("A"))
        assert (alpha == 255).sum() > 50000
