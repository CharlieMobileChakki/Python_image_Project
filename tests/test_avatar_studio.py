"""Unit tests for AI Stylized Avatar Profile Picture Generator ("Avatar Studio")."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from app.core.avatar import (
    analyze_avatar_identity,
    build_avatar_stylized_prompt,
    crop_bust_up_portrait,
    detect_face_bbox_yunet,
    process_avatar_pipeline,
    resolve_halo_and_doodle,
)
from app.main import STATIC_DIR, app as main_app
from app.schemas import AvatarGenerationResponse, AvatarIdentityFeatures


def _create_synthetic_test_photo(width: int = 600, height: int = 800) -> Image.Image:
    """Create a synthetic portrait photo with distinct face and shoulders."""
    arr = np.full((height, width, 3), 200, dtype=np.uint8)
    arr[: int(height * 0.4), :] = [180, 210, 235]
    arr[int(height * 0.5):, int(width * 0.15): int(width * 0.85)] = [30, 40, 70]
    cy, cx = int(height * 0.32), int(width * 0.5)
    y_idx, x_idx = np.ogrid[:height, :width]
    mask = ((x_idx - cx) / 100.0) ** 2 + ((y_idx - cy) / 130.0) ** 2 <= 1.0
    arr[mask] = [230, 190, 160]
    return Image.fromarray(arr, mode="RGB")


def _image_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class TestAvatarStudio(unittest.TestCase):
    """Test suite for Avatar Studio cropping, prompt generation, pipeline, and API endpoint."""

    def test_crop_bust_up_portrait_proportions(self):
        """Verify bust-up portrait crop preserves head and upper torso centered."""
        img = _create_synthetic_test_photo(600, 800)
        # 1. With vision face_box_2d [ymin, xmin, ymax, xmax] (0..1000 scale)
        cropped_vision = crop_bust_up_portrait(img, face_box_2d=[160, 330, 480, 670], target_size=1024)
        self.assertIsInstance(cropped_vision, Image.Image)
        self.assertEqual(cropped_vision.mode, "RGB")
        self.assertGreaterEqual(cropped_vision.width, 200)
        self.assertGreaterEqual(cropped_vision.height, 200)

        # 2. With fallback / automated detection
        cropped_auto = crop_bust_up_portrait(img, face_box_2d=None, target_size=1024)
        self.assertIsInstance(cropped_auto, Image.Image)
        self.assertGreaterEqual(cropped_auto.width, 200)

    def test_resolve_halo_and_doodle(self):
        """Verify halo color and doodle accent resolution logic."""
        halo, doodle = resolve_halo_and_doodle({"gender": "female", "recommended_halo_color": "lavender"}, "auto", "auto")
        self.assertEqual(halo["name"], "lavender")
        self.assertIn("heart", doodle)

        halo_m, doodle_m = resolve_halo_and_doodle({"gender": "male", "recommended_halo_color": "soft_blue"}, "auto", "auto")
        self.assertEqual(halo_m["name"], "soft_blue")
        self.assertIn("crown", doodle_m)

        halo_exp, doodle_exp = resolve_halo_and_doodle({"gender": "female"}, "mint_green", "sparks")
        self.assertEqual(halo_exp["name"], "mint_green")
        self.assertIn("sparkle", doodle_exp)

    def test_build_avatar_stylized_prompt_male_suit_glasses(self):
        """Verify prompt construction for masculine avatar with suit, tie, glasses, and crown."""
        avatar_info = {
            "gender": "male",
            "expression": "confident subtle smile",
            "hair_details": "thick voluminous dark styled textured hair with fringe",
            "eyewear": "thin rectangular wireframe eyeglasses",
            "facial_hair": "neat dark moustache and light chin stubble",
            "upper_garment": "black tailored suit blazer, white collared shirt, vibrant red necktie, and pocket square",
            "distinctive_features": None,
            "recommended_halo_color": "soft_blue",
            "recommended_doodle": "crown",
        }
        prompt, halo, doodle = build_avatar_stylized_prompt(
            avatar_info=avatar_info,
            style="vector_art",
            halo_color="soft_blue",
            doodle_accent="crown",
        )

        self.assertEqual(halo, "soft_blue")
        self.assertEqual(doodle, "crown")
        self.assertIn("STYLIZED 2D AVATAR PROFILE ILLUSTRATION", prompt)
        self.assertIn("EXACT FACIAL LIKENESS MANDATE", prompt)
        self.assertIn("MANDATORY EYEWEAR LOCK", prompt)
        self.assertIn("thin rectangular wireframe eyeglasses", prompt)
        self.assertIn("red necktie", prompt)
        self.assertIn("circular disc badge", prompt)
        self.assertIn("crown doodle", prompt)
        self.assertIn("ABSOLUTELY NO ARTIFICIAL FACIAL SLIMMING", prompt)
        self.assertIn("NO photorealistic skin pores", prompt)

    def test_build_avatar_stylized_prompt_female_floral_dress(self):
        """Verify prompt construction for feminine avatar with floral dress, bindi, and heart doodle."""
        avatar_info = {
            "gender": "female",
            "expression": "radiant warm smile showing upper teeth",
            "hair_details": "long silky straight-wavy dark hair parted in middle cascading past shoulders",
            "eyewear": None,
            "facial_hair": None,
            "upper_garment": "dark navy blue dress with yellow floral blossom print and square neckline",
            "distinctive_features": "small dark bindi between eyebrows",
            "recommended_halo_color": "periwinkle",
            "recommended_doodle": "heart",
        }
        prompt, halo, doodle = build_avatar_stylized_prompt(
            avatar_info=avatar_info,
            style="vector_art",
            halo_color="auto",
            doodle_accent="auto",
        )

        self.assertEqual(halo, "periwinkle")
        self.assertEqual(doodle, "heart")
        self.assertIn("dark navy blue dress with yellow floral blossom print", prompt)
        self.assertIn("small dark bindi between eyebrows", prompt)
        self.assertIn("outlined heart doodle", prompt)
        self.assertIn("periwinkle", prompt)

    def test_schema_serialization(self):
        """Verify Pydantic schemas validate correctly."""
        identity = AvatarIdentityFeatures(
            gender="male",
            expression="subtle smile",
            hair_details="styled voluminous hair",
            eyewear="wireframe glasses",
            facial_hair="light stubble",
            upper_garment="black blazer with red tie",
            distinctive_features=None,
        )
        resp = AvatarGenerationResponse(
            status="success",
            success=True,
            job_id="avatar_test123",
            avatar_url="https://example.com/outputs/avatar_test123_avatar.png",
            style="vector_art",
            halo_color="soft_blue",
            doodle_accent="crown",
            identity_features=identity,
            resolution="1024x1024",
            processing_time_sec=1.45,
            execution_time_ms=1450.0,
            engine="Qwen/Qwen-Image-2.1-Turbo",
        )
        dumped = resp.model_dump()
        self.assertEqual(dumped["status"], "success")
        self.assertEqual(dumped["identity_features"]["gender"], "male")
        self.assertEqual(dumped["resolution"], "1024x1024")

    @patch("app.core.avatar.avatar_pipeline.analyze_avatar_identity")
    @patch("app.core.avatar.avatar_pipeline.generate_garment_via_gpu")
    def test_full_avatar_pipeline_execution(self, mock_gpu, mock_analyze):
        """Verify end-to-end process_avatar_pipeline saves 1024x1024 PNG and outputs clean metadata."""
        mock_analyze.return_value = {
            "gender": "male",
            "expression": "subtle smile",
            "hair_details": "styled voluminous dark hair",
            "eyewear": "thin rectangular wireframe eyeglasses",
            "facial_hair": "light stubble",
            "upper_garment": "black suit and red tie",
            "distinctive_features": None,
            "recommended_halo_color": "soft_blue",
            "recommended_doodle": "crown",
            "face_box_2d": [150, 300, 500, 700],
        }
        synthetic_result = Image.new("RGB", (1024, 1024), color=(250, 248, 245))
        mock_gpu.return_value = (synthetic_result, {"engine": "NVIDIA A40 Turbo", "gpu_name": "A40"})

        raw_bytes = _image_to_jpeg_bytes(_create_synthetic_test_photo(500, 600))
        result = process_avatar_pipeline(
            image_bytes=raw_bytes,
            style="vector_art",
            halo_color="soft_blue",
            doodle_accent="crown",
            include_base64=True,
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["success"])
        self.assertTrue(result["avatar_url"].startswith("/outputs/"))
        self.assertTrue(result["avatar_url"].endswith(".png"))
        self.assertIsNotNone(result["avatar_base64"])
        self.assertTrue(result["avatar_base64"].startswith("data:image/png;base64,"))
        self.assertEqual(result["resolution"], "1024x1024")
        self.assertEqual(result["halo_color"], "soft_blue")
        self.assertEqual(result["doodle_accent"], "crown")
        mock_gpu.assert_called_once()

    def test_routes_registered_in_fastapi_openapi(self):
        """Verify endpoints are cleanly registered in FastAPI schema."""
        openapi = main_app.openapi()
        paths = openapi.get("paths", {})
        self.assertIn("/api/generate-avatar", paths)
        post_op = paths["/api/generate-avatar"].get("post", {})
        self.assertEqual(post_op.get("summary"), "Generate Stylized 2D Vector Avatar Profile Picture")

    def test_frontend_ui_avatar_studio_tab_integration(self):
        """Verify Avatar Studio workspace, controls, and script are integrated into Web UI."""
        index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn("modeBtnAvatar", index_html)
        self.assertIn("avatarWorkspace", index_html)
        self.assertIn("avatarStyleSelect", index_html)
        self.assertIn("avatarHaloSelect", index_html)
        self.assertIn("avatarDoodleSelect", index_html)
        self.assertIn("generateStylizedAvatar()", index_html)
        self.assertIn("/api/generate-avatar", index_html)
        self.assertIn("switchAvatarOutputView", index_html)


if __name__ == "__main__":
    unittest.main()
