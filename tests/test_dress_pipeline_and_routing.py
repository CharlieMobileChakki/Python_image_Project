"""Comprehensive validation tests for Dress Generation latency optimizations and routing rules."""
import io
import time
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import pytest

from app.core.pipeline.pipeline_router import resolve_execution_flow, ExecutionFlow
from app.core.pipeline.unified_pipeline import process_garment_pipeline
from app.core.flow_b.qwen_client import generate_garment_via_gpu


def create_dummy_image(width=512, height=512, color=(200, 100, 100)) -> bytes:
    """Create dummy test image bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestDressOptimizationAndRouting:

    def test_routing_resolution(self):
        """Verify routing between Flow A (Without Person) and Flow B (With Person)."""
        # Flat-lay / without-person cases
        assert resolve_execution_flow(person_mode="without_person") == ExecutionFlow.FLOW_A_WITHOUT_PERSON.value
        assert resolve_execution_flow(person_mode="flat_lay") == ExecutionFlow.FLOW_A_WITHOUT_PERSON.value
        assert resolve_execution_flow(has_human=False) == ExecutionFlow.FLOW_A_WITHOUT_PERSON.value
        assert resolve_execution_flow(mode_hint="isolated") == ExecutionFlow.FLOW_A_WITHOUT_PERSON.value

        # Model-worn / with-person cases
        assert resolve_execution_flow(person_mode="with_person") == ExecutionFlow.FLOW_B_WITH_PERSON.value
        assert resolve_execution_flow(has_human=True) == ExecutionFlow.FLOW_B_WITH_PERSON.value
        assert resolve_execution_flow() == ExecutionFlow.FLOW_B_WITH_PERSON.value

    @patch("app.core.pipeline.unified_pipeline.extract_isolated_flow_a")
    @patch("app.core.pipeline.unified_pipeline.analyze_garment_and_build_prompts")
    @patch("app.core.pipeline.unified_pipeline.generate_garment_via_gpu")
    def test_flat_lay_dress_uses_raw_cutout_zero_vision_zero_qwen(
        self, mock_qwen, mock_vision, mock_birefnet
    ):
        """When user requests background removal for a flat-lay dress (without person),
        verify zero Vision and zero Qwen calls are made.
        """
        mock_birefnet.return_value = Image.new("RGBA", (512, 512), (200, 50, 50, 255))
        img_bytes = create_dummy_image(512, 512)

        # 1. Test with person_mode='without_person'
        res1 = process_garment_pipeline(
            image_bytes=img_bytes,
            person_mode="without_person",
            target_garment="dress",
        )
        assert res1["status"] == "success"
        assert res1["mode"] == "without_person"
        mock_vision.assert_not_called()
        mock_qwen.assert_not_called()

        # 2. Test with mode_hint='raw_cutout_only'
        res2 = process_garment_pipeline(
            image_bytes=img_bytes,
            person_mode="with_person",
            mode_hint="raw_cutout_only",
            target_garment="dress",
        )
        assert res2["status"] == "success"
        assert res2["mode"] == "without_person"
        mock_vision.assert_not_called()
        mock_qwen.assert_not_called()

    @patch("app.core.pipeline.unified_pipeline.extract_isolated_flow_a")
    @patch("app.core.pipeline.unified_pipeline.analyze_garment_and_build_prompts")
    @patch("app.core.pipeline.unified_pipeline.generate_garment_via_gpu")
    def test_single_dress_target_submits_exactly_one_gpu_job(
        self, mock_qwen, mock_vision, mock_birefnet
    ):
        """When target_garment=dress and input has dress, verify exactly ONE GPU job is submitted with steps=8."""
        mock_vision.return_value = [
            {
                "category": "dress",
                "recognized_label": "Floral Maxi Dress",
                "prompt": "Reconstruct dress",
                "prompt_source": "test",
            },
            {
                "category": "dress",
                "recognized_label": "Secondary dress component",
                "prompt": "Reconstruct component",
                "prompt_source": "test",
            },
        ]
        mock_qwen.return_value = (
            Image.new("RGB", (512, 512), (255, 255, 255)),
            {"model": "Qwen-Image-Edit", "gpu_name": "RTX 3090", "inference_time_sec": "7.2"},
        )
        mock_birefnet.return_value = Image.new("RGBA", (512, 512), (200, 50, 50, 255))

        img_bytes = create_dummy_image(512, 512)

        res = process_garment_pipeline(
            image_bytes=img_bytes,
            person_mode="with_person",
            target_garment="dress",
        )

        assert res["status"] == "success"
        # Exactly one GPU call made!
        assert mock_qwen.call_count == 1
        # Passed steps=8 for apparel (Qwen Turbo native)
        assert mock_qwen.call_args.kwargs["steps"] == 8
        assert mock_qwen.call_args.kwargs["guidance_scale"] == 1.0
        assert len(res["garments"]) == 1
        assert res["garments"][0]["category"] == "dress"

    @patch("app.core.pipeline.unified_pipeline.extract_isolated_flow_a")
    @patch("app.core.pipeline.unified_pipeline.analyze_garment_and_build_prompts")
    @patch("app.core.pipeline.unified_pipeline.generate_garment_via_gpu")
    def test_all_garments_preserves_multiple_items(
        self, mock_qwen, mock_vision, mock_birefnet
    ):
        """When target_garment='all', verify multiple garments are preserved."""
        mock_vision.return_value = [
            {
                "category": "upper_clothes",
                "recognized_label": "Silk Shirt",
                "prompt": "Reconstruct shirt",
                "prompt_source": "test",
            },
            {
                "category": "pants",
                "recognized_label": "Chino Trousers",
                "prompt": "Reconstruct pants",
                "prompt_source": "test",
            },
        ]
        mock_qwen.return_value = (
            Image.new("RGB", (512, 512), (255, 255, 255)),
            {"model": "Qwen-Image-Edit", "gpu_name": "RTX 3090", "inference_time_sec": "7.2"},
        )
        mock_birefnet.return_value = Image.new("RGBA", (512, 512), (200, 50, 50, 255))

        img_bytes = create_dummy_image(512, 512)

        res = process_garment_pipeline(
            image_bytes=img_bytes,
            person_mode="with_person",
            target_garment="all",
        )

        assert res["status"] == "success"
        # Preserved 2 jobs when 'all' is requested!
        assert mock_qwen.call_count == 2
        assert len(res["garments"]) == 2

    @patch("app.core.pipeline.unified_pipeline.extract_isolated_flow_a")
    @patch("app.core.pipeline.unified_pipeline.generate_garment_via_gpu")
    def test_jewelry_preserves_26_steps_and_stands(
        self, mock_qwen, mock_birefnet
    ):
        """Verify that jewelry behavior (26 steps, boutique stands) is strictly preserved."""
        mock_qwen.return_value = (
            Image.new("RGB", (512, 512), (240, 230, 200)),
            {"model": "Qwen-Image-Edit", "gpu_name": "RTX 3090", "inference_time_sec": "15.0"},
        )
        mock_birefnet.return_value = Image.new("RGBA", (512, 512), (240, 230, 200, 255))

        img_bytes = create_dummy_image(512, 512)

        res = process_garment_pipeline(
            image_bytes=img_bytes,
            person_mode="jewelry",
            target_garment="necklace",
            presentation_mode="mode_c_display_stand",
        )

        assert res["status"] == "success"
        assert mock_qwen.call_count == 1
        # Jewelry retains 26 steps and 2.5 guidance scale!
        assert mock_qwen.call_args.kwargs["steps"] == 26
        assert mock_qwen.call_args.kwargs["guidance_scale"] == 2.5
        assert res["garments"][0]["stand_type"] == "beige_velvet_brass_neck_bust"
