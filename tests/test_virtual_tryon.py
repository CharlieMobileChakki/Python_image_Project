"""Unit tests for Multi-Image Virtual Try-On, ChatGPT Vision Stylist, GPU Worker, and UI."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


def _make_image_bytes(color=(120, 140, 200), size=(64, 64), fmt="PNG") -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestMultiImageGpuWorkerAndClient(unittest.TestCase):
    def test_qwen_client_multi_image_runpod_payload(self):
        from app.core.flow_b import qwen_client

        out_b64 = base64.b64encode(_make_image_bytes((40, 120, 80), (768, 1024))).decode("ascii")
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "id": "rp-tryon-1",
            "status": "COMPLETED",
            "output": {
                "image_base64": out_b64,
                "metadata": {"model": "Qwen/Qwen-Image-2.1-Turbo", "gpu_name": "NVIDIA A40"},
            },
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_resp

        refs = [
            Image.new("RGB", (200, 300), (210, 180, 160)),
            Image.new("RGB", (200, 200), (20, 40, 110)),
            Image.new("RGB", (200, 200), (60, 60, 60)),
            Image.new("RGB", (150, 150), (190, 170, 90)),
        ]
        with patch.dict(
            os.environ,
            {"QWEN_REMOTE_URL": "https://api.runpod.ai/v2/ep_tryon", "RUNPOD_API_KEY": "rpa_test"},
            clear=False,
        ), patch("app.core.flow_b.qwen_client.httpx.Client", return_value=mock_client):
            out_img, meta = qwen_client.generate_garment_via_gpu(
                refs,
                "Portrait of <image1> wearing <image2>, <image3>, and <image4>",
                width=768,
                height=1024,
            )

        self.assertEqual(out_img.size, (768, 1024))
        self.assertEqual(meta["execution_mode"], "runpod_serverless")
        sent_input = mock_client.__enter__.return_value.post.call_args.kwargs["json"]["input"]
        self.assertIn("image_base64", sent_input)
        self.assertIn("images_base64", sent_input)
        self.assertEqual(len(sent_input["images_base64"]), 4)
        self.assertEqual(sent_input["width"], 768)
        self.assertEqual(sent_input["height"], 1024)

    def test_qwen_client_multi_image_http_multipart_files(self):
        from app.core.flow_b import qwen_client

        out_bytes = _make_image_bytes((80, 80, 190), (768, 1024))
        good_resp = MagicMock(
            status_code=200,
            headers={"content-type": "image/png", "X-GPU-Name": "Node-TryOn", "X-Inference-Time-Sec": "9.1"},
            content=out_bytes,
        )
        client_cm = MagicMock()
        client_cm.__enter__.return_value.post.return_value = good_resp

        refs = [
            Image.new("RGB", (128, 128), (220, 190, 170)),
            Image.new("RGB", (128, 128), (30, 50, 140)),
        ]
        with patch.dict(
            os.environ,
            {"QWEN_REMOTE_URL": "http://gpu-node:8001", "QWEN_API_KEY": "secret"},
            clear=False,
        ), patch("app.core.flow_b.qwen_client.httpx.Client", return_value=client_cm):
            out_img, meta = qwen_client.generate_garment_via_gpu(refs, "<image1> wearing <image2>")

        self.assertEqual(out_img.size, (768, 1024))
        self.assertEqual(meta["gpu_name"], "Node-TryOn")
        sent_files = client_cm.__enter__.return_value.post.call_args.kwargs["files"]
        self.assertIsInstance(sent_files, list)
        self.assertEqual([field for field, _ in sent_files], ["images", "images"])

    def test_rp_handler_and_engine_multi_image(self):
        worker_dir = str(Path(__file__).resolve().parent.parent / "gpu_worker")
        if worker_dir not in sys.path:
            sys.path.insert(0, worker_dir)

        fake_runpod = types.ModuleType("runpod")
        fake_runpod.serverless = types.SimpleNamespace(start=MagicMock())
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.Generator = lambda dev: MagicMock(manual_seed=lambda s: "gen")
        fake_torch.inference_mode = MagicMock

        with patch.dict(sys.modules, {"runpod": fake_runpod, "torch": fake_torch}):
            import engine
            import rp_handler

            raw_imgs = [_make_image_bytes((i * 40, 100, 150), (1000, 1000)) for i in range(3)]
            prepared = engine.prepare_ref_images(raw_imgs, max_in=768)
            self.assertEqual(len(prepared), 3)
            self.assertEqual(prepared[0].size, (768, 768))

            raw_five = [_make_image_bytes((i * 30, 110, 140), (1000, 1000)) for i in range(5)]
            adaptive_five = engine.prepare_ref_images(raw_five)
            self.assertEqual(adaptive_five[0].size, (512, 512))
            self.assertEqual(adaptive_five[1].size, (448, 448))

            fake_pipe = MagicMock(
                return_value=types.SimpleNamespace(images=[Image.new("RGB", (768, 1024), (250, 250, 250))])
            )
            fake_pipe.__call__ = fake_pipe
            with patch.object(engine, "load_qwen21_pipeline", return_value=fake_pipe):
                png_bytes, _, _ = engine.run_inference_sync(prepared, "<image1> tries <image2>", width=768, height=1024)
            self.assertGreater(len(png_bytes), 100)
            self.assertIsInstance(fake_pipe.call_args.kwargs["image"], list)
            self.assertEqual(len(fake_pipe.call_args.kwargs["image"]), 3)
            self.assertFalse(fake_pipe.call_args.kwargs["use_kv_cache"])

            # Verify automatic OOM rescue downscales to 384px, slices attention, and retries cleanly
            oom_pipe = MagicMock(
                side_effect=[
                    RuntimeError("CUDA out of memory. Tried to allocate 568.00 MiB"),
                    types.SimpleNamespace(images=[Image.new("RGB", (768, 1024), (245, 245, 245))]),
                ]
            )
            oom_pipe.__call__ = oom_pipe
            oom_pipe.enable_attention_slicing = MagicMock()
            oom_pipe.disable_attention_slicing = MagicMock()
            with patch.object(engine, "load_qwen21_pipeline", return_value=oom_pipe):
                rescued_bytes, _, _ = engine.run_inference_sync(adaptive_five, "<image1> + 4 refs", width=768, height=1024)
            self.assertGreater(len(rescued_bytes), 100)
            self.assertEqual(oom_pipe.call_count, 2)
            oom_pipe.enable_attention_slicing.assert_called_once_with("max")
            oom_pipe.disable_attention_slicing.assert_called_once()
            self.assertEqual(oom_pipe.call_args.kwargs["image"][0].size, (384, 384))

            b64_list = [base64.b64encode(_make_image_bytes((10, 20, 30), (64, 64))).decode("ascii") for _ in range(3)]
            out_bytes = _make_image_bytes((200, 210, 220), (768, 1024))
            with patch.object(rp_handler, "run_inference_sync", return_value=(out_bytes, {}, {"gpu_name": "A40"})) as mock_inf:
                res = rp_handler.handler(
                    {"input": {"action": "generate", "images_base64": b64_list, "prompt": "tryon look", "width": 768, "height": 1024}}
                )

            self.assertIn("image_base64", res)
            passed_ref = mock_inf.call_args.kwargs["ref_img"]
            self.assertIsInstance(passed_ref, list)
            self.assertEqual(len(passed_ref), 3)


class TestTryonStylistAndPipeline(unittest.TestCase):
    def test_stylist_fallback_pairs_shirts_pants_and_includes_accessories(self):
        from app.core.flow_b.tryon_stylist import analyze_wardrobe_and_plan_tryon, build_qwen_tryon_prompt

        avatar = Image.new("RGB", (256, 384), (215, 185, 165))
        garments = [
            {"index": 0, "image": Image.new("RGB", (200, 200), (240, 240, 250)), "role_hint": "top", "filename": "white_floral_shirt.jpg"},
            {"index": 1, "image": Image.new("RGB", (200, 200), (20, 50, 90)), "role_hint": "top", "filename": "navy_linen_shirt.jpg"},
            {"index": 2, "image": Image.new("RGB", (200, 200), (190, 175, 150)), "role_hint": "bottom", "filename": "beige_chinos.jpg"},
            {"index": 3, "image": Image.new("RGB", (200, 200), (35, 45, 70)), "role_hint": "bottom", "filename": "dark_denim_jeans.jpg"},
        ]
        accessories = [
            {"index": 0, "image": Image.new("RGB", (120, 120), (160, 130, 80)), "filename": "gold_chronograph_watch.jpg"},
            {"index": 1, "image": Image.new("RGB", (120, 120), (235, 235, 235)), "filename": "white_leather_sneakers.jpg"},
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            plan = analyze_wardrobe_and_plan_tryon(
                avatar_img=avatar,
                garments=garments,
                accessories=accessories,
                max_combinations=2,
                occasion="smart_casual",
            )

        self.assertEqual(len(plan["combinations"]), 2)
        c1, c2 = plan["combinations"]
        self.assertEqual(c1["selected_garment_indices"], [0, 2])
        self.assertEqual(c2["selected_garment_indices"], [1, 3])
        self.assertEqual(c1["selected_accessory_indices"], [0, 1])
        self.assertEqual(c2["selected_accessory_indices"], [0, 1])

        prompt, total_refs = build_qwen_tryon_prompt(
            plan["avatar_summary"],
            [plan["garments"][0], plan["garments"][2]],
            plan["accessories"],
            occasion="smart_casual",
        )
        self.assertEqual(total_refs, 5)
        for tag in ("<image1>", "<image2>", "<image3>", "<image4>", "<image5>"):
            self.assertIn(tag, prompt)
        self.assertIn("IDENTITY & FACE LOCK", prompt)

    def test_stylist_vision_normalizes_and_retains_missed_accessories(self):
        from app.core.flow_b.tryon_stylist import analyze_wardrobe_and_plan_tryon

        vision_json = {
            "avatar_summary": "South Asian young man with medium warm skin tone, short black side-parted hair, trim beard",
            "garments": [
                {"garment_index": 0, "role": "top", "title": "Indigo Block-Print Cotton Shirt", "colors": "deep indigo & off-white", "patterns_and_design": "hand-stamped floral block motifs, mandarin collar, wooden buttons"},
                {"garment_index": 1, "role": "bottom", "title": "Sand Cotton Twill Chinos", "colors": "warm sand beige", "patterns_and_design": "slim tapered fit, slant pockets, matte twill weave"},
            ],
            "accessories": [
                {"accessory_index": 0, "accessory_type": "wristwatch", "body_placement": "left wrist", "title": "Silver Steel Dress Watch", "design_details": "sunburst blue dial, brushed steel bracelet"},
                {"accessory_index": 1, "accessory_type": "sneakers", "body_placement": "both feet", "title": "White Low-Top Sneakers", "design_details": "matte white leather upper, gum rubber sole"},
            ],
            "combinations": [
                {
                    "title": "Artisanal Indigo & Sand",
                    "occasion": "smart_casual",
                    "style_reason": "Indigo block print pops cleanly over warm sand chinos.",
                    "selected_garment_indices": [0, 1, 99],
                    "selected_accessory_indices": [0],
                }
            ],
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(vision_json)}}]}
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_resp

        avatar = Image.new("RGB", (200, 300), (200, 170, 150))
        garments = [
            {"index": 0, "image": Image.new("RGB", (150, 150), (30, 40, 100)), "role_hint": "top", "filename": "shirt.jpg"},
            {"index": 1, "image": Image.new("RGB", (150, 150), (200, 190, 160)), "role_hint": "bottom", "filename": "pants.jpg"},
        ]
        accessories = [
            {"index": 0, "image": Image.new("RGB", (100, 100), (180, 180, 190)), "filename": "watch.jpg"},
            {"index": 1, "image": Image.new("RGB", (100, 100), (245, 245, 245)), "filename": "shoes.jpg"},
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.core.flow_b.tryon_stylist.httpx.Client", return_value=mock_client
        ):
            plan = analyze_wardrobe_and_plan_tryon(avatar, garments, accessories, max_combinations=2)

        combo = plan["combinations"][0]
        self.assertEqual(combo["selected_garment_indices"], [0, 1])
        self.assertEqual(sorted(combo["selected_accessory_indices"]), [0, 1])
        self.assertIn("hand-stamped floral block motifs", plan["garments"][0]["patterns_and_design"])

    def test_full_virtual_tryon_pipeline_saves_pngs_and_preserves_routes_and_ui(self):
        from app.core.pipeline import virtual_tryon
        from app.main import app as main_app

        captured_calls = []

        def fake_gpu(reference_image, prompt, width=None, height=None, seed=None, **kwargs):
            captured_calls.append((reference_image, prompt, width, height))
            return Image.new("RGB", (width or 768, height or 1024), (230, 232, 238)), {
                "model": "Qwen/Qwen-Image-2.1-Turbo",
                "gpu_name": "NVIDIA A40",
            }

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            virtual_tryon, "OUTPUT_DIR", Path(tmpdir)
        ), patch.object(
            virtual_tryon, "is_gpu_worker_configured", return_value=(True, "https://api.runpod.ai/v2/ep_ok")
        ), patch.object(
            virtual_tryon, "generate_garment_via_gpu", side_effect=fake_gpu
        ), patch.dict(
            os.environ, {"OPENAI_API_KEY": ""}, clear=False
        ):
            res = virtual_tryon.process_virtual_tryon_pipeline(
                avatar_bytes=_make_image_bytes((210, 180, 160), (256, 384)),
                garment_entries=[
                    {"bytes": _make_image_bytes((240, 240, 255)), "role_hint": "top", "filename": "shirt_a.jpg"},
                    {"bytes": _make_image_bytes((20, 40, 90)), "role_hint": "top", "filename": "shirt_b.jpg"},
                    {"bytes": _make_image_bytes((190, 175, 145)), "role_hint": "bottom", "filename": "pant_a.jpg"},
                    {"bytes": _make_image_bytes((40, 40, 50)), "role_hint": "bottom", "filename": "pant_b.jpg"},
                ],
                accessory_entries=[
                    {"bytes": _make_image_bytes((180, 160, 100)), "role_hint": "accessory", "filename": "watch.jpg"},
                ],
                max_combinations=2,
                occasion="smart_casual",
                base_url="https://bg.arangtik.com",
            )

            self.assertEqual(res["total_combinations"], 2)
            self.assertEqual(len(captured_calls), 2)
            for refs, prompt, w, h in captured_calls:
                self.assertEqual((w, h), (768, 1024))
                self.assertEqual(len(refs), 4)  # avatar + top + bottom + watch
                self.assertIn("<image4>", prompt)
            for look in res["combinations"]:
                self.assertTrue(look["url"].startswith("https://bg.arangtik.com/api/outputs/"))
                self.assertEqual(look["resolution"], "768x1024")
                saved_name = look["url"].rsplit("/", 1)[-1]
                self.assertTrue((Path(tmpdir) / saved_name).exists())

        from app.api.routes import router as api_router
        from app.api.virtual_tryon import virtual_tryon_router

        openapi_paths = set(main_app.openapi()["paths"].keys())
        self.assertTrue(
            {
                "/api/virtual-tryon",
                "/api/background-removal",
                "/api/qwen-image",
                "/api/jewelry-studio",
                "/api/forward-downstream",
                "/api/outputs/{filename}",
            }.issubset(openapi_paths)
        )
        self.assertIn("/generate-garment", {getattr(r, "path", None) for r in api_router.routes})
        self.assertIn("/virtual-tryon", {getattr(r, "path", None) for r in virtual_tryon_router.routes})

        index_html = (Path(__file__).resolve().parent.parent / "app" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("/api/background-removal", index_html)
        self.assertIn("/api/virtual-tryon", index_html)
        self.assertIn("tryonAvatarInput", index_html)
        self.assertIn("tryonGarmentsInput", index_html)
        self.assertIn("tryonAccessoriesInput", index_html)


if __name__ == "__main__":
    unittest.main()
