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
            self.assertEqual(adaptive_five[0].size, (576, 576))
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
            self.assertEqual(fake_pipe.call_args.kwargs["image"][0].size, (576, 576))
            self.assertEqual(fake_pipe.call_args.kwargs["image"][1].size, (512, 512))
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
        self.assertIn("STRICT FACE & IDENTITY PRESERVATION", prompt)
        self.assertIn("COMPLETE GARMENT DRESSING", prompt)
        self.assertNotIn("DEFAULT LOWER GARMENT", prompt)

        # When NO top shirt is uploaded (pants/jeans ONLY!), never generate shirtless — add modest complement top
        bottom_only_prompt, bottom_only_refs = build_qwen_tryon_prompt(
            plan["avatar_summary"],
            [plan["garments"][2]],
            [],
            occasion="casual",
            avatar_gender="male",
        )
        self.assertEqual(bottom_only_refs, 2)
        self.assertIn("NO SHIRTLESS TORSO", bottom_only_prompt)
        self.assertIn("T-shirt", bottom_only_prompt)

        # When NO bottom pant is uploaded with a top, dynamic complementary trousers/jeans must be applied
        top_only_prompt, top_only_refs = build_qwen_tryon_prompt(
            plan["avatar_summary"],
            [plan["garments"][0]],
            [],
            occasion="smart_casual",
            avatar_gender="male",
        )
        self.assertEqual(top_only_refs, 2)
        self.assertIn("DEFAULT LOWER GARMENT (NO BOTTOM UPLOADED)", top_only_prompt)
        self.assertIn("adult male fashion model", top_only_prompt)

    def test_stylist_isolated_parallel_vision_and_gender_accessory_filtering(self):
        from app.core.flow_b.tryon_stylist import analyze_wardrobe_and_plan_tryon
        from app.core.pipeline.virtual_tryon import prepare_face_identity_ref

        seen_sys_prompts = []

        def fake_post(url, **kwargs):
            body = kwargs["json"]
            sys_txt = body["messages"][0]["content"]
            user_content = body["messages"][1]["content"]
            seen_sys_prompts.append((sys_txt, user_content))

            # Guarantee garment analysis calls ONLY ever see 1 image and NEVER see [AVATAR]
            if "ONE standalone garment" in sys_txt:
                self.assertEqual(sum(1 for p in user_content if p["type"] == "image_url"), 1)
                tag_text = user_content[0]["text"]
                if "[GARMENT_0]" in tag_text:
                    res = {"role": "top", "target_gender": "unisex", "title": "White Crimson-Stripe Football Jersey T-Shirt", "colors": "optic white, crimson red, dark navy", "patterns_and_design": "crew-neck jersey, right-chest Adidas 3-bar logo, left-chest gold club crest, red shoulder 3-stripes, zero buttons"}
                else:
                    res = {"role": "bottom", "target_gender": "unisex", "title": "Sand Cotton Twill Chinos", "colors": "warm sand beige", "patterns_and_design": "slim tapered fit, slant pockets"}
            elif "forensic facial & portrait biometric analyst" in sys_txt:
                self.assertEqual(sum(1 for p in user_content if p["type"] == "image_url"), 1)
                res = {
                    "avatar_gender": "male",
                    "avatar_summary": "50s Indian man with warm wheatish skin tone and average build",
                    "forensic_face_details": "receding salt-and-pepper grey-black hair, soft under-eye contour, broad nose bridge, no glasses worn on face",
                    "original_worn_clothes": "light blue tropical floral collared shirt",
                    "avatar_face_box_2d": [100, 350, 380, 650],
                }
            elif "ONE standalone fashion accessory" in sys_txt:
                tag_text = user_content[0]["text"]
                if "[ACCESSORY_0]" in tag_text:
                    res = {"accessory_type": "wristwatch", "target_gender": "unisex", "body_placement": "left wrist", "title": "Silver Steel Dress Watch", "design_details": "white dial, black leather strap"}
                elif "[ACCESSORY_1]" in tag_text:
                    res = {"accessory_type": "jhumkas", "target_gender": "women", "body_placement": "both earlobes", "title": "Gold Bridal Jhumka Earrings", "design_details": "22K yellow gold bell drops with pearl fringe"}
                else:
                    res = {"accessory_type": "sneakers", "target_gender": "unisex", "body_placement": "worn on both feet", "title": "White Low-Top Sneakers", "design_details": "matte white leather upper, grey sole"}
            else:
                res = {"combinations": [{"title": "Sporty Crisp Look", "occasion": "smart_casual", "style_reason": "Clean jersey + beige chinos.", "selected_garment_indices": [1, 0, 99], "selected_accessory_indices": [0, 1]}]}

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(res)}}]}
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = fake_post

        # Upper head warm skin (100..380), lower torso black shirt (500..999)
        avatar = Image.new("RGB", (400, 600), (15, 15, 18))
        for x in range(140, 260):
            for y in range(60, 228):
                avatar.putpixel((x, y), (205, 170, 145))

        garments = [
            {"index": 0, "image": Image.new("RGB", (150, 150), (245, 245, 245)), "role_hint": "auto", "filename": "white_jersey.jpg"},
            {"index": 1, "image": Image.new("RGB", (150, 150), (200, 190, 160)), "role_hint": "auto", "filename": "beige_chinos.jpg"},
        ]
        accessories = [
            {"index": 0, "image": Image.new("RGB", (100, 100), (180, 180, 190)), "filename": "watch.jpg"},
            {"index": 1, "image": Image.new("RGB", (100, 100), (210, 180, 70)), "filename": "gold_jhumka.jpg"},
            {"index": 2, "image": Image.new("RGB", (100, 100), (245, 245, 245)), "filename": "shoes.jpg"},
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.core.flow_b.tryon_stylist.httpx.Client", return_value=mock_client
        ):
            plan = analyze_wardrobe_and_plan_tryon(avatar, garments, accessories, max_combinations=2)

        combo = plan["combinations"][0]
        self.assertEqual(plan["avatar_gender"], "male")
        self.assertEqual(combo["selected_garment_indices"], [0, 1])
        # Women-only jhumka earrings (1) excluded automatically for male avatar; watch (0) and sneakers (2) kept!
        self.assertEqual(sorted(combo["selected_accessory_indices"]), [0, 2])
        self.assertEqual([a["title"] for a in plan["excluded_accessories"]], ["Gold Bridal Jhumka Earrings"])
        self.assertEqual(plan["avatar_face_box_2d"], [100, 350, 380, 650])
        self.assertEqual(plan["original_worn_clothes"], "light blue tropical floral collared shirt")
        self.assertIn("Adidas 3-bar logo", plan["garments"][0]["patterns_and_design"])

        # High-fill face identity crop (90% of 576px card) retains sharp face detail and pads cleanly around corners
        id_ref = prepare_face_identity_ref(avatar, plan["avatar_face_box_2d"], canvas_size=576)
        self.assertEqual(id_ref.size, (576, 576))
        self.assertEqual(id_ref.getpixel((5, 5)), (244, 241, 236))
        self.assertGreater(id_ref.getpixel((288, 288))[0], 150)

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
        ), patch.object(
            virtual_tryon,
            "enforce_quality_and_refine",
            return_value=(
                Image.new("RGB", (832, 1152), (230, 232, 238)),
                {"passed": True, "face_similarity": 0.82, "validation_grade": "A+", "clothing_coverage": {"passed": True}},
            ),
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
                self.assertEqual((w, h), (832, 1152))
                self.assertEqual(len(refs), 4)  # avatar + top + bottom + watch
                self.assertIn("<image4>", prompt)
            for look in res["combinations"]:
                self.assertTrue(look["url"].startswith("https://bg.arangtik.com/api/outputs/"))
                self.assertEqual(look["resolution"], "832x1152")
                saved_name = look["url"].rsplit("/", 1)[-1]
                self.assertTrue((Path(tmpdir) / saved_name).exists())

            # Top-only upload without pants dynamically applies complementary trousers/jeans
            top_only_res = virtual_tryon.process_virtual_tryon_pipeline(
                avatar_bytes=_make_image_bytes((210, 180, 160), (256, 384)),
                garment_entries=[
                    {"bytes": _make_image_bytes((240, 240, 255)), "role_hint": "top", "filename": "polo_shirt.jpg"},
                ],
                max_combinations=1,
            )
            self.assertTrue(top_only_res["combinations"][0]["garment_details"][-1]["is_complementary"])
            self.assertIn("DEFAULT LOWER GARMENT (NO BOTTOM UPLOADED)", top_only_res["combinations"][0]["prompt_used"])

            # Bottom-only (jeans ONLY, no shirt) automatically adds modest upper top (never shirtless)
            jeans_only_res = virtual_tryon.process_virtual_tryon_pipeline(
                avatar_bytes=_make_image_bytes((210, 180, 160), (256, 384)),
                garment_entries=[
                    {"bytes": _make_image_bytes((40, 70, 130)), "role_hint": "bottom", "filename": "blue_denim_jeans.jpg"},
                ],
                max_combinations=1,
            )
            self.assertEqual(jeans_only_res["combinations"][0]["garment_details"][0]["label"], "Default Modest Complementary Top")
            self.assertIn("NO SHIRTLESS TORSO", jeans_only_res["combinations"][0]["prompt_used"])

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
        self.assertIn("tryonPhotographyStyle", index_html)
        self.assertIn("tryonPosePreference", index_html)

    def test_dynamic_poses_biometrics_and_photography_styles(self):
        from app.core.flow_b.tryon_stylist import (
            _determine_dynamic_pose,
            _determine_photography_style,
            build_qwen_tryon_prompt,
        )

        garments = [{"role": "top", "title": "White T-Shirt", "colors": "white", "patterns_and_design": "crew-neck"}]
        accessories = [{"accessory_type": "wristwatch", "title": "Black Chrono Watch", "body_placement": "left wrist", "design_details": "black leather strap"}]
        pose_casual = _determine_dynamic_pose("male", "smart_casual", garments, accessories, pose_preference="auto")
        self.assertIn("contrapposto", pose_casual.lower())
        self.assertIn("wrist", pose_casual.lower())

        pose_formal = _determine_dynamic_pose("male", "formal", garments, accessories, pose_preference="confident_executive")
        self.assertIn("executive", pose_formal.lower())

        garment_kurta = [{"role": "full_outfit", "title": "White Silk Kurta", "colors": "white", "patterns_and_design": "mandarin collar"}]
        pose_trad = _determine_dynamic_pose("male", "festive", garment_kurta, [], pose_preference="auto")
        self.assertIn("stately", pose_trad.lower())

        label_ed, photo_ed = _determine_photography_style("editorial_lookbook", "formal")
        self.assertIn("Hasselblad", photo_ed)
        label_st, photo_st = _determine_photography_style("premium_studio", "smart_casual")
        self.assertIn("softbox", photo_st.lower())

        avatar_summary = "early-20s Indian young man, slim build, wheatish complexion"
        forensic_face = "styled forward messy fringe hair, thin rectangular wireframe glasses with clear lenses, neat mustache"
        prompt, refs = build_qwen_tryon_prompt(
            avatar_summary=avatar_summary,
            selected_garments=garments,
            selected_accessories=accessories,
            occasion="smart_casual",
            avatar_gender="male",
            forensic_face_details=forensic_face,
            photography_style="premium_studio",
            pose_preference="relaxed_casual",
        )
        self.assertIn("messy fringe", prompt)
        self.assertIn("MANDATORY EYEWEAR LOCK", prompt)
        self.assertIn("contrapposto", prompt.lower())
        self.assertNotIn("0-degree direct front-facing studio lookbook photograph", prompt)

    def test_face_analyzer_yunet_and_sface_embeddings(self):
        import numpy as np
        from app.core.image_processing.face_analyzer import (
            detect_face_and_landmarks,
            extract_face_embedding,
            compute_face_similarity,
            refine_face_identity,
        )
        # Test synthetic image
        img = Image.new("RGB", (256, 256), (120, 120, 120))
        info = detect_face_and_landmarks(img)
        # Synthetic flat image has no face
        self.assertIsNone(info)

        # Test dummy embedding cosine similarity math
        f1 = np.ones(128, dtype=np.float32)
        f2 = np.ones(128, dtype=np.float32)
        sim = compute_face_similarity(f1, f2)
        self.assertAlmostEqual(sim, 1.0, places=2)

        # Orthogonal embeddings
        f3 = np.zeros(128, dtype=np.float32)
        f3[0] = 1.0
        f4 = np.zeros(128, dtype=np.float32)
        f4[1] = 1.0
        sim_ortho = compute_face_similarity(f3, f4)
        self.assertTrue(0.0 <= sim_ortho <= 0.5)

    def test_tryon_validator_coverage_and_delta_e(self):
        from app.core.pipeline.tryon_validator import (
            validate_clothing_coverage,
            compute_garment_color_drift_delta_e,
            validate_virtual_tryon_output,
        )
        # Fully clothed grey model
        img_clothed = Image.new("RGB", (832, 1152), (80, 80, 80))
        cov_clothed = validate_clothing_coverage(img_clothed)
        self.assertTrue(cov_clothed["passed"])
        self.assertTrue(cov_clothed["torso_covered"])
        self.assertTrue(cov_clothed["legs_covered"])

        # Nude-toned / skin-colored torso simulation
        img_skin = Image.new("RGB", (832, 1152), (210, 160, 130))
        cov_skin = validate_clothing_coverage(img_skin)
        self.assertFalse(cov_skin["passed"])

        # Color fidelity delta-E
        g_meta = [{"role": "top", "title": "Dark Navy Shirt", "colors": "navy blue"}]
        delta_e = compute_garment_color_drift_delta_e(img_clothed, g_meta)
        self.assertIsInstance(delta_e, float)

    def test_bounded_retry_triggered_on_validation_failure(self):
        from app.core.pipeline import virtual_tryon

        captured_calls = []

        def fake_gpu(reference_image, prompt, width=None, height=None, seed=None, **kwargs):
            captured_calls.append(seed)
            return Image.new("RGB", (width or 832, height or 1152), (200, 200, 200)), {
                "model": "Qwen/Qwen-Image-2.1-Turbo",
                "gpu_name": "NVIDIA A40",
            }

        call_count = [0]

        def fake_enforce(generated_img, avatar_img, selected_garments, selected_accessories, avatar_gender):
            call_count[0] += 1
            if call_count[0] == 1:
                # First run fails quality validation (e.g. torso uncovered)
                return generated_img, {
                    "passed": False,
                    "face_similarity": 0.25,
                    "validation_grade": "F",
                    "defects": ["uncovered_torso"],
                    "clothing_coverage": {"passed": False, "torso_covered": False, "legs_covered": True},
                }
            # Second run (retry) passes
            return generated_img, {
                "passed": True,
                "face_similarity": 0.85,
                "validation_grade": "A+",
                "defects": [],
                "clothing_coverage": {"passed": True, "torso_covered": True, "legs_covered": True},
            }

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            virtual_tryon, "OUTPUT_DIR", Path(tmpdir)
        ), patch.object(
            virtual_tryon, "is_gpu_worker_configured", return_value=(True, "https://api.runpod.ai/v2/ep_ok")
        ), patch.object(
            virtual_tryon, "generate_garment_via_gpu", side_effect=fake_gpu
        ), patch.object(
            virtual_tryon, "enforce_quality_and_refine", side_effect=fake_enforce
        ), patch.dict(
            os.environ, {"OPENAI_API_KEY": ""}, clear=False
        ):
            res = virtual_tryon.process_virtual_tryon_pipeline(
                avatar_bytes=_make_image_bytes((210, 180, 160), (256, 384)),
                garment_entries=[
                    {"bytes": _make_image_bytes((240, 240, 255)), "role_hint": "top", "filename": "shirt.jpg"},
                    {"bytes": _make_image_bytes((40, 40, 50)), "role_hint": "bottom", "filename": "pant.jpg"},
                ],
                max_combinations=1,
                seed=42,
            )
            # Both initial and retry calls occurred
            self.assertEqual(len(captured_calls), 2)
            # The second call used seed increment + 101
            self.assertEqual(captured_calls[1], captured_calls[0] + 101)
            # Final look report reflects passing retry
            look = res["combinations"][0]
            self.assertTrue(look["validation_metrics"]["retried"])
            self.assertEqual(look["validation_metrics"]["validation_grade"], "A+")


if __name__ == "__main__":
    unittest.main()
