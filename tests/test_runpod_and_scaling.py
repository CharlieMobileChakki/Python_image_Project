"""Verification tests for RunPod Serverless, provider-agnostic GPU routing, and direct 1024x1024 studio flow."""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


def _make_png_bytes(color=(255, 255, 255), size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestProviderAgnosticQwenClient(unittest.TestCase):
    def test_runpod_url_normalization_and_detection(self):
        from app.core.flow_b import qwen_client

        with patch.dict(
            os.environ,
            {"QWEN_REMOTE_URL": "https://api.runpod.ai/v2/abc123xyz/runsync"},
            clear=False,
        ):
            ok, url = qwen_client.is_gpu_worker_configured()
            self.assertTrue(ok)
            self.assertEqual(url, "https://api.runpod.ai/v2/abc123xyz")
            self.assertTrue(qwen_client._is_runpod_url(url))

    def test_runpod_endpoint_id_fallback(self):
        from app.core.flow_b import qwen_client

        with patch.dict(
            os.environ,
            {"QWEN_REMOTE_URL": "", "QWEN_GPU_URL": "", "RUNPOD_ENDPOINT_ID": "ep_999"},
            clear=False,
        ):
            ok, url = qwen_client.is_gpu_worker_configured()
            self.assertTrue(ok)
            self.assertEqual(url, "https://api.runpod.ai/v2/ep_999")

    def test_runpod_generate_completed_runsync(self):
        from app.core.flow_b import qwen_client

        out_b64 = base64.b64encode(_make_png_bytes((20, 180, 40), (128, 128))).decode("ascii")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "rp-job-1",
            "status": "COMPLETED",
            "output": {
                "image_base64": out_b64,
                "metadata": {
                    "model": "Qwen/Qwen-Image-2.1-Turbo",
                    "gpu_name": "NVIDIA RTX A6000",
                    "vram_total_gb": 48.0,
                    "inference_time_sec": "7.65",
                    "seed": "42",
                },
            },
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_resp

        with patch.dict(
            os.environ,
            {"QWEN_REMOTE_URL": "https://api.runpod.ai/v2/ep_test", "RUNPOD_API_KEY": "rpa_test_key"},
            clear=False,
        ), patch("app.core.flow_b.qwen_client.httpx.Client", return_value=mock_client):
            ref = Image.new("RGB", (256, 256), (255, 255, 255))
            img, meta = qwen_client.generate_garment_via_gpu(ref, "studio kurta prompt", width=1024, height=1024)

        self.assertEqual(img.size, (128, 128))
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(meta["execution_mode"], "runpod_serverless")
        self.assertEqual(meta["gpu_name"], "NVIDIA RTX A6000")

        sent_input = mock_client.__enter__.return_value.post.call_args.kwargs["json"]["input"]
        self.assertNotIn("transparent", sent_input)
        self.assertNotIn("pre_clean", sent_input)
        self.assertEqual(sent_input["width"], 1024)
        self.assertEqual(sent_input["height"], 1024)

    def test_runpod_in_queue_polls_same_job_id_without_duplicate_run(self):
        from app.core.flow_b import qwen_client

        out_b64 = base64.b64encode(_make_png_bytes((30, 30, 200))).decode("ascii")
        sync_resp = MagicMock(status_code=200)
        sync_resp.json.return_value = {"id": "rp-cold-77", "status": "IN_QUEUE"}

        status_resp = MagicMock(status_code=200)
        status_resp.json.return_value = {
            "id": "rp-cold-77",
            "status": "COMPLETED",
            "output": {"image_base64": out_b64, "metadata": {"gpu_name": "NVIDIA A40"}},
        }

        single_client = MagicMock()
        single_client.__enter__.return_value.post.return_value = sync_resp
        single_client.__enter__.return_value.get.return_value = status_resp

        with patch.dict(
            os.environ,
            {"QWEN_REMOTE_URL": "https://api.runpod.ai/v2/ep_test", "RUNPOD_API_KEY": "rpa_test"},
            clear=False,
        ), patch("app.core.flow_b.qwen_client.httpx.Client", return_value=single_client), patch(
            "app.core.flow_b.qwen_client.time.sleep"
        ):
            ref = Image.new("RGB", (128, 128), (255, 255, 255))
            img, meta = qwen_client.generate_garment_via_gpu(ref, "studio saree prompt")

        single_client.__enter__.return_value.post.assert_called_once()
        single_client.__enter__.return_value.get.assert_called_once()
        called_url = single_client.__enter__.return_value.get.call_args[0][0]
        self.assertTrue(called_url.endswith("/status/rp-cold-77"))
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(meta["gpu_name"], "NVIDIA A40")

    def test_multi_http_round_robin_failover(self):
        from app.core.flow_b import qwen_client

        png_bytes = _make_png_bytes((90, 90, 90))
        good_resp = MagicMock(
            status_code=200,
            headers={"content-type": "image/png", "X-GPU-Name": "Node-B", "X-Inference-Time-Sec": "6.5"},
            content=png_bytes,
        )

        call_urls = []

        def fake_post(url, **kwargs):
            call_urls.append(url)
            if "node1" in url:
                raise qwen_client.httpx.ConnectError("node1 down")
            return good_resp

        client_cm = MagicMock()
        client_cm.__enter__.return_value.post.side_effect = fake_post

        with patch.dict(
            os.environ,
            {"QWEN_REMOTE_URL": "http://node1:8001,http://node2:8001", "QWEN_API_KEY": "secret"},
            clear=False,
        ), patch("app.core.flow_b.qwen_client._rr_counter", 0), patch(
            "app.core.flow_b.qwen_client.httpx.Client", return_value=client_cm
        ), patch("app.core.flow_b.qwen_client.time.sleep"):
            ref = Image.new("RGB", (128, 128), (255, 255, 255))
            img, meta = qwen_client.generate_garment_via_gpu(ref, "shirt")

        self.assertIn("http://node1:8001/generate", call_urls)
        self.assertIn("http://node2:8001/generate", call_urls)
        self.assertEqual(meta["gpu_name"], "Node-B")
        self.assertEqual(img.size, (64, 64))
        self.assertEqual(img.mode, "RGB")

    def test_concurrency_auto_scales_with_runpod_max_workers(self):
        from app.api import routes

        with patch.dict(
            os.environ,
            {
                "MAX_CONCURRENT_JOBS": "auto",
                "QWEN_REMOTE_URL": "https://api.runpod.ai/v2/ep_scale",
                "RUNPOD_MAX_WORKERS": "8",
            },
            clear=False,
        ):
            self.assertEqual(routes._resolve_concurrency_limit(), 16)


class TestGpuWorkerAndStudioPipeline(unittest.TestCase):
    def test_runpod_cached_model_snapshot_resolution(self):
        worker_dir = str(Path(__file__).resolve().parent.parent / "gpu_worker")
        if worker_dir not in sys.path:
            sys.path.insert(0, worker_dir)

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"

        with tempfile.TemporaryDirectory() as tmpdir:
            snap = Path(tmpdir) / "models--Qwen--Qwen-Image-2.1-Turbo" / "snapshots" / "rev123"
            snap.mkdir(parents=True)
            (snap / "model_index.json").write_text("{}", encoding="utf-8")

            with patch.dict(sys.modules, {"torch": fake_torch}):
                import engine
                self.assertEqual(engine.OUTPUT_WIDTH, 1024)
                self.assertEqual(engine.OUTPUT_HEIGHT, 1024)
                with patch.object(engine, "_hub_cache_roots", return_value=[tmpdir]):
                    src, cached = engine.resolve_model_source("Qwen/Qwen-Image-2.1-Turbo", "model_index.json")
                    self.assertTrue(cached)
                    self.assertEqual(Path(src), snap)

    def test_rp_handler_generate_and_health(self):
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

        with patch.dict(sys.modules, {"runpod": fake_runpod, "torch": fake_torch}):
            import rp_handler

            health_out = rp_handler.handler({"input": {"action": "health"}})
            self.assertEqual(health_out["status"], "ok")
            self.assertEqual(health_out["output_size"], "1024x1024")
            self.assertNotIn("rmbg_loaded", health_out)

            in_b64 = base64.b64encode(_make_png_bytes((255, 255, 255), (32, 32))).decode("ascii")
            out_bytes = _make_png_bytes((240, 240, 240), (1024, 1024))
            fake_meta = {"model": "Qwen/Qwen-Image-2.1-Turbo", "gpu_name": "RTX A6000", "inference_time_sec": "12.5"}

            with patch.object(rp_handler, "run_inference_sync", return_value=(out_bytes, {}, fake_meta)):
                res = rp_handler.handler(
                    {"input": {"action": "generate", "image_base64": in_b64, "prompt": "kurta"}}
                )

            self.assertIn("image_base64", res)
            self.assertEqual(res["metadata"]["gpu_name"], "RTX A6000")

    def test_http_worker_and_backend_routes_clean(self):
        worker_dir = str(Path(__file__).resolve().parent.parent / "gpu_worker")
        if worker_dir not in sys.path:
            sys.path.insert(0, worker_dir)

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"

        with patch.dict(sys.modules, {"torch": fake_torch}):
            import server
            route_paths = {r.path for r in server.app.routes}
            self.assertTrue({"/health", "/generate", "/jobs"}.issubset(route_paths))
            self.assertNotIn("/remove-background", route_paths)

        from app.main import app as main_app
        main_paths = {getattr(r, "path", None) for r in main_app.routes}
        self.assertTrue({"/health", "/api/health"}.issubset(main_paths))

        from app.api.routes import router as api_router
        from app.api.reconstruction import reconstruction_router
        api_paths = {getattr(r, "path", None) for r in api_router.routes}
        recon_paths = {getattr(r, "path", None) for r in reconstruction_router.routes}
        self.assertTrue({"/background-removal", "/generate-garment", "/jewelry-studio", "/forward-downstream", "/set-gpu-url"}.issubset(api_paths))
        self.assertIn("/qwen-image", recon_paths)

    def test_direct_1024_studio_endpoint_and_ui(self):
        from app.core.pipeline import unified_pipeline
        from app.main import STATIC_DIR, app as main_app

        openapi = main_app.openapi()
        paths = openapi["paths"]
        self.assertIn("/api/background-removal", paths)
        self.assertNotIn("/api/generate-garment", paths)
        self.assertIn("/api/qwen-image", paths)
        self.assertIn("/api/jewelry-studio", paths)
        self.assertIn("/api/forward-downstream", paths)

        ui_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn("/api/background-removal", ui_html)
        self.assertIn("trackerStep2", ui_html)
        self.assertNotIn("trackerStep3", ui_html)
        self.assertNotIn("/api/generate-garment", ui_html)
        self.assertNotIn("checkerboard", ui_html)

        style_css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
        self.assertIn(".studio-output-bg", style_css)
        self.assertNotIn(".checkerboard-bg", style_css)

        root_dir = Path(__file__).resolve().parent.parent
        dockerfile_text = (root_dir / "gpu_worker" / "Dockerfile").read_text(encoding="utf-8")
        reqs_gpu_text = (root_dir / "gpu_worker" / "requirements-gpu.txt").read_text(encoding="utf-8")
        self.assertIn("cuda12.4-cudnn9-runtime", dockerfile_text)
        self.assertIn("ENTRYPOINT []", dockerfile_text)
        self.assertNotIn("torchaudio", dockerfile_text)
        self.assertNotIn("RMBG", dockerfile_text)
        self.assertNotIn("scikit-image", reqs_gpu_text)

        fake_plan = [{
            "category": "upper_clothes",
            "recognized_label": "Studio Shirt",
            "prompt": "Clean white studio shirt",
            "box_2d": None,
            "item_count": 1,
        }]
        fake_generated = Image.new("RGB", (1024, 1024), (255, 255, 255))

        with patch.object(unified_pipeline, "analyze_garment_and_build_prompts", return_value=fake_plan), patch.object(
            unified_pipeline,
            "generate_garment_via_gpu",
            return_value=(fake_generated, {"model": "Qwen/Qwen-Image-2.1-Turbo", "gpu_name": "RTX A6000"}),
        ) as mock_gpu:
            res = unified_pipeline.process_garment_pipeline(
                image_bytes=_make_png_bytes((220, 220, 220), (512, 512)),
                target_garment="upper_clothes",
            )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_garments"], 1)
        self.assertEqual(res["mode"], "studio_generation")
        self.assertNotIn("background_removal_status", res["garments"][0])
        self.assertEqual(mock_gpu.call_args.kwargs["width"], 1024)
        self.assertEqual(mock_gpu.call_args.kwargs["height"], 1024)

        out_path = unified_pipeline.OUTPUT_DIR / res["garments"][0]["image"]
        try:
            with Image.open(out_path) as saved:
                self.assertEqual(saved.size, (1024, 1024))
                self.assertEqual(saved.mode, "RGB")
        finally:
            if out_path.exists():
                out_path.unlink()


if __name__ == "__main__":
    unittest.main()
