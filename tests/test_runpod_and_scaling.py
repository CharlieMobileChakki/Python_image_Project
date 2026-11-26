"""Verification tests for RunPod Serverless, provider-agnostic GPU routing, and concurrency scaling."""
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


def _make_png_bytes(color=(200, 40, 40, 255), size=(64, 64)) -> bytes:
    img = Image.new("RGBA", size, color)
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

        out_b64 = base64.b64encode(_make_png_bytes((20, 180, 40, 255), (128, 128))).decode("ascii")
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
                    "preclean_time_sec": "0.02",
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
            img, meta = qwen_client.generate_garment_via_gpu(ref, "studio kurta prompt", pre_clean=True)

        self.assertEqual(img.size, (128, 128))
        self.assertEqual(img.mode, "RGBA")
        self.assertEqual(meta["execution_mode"], "runpod_serverless")
        self.assertEqual(meta["gpu_name"], "NVIDIA RTX A6000")

    def test_runpod_in_queue_polls_same_job_id_without_duplicate_run(self):
        from app.core.flow_b import qwen_client

        out_b64 = base64.b64encode(_make_png_bytes((30, 30, 200, 255))).decode("ascii")
        sync_resp = MagicMock(status_code=200)
        sync_resp.json.return_value = {"id": "rp-cold-77", "status": "IN_QUEUE"}

        status_resp = MagicMock(status_code=200)
        status_resp.json.return_value = {
            "id": "rp-cold-77",
            "status": "COMPLETED",
            "output": {"image_base64": out_b64, "metadata": {"gpu_name": "NVIDIA A40"}},
        }

        sync_client = MagicMock()
        sync_client.__enter__.return_value.post.return_value = sync_resp
        poll_client = MagicMock()
        poll_client.__enter__.return_value.get.return_value = status_resp

        with patch.dict(
            os.environ,
            {"QWEN_REMOTE_URL": "https://api.runpod.ai/v2/ep_test", "RUNPOD_API_KEY": "rpa_test"},
            clear=False,
        ), patch("app.core.flow_b.qwen_client.httpx.Client", side_effect=[sync_client, poll_client]), patch(
            "app.core.flow_b.qwen_client.time.sleep"
        ):
            ref = Image.new("RGB", (128, 128), (255, 255, 255))
            img, meta = qwen_client.generate_garment_via_gpu(ref, "studio saree prompt")

        poll_client.__enter__.return_value.post.assert_not_called()
        poll_client.__enter__.return_value.get.assert_called_once()
        called_url = poll_client.__enter__.return_value.get.call_args[0][0]
        self.assertTrue(called_url.endswith("/status/rp-cold-77"))
        self.assertEqual(img.mode, "RGBA")
        self.assertEqual(meta["gpu_name"], "NVIDIA A40")

    def test_multi_http_round_robin_failover(self):
        from app.core.flow_b import qwen_client

        png_bytes = _make_png_bytes((90, 90, 90, 255))
        good_resp = MagicMock(
            status_code=200,
            headers={"content-type": "image/png", "X-GPU-Name": "Node-B", "X-Inference-Time-Sec": "6.5"},
            content=png_bytes,
        )

        # First HTTP node fails, fails over seamlessly to second node
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


class TestGpuWorkerModules(unittest.TestCase):
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

            in_b64 = base64.b64encode(_make_png_bytes((255, 255, 255, 255), (32, 32))).decode("ascii")
            out_bytes = _make_png_bytes((10, 20, 30, 255), (64, 64))
            fake_meta = {"model": "Qwen/Qwen-Image-2.1-Turbo", "gpu_name": "RTX A6000", "inference_time_sec": "7.6"}

            with patch.object(rp_handler, "run_inference_sync", return_value=(out_bytes, {}, fake_meta)):
                res = rp_handler.handler(
                    {"input": {"action": "generate", "image_base64": in_b64, "prompt": "kurta", "transparent": True}}
                )

            self.assertIn("image_base64", res)
            self.assertEqual(res["metadata"]["gpu_name"], "RTX A6000")

    def test_http_worker_and_backend_routes_import_cleanly(self):
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
            self.assertTrue({"/health", "/generate", "/jobs", "/remove-background"}.issubset(route_paths))

        from app.main import app as main_app
        main_paths = {getattr(r, "path", None) for r in main_app.routes}
        self.assertTrue({"/health", "/api/health"}.issubset(main_paths))

        from app.api.routes import router as api_router
        from app.api.reconstruction import reconstruction_router
        api_paths = {getattr(r, "path", None) for r in api_router.routes}
        recon_paths = {getattr(r, "path", None) for r in reconstruction_router.routes}
        self.assertTrue({"/background-removal", "/jewelry-studio", "/set-gpu-url"}.issubset(api_paths))
        self.assertIn("/qwen-image", recon_paths)


if __name__ == "__main__":
    unittest.main()
