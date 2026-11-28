"""Provider-Agnostic Remote GPU Client for Arangtik Studio (Qwen-Image-2.1-Turbo).

Supports seamless zero-code switching via `.env`:
1. RunPod Serverless Queue (`QWEN_REMOTE_URL=https://api.runpod.ai/v2/<ENDPOINT_ID>`
   or `RUNPOD_ENDPOINT_ID=<ENDPOINT_ID>` with `RUNPOD_API_KEY` / `QWEN_API_KEY`).
2. Standard HTTP GPU Servers (`QWEN_REMOTE_URL=http://<IP>:8001` or comma-separated
   multi-server URLs `http://ip1:8001,http://ip2:8001` with round-robin + failover).
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
import threading
import time
from typing import Any

from dotenv import load_dotenv
import httpx
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_CODES = (429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 530)
_RUNPOD_URL_RE = re.compile(r"^(https?://api\.runpod\.ai/v2/[^/?#]+)", re.IGNORECASE)

_rr_lock = threading.Lock()
_rr_counter = 0

_capacity_lock = threading.Lock()
_capacity_cache: dict[str, Any] = {"value": 1, "ts": 0.0}


def _normalize_worker_url(raw_url: str) -> str:
    cleaned = raw_url.strip().rstrip("/")
    match = _RUNPOD_URL_RE.match(cleaned)
    if match:
        return match.group(1).rstrip("/")
    return cleaned


def _get_remote_urls() -> list[str]:
    raw = (os.getenv("QWEN_REMOTE_URL") or os.getenv("QWEN_GPU_URL") or "").strip()
    urls = [_normalize_worker_url(u) for u in raw.split(",") if u.strip()]
    if urls:
        return urls

    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
    if endpoint_id:
        return [f"https://api.runpod.ai/v2/{endpoint_id}"]
    return []


def _get_remote_url() -> str:
    urls = _get_remote_urls()
    return urls[0] if urls else ""


def _is_runpod_url(url: str) -> bool:
    return "api.runpod.ai" in url.lower()


def _get_api_key(for_runpod: bool = False) -> str:
    if for_runpod:
        return (os.getenv("RUNPOD_API_KEY") or os.getenv("QWEN_API_KEY") or "").strip()
    return (os.getenv("QWEN_API_KEY") or os.getenv("RUNPOD_API_KEY") or "").strip()


def _get_timeout() -> float:
    try:
        return float(os.getenv("QWEN_TIMEOUT_SECONDS", "300"))
    except ValueError:
        return 300.0


def _ordered_urls_for_request() -> list[str]:
    """Return configured worker URLs rotated cleanly for round-robin load sharing."""
    urls = _get_remote_urls()
    if len(urls) <= 1:
        return urls
    global _rr_counter
    with _rr_lock:
        start_idx = _rr_counter % len(urls)
        _rr_counter += 1
    return urls[start_idx:] + urls[:start_idx]


def is_gpu_worker_configured() -> tuple[bool, str]:
    """Check whether a remote GPU endpoint (RunPod Serverless or HTTP) is configured."""
    urls = _get_remote_urls()
    if not urls:
        return (
            False,
            "No GPU worker configured. Set QWEN_REMOTE_URL (e.g. https://api.runpod.ai/v2/<ENDPOINT_ID> or http://<GPU_IP>:8001) in .env.",
        )
    if len(urls) == 1:
        return True, urls[0]
    return True, ",".join(urls)


def _encode_image_bytes(image: Image.Image, max_dim: int = 1600, quality: int = 98) -> bytes:
    """Convert PIL image to RGB JPEG bytes, capping longest edge cleanly."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _runpod_headers() -> dict[str, str]:
    api_key = _get_api_key(for_runpod=True)
    if not api_key:
        raise RuntimeError(
            "RunPod API key missing. Set RUNPOD_API_KEY (or QWEN_API_KEY) in .env."
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _http_headers() -> dict[str, str]:
    api_key = _get_api_key(for_runpod=False)
    return {"X-API-Key": api_key} if api_key else {}


def _probe_single_worker_health(url: str) -> dict[str, Any]:
    if _is_runpod_url(url):
        try:
            headers = _runpod_headers()
        except RuntimeError as exc:
            return {"configured": True, "online": False, "url": url, "backend": "runpod_serverless", "detail": str(exc)}

        try:
            with httpx.Client(timeout=10.0) as client:
                # RunPod built-in /health checks endpoint state without booting/billing a cold GPU
                resp = client.get(f"{url}/health", headers=headers)
                if resp.status_code == 200:
                    payload = resp.json()
                    workers = payload.get("workers") or {}
                    live_workers = int(workers.get("ready", 0)) + int(workers.get("idle", 0)) + int(workers.get("running", 0))
                    max_workers = max(1, int(os.getenv("RUNPOD_MAX_WORKERS", "5")))
                    return {
                        "configured": True,
                        "online": True,
                        "url": url,
                        "backend": "runpod_serverless",
                        "gpu_name": os.getenv("RUNPOD_GPU_LABEL", "RunPod Serverless (48GB A6000/L40S)"),
                        "vram_total_gb": float(os.getenv("RUNPOD_VRAM_GB", "48.0")),
                        "workers_live": live_workers,
                        "workers_total": max_workers,
                        "runpod_health": payload,
                    }
                return {
                    "configured": True,
                    "online": False,
                    "url": url,
                    "backend": "runpod_serverless",
                    "status_code": resp.status_code,
                    "detail": f"RunPod /health HTTP {resp.status_code}: {resp.text[:160]}",
                }
        except Exception as exc:
            return {"configured": True, "online": False, "url": url, "backend": "runpod_serverless", "detail": str(exc)}

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{url}/health", headers=_http_headers())
            if resp.status_code == 200:
                payload = resp.json()
                return {
                    "configured": True,
                    "online": True,
                    "url": url,
                    "backend": "http_worker",
                    "workers_total": int(payload.get("workers_total", 1)),
                    **payload,
                }
            return {
                "configured": True,
                "online": False,
                "url": url,
                "backend": "http_worker",
                "status_code": resp.status_code,
                "detail": f"HTTP {resp.status_code}: {resp.text[:120]}",
            }
    except Exception as exc:
        return {"configured": True, "online": False, "url": url, "backend": "http_worker", "detail": str(exc)}


def check_gpu_worker_health() -> dict[str, Any]:
    """Probe configured remote GPU worker(s) and aggregate online capacity."""
    configured, info = is_gpu_worker_configured()
    if not configured:
        return {"configured": False, "online": False, "workers_total": 0, "detail": info}

    urls = _get_remote_urls()
    results = [_probe_single_worker_health(u) for u in urls]
    online_results = [r for r in results if r.get("online")]

    with _capacity_lock:
        if online_results:
            total_workers = sum(max(1, int(r.get("workers_total", 1))) for r in online_results)
        else:
            total_workers = 0
        _capacity_cache["value"] = max(1, total_workers)
        _capacity_cache["ts"] = time.monotonic()

    if not online_results:
        first = results[0]
        return {**first, "url": info, "workers_total": 0}

    primary = dict(online_results[0])
    primary["url"] = info
    primary["workers_total"] = total_workers
    primary["nodes_online"] = len(online_results)
    primary["nodes_total"] = len(urls)
    return primary


def get_worker_capacity(force_refresh: bool = False) -> int:
    """Return total concurrent worker slots across configured RunPod/HTTP endpoints."""
    urls = _get_remote_urls()
    if not urls:
        return 1

    # Pure RunPod Serverless can always auto-scale up to configured RUNPOD_MAX_WORKERS
    if all(_is_runpod_url(u) for u in urls):
        return max(1, int(os.getenv("RUNPOD_MAX_WORKERS", "5")) * len(urls))

    with _capacity_lock:
        age = time.monotonic() - _capacity_cache["ts"]
        if not force_refresh and age < 15.0:
            return int(_capacity_cache["value"])

    health = check_gpu_worker_health()
    return max(1, int(health.get("workers_total") or len(urls)))


def _cancel_runpod_job(client: httpx.Client, base_url: str, job_id: str, headers: dict[str, str]) -> None:
    """Best-effort cancel on timeout/failure so stale jobs do not pile up in RunPod queue."""
    try:
        client.post(f"{base_url}/cancel/{job_id}", headers=headers, timeout=5.0)
        logger.info("Cancelled timed-out RunPod job %s.", job_id)
    except Exception:
        pass


def _call_runpod_endpoint(
    base_url: str,
    input_payload: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any]:
    """Execute a job via RunPod Serverless `/run` + fast `/status/{id}` polling (zero duplicate jobs)."""
    headers = _runpod_headers()
    t_start = time.perf_counter()

    with httpx.Client(timeout=45.0) as client:
        submit_resp = None
        for attempt in range(1, 4):
            try:
                submit_resp = client.post(
                    f"{base_url}/run",
                    headers=headers,
                    json={"input": input_payload},
                )
                if submit_resp.status_code == 200:
                    break
                if submit_resp.status_code in _TRANSIENT_HTTP_CODES and attempt < 3:
                    time.sleep(2.5)
                    continue
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                if attempt < 3:
                    time.sleep(2.5)
                    continue
                raise RuntimeError(f"Cannot connect to RunPod endpoint {base_url}: {exc}") from exc

        if submit_resp is None or submit_resp.status_code != 200:
            err_txt = submit_resp.text[:300] if submit_resp is not None else "No response"
            code = submit_resp.status_code if submit_resp is not None else "None"
            raise RuntimeError(f"RunPod /run returned HTTP {code}: {err_txt}")

        data = submit_resp.json()
        if str(data.get("status") or "").upper() == "COMPLETED" and "output" in data:
            output = data.get("output") or {}
            if isinstance(output, dict) and output.get("error"):
                raise RuntimeError(f"RunPod worker error: {output['error']}")
            return output

        job_id = str(data["id"])
        logger.info("RunPod job %s queued; polling status...", job_id)

        consecutive_errors = 0
        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed > timeout_sec:
                _cancel_runpod_job(client, base_url, job_id, headers)
                raise RuntimeError(f"RunPod job {job_id} timed out after {timeout_sec:.0f}s.")

            time.sleep(0.5 if elapsed < 25.0 else 1.0)
            try:
                status_resp = client.get(f"{base_url}/status/{job_id}", headers=headers)
            except (httpx.RequestError, httpx.TimeoutException) as poll_exc:
                consecutive_errors += 1
                if consecutive_errors > 8:
                    _cancel_runpod_job(client, base_url, job_id, headers)
                    raise RuntimeError(f"RunPod status polling failed after retries: {poll_exc}") from poll_exc
                continue

            if status_resp.status_code in _TRANSIENT_HTTP_CODES:
                consecutive_errors += 1
                if consecutive_errors > 8:
                    _cancel_runpod_job(client, base_url, job_id, headers)
                    raise RuntimeError(
                        f"RunPod status polling failed ({status_resp.status_code}): {status_resp.text[:200]}"
                    )
                time.sleep(1.5)
                continue

            if status_resp.status_code != 200:
                raise RuntimeError(
                    f"RunPod status check failed ({status_resp.status_code}): {status_resp.text[:200]}"
                )

            consecutive_errors = 0
            state = status_resp.json()
            status = str(state.get("status") or "").upper()
            if status == "COMPLETED":
                logger.info("RunPod job %s completed in %.2fs.", job_id, time.perf_counter() - t_start)
                output = state.get("output") or {}
                if isinstance(output, dict) and output.get("error"):
                    raise RuntimeError(f"RunPod worker error: {output['error']}")
                return output
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod job {job_id} {status}: {state.get('error') or state}")


def _build_http_files_payload(
    img_bytes: bytes | list[bytes],
) -> dict[str, tuple[str, bytes, str]] | list[tuple[str, tuple[str, bytes, str]]]:
    """Build multipart files payload for single-image (`image`) or multi-image (`images`) worker endpoints."""
    if isinstance(img_bytes, list):
        if len(img_bytes) == 1:
            return {"image": ("reference.jpg", img_bytes[0], "image/jpeg")}
        return [
            ("images", (f"reference_{idx}.jpg", raw, "image/jpeg"))
            for idx, raw in enumerate(img_bytes[:10])
        ]
    return {"image": ("reference.jpg", img_bytes, "image/jpeg")}


def _call_http_worker_generate(
    remote_url: str,
    img_bytes: bytes | list[bytes],
    form_data: dict[str, str],
    seed: int | None,
    timeout_sec: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Call standard FastAPI HTTP GPU worker (`/generate` fast-path with `/jobs` fallback)."""
    headers = _http_headers()
    files_payload = _build_http_files_payload(img_bytes)
    t_start = time.perf_counter()
    is_cloudflare = "trycloudflare.com" in remote_url
    direct_timeout = min(timeout_sec, 60.0 if is_cloudflare else 150.0)

    try:
        with httpx.Client(timeout=direct_timeout) as fast_client:
            logger.info("Calling direct /generate on GPU worker %s (timeout=%.0fs)...", remote_url, direct_timeout)
            gen_resp = fast_client.post(
                f"{remote_url}/generate",
                headers=headers,
                files=files_payload,
                data=form_data,
            )
            if gen_resp.status_code == 200 and gen_resp.headers.get("content-type", "").startswith("image/"):
                out_img = Image.open(io.BytesIO(gen_resp.content)).convert("RGB")
                gpu_time = gen_resp.headers.get("X-Inference-Time-Sec", str(round(time.perf_counter() - t_start, 2)))
                meta = {
                    "model": gen_resp.headers.get("X-Model-Name", "Qwen/Qwen-Image-2.1-Turbo"),
                    "gpu_name": gen_resp.headers.get("X-GPU-Name", "NVIDIA GPU"),
                    "vram_gb": gen_resp.headers.get("X-VRAM-GB", ""),
                    "inference_time_sec": str(gpu_time),
                    "seed": gen_resp.headers.get("X-Seed", str(seed or "")),
                    "execution_mode": "http_direct_sync",
                }
                return out_img, meta
    except httpx.ReadTimeout as timeout_exc:
        if not is_cloudflare:
            raise RuntimeError(
                f"GPU Worker /generate timed out after {direct_timeout:.0f}s: {timeout_exc}"
            ) from timeout_exc
        logger.info("Direct /generate hit tunnel cutoff (%s), switching to async /jobs...", timeout_exc)
    except Exception as fast_exc:
        logger.info("Direct /generate unavailable (%s), falling back to /jobs queue...", fast_exc)

    with httpx.Client(timeout=45.0) as client:
        submit_resp = None
        for attempt in range(1, 4):
            try:
                submit_resp = client.post(
                    f"{remote_url}/jobs",
                    headers=headers,
                    files=files_payload,
                    data=form_data,
                )
                if submit_resp.status_code == 200:
                    break
                if submit_resp.status_code in _TRANSIENT_HTTP_CODES and attempt < 3:
                    time.sleep(3.0)
                    continue
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                if attempt < 3:
                    time.sleep(3.0)
                    continue
                raise RuntimeError(f"Cannot connect to GPU Worker at {remote_url}: {exc}") from exc

        if submit_resp is None or submit_resp.status_code != 200:
            err_text = submit_resp.text[:300] if submit_resp is not None else "No response"
            raise RuntimeError(
                f"GPU Worker /jobs returned HTTP {submit_resp.status_code if submit_resp else 'None'}: {err_text}"
            )

        job_id = submit_resp.json()["job_id"]
        consecutive_errors = 0
        while True:
            if time.perf_counter() - t_start > timeout_sec:
                raise RuntimeError(f"GPU Worker job {job_id} timed out after {timeout_sec:.0f}s.")

            time.sleep(1.0)
            try:
                status_resp = client.get(f"{remote_url}/jobs/{job_id}", headers=headers)
            except (httpx.RequestError, httpx.TimeoutException) as poll_exc:
                consecutive_errors += 1
                if consecutive_errors > 8:
                    raise RuntimeError(f"GPU Worker status polling failed: {poll_exc}") from poll_exc
                continue

            if status_resp.status_code in _TRANSIENT_HTTP_CODES:
                consecutive_errors += 1
                if consecutive_errors > 8:
                    raise RuntimeError(
                        f"GPU Worker status failed ({status_resp.status_code}): {status_resp.text[:200]}"
                    )
                time.sleep(2.5)
                continue

            if status_resp.status_code != 200:
                raise RuntimeError(
                    f"GPU Worker status failed ({status_resp.status_code}): {status_resp.text[:200]}"
                )

            consecutive_errors = 0
            state = status_resp.json()
            if state.get("status") == "done":
                break
            if state.get("status") == "error":
                raise RuntimeError(f"GPU Worker job failed: {state.get('error', 'Unknown GPU error')}")

        result_resp = client.get(f"{remote_url}/jobs/{job_id}/result", headers=headers, timeout=60.0)
        if result_resp.status_code != 200:
            raise RuntimeError(f"GPU Worker result fetch failed ({result_resp.status_code}): {result_resp.text[:200]}")

    generated_img = Image.open(io.BytesIO(result_resp.content)).convert("RGB")
    metadata = {
        "model": result_resp.headers.get("X-Model-Name", "Qwen/Qwen-Image-2.1-Turbo"),
        "inference_time_sec": result_resp.headers.get("X-Inference-Time-Sec", "0"),
        "gpu_name": result_resp.headers.get("X-GPU-Name", "NVIDIA GPU"),
        "vram_gb": result_resp.headers.get("X-VRAM-GB", "48.0"),
        "seed": result_resp.headers.get("X-Seed", str(seed or "")),
        "execution_mode": "http_async_queue",
    }
    return generated_img, metadata


def generate_garment_via_gpu(
    reference_image: Image.Image | list[Image.Image] | None = None,
    prompt: str = "",
    steps: int = 12,
    guidance_scale: float = 1.4,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs: Any,
) -> tuple[Image.Image, dict[str, Any]]:
    """Generate studio PNG via configured GPU backend (single or multi-image RunPod/HTTP cluster)."""
    if reference_image is None:
        reference_image = kwargs.get("condition_images") or kwargs.get("condition_image")
    if reference_image is None:
        raise ValueError("reference_image must be provided.")
    configured, msg = is_gpu_worker_configured()
    if not configured:
        raise RuntimeError(msg)

    if isinstance(reference_image, list):
        refs = [img for img in reference_image if img is not None][:10]
        if not refs:
            raise ValueError("reference_image list cannot be empty.")
        is_multi = len(refs) > 1
        img_bytes_list = [
            _encode_image_bytes(
                img,
                max_dim=800 if is_multi else 1600,
                quality=96 if is_multi else 98,
            )
            for img in refs
        ]
        http_files_input: bytes | list[bytes] = img_bytes_list if is_multi else img_bytes_list[0]
    else:
        is_multi = False
        single_bytes = _encode_image_bytes(reference_image, max_dim=1600, quality=98)
        img_bytes_list = [single_bytes]
        http_files_input = single_bytes

    timeout_sec = _get_timeout()
    urls = _ordered_urls_for_request()
    last_exc: Exception | None = None

    for idx, url in enumerate(urls):
        try:
            if _is_runpod_url(url):
                payload: dict[str, Any] = {
                    "action": "generate",
                    "image_base64": base64.b64encode(img_bytes_list[0]).decode("ascii"),
                    "prompt": prompt,
                    "steps": int(steps),
                    "guidance_scale": float(guidance_scale),
                }
                if is_multi:
                    payload["images_base64"] = [
                        base64.b64encode(raw).decode("ascii") for raw in img_bytes_list
                    ]
                if seed is not None:
                    payload["seed"] = int(seed)
                if width and width > 0:
                    payload["width"] = int(width)
                if height and height > 0:
                    payload["height"] = int(height)

                output = _call_runpod_endpoint(url, payload, timeout_sec=timeout_sec)
                b64_out = output.get("image_base64")
                if not b64_out:
                    raise RuntimeError("RunPod completed without 'image_base64' in output.")
                out_img = Image.open(io.BytesIO(base64.b64decode(b64_out))).convert("RGB")
                raw_meta = output.get("metadata") or {}
                meta = {
                    "model": raw_meta.get("model", "Qwen/Qwen-Image-2.1-Turbo"),
                    "gpu_name": raw_meta.get("gpu_name", "RunPod Serverless GPU"),
                    "vram_gb": str(raw_meta.get("vram_total_gb", "48.0")),
                    "inference_time_sec": str(raw_meta.get("inference_time_sec", "0")),
                    "seed": str(raw_meta.get("seed", seed or "")),
                    "execution_mode": "runpod_serverless",
                }
                return out_img, meta

            form_data: dict[str, str] = {
                "prompt": prompt,
                "steps": str(steps),
                "guidance_scale": str(guidance_scale),
            }
            if seed is not None:
                form_data["seed"] = str(seed)
            if width and width > 0:
                form_data["width"] = str(width)
            if height and height > 0:
                form_data["height"] = str(height)

            return _call_http_worker_generate(url, http_files_input, form_data, seed=seed, timeout_sec=timeout_sec)

        except Exception as exc:
            last_exc = exc
            if idx + 1 < len(urls):
                logger.warning("GPU worker %s failed (%s); failing over to next endpoint...", url, exc)
                continue
            raise

    raise RuntimeError(f"All configured GPU workers failed: {last_exc}")
