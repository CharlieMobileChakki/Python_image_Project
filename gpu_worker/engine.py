"""Shared GPU Inference Core for Arangtik Flow B (Qwen/Qwen-Image-2.1-Turbo + BRIA RMBG-1.4).

Used by both:
- RunPod Serverless Queue Handler (`rp_handler.py`)
- FastAPI HTTP Microservice (`server.py` for Pods / Dedicated / Multi-GPU nodes)

Zero duplicated model loading, preprocessing, or matting logic.
"""
from __future__ import annotations

import glob
import io
import inspect
import logging
import os
import threading
import time
from typing import Any, Optional

# Lazy CUDA module loading speeds up cold start; avoid expandable_segments:True which can stall large .to("cuda") transfers
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

from dotenv import load_dotenv
from PIL import Image, ImageOps
import torch

load_dotenv()

logger = logging.getLogger("qwen21-gpu-worker")

MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1-Turbo").strip()
RMBG_MODEL_ID = os.getenv("RMBG_MODEL_ID", "briaai/RMBG-1.4").strip()
API_KEY = os.getenv("QWEN_API_KEY", "").strip()
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "false").strip().lower() in ("1", "true", "yes")
AUTO_WARMUP_QWEN = os.getenv("AUTO_WARMUP_QWEN", "true").strip().lower() in ("1", "true", "yes")
OUTPUT_WIDTH = int(os.getenv("QWEN_OUTPUT_WIDTH", "896"))
OUTPUT_HEIGHT = int(os.getenv("QWEN_OUTPUT_HEIGHT", "896"))

_pipeline = None
_pipeline_warming = False
_pipe_lock = threading.Lock()

_rmbg_model = None
_rmbg_lock = threading.Lock()


def _hub_cache_roots() -> list[str]:
    """Return candidate HuggingFace hub cache directories across RunPod, Pod, and local hosts."""
    roots: list[str] = [
        "/runpod-volume/huggingface-cache/hub",
    ]
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        roots.append(os.path.join(hf_home, "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    # Preserve order while deduplicating
    seen: set[str] = set()
    unique: list[str] = []
    for root in roots:
        norm = os.path.normpath(root)
        if norm not in seen:
            seen.add(norm)
            unique.append(root)
    return unique


def resolve_model_source(
    model_id: str,
    required_file: str,
    explicit_env_var: str = "",
) -> tuple[str, bool]:
    """Resolve a HuggingFace model ID to a local cached snapshot directory when available.

    Supports:
    - RunPod Serverless Cached Models (`/runpod-volume/huggingface-cache/hub/models--org--repo/snapshots/...`)
    - Custom mounted directories (`QWEN_LOCAL_MODEL_DIR` / `RMBG_LOCAL_MODEL_DIR`)
    - Standard HuggingFace hub cache (`~/.cache/huggingface/hub`)
    """
    if explicit_env_var:
        explicit_dir = os.getenv(explicit_env_var, "").strip()
        if explicit_dir and os.path.exists(os.path.join(explicit_dir, required_file)):
            return explicit_dir, True

    hf_folder_name = f"models--{model_id.replace('/', '--')}"
    for hub_root in _hub_cache_roots():
        snapshots_glob = os.path.join(hub_root, hf_folder_name, "snapshots", "*")
        for snap_dir in sorted(glob.glob(snapshots_glob), reverse=True):
            if os.path.exists(os.path.join(snap_dir, required_file)):
                return snap_dir, True

    return model_id, False


def has_local_qwen_cache() -> bool:
    """Return True if Qwen model weights are already present on disk."""
    _, has_local = resolve_model_source(MODEL_ID, "model_index.json", "QWEN_LOCAL_MODEL_DIR")
    return has_local


def _from_pretrained_compat(loader_cls: Any, source: str, dtype: torch.dtype, **kwargs: Any) -> Any:
    """Call `.from_pretrained()` handling both `dtype` and legacy `torch_dtype` kwargs."""
    try:
        return loader_cls.from_pretrained(source, dtype=dtype, **kwargs)
    except TypeError:
        return loader_cls.from_pretrained(source, torch_dtype=dtype, **kwargs)


def get_gpu_info() -> dict[str, Any]:
    """Return CUDA device telemetry (name, total VRAM, allocated VRAM)."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "gpu_name": "CPU (No CUDA)", "vram_total_gb": 0.0, "vram_used_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    total_gb = round(props.total_memory / (1024 ** 3), 2)
    used_gb = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
    return {
        "cuda_available": True,
        "gpu_name": props.name,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
    }


def is_pipeline_loaded() -> bool:
    return _pipeline is not None


def is_pipeline_warming() -> bool:
    return _pipeline_warming


def is_rmbg_loaded() -> bool:
    return _rmbg_model is not None


def load_rmbg_model():
    """Load BRIA RMBG-1.4 directly into CUDA VRAM for sub-50ms background removal."""
    global _rmbg_model
    if _rmbg_model is not None:
        return _rmbg_model

    with _rmbg_lock:
        if _rmbg_model is not None:
            return _rmbg_model

        import torch.nn as nn
        if not hasattr(nn.Module, "all_tied_weights_keys"):
            nn.Module.all_tied_weights_keys = {}

        from transformers import AutoModelForImageSegmentation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

        source, has_local = resolve_model_source(RMBG_MODEL_ID, "config.json", "RMBG_LOCAL_MODEL_DIR")
        logger.info("Loading GPU RMBG-1.4 background removal model from '%s' (cached=%s)...", source, has_local)

        model = None
        attempts = (True,) if has_local else (True, False)
        for local_only in attempts:
            try:
                model = _from_pretrained_compat(
                    AutoModelForImageSegmentation,
                    source,
                    dtype=dtype,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                break
            except Exception:
                if not local_only:
                    raise
                logger.info("RMBG-1.4 not found in local cache; fetching from HuggingFace...")

        model.to(device)
        model.eval()
        _rmbg_model = model
        logger.info("RMBG-1.4 background removal model loaded on %s (%s).", device, dtype)
        return _rmbg_model


def remove_background_gpu(image: Image.Image) -> Image.Image:
    """GPU-accelerated salient background removal using official BRIA RMBG-1.4 preprocessing."""
    import numpy as np
    from torchvision.transforms.functional import normalize

    model = load_rmbg_model()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    orig_w, orig_h = image.size
    rgb_img = image.convert("RGB")

    # Official BRIA RMBG-1.4 preprocessing: 1024x1024 float [0, 1], mean=[0.5,0.5,0.5], std=[1.0,1.0,1.0]
    im_np = np.array(rgb_img)
    if len(im_np.shape) < 3:
        im_np = im_np[:, :, np.newaxis]
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    im_tensor = torch.nn.functional.interpolate(im_tensor, size=(1024, 1024), mode="bilinear")
    im_tensor = im_tensor / 255.0
    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(device=device, dtype=dtype)

    with torch.inference_mode():
        output = model(im_tensor)
        pred = output
        while isinstance(pred, (tuple, list)):
            pred = pred[0]

        while pred.dim() > 4:
            pred = pred.squeeze(0)
        while pred.dim() < 4:
            pred = pred.unsqueeze(0)

        # Official BRIA RMBG-1.4 post-processing (no sigmoid; already outputs normalized salience map)
        pred = torch.nn.functional.interpolate(
            pred.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        ma = torch.max(pred)
        mi = torch.min(pred)
        if (ma - mi) > 1e-6:
            pred = (pred - mi) / (ma - mi)

        # Clear faint background haze under 0.15 so empty canvas areas are pure 0 alpha
        pred = torch.where(pred < 0.15, torch.zeros_like(pred), pred)
        pred_np = pred[0, 0].cpu().numpy()

    mask_arr = (np.clip(pred_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask = Image.fromarray(mask_arr, mode="L")

    rgba_img = rgb_img.copy()
    rgba_img.putalpha(mask)
    return rgba_img


def load_qwen21_pipeline():
    """Load official Qwen/Qwen-Image-2.1-Turbo in lossless bfloat16 directly into GPU VRAM."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipe_lock:
        if _pipeline is not None:
            return _pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected on this server.")

        import gc
        from diffusers import QwenImage21Pipeline

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_source, has_local = resolve_model_source(MODEL_ID, "model_index.json", "QWEN_LOCAL_MODEL_DIR")
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(
            "Loading %s from '%s' (cached=%s, precision=%s, VRAM=%.1f GB)...",
            MODEL_ID,
            model_source,
            has_local,
            dtype,
            total_vram_gb,
        )
        t0 = time.perf_counter()

        pipe = None
        loaded_via_device_map = False

        # Prefer direct device_map="cuda" on >36 GB GPUs to stream weights straight into VRAM
        if total_vram_gb > 36:
            try:
                t_dm = time.perf_counter()
                logger.info("Attempting direct CUDA stream load (device_map='cuda')...")
                pipe = _from_pretrained_compat(
                    QwenImage21Pipeline,
                    model_source,
                    dtype=dtype,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    local_files_only=has_local,
                )
                loaded_via_device_map = True
                logger.info("Direct CUDA stream load succeeded in %.2fs.", time.perf_counter() - t_dm)
            except Exception as dm_exc:
                logger.warning("Direct device_map='cuda' unsupported (%s); falling back to staged component load.", dm_exc)
                pipe = None
                gc.collect()
                torch.cuda.empty_cache()

        if pipe is None:
            pipe = _from_pretrained_compat(
                QwenImage21Pipeline,
                model_source,
                dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=has_local,
            )

        if total_vram_gb > 36 and not loaded_via_device_map:
            logger.info("High VRAM GPU detected (%.1f GB). Moving components to CUDA...", total_vram_gb)
            for comp_name, comp in pipe.components.items():
                if isinstance(comp, torch.nn.Module):
                    tc = time.perf_counter()
                    for p in (*comp.parameters(), *comp.buffers()):
                        if p.device.type == "cpu" and not p.is_contiguous():
                            p.data = p.data.contiguous()
                    comp.to("cuda")
                    torch.cuda.synchronize()
                    gc.collect()
                    logger.info("Moved '%s' to CUDA in %.2fs.", comp_name, time.perf_counter() - tc)
            logger.info("Model 100%% resident in CUDA VRAM.")
        elif total_vram_gb <= 36:
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
                logger.info("Enabled model_cpu_offload (safe fallback for <=36GB VRAM GPUs).")
            else:
                pipe.to("cuda")

        # Tensor Core TF32 acceleration & cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        if hasattr(pipe, "disable_attention_slicing"):
            pipe.disable_attention_slicing()

        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "disable_slicing"):
                pipe.vae.disable_slicing()
            if hasattr(pipe.vae, "disable_tiling"):
                pipe.vae.disable_tiling()

        _pipeline = pipe
        logger.info("Model %s ready in %.2fs.", MODEL_ID, time.perf_counter() - t0)
        return _pipeline


def preload_models(background_qwen: bool = False) -> None:
    """Preload RMBG-1.4 and Qwen-Image-2.1-Turbo into VRAM (blocking or background Qwen warmup)."""
    try:
        logger.info("Preloading GPU RMBG-1.4 into VRAM...")
        load_rmbg_model()
    except Exception as exc:
        logger.warning("RMBG preload failed (will retry on first request): %s", exc)

    if not background_qwen:
        load_qwen21_pipeline()
        return

    should_warmup = PRELOAD_MODEL or (AUTO_WARMUP_QWEN and has_local_qwen_cache())
    if should_warmup:
        def _bg():
            global _pipeline_warming
            _pipeline_warming = True
            try:
                logger.info("Background Qwen warmup thread started...")
                load_qwen21_pipeline()
                logger.info("Background Qwen warmup complete - GPU pipeline resident.")
            except Exception as exc:
                logger.error("Background Qwen warmup failed (will retry on first request): %s", exc)
            finally:
                _pipeline_warming = False

        threading.Thread(target=_bg, daemon=True, name="qwen-warmup").start()


def prepare_ref_image(raw_bytes: bytes) -> Image.Image:
    """Decode and preserve RGBA format capped cleanly to 896px max edge."""
    ref_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGBA")
    w, h = ref_img.size
    max_in = 896
    if max(w, h) > max_in:
        scale = max_in / float(max(w, h))
        ref_img = ref_img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.LANCZOS)
    return ref_img


def run_inference_sync(
    ref_img: Image.Image,
    prompt: str,
    steps: int = 8,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    transparent: bool = False,
    pre_clean: bool = False,
) -> tuple[bytes, dict[str, str], dict[str, Any]]:
    """Run Qwen-Image-2.1-Turbo 8-step editing with KV cache and optional in-GPU pre-clean + matting."""
    import numpy as np

    pipe = load_qwen21_pipeline()
    used_seed = seed if seed is not None else int(time.time()) % 1000000
    generator = torch.Generator("cpu").manual_seed(used_seed)

    target_w = int(width) if width and int(width) > 0 else OUTPUT_WIDTH
    target_h = int(height) if height and int(height) > 0 else OUTPUT_HEIGHT

    t0 = time.perf_counter()

    preclean_sec = 0.0
    if pre_clean:
        t_pre = time.perf_counter()
        try:
            rgba_ref = remove_background_gpu(ref_img)
            alpha_ch = rgba_ref.split()[3]
            if np.sum(np.array(alpha_ch) > 30) > 0.05 * (rgba_ref.width * rgba_ref.height):
                white_bg_ref = Image.new("RGB", rgba_ref.size, (255, 255, 255))
                white_bg_ref.paste(rgba_ref, (0, 0), mask=alpha_ch)
                ref_img = white_bg_ref.convert("RGBA")
                preclean_sec = round(time.perf_counter() - t_pre, 3)
                logger.info("In-GPU reference pre-clean completed in %.3fs.", preclean_sec)
        except Exception as exc:
            logger.warning("In-GPU reference pre-clean failed (%s); using original reference image.", exc)

    logger.info(
        "Running Qwen-Image-2.1-Turbo (size=%dx%d, in=%s, pre_clean=%s, transparent=%s) | prompt: %.200s...",
        target_w,
        target_h,
        ref_img.size,
        pre_clean,
        transparent,
        prompt,
    )

    # Adaptive VRAM cache threshold so 48GB, 80GB, and 96GB GPUs reclaim fragmented blocks only near capacity
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024 ** 3)
        mem_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        if mem_gb > max(20.0, total_vram_gb * 0.85):
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    with _pipe_lock, torch.inference_mode():
        pipe_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": ref_img,
            "width": target_w,
            "height": target_h,
            "use_kv_cache": True,
            "generator": generator,
        }
        call_params = inspect.signature(pipe.__call__).parameters
        if "num_inference_steps" in call_params and steps and steps > 0:
            pipe_kwargs["num_inference_steps"] = int(steps)
        if "true_cfg_scale" in call_params and guidance_scale and guidance_scale > 0:
            pipe_kwargs["true_cfg_scale"] = float(guidance_scale)
        elif "guidance_scale" in call_params and guidance_scale and guidance_scale > 0:
            pipe_kwargs["guidance_scale"] = float(guidance_scale)

        t_pipe = time.perf_counter()
        result = pipe(**pipe_kwargs)
        out_img: Image.Image = result.images[0].copy()
        del result
        logger.info("Qwen pipe execution finished in %.2fs", time.perf_counter() - t_pipe)

    if transparent:
        t_rmbg = time.perf_counter()
        try:
            out_img = remove_background_gpu(out_img)
            logger.info("GPU background removal finished in %.3fs", time.perf_counter() - t_rmbg)
        except Exception as exc:
            logger.exception("GPU background removal failed (%s); returning original RGB image.", exc)

    t_save = time.perf_counter()
    buf = io.BytesIO()
    out_img.save(buf, format="PNG", compress_level=1)
    logger.info("PNG encode finished in %.2fs", time.perf_counter() - t_save)

    elapsed = round(time.perf_counter() - t0, 2)
    gpu_info = get_gpu_info()

    headers = {
        "X-Model-Name": MODEL_ID,
        "X-GPU-Name": str(gpu_info["gpu_name"]),
        "X-VRAM-GB": str(gpu_info["vram_total_gb"]),
        "X-Inference-Time-Sec": str(elapsed),
        "X-Preclean-Time-Sec": str(preclean_sec),
        "X-Seed": str(used_seed),
        "X-Transparent": str(transparent).lower(),
    }
    metadata = {
        "model": MODEL_ID,
        "gpu_name": gpu_info["gpu_name"],
        "vram_total_gb": gpu_info["vram_total_gb"],
        "inference_time_sec": str(elapsed),
        "preclean_time_sec": str(preclean_sec),
        "seed": str(used_seed),
        "transparent": bool(transparent),
    }
    return buf.getvalue(), headers, metadata


def run_rmbg_sync(img: Image.Image) -> tuple[bytes, float]:
    """Run standalone RMBG-1.4 background removal and encode PNG bytes."""
    t0 = time.perf_counter()
    rgba = remove_background_gpu(img)
    elapsed = round(time.perf_counter() - t0, 3)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), elapsed
