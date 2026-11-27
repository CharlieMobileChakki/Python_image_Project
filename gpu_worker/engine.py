"""Shared GPU Inference Core for Arangtik Studio (Qwen/Qwen-Image-2.1-Turbo).

Used by both:
- RunPod Serverless Queue Handler (`rp_handler.py`)
- FastAPI HTTP Microservice (`server.py` for Pods / Dedicated / Multi-GPU nodes)

Zero duplicated model loading or inference logic.
"""
from __future__ import annotations

import gc
import glob
import inspect
import io
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
API_KEY = os.getenv("QWEN_API_KEY", "").strip()
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "false").strip().lower() in ("1", "true", "yes")
AUTO_WARMUP_QWEN = os.getenv("AUTO_WARMUP_QWEN", "true").strip().lower() in ("1", "true", "yes")
OUTPUT_WIDTH = int(os.getenv("QWEN_OUTPUT_WIDTH", "1024"))
OUTPUT_HEIGHT = int(os.getenv("QWEN_OUTPUT_HEIGHT", "1024"))

_pipeline = None
_pipeline_warming = False
_pipe_lock = threading.Lock()


def _hub_cache_roots() -> list[str]:
    """Return candidate HuggingFace hub cache directories across RunPod, Pod, and local hosts."""
    roots: list[str] = [
        "/runpod-volume/huggingface-cache/hub",
    ]
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        roots.append(os.path.join(hf_home, "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
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
    - Custom mounted directories (`QWEN_LOCAL_MODEL_DIR`)
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
    """Preload Qwen-Image-2.1-Turbo into VRAM (blocking or background warmup)."""
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


def prepare_ref_image(raw_bytes: bytes, max_in: int = 896) -> Image.Image:
    """Decode EXIF-corrected RGB reference image capped cleanly to ``max_in`` max edge."""
    ref_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    w, h = ref_img.size
    if max(w, h) > max_in:
        scale = max_in / float(max(w, h))
        ref_img = ref_img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.LANCZOS)
    return ref_img


def prepare_ref_images(raw_bytes_list: list[bytes], max_in: int = 768) -> list[Image.Image]:
    """Decode up to 10 EXIF-corrected RGB condition images for Qwen-Image-2.1 multi-image inference."""
    if not raw_bytes_list:
        raise ValueError("At least one reference image is required.")
    return [prepare_ref_image(raw, max_in=max_in) for raw in raw_bytes_list[:10]]


def run_inference_sync(
    ref_img: Image.Image | list[Image.Image],
    prompt: str,
    steps: int = 8,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> tuple[bytes, dict[str, str], dict[str, Any]]:
    """Run direct Qwen-Image-2.1-Turbo single or multi-image generation and encode RGB PNG bytes."""
    pipe = load_qwen21_pipeline()
    used_seed = seed if seed is not None else int(time.time()) % 1000000
    generator = torch.Generator("cpu").manual_seed(used_seed)

    ref_list = [ref_img] if isinstance(ref_img, Image.Image) else list(ref_img)[:10]
    if not ref_list:
        raise ValueError("No reference images provided for Qwen-Image-2.1-Turbo inference.")
    pipe_image: Image.Image | list[Image.Image] = ref_list[0] if len(ref_list) == 1 else ref_list

    target_w = int(width) if width and int(width) > 0 else OUTPUT_WIDTH
    target_h = int(height) if height and int(height) > 0 else OUTPUT_HEIGHT

    t0 = time.perf_counter()

    in_sizes = ref_list[0].size if len(ref_list) == 1 else [img.size for img in ref_list]
    logger.info(
        "Running Qwen-Image-2.1-Turbo (size=%dx%d, refs=%d, in=%s) | prompt: %.200s...",
        target_w,
        target_h,
        len(ref_list),
        in_sizes,
        prompt,
    )

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024 ** 3)
        mem_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        if mem_gb > max(20.0, total_vram_gb * 0.85):
            gc.collect()
            torch.cuda.empty_cache()

    with _pipe_lock, torch.inference_mode():
        pipe_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": pipe_image,
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
        out_img = result.images[0].convert("RGB")
        del result
        logger.info("Qwen pipe execution finished in %.2fs", time.perf_counter() - t_pipe)

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
        "X-Seed": str(used_seed),
    }
    metadata = {
        "model": MODEL_ID,
        "gpu_name": gpu_info["gpu_name"],
        "vram_total_gb": gpu_info["vram_total_gb"],
        "inference_time_sec": str(elapsed),
        "seed": str(used_seed),
    }
    return buf.getvalue(), headers, metadata
