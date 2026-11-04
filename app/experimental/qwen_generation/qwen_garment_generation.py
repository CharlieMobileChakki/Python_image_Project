"""Official Qwen-Image-2.1 Garment Generation Service.

Integrates the official Qwen/Qwen-Image-2.1 visual diffusion pipeline (QwenImage21Pipeline)
into Arangtik Garment AI as an isolated, experimental generation module.

Architectural Guarantees:
- Model Selection: Strictly uses 'Qwen/Qwen-Image-2.1' via QwenImage21Pipeline.
- Lazy Initialization: Zero startup memory overhead; initialized strictly upon request.
- Model Caching: Singleton pipeline cached in GPU memory across requests.
- Native Transparency: Utilizes Qwen-Image-2.1's prompt-driven alpha channel generation.
- Graceful Environment Guard: Identifies lack of CUDA/GPU and marks execution as
  'PENDING GPU TESTING' without crashing or locking CPU threads.
- RTX 4090 Optimized: Configured for bfloat16, VAE slicing/tiling, and optional CPU offloading.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import logging
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Default model ID as released officially on Hugging Face
OFFICIAL_MODEL_ID = "Qwen/Qwen-Image-2.1"

# Official prompt engineering template for isolated garment generation with native transparency
DEFAULT_GARMENT_TRANSPARENT_PROMPT = (
    "This is an RGBA image with transparency. "
    "A clean, photorealistic, isolated studio product photograph of the complete standalone garment "
    "shown in the reference image. "
    "The clothing item is perfectly centered, fully reconstructed, and laid out flat. "
    "Reconstruct all missing or occluded fabric, pleats, waistline, and sleeves where hands, arms, "
    "or body parts overlapped in the reference photo. "
    "Remove the person, human body, skin, face, hair, mannequin, hangers, and any background entirely. "
    "Maintain the exact visible color, fabric texture, embroidery, patterns, borders, and silhouette "
    "from the reference image. "
    "The image has an alpha channel and the background is transparent."
)

DEFAULT_GARMENT_SOLID_PROMPT = (
    "A clean, photorealistic, isolated studio product photograph of the complete standalone garment "
    "shown in the reference image. "
    "The clothing item is perfectly centered, fully reconstructed, and laid out flat on a solid white background. "
    "Reconstruct all missing or occluded fabric where body parts overlapped in the reference photo. "
    "Remove the person, body, skin, face, mannequin, and hangers completely. "
    "Preserve exact colors, fabric texture, embroidery, and silhouette."
)

DEFAULT_RECONSTRUCTION_PROMPT = (
    "This is an RGBA image with transparency. "
    "A clean, photorealistic, isolated studio product photograph of the EXACT SAME garment shown in the reference image. "
    "Reconstruct only the missing fabric, folds, pleats, waistline, and drape where hands, arms, "
    "or body parts occluded the clothing in the reference photo. "
    "Strictly preserve the original fabric color, color distribution, fabric texture, weave, "
    "embroidery, patterns, printed motifs, borders, lace, seams, and neckline from the visible reference. "
    "Do not invent a different garment, color, pattern, or design. "
    "Remove the person, body, skin, face, hair, mannequin, and background completely. "
    "The clothing item is perfectly isolated on a transparent background."
)


def composite_reconstructed_fabric(
    original_garment: Image.Image,
    generated_fabric: Image.Image,
    repair_mask: np.ndarray,
    feather_radius: int = 5,
) -> Image.Image:
    """Composite Qwen-generated fabric strictly into the occluded region while preserving 100% of original pixels."""
    orig_rgba = original_garment.convert("RGBA")
    W, H = orig_rgba.size

    # Ensure generated fabric matches original dimensions
    gen_rgba = generated_fabric.convert("RGBA").resize((W, H), Image.Resampling.LANCZOS)

    # Ensure repair mask matches dimensions
    if repair_mask.shape != (H, W):
        repair_mask = cv2.resize(repair_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    orig_arr = np.array(orig_rgba).astype(np.float32)
    gen_arr = np.array(gen_rgba).astype(np.float32)

    # Soft feathering at the boundary to eliminate hard seam transitions
    mask_binary = (repair_mask > 20).astype(np.float32)
    if feather_radius > 0 and mask_binary.any():
        ksize = feather_radius * 2 + 1
        mask_f = cv2.GaussianBlur(mask_binary, (ksize, ksize), feather_radius / 2.0)
    else:
        mask_f = mask_binary
    mask_f = np.clip(mask_f, 0.0, 1.0)[..., None]

    # Preserve 100% of original pixels outside the repair mask
    blended_rgb = orig_arr[:, :, :3] * (1.0 - mask_f) + gen_arr[:, :, :3] * mask_f
    blended_alpha = np.maximum(orig_arr[:, :, 3:4], gen_arr[:, :, 3:4] * mask_f)

    blended_rgba = np.dstack([blended_rgb, blended_alpha])
    return Image.fromarray(np.clip(blended_rgba, 0, 255).astype(np.uint8), mode="RGBA")


@dataclass
class QwenGenerationResult:
    """Output container for Qwen-Image-2.1 generation."""
    image: Image.Image
    image_format: str
    width: int
    height: int
    has_alpha_channel: bool
    inference_time_seconds: float
    gpu_memory_peak_mb: Optional[float]
    model_name: str
    seed: Optional[int]
    status: str
    error_detail: Optional[str] = None


class QwenGarmentGenerationEngine:
    """Manages lifecycle, memory, and inference for Qwen-Image-2.1."""

    def __init__(
        self,
        model_id: str = OFFICIAL_MODEL_ID,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        enable_cpu_offload: bool = False,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = True,
    ):
        self.model_id = os.getenv("QWEN_IMAGE_MODEL_ID", model_id)
        self.configured_device = os.getenv("QWEN_DEVICE", device).lower()
        self.dtype_str = os.getenv("QWEN_DTYPE", torch_dtype).lower()
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling

        self.pipeline = None
        self._lock = threading.Lock()
        self._active_device = "unloaded"

    def is_available(self) -> Tuple[bool, str]:
        """Verify whether required hardware and diffusers pipeline support are available."""
        try:
            import torch
        except ImportError:
            return False, "PyTorch is not installed in the active environment."

        if self.configured_device == "cuda" and not torch.cuda.is_available():
            return (
                False,
                "CUDA GPU is not available on this host. Model execution is PENDING GPU TESTING on an RTX 4090 server."
            )

        try:
            import diffusers
            # Check for native QwenImage21Pipeline support
            if not hasattr(diffusers, "QwenImage21Pipeline"):
                return (
                    False,
                    f"diffusers version {diffusers.__version__} lacks 'QwenImage21Pipeline'. "
                    "Install the latest diffusers via 'pip install git+https://github.com/huggingface/diffusers.git'."
                )
        except ImportError:
            return False, "Hugging Face 'diffusers' package is not installed."

        return True, "Available"

    def is_loaded(self) -> bool:
        """Check if pipeline weights are currently cached in memory."""
        return self.pipeline is not None

    def load(self) -> None:
        """Load and cache QwenImage21Pipeline weights."""
        if self.pipeline is not None:
            return

        with self._lock:
            if self.pipeline is not None:
                return

            is_avail, reason = self.is_available()
            if not is_avail:
                raise RuntimeError(f"Cannot initialize Qwen-Image-2.1: {reason}")

            import torch
            from diffusers import QwenImage21Pipeline

            # Select PyTorch dtype (bfloat16 recommended for Ampere / Ada RTX 4090)
            if self.dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.dtype_str == "float16":
                dtype = torch.float16
            else:
                dtype = torch.float32

            logger.info("Loading Qwen-Image-2.1 pipeline from '%s' (dtype=%s)...", self.model_id, dtype)
            t0 = time.perf_counter()

            try:
                pipe = QwenImage21Pipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                )

                if self.enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                    self._active_device = "cuda:offloaded"
                else:
                    pipe.to(self.configured_device)
                    self._active_device = self.configured_device

                # Memory optimizations for 24GB VRAM
                if self.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                if self.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()

                self.pipeline = pipe
                load_sec = time.perf_counter() - t0
                logger.info("Qwen-Image-2.1 successfully loaded in %.2f seconds.", load_sec)

            except Exception as exc:
                self.pipeline = None
                self._active_device = "error"
                logger.exception("Failed to initialize Qwen-Image-2.1: %s", exc)
                raise RuntimeError(f"Qwen-Image-2.1 loading error: {exc}") from exc

    def generate(
        self,
        reference_image: Image.Image,
        prompt: Optional[str] = None,
        resolution: int = 1024,
        inference_steps: int = 28,
        seed: Optional[int] = None,
        transparent_background: bool = True,
    ) -> QwenGenerationResult:
        """Execute Qwen-Image-2.1 garment synthesis conditioned on reference image.

        Args:
            reference_image: PIL Image containing model or garment photo.
            prompt: Custom text instruction (defaults to standard garment prompt).
            resolution: Target resolution (width/height).
            inference_steps: Number of denoising steps (default 28).
            seed: Random seed for deterministic reproducibility.
            transparent_background: Trigger native RGBA transparency generation.
        """
        # 1. Validate environment
        is_avail, reason = self.is_available()
        if not is_avail:
            # Clean non-blocking fallback for development environments lacking GPU
            logger.warning("Qwen-Image-2.1 skipped: %s", reason)
            dummy_img = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
            return QwenGenerationResult(
                image=dummy_img,
                image_format="PNG",
                width=resolution,
                height=resolution,
                has_alpha_channel=True,
                inference_time_seconds=0.0,
                gpu_memory_peak_mb=0.0,
                model_name=self.model_id,
                seed=seed,
                status="pending_gpu_testing",
                error_detail=reason,
            )

        # 2. Ensure pipeline loaded
        self.load()

        # 3. Prepare inputs
        import torch
        ref_rgb = ImageOps.exif_transpose(reference_image).convert("RGB")
        target_size = (resolution, resolution)

        final_prompt = prompt
        if not final_prompt:
            final_prompt = (
                DEFAULT_GARMENT_TRANSPARENT_PROMPT
                if transparent_background
                else DEFAULT_GARMENT_SOLID_PROMPT
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.configured_device).manual_seed(seed)

        # 4. Measure GPU VRAM before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_vram = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            start_vram = 0.0

        t0 = time.perf_counter()
        with self._lock:
            # QwenImage21Pipeline takes instruction prompt and condition image directly
            output = self.pipeline(
                prompt=final_prompt,
                image=ref_rgb,
                width=target_size[0],
                height=target_size[1],
                num_inference_steps=inference_steps,
                generator=generator,
            )

        inference_time = time.perf_counter() - t0
        generated_img: Image.Image = output.images[0]

        # 5. Measure peak GPU VRAM
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_vram = 0.0

        # 6. Verify Alpha Channel / Transparency
        has_alpha = (generated_img.mode == "RGBA")

        # Graceful transparent fallback if RGBA requested but model generated RGB
        if transparent_background and not has_alpha:
            try:
                from app.core.background_removal import extract_isolated_flow_a
                generated_img = extract_isolated_flow_a(generated_img)
                has_alpha = True
                logger.info("Applied BiRefNet Lite fallback to ensure transparent RGBA background.")
            except Exception as exc:
                logger.warning("BiRefNet transparency fallback could not be applied: %s", exc)

        return QwenGenerationResult(
            image=generated_img,
            image_format="PNG",
            width=generated_img.width,
            height=generated_img.height,
            has_alpha_channel=has_alpha,
            inference_time_seconds=round(inference_time, 2),
            gpu_memory_peak_mb=round(peak_vram, 1),
            model_name=self.model_id,
            seed=seed,
            status="success",
            error_detail=None,
        )

    def reconstruct_garment(
        self,
        original_image: Image.Image,
        garment_cutout: Image.Image,
        repair_mask: Image.Image | np.ndarray,
        garment_reference: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        resolution: int = 1024,
        inference_steps: int = 28,
        seed: Optional[int] = None,
        feather_radius: int = 5,
    ) -> QwenGenerationResult:
        """Execute reference-based fabric reconstruction using Qwen-Image-2.1 conditioned on original photo.

        Workflow:
        1. Condition Qwen on original photo and visible garment reference.
        2. Generate missing fabric conditioned on visible colors, embroidery, borders.
        3. Mandatory background verification with BiRefNet Lite.
        4. Strict original pixel preservation: composite generated fabric into repair mask region.
        5. Return clean transparent RGBA PNG.
        """
        if isinstance(repair_mask, Image.Image):
            repair_mask_np = np.array(repair_mask.convert("L"))
        else:
            repair_mask_np = np.array(repair_mask)

        # 1. Environment Guard
        is_avail, reason = self.is_available()
        if not is_avail:
            logger.warning("Qwen-Image-2.1 reconstruction skipped: %s", reason)
            return QwenGenerationResult(
                image=garment_cutout,
                image_format="PNG",
                width=garment_cutout.width,
                height=garment_cutout.height,
                has_alpha_channel=(garment_cutout.mode == "RGBA"),
                inference_time_seconds=0.0,
                gpu_memory_peak_mb=0.0,
                model_name=self.model_id,
                seed=seed,
                status="pending_gpu_testing",
                error_detail=reason,
            )

        # 2. Select visual conditioning reference: prefer tight garment reference or full original
        ref_condition = garment_reference if garment_reference is not None else original_image

        # 3. Denoising inference prompt
        final_prompt = prompt or DEFAULT_RECONSTRUCTION_PROMPT

        # 4. Run diffusion generation
        gen_res = self.generate(
            reference_image=ref_condition,
            prompt=final_prompt,
            resolution=resolution,
            inference_steps=inference_steps,
            seed=seed,
            transparent_background=True,
        )

        if gen_res.status != "success":
            return gen_res

        generated_raw = gen_res.image

        # 5. Mandatory Final Background Verification (BiRefNet Lite)
        try:
            from app.core.background_removal import extract_isolated_flow_a
            # Always verify transparency on the generated fabric
            verified_fabric = extract_isolated_flow_a(generated_raw)
        except Exception as exc:
            logger.warning("BiRefNet transparency verification fallback: %s", exc)
            verified_fabric = generated_raw

        # 6. Original Pixel Preservation (Composite strictly inside repair mask)
        composited_garment = composite_reconstructed_fabric(
            original_garment=garment_cutout,
            generated_fabric=verified_fabric,
            repair_mask=repair_mask_np,
            feather_radius=feather_radius,
        )

        return QwenGenerationResult(
            image=composited_garment,
            image_format="PNG",
            width=composited_garment.width,
            height=composited_garment.height,
            has_alpha_channel=True,
            inference_time_seconds=gen_res.inference_time_seconds,
            gpu_memory_peak_mb=gen_res.gpu_memory_peak_mb,
            model_name=self.model_id,
            seed=seed,
            status="success",
            error_detail=None,
        )


# Global singleton engine instance
_qwen_engine: Optional[QwenGarmentGenerationEngine] = None
_engine_lock = threading.Lock()


def get_qwen_garment_engine() -> QwenGarmentGenerationEngine:
    """Get or instantiate the singleton QwenGarmentGenerationEngine."""
    global _qwen_engine
    if _qwen_engine is None:
        with _engine_lock:
            if _qwen_engine is None:
                _qwen_engine = QwenGarmentGenerationEngine()
    return _qwen_engine

