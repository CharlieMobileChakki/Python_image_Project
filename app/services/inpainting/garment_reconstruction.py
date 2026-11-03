"""Optional advanced garment reconstruction engine using SDXL Inpainting + IP-Adapter Plus.

Extracted from arangtik_garment_reconstruction standalone project.
Architectural guarantees:
- Lazy initialization: Never loaded during server startup unless explicitly invoked.
- Independent from LaMa: Existing LaMa ONNX remains the primary, zero-overhead engine.
- Device-adaptive: Automatically detects CUDA GPU; raises informative errors on CPU if unconfigured.
- Reference conditioning: Conditions visual synthesis strictly on garment fabric reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Optional

from PIL import Image

from app.services.inpainting.reconstruction_utils import (
    composite_garment,
    prepare_mask,
    resize_to_multiple_of_8,
    validate_repair_mask,
)

logger = logging.getLogger(__name__)

GARMENT_PROMPT = (
    "Photorealistic isolated product photograph of the SAME garment shown in the reference image. "
    "Reconstruct only the missing fabric region of the existing garment. "
    "Match the reference garment's exact visible color family, fabric, weave, embroidery, "
    "neckline, sleeve construction, decorative border, stitching, motifs, folds and drape. "
    "Continue the garment naturally from the preserved surrounding fabric. "
    "Show the complete clothing item only, with no wearer and no human parts. "
    "The result is a realistic flat invisible-form garment product cutout, not a person wearing it."
)

GARMENT_NEGATIVE_PROMPT = (
    "person, human, woman, man, face, head, hair, neck, skin, hand, hands, arm, arms, legs, "
    "feet, body, mannequin, model wearing clothes, portrait, human silhouette, extra garment, "
    "different outfit, changed garment color, changed embroidery, invented logos, text, watermark, "
    "background, room, furniture, hanger, duplicate clothing, floating body parts, "
    "cartoon, illustration, 3d character, plastic fabric, blurry, low resolution, distorted seams"
)


@dataclass
class ReconstructionResult:
    image: Image.Image
    inference_seconds: float
    seed: int
    device: str
    width: int
    height: int


class GarmentReconstructionEngine:
    """Lazy-loaded SDXL Inpainting + IP-Adapter Plus engine for advanced fabric reconstruction."""

    def __init__(
        self,
        sdxl_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_subfolder: str = "sdxl_models",
        ip_adapter_weight: str = "ip-adapter-plus_sdxl_vit-h.safetensors",
        ip_adapter_image_encoder_subfolder: str = "models/image_encoder",
        device: str = "cuda",
        allow_cpu_fallback: bool = False,
        use_model_cpu_offload: bool = False,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = True,
        max_image_side: int = 1024,
        mask_blur_radius: float = 1.5,
        default_steps: int = 30,
        default_guidance_scale: float = 5.5,
        default_strength: float = 0.95,
        default_ip_adapter_scale: float = 0.75,
    ):
        self.sdxl_model_id = os.getenv("SDXL_MODEL_ID", sdxl_model_id)
        self.ip_adapter_repo = os.getenv("IP_ADAPTER_REPO", ip_adapter_repo)
        self.ip_adapter_subfolder = ip_adapter_subfolder
        self.ip_adapter_weight = ip_adapter_weight
        self.ip_adapter_image_encoder_subfolder = ip_adapter_image_encoder_subfolder
        self.configured_device = os.getenv("RECONSTRUCTION_DEVICE", device).lower()
        self.allow_cpu_fallback = os.getenv("ALLOW_CPU_FALLBACK", str(allow_cpu_fallback)).lower() == "true"
        self.use_model_cpu_offload = use_model_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling
        self.max_image_side = max_image_side
        self.mask_blur_radius = mask_blur_radius
        self.default_steps = default_steps
        self.default_guidance_scale = default_guidance_scale
        self.default_strength = default_strength
        self.default_ip_adapter_scale = default_ip_adapter_scale

        self.pipeline = None
        self.active_device = "cpu"
        self._inference_lock = threading.Lock()

    def is_available(self) -> tuple[bool, str]:
        """Check whether diffusers and PyTorch CUDA are available for SDXL."""
        try:
            import torch
            import diffusers
            import transformers
        except ImportError as exc:
            return False, f"Missing required dependencies ({exc.name})."

        import torch
        if self.configured_device == "cuda" and not torch.cuda.is_available():
            if not self.allow_cpu_fallback:
                return False, "CUDA is not available on this machine (PyTorch CPU build or no NVIDIA GPU)."

        return True, "Available"

    def load(self) -> None:
        """Lazy load the pipeline on first demand."""
        if self.pipeline is not None:
            return

        is_avail, reason = self.is_available()
        if not is_avail:
            raise RuntimeError(f"SDXL Garment Reconstruction cannot be loaded: {reason}")

        import torch
        from diffusers import AutoPipelineForInpainting
        from transformers import CLIPVisionModelWithProjection

        if self.configured_device == "cuda" and torch.cuda.is_available():
            self.active_device = "cuda"
            dtype = torch.float16
            torch.backends.cuda.matmul.allow_tf32 = True
        elif self.allow_cpu_fallback:
            self.active_device = "cpu"
            dtype = torch.float32
            logger.warning("Loading SDXL on CPU; inference will be very slow.")
        else:
            raise RuntimeError("CUDA was requested but is unavailable. Set ALLOW_CPU_FALLBACK=true to test on CPU.")

        logger.info("Loading IP-Adapter image encoder: %s", self.ip_adapter_repo)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.ip_adapter_repo,
            subfolder=self.ip_adapter_image_encoder_subfolder,
            torch_dtype=dtype,
        )

        logger.info("Loading SDXL inpainting pipeline: %s", self.sdxl_model_id)
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.sdxl_model_id,
            image_encoder=image_encoder,
            torch_dtype=dtype,
            variant="fp16" if self.active_device == "cuda" else None,
            use_safetensors=True,
        )

        logger.info("Loading IP-Adapter Plus weights: %s", self.ip_adapter_weight)
        pipe.load_ip_adapter(
            self.ip_adapter_repo,
            subfolder=self.ip_adapter_subfolder,
            weight_name=self.ip_adapter_weight,
        )
        pipe.set_ip_adapter_scale(self.default_ip_adapter_scale)

        if self.enable_vae_slicing:
            pipe.enable_vae_slicing()
        if self.enable_vae_tiling:
            pipe.enable_vae_tiling()

        if self.active_device == "cuda":
            if self.use_model_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to("cpu")

        self.pipeline = pipe
        logger.info("SDXL Garment Reconstruction Engine loaded successfully on %s", self.active_device)

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def reconstruct(
        self,
        original_reference: Image.Image,
        damaged_garment: Image.Image,
        repair_mask: Image.Image,
        garment_reference: Optional[Image.Image] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        ip_adapter_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> ReconstructionResult:
        if self.pipeline is None:
            self.load()

        import torch

        started_at = time.perf_counter()
        damaged_rgba = damaged_garment.convert("RGBA")
        original_size = damaged_rgba.size

        mask_aligned = repair_mask.convert("L").resize(original_size, Image.Resampling.NEAREST)
        validate_repair_mask(mask_aligned)

        # Condition strictly on visible garment fabric reference crop
        reference = (garment_reference or original_reference).convert("RGB")
        if reference.width < 32 or reference.height < 32:
            raise ValueError("Garment reference image is too small. Please provide a visible garment crop.")

        damaged_rgb = damaged_rgba.convert("RGB")
        damaged_rgb, _ = resize_to_multiple_of_8(damaged_rgb, self.max_image_side)
        mask_resized, _ = resize_to_multiple_of_8(mask_aligned, self.max_image_side)
        mask_resized = prepare_mask(mask_resized, damaged_rgb.size, self.mask_blur_radius)
        reference, _ = resize_to_multiple_of_8(reference, self.max_image_side)

        selected_seed = seed if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
        selected_steps = steps if steps is not None else self.default_steps
        selected_guidance = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        selected_strength = strength if strength is not None else self.default_strength
        selected_adapter_scale = ip_adapter_scale if ip_adapter_scale is not None else self.default_ip_adapter_scale

        self.pipeline.set_ip_adapter_scale(selected_adapter_scale)
        generator_device = "cuda" if self.active_device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(selected_seed))

        logger.info(
            "Starting SDXL garment reconstruction: size=%s, steps=%s, seed=%s, device=%s",
            damaged_rgb.size, selected_steps, selected_seed, self.active_device,
        )

        with self._inference_lock:
            with torch.inference_mode():
                generated = self.pipeline(
                    prompt=GARMENT_PROMPT,
                    negative_prompt=GARMENT_NEGATIVE_PROMPT,
                    image=damaged_rgb,
                    mask_image=mask_resized,
                    ip_adapter_image=reference,
                    num_inference_steps=selected_steps,
                    guidance_scale=selected_guidance,
                    strength=selected_strength,
                    generator=generator,
                    padding_mask_crop=32,
                ).images[0]

        generated = generated.resize(original_size, Image.Resampling.LANCZOS)
        final_image = composite_garment(damaged_rgba, generated, mask_aligned)
        elapsed = time.perf_counter() - started_at

        if self.active_device == "cuda":
            torch.cuda.empty_cache()

        return ReconstructionResult(
            image=final_image,
            inference_seconds=round(elapsed, 3),
            seed=int(selected_seed),
            device=self.active_device,
            width=final_image.width,
            height=final_image.height,
        )


# Global singleton instance for optional lazy loading
_global_sdxl_engine: GarmentReconstructionEngine | None = None


def get_sdxl_reconstruction_engine() -> GarmentReconstructionEngine:
    global _global_sdxl_engine
    if _global_sdxl_engine is None:
        _global_sdxl_engine = GarmentReconstructionEngine()
    return _global_sdxl_engine

