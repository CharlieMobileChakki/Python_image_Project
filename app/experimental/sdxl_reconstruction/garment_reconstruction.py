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

from app.experimental.sdxl_reconstruction.reconstruction_utils import (
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

    def is_loaded(self) -> bool:
        """Return True if pipeline weights are currently loaded in memory."""
        return self.pipeline is not None

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

        logger.info("Loading SDXL inpainting pipeline (%s) on %s...", self.sdxl_model_id, self.active_device)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.ip_adapter_repo,
            subfolder=self.ip_adapter_image_encoder_subfolder,
            torch_dtype=dtype,
        )

        pipe = AutoPipelineForInpainting.from_pretrained(
            self.sdxl_model_id,
            image_encoder=image_encoder,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        pipe.load_ip_adapter(
            self.ip_adapter_repo,
            subfolder=self.ip_adapter_subfolder,
            weight_name=self.ip_adapter_weight,
        )

        if self.enable_vae_slicing:
            pipe.enable_vae_slicing()
        if self.enable_vae_tiling:
            pipe.enable_vae_tiling()

        if self.use_model_cpu_offload and self.active_device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.active_device)

        self.pipeline = pipe
        logger.info("SDXL Inpainting + IP-Adapter Plus pipeline ready.")

    def reconstruct(
        self,
        original_reference: Image.Image,
        damaged_garment: Image.Image,
        repair_mask: Image.Image,
        garment_reference: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        ip_adapter_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> ReconstructionResult:
        """Run SDXL inpainting conditioned on garment reference."""
        validate_repair_mask(repair_mask)
        self.load()

        import torch

        steps = steps or self.default_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        strength = strength if strength is not None else self.default_strength
        ip_adapter_scale = ip_adapter_scale if ip_adapter_scale is not None else self.default_ip_adapter_scale

        prompt_str = prompt or GARMENT_PROMPT
        neg_prompt_str = negative_prompt or GARMENT_NEGATIVE_PROMPT

        # Determine reference image: prefer tight crop, fallback to full original
        ref_image = garment_reference if garment_reference is not None else original_reference
        ref_image = ref_image.convert("RGB")

        # Resize damaged garment & mask to multi-8 dimensions
        damaged_resized, original_size = resize_to_multiple_of_8(damaged_garment, self.max_image_side)
        target_size = damaged_resized.size
        mask_prepared = prepare_mask(repair_mask, target_size, blur_radius=self.mask_blur_radius)

        gen_seed = seed if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator = torch.Generator(device=self.active_device).manual_seed(gen_seed)

        start_time = time.perf_counter()
        with self._inference_lock:
            self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
            output = self.pipeline(
                prompt=prompt_str,
                negative_prompt=neg_prompt_str,
                image=damaged_resized.convert("RGB"),
                mask_image=mask_prepared,
                ip_adapter_image=ref_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            )

        inference_seconds = round(time.perf_counter() - start_time, 2)
        generated_rgb = output.images[0]

        # Composite newly generated fabric only inside repair mask
        final_garment = composite_garment(damaged_garment, generated_rgb, repair_mask)

        return ReconstructionResult(
            image=final_garment,
            inference_seconds=inference_seconds,
            seed=gen_seed,
            device=self.active_device,
            width=final_garment.width,
            height=final_garment.height,
        )


_sdxl_engine: Optional[GarmentReconstructionEngine] = None
_sdxl_lock = threading.Lock()


def get_sdxl_reconstruction_engine() -> GarmentReconstructionEngine:
    """Get or initialize singleton SDXL GarmentReconstructionEngine."""
    global _sdxl_engine
    if _sdxl_engine is None:
        with _sdxl_lock:
            if _sdxl_engine is None:
                _sdxl_engine = GarmentReconstructionEngine()
    return _sdxl_engine

