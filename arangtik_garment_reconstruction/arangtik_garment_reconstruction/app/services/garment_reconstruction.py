from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting
from transformers import CLIPVisionModelWithProjection

from app.config import Settings
from app.services.image_utils import composite_garment, prepare_mask, resize_to_sdxl, validate_repair_mask

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
    """Loads SDXL Inpainting + IP-Adapter Plus once and reuses it for API requests."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline = None
        self.device = "cpu"
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        if self.pipeline is not None:
            return

        requested_device = self.settings.device.lower()
        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
            torch.backends.cuda.matmul.allow_tf32 = True
        elif self.settings.allow_cpu_fallback:
            self.device = "cpu"
            dtype = torch.float32
            logger.warning("CUDA is unavailable. Loading SDXL on CPU; inference will be very slow.")
        else:
            raise RuntimeError(
                "CUDA GPU was requested but PyTorch cannot access CUDA. "
                "Install a CUDA-enabled PyTorch build and verify with nvidia-smi, "
                "or set ALLOW_CPU_FALLBACK=true only for a slow smoke test."
            )

        logger.info("Loading IP-Adapter image encoder: %s", self.settings.ip_adapter_repo)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.settings.ip_adapter_repo,
            subfolder=self.settings.ip_adapter_image_encoder_subfolder,
            torch_dtype=dtype,
        )

        logger.info("Loading SDXL inpainting checkpoint: %s", self.settings.sdxl_model_id)
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.settings.sdxl_model_id,
            image_encoder=image_encoder,
            torch_dtype=dtype,
            variant="fp16" if self.device == "cuda" else None,
            use_safetensors=True,
        )

        logger.info("Loading IP-Adapter Plus weights: %s", self.settings.ip_adapter_weight)
        pipe.load_ip_adapter(
            self.settings.ip_adapter_repo,
            subfolder=self.settings.ip_adapter_subfolder,
            weight_name=self.settings.ip_adapter_weight,
        )
        pipe.set_ip_adapter_scale(self.settings.default_ip_adapter_scale)

        if self.settings.enable_vae_slicing:
            pipe.enable_vae_slicing()
        if self.settings.enable_vae_tiling:
            pipe.enable_vae_tiling()

        if self.device == "cuda":
            if self.settings.use_model_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to("cpu")

        self.pipeline = pipe
        logger.info("Garment reconstruction model is ready on %s", self.device)

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
            raise RuntimeError("Model is not loaded. Wait for application startup to finish.")

        started_at = time.perf_counter()
        damaged_rgba = damaged_garment.convert("RGBA")
        original_size = damaged_rgba.size

        mask_aligned = repair_mask.convert("L").resize(original_size, Image.Resampling.NEAREST)
        validate_repair_mask(mask_aligned)

        # The user should provide a tight garment crop as garment_reference whenever possible.
        # If not provided, the full original photo is used as the visual reference.
        reference = (garment_reference or original_reference).convert("RGB")
        if reference.width < 32 or reference.height < 32:
            raise ValueError("Garment reference image is too small. Upload a visible garment crop.")

        damaged_rgb = damaged_rgba.convert("RGB")
        # Transparent canvas pixels are neutralized before diffusion; output alpha is restored later.
        damaged_rgb, _ = resize_to_sdxl(damaged_rgb, self.settings.max_image_side)
        mask_resized, _ = resize_to_sdxl(mask_aligned, self.settings.max_image_side)
        mask_resized = prepare_mask(mask_resized, damaged_rgb.size, self.settings.mask_blur_radius)
        reference, _ = resize_to_sdxl(reference, self.settings.max_image_side)

        selected_seed = seed if seed is not None else self.settings.default_seed
        if selected_seed is None:
            selected_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        selected_steps = steps if steps is not None else self.settings.default_steps
        selected_guidance = guidance_scale if guidance_scale is not None else self.settings.default_guidance_scale
        selected_strength = strength if strength is not None else self.settings.default_strength
        selected_adapter_scale = (
            ip_adapter_scale if ip_adapter_scale is not None else self.settings.default_ip_adapter_scale
        )

        if not 10 <= selected_steps <= 80:
            raise ValueError("steps must be between 10 and 80.")
        if not 1.0 <= selected_guidance <= 12.0:
            raise ValueError("guidance_scale must be between 1 and 12.")
        if not 0.1 <= selected_strength <= 1.0:
            raise ValueError("strength must be between 0.1 and 1.0.")
        if not 0.0 <= selected_adapter_scale <= 1.0:
            raise ValueError("ip_adapter_scale must be between 0 and 1.")

        self.pipeline.set_ip_adapter_scale(selected_adapter_scale)
        generator_device = "cuda" if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(selected_seed))

        logger.info(
            "Starting garment reconstruction: size=%s, steps=%s, seed=%s, device=%s",
            damaged_rgb.size, selected_steps, selected_seed, self.device,
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

        # Resize generated pixels back to source dimensions and composite only inside the mask.
        generated = generated.resize(original_size, Image.Resampling.LANCZOS)
        final_image = composite_garment(damaged_rgba, generated, mask_aligned)
        elapsed = time.perf_counter() - started_at

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return ReconstructionResult(
            image=final_image,
            inference_seconds=round(elapsed, 3),
            seed=int(selected_seed),
            device=self.device,
            width=final_image.width,
            height=final_image.height,
        )
