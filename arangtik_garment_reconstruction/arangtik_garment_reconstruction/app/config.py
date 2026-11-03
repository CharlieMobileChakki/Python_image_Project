from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Arangtik Garment Reconstruction API"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    sdxl_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    ip_adapter_image_encoder_subfolder: str = "models/image_encoder"

    device: str = "cuda"
    allow_cpu_fallback: bool = False
    use_model_cpu_offload: bool = False
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True

    default_steps: int = 30
    default_guidance_scale: float = 5.5
    default_strength: float = 0.95
    default_ip_adapter_scale: float = 0.75
    max_image_side: int = 1024
    max_upload_mb: int = 25
    mask_blur_radius: float = 1.5
    default_seed: Optional[int] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
