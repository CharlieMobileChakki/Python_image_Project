from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from app.api.reconstruction import router as reconstruction_router
from app.config import get_settings
from app.services.garment_reconstruction import GarmentReconstructionEngine

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", settings.app_name)
    engine = GarmentReconstructionEngine(settings)
    app.state.settings = settings
    app.state.reconstruction_engine = engine
    # Load once at startup; do not reload the  SDXL checkpoint for each request.
    engine.load()
    yield
    logger.info("Shutting down garment reconstruction API")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Reference-guided garment repair. Input a damaged transparent garment, "
        "the original photo, and a white-on-black repair mask. Returns a transparent PNG."
    ),
    lifespan=lifespan,
)

app.include_router(reconstruction_router)


@app.get("/", tags=["System"])
def root():
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": "/health",
        "reconstruction_endpoint": "/api/v1/reconstruct-garment",
    }


@app.get("/health", tags=["System"])
def health():
    engine = getattr(app.state, "reconstruction_engine", None)
    loaded = bool(engine and engine.is_loaded())
    return {
        "status": "ok" if loaded else "loading_or_unavailable",
        "model_loaded": loaded,
        "device": engine.device if engine else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
