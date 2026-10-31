"""Flow B: FastAPI Service for Semantic Garment Parsing and Extraction on Worn Photos."""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from flowb_service.extraction import (
    EXPLICIT_ALIASES,
    MODEL_ID,
    extract_garment_worn,
    is_model_loaded,
    load_flowb_model,
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1")))
PROCESS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load SegFormer model once at application startup
    await run_in_threadpool(load_flowb_model)
    yield


app = FastAPI(
    title="Arangtik Garment Extraction API — Flow B",
    version="0.1.0",
    description="Experimental API for Flow B: Semantic parsing and extraction of worn clothing items.",
    lifespan=lifespan,
)

# Configure CORS allowing local test interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "X-Model-Name",
        "X-Processing-Time-Ms",
        "X-Original-Width",
        "X-Original-Height",
        "X-Target-Classes",
        "X-Mask-Coverage",
    ],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "arangtik-garment-flow-b",
        "docs": "/docs",
        "health": "/health",
        "endpoint": "/api/v1/extract-garment/worn",
    }


@app.get("/health")
def health() -> Response:
    loaded = is_model_loaded()
    payload = {
        "status": "ok" if loaded else "initializing",
        "service": "arangtik-garment-flow-b",
        "flow": "B",
        "model": MODEL_ID,
        "model_loaded": loaded,
        "supported_aliases": list(EXPLICIT_ALIASES.keys()),
    }
    status_code = status.HTTP_200_OK if loaded else status.HTTP_503_SERVICE_UNAVAILABLE
    return Response(
        content=__import__("json").dumps(payload),
        media_type="application/json",
        status_code=status_code,
    )


@app.post("/api/v1/extract-garment/worn")
async def extract_garment_worn_endpoint(
    file: UploadFile = File(...),
    class_name: str = Form(...),
    alpha_mode: str = Form("soft"),
    alpha_low: float = Form(0.20),
    alpha_high: float = Form(0.80),
) -> Response:
    """Extract specified garment(s) from a photo of a person wearing clothes.
    
    Returns a normalized, centered 1024x1024 transparent PNG.
    """
    # 1. Content type validation
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Supported image types are JPG, PNG, and WEBP.",
        )

    # 2. File size and emptiness validation
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    # 3. Model readiness check
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing. Please retry shortly.",
        )

    # 4. Input parameter validation
    clean_mode = alpha_mode.strip().lower()
    if clean_mode not in {"soft", "hard"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="alpha_mode must be either 'soft' or 'hard'.",
        )

    # 5. Execute extraction with semaphore protection
    try:
        async with PROCESS_SEMAPHORE:
            result = await run_in_threadpool(
                extract_garment_worn,
                image_bytes=image_bytes,
                class_name=class_name,
                alpha_mode=clean_mode,
                alpha_low=alpha_low,
                alpha_high=alpha_high,
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()

    return Response(
        content=result.png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="arangtik-garment-worn.png"',
            "X-Model-Name": MODEL_ID,
            "X-Processing-Time-Ms": str(result.processing_time_ms),
            "X-Original-Width": str(result.original_size[0]),
            "X-Original-Height": str(result.original_size[1]),
            "X-Target-Classes": ",".join(result.target_classes),
            "X-Mask-Coverage": f"{result.mask_coverage_pct}%",
            "Cache-Control": "no-store",
        },
    )

