"""Main FastAPI application for Arangtik Garment AI.

Provides:
- Option 1: Browser-based interactive UI (served at / and /ui).
- Option 2: Machine-to-Machine API for external platforms (/api/v1/process-garment).
- Flow A legacy extraction endpoint (/api/v1/extract-garment).
- Direct file download for normalized 1024x1024 transparent PNGs (/api/v1/outputs/{filename}).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.api.reconstruction import reconstruction_router
from app.api.routes import router as api_router
from app.core.background_removal import extract_isolated_flow_a
from app.core.model_manager import load_inpainter_model, load_segmenter_models
from app.services.extraction import MODEL_NAME as FLOW_A_MODEL_NAME, extract_garment, load_model

MODEL_NAME = FLOW_A_MODEL_NAME
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload AI models once during server startup."""
    print("Preloading Arangtik Garment AI models...")
    await run_in_threadpool(load_model)
    try:
        await run_in_threadpool(load_segmenter_models)
        print("  - SegFormer B2 Clothes & BiRefNet Lite loaded.")
    except Exception as exc:
        print(f"  - Warning: Failed to preload segmentation models: {exc}")

    try:
        await run_in_threadpool(load_inpainter_model)
        print("  - LaMa Fast Fourier Inpainter ONNX loaded.")
    except Exception as exc:
        print(f"  - Warning: Failed to preload inpainter model: {exc}")

    print("Arangtik Garment AI ready for requests.")
    yield


app = FastAPI(
    title="Arangtik Garment AI",
    version="1.0.0",
    description="Unified AI pipeline for garment extraction, human parsing, fabric inpainting, and 1024x1024 normalization.",
    lifespan=lifespan,
)

# Enable CORS for local UI and external client apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Garment-Category",
        "X-Garment-Label",
        "X-Execution-Time-Ms",
        "X-Model-Name",
    ],
)

# Include core API router under /api/v1
app.include_router(api_router, prefix="/api/v1")
app.include_router(reconstruction_router, prefix="/api/v1")

# Mount outputs directory to serve generated 1024x1024 PNGs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount static directory for UI assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root_ui(request: Request):
    """Option 1: Interactive Browser UI or API discovery."""
    accept = request.headers.get("accept", "")
    if "application/json" in accept or "*/*" in accept and "text/html" not in accept:
        return {"service": "arangtik-garment-extraction", "docs": "/docs", "health": "/health"}

    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(content=index_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Arangtik Garment AI</h1><p>Visit <a href='/docs'>/docs</a>.</p>")


@app.get("/ui", response_class=HTMLResponse)
async def ui_redirect():
    """Direct shortcut to Option 1 Web UI."""
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(content=index_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Arangtik Garment AI</h1><p>UI file not found.</p>")


@app.get("/health")
async def health_check():
    """Root health check probe."""
    return {
        "status": "ok",
        "service": "arangtik-garment-extraction",
        "flow": "A",
        "model": MODEL_NAME,
        "version": "1.0.0",
        "docs_url": "/docs",
        "ui_url": "/ui",
    }


# Backward-compatible Flow A endpoint
@app.post("/api/v1/extract-garment")
async def legacy_extract_garment(file: UploadFile = File(...)) -> Response:
    """Legacy Flow A endpoint returning transparent PNG directly."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Supported image types are JPG, PNG, and WEBP.")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")

    import time
    started = time.perf_counter()
    try:
        png_bytes, original_size = await run_in_threadpool(extract_garment, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="arangtik-garment.png"',
            "X-Model-Name": MODEL_NAME,
            "X-Processing-Time-Ms": str(elapsed_ms),
            "X-Original-Width": str(original_size[0]),
            "X-Original-Height": str(original_size[1]),
            "Cache-Control": "no-store",
        },
    )
