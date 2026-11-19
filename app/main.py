"""Main FastAPI application for Arangtik Garment AI.

Provides:
- Option 1: Browser-based interactive UI (served at / and /ui).
- Option 2: Machine-to-Machine API:
  - POST /api/background-removal (BiRefNet Lite + SegFormer B2 Clothes)
  - POST /api/qwen-image (Qwen-Image-2.1 Fabric Reconstruction)
  - GET /api/outputs/{filename} (Direct 1024x1024 transparent PNG retrieval)
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
from app.core.flow_b import check_gpu_worker_health
from app.core.model_manager import are_models_loaded, load_flow_a
from app.core.model_manager.model_registry import FLOW_A_MODEL_NAME
from app.schemas import HealthResponse

MODEL_NAME = FLOW_A_MODEL_NAME
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload Flow A (BiRefNet Lite) model once during server startup."""
    print("Preloading Arangtik Garment AI (Flow A: BiRefNet Lite)...")
    try:
        await run_in_threadpool(load_flow_a)
        print("  - BiRefNet Lite loaded.")
    except Exception as exc:
        print(f"  - Warning: Failed to preload BiRefNet Lite: {exc}")

    print("Arangtik Garment AI ready for requests (Flow B delegates to ChatGPT Vision + Remote GPU Qwen).")
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

# Include core API routers under /api (No /v1 prefix, single API per function)
app.include_router(api_router, prefix="/api")
app.include_router(reconstruction_router, prefix="/api")

# Mount outputs directory to serve generated 1024x1024 PNGs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount static directory for UI assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    fav = STATIC_DIR / "favicon.svg"
    if fav.exists():
        return FileResponse(fav, media_type="image/svg+xml")
    raise HTTPException(status_code=404)


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


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Unified system health check and Flow A / Flow B GPU Worker readiness status."""
    flow_a_status = are_models_loaded()
    gpu_health = await run_in_threadpool(check_gpu_worker_health)
    openai_ready = bool(os.getenv("OPENAI_API_KEY", "").strip())

    remote_gpu_online = bool(gpu_health.get("online", False))
    gpu_name = gpu_health.get("gpu_name", "Remote GPU Offline") if remote_gpu_online else "Not Connected"

    return HealthResponse(
        status="ok",
        service="arangtik-garment-ai",
        flow="Flow A (BiRefNet Lite) + Flow B (ChatGPT Vision + Remote GPU Qwen)",
        version="2.0.0",
        model=MODEL_NAME,
        docs_url="/docs",
        ui_url="/ui",
        models_loaded={
            **flow_a_status,
            "openai_vision_configured": openai_ready,
            "qwen_gpu_worker_online": remote_gpu_online,
        },
        hardware={
            "cpu_cores": os.cpu_count() or 1,
            "device": "remote_cuda" if remote_gpu_online else "cpu",
            "cuda_available": remote_gpu_online,
            "gpu_name": gpu_name,
            "gpu_worker_url": gpu_health.get("url", "not_set"),
            "gpu_worker_detail": gpu_health.get("detail", "Connected"),
        },
    )
