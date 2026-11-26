"""Main FastAPI application for Arangtik Garment AI.

Provides:
- Browser Studio UI (served at / and /ui).
- Production APIs:
  - POST /api/generate-garment (ChatGPT Vision + Remote GPU Qwen-Image-2.1-Turbo -> 1024x1024 Studio PNG)
  - POST /api/qwen-image (Direct single-image Qwen 2.1 PNG endpoint)
  - POST /api/jewelry-studio (Multi-angle luxury jewelry generator)
  - POST /api/forward-downstream (Forward generated PNGs to external API)
  - GET /api/outputs/{filename} (Direct 1024x1024 studio PNG retrieval)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.api.reconstruction import reconstruction_router
from app.api.routes import router as api_router
from app.core.flow_b import check_gpu_worker_health
from app.schemas import HealthResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup banner for lightweight API orchestrator (delegates inference to remote GPU)."""
    print("Arangtik Garment AI ready (Direct 1024x1024 Studio Pipeline -> ChatGPT Vision + Remote GPU Qwen 2.1).")
    yield


app = FastAPI(
    title="Arangtik Garment AI",
    version="4.1.0",
    description="Direct 1024x1024 AI Studio Generator via ChatGPT Vision and Qwen-Image-2.1-Turbo.",
    lifespan=lifespan,
)

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

app.include_router(api_router, prefix="/api")
app.include_router(reconstruction_router, prefix="/api")

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

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
    """Serve Interactive Browser Studio UI or API discovery JSON."""
    accept = request.headers.get("accept", "")
    if "application/json" in accept or ("*/*" in accept and "text/html" not in accept):
        return {"service": "arangtik-garment-ai", "docs": "/docs", "health": "/health"}

    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(content=index_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Arangtik Garment AI</h1><p>Visit <a href='/docs'>/docs</a>.</p>")


@app.get("/ui", response_class=HTMLResponse)
async def ui_redirect():
    """Direct shortcut to Studio Web UI."""
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(content=index_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Arangtik Garment AI</h1><p>UI file not found.</p>")


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Unified system health check and Remote GPU Worker readiness status."""
    gpu_health = await run_in_threadpool(check_gpu_worker_health)
    openai_ready = bool(os.getenv("OPENAI_API_KEY", "").strip())

    remote_gpu_online = bool(gpu_health.get("online", False))
    gpu_name = gpu_health.get("gpu_name", "Remote GPU Offline") if remote_gpu_online else "Not Connected"
    workers_total = int(gpu_health.get("workers_total") or (1 if remote_gpu_online else 0))
    batch_concurrency = max(1, min(workers_total, 10)) if remote_gpu_online else 1
    backend_type = gpu_health.get("backend", "unconfigured")

    return HealthResponse(
        status="ok",
        service="arangtik-garment-ai",
        flow="Direct 1024x1024 Studio Pipeline (ChatGPT Vision + Remote GPU Qwen 2.1)",
        version="4.1.0",
        model="Qwen/Qwen-Image-2.1-Turbo",
        docs_url="/docs",
        ui_url="/ui",
        models_loaded={
            "openai_vision_configured": openai_ready,
            "qwen_gpu_worker_online": remote_gpu_online,
        },
        hardware={
            "cpu_cores": os.cpu_count() or 1,
            "device": backend_type if remote_gpu_online else "cpu",
            "backend": backend_type,
            "gpu_worker_url": gpu_health.get("url", ""),
            "cuda_available": remote_gpu_online,
            "gpu_name": gpu_name,
            "vram_gb": gpu_health.get("vram_total_gb", 48.0 if remote_gpu_online else 0.0),
            "workers_total": workers_total,
            "batch_concurrency": batch_concurrency,
        },
    )
