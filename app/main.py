from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

load_dotenv()

from app.services.extraction import MODEL_NAME, extract_garment, load_model  # noqa: E402

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
# Keep local testing predictable on CPU/GPU machines with limited memory.
PROCESS_SEMAPHORE = asyncio.Semaphore(max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1"))))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load weights once, rather than reloading the model for every uploaded image.
    await run_in_threadpool(load_model)
    yield


app = FastAPI(
    title="Arangtik Garment Extraction API",
    version="0.1.0",
    description="Prototype API for Flow A: single clothing image to transparent PNG.",
    lifespan=lifespan,
)

# Convenient for the local HTML file. In production, restrict origins to your real frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Model-Name", "X-Processing-Time-Ms", "X-Original-Width", "X-Original-Height"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "arangtik-garment-extraction", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "service": "arangtik-garment-extraction", "flow": "A", "model": MODEL_NAME}


@app.post("/api/v1/extract-garment")
async def extract_garment_endpoint(file: UploadFile = File(...)) -> Response:
    """Accept multipart field `file` and return a transparent PNG directly."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Supported image types are JPG, PNG, and WEBP.")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")

    started = time.perf_counter()
    try:
        async with PROCESS_SEMAPHORE:
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
