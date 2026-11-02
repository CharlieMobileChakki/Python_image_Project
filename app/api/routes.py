"""FastAPI router defining all endpoints for Arangtik Garment AI."""
from __future__ import annotations

import asyncio
from io import BytesIO
import os
from pathlib import Path
import time
from typing import Any
import zipfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.api.schemas import HealthResponse, ProcessGarmentResponse
from app.core.background_removal import extract_isolated_flow_a
from app.core.image_processing import detect_rotation_angle, normalize_to_1024, straighten_garment
from app.core.inpainting import repair_garment_cutout
from app.core.model_manager import are_models_loaded, is_inpainter_loaded
from app.core.pipeline import OUTPUT_DIR, process_garment_pipeline

router = APIRouter()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1"))))
    return _semaphore


async def validate_and_read_image(file: UploadFile) -> bytes:
    """Validate content type and size limits, returning raw bytes."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Supported image formats are JPG, PNG, and WEBP.")
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds upload limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return image_bytes


@router.post("/process-garment", response_model=ProcessGarmentResponse)
async def process_garment_endpoint(
    file: UploadFile = File(..., description="Input image of garment or worn model"),
    target_garment: str = Form("all", description="Target garment filter: 'all', 'upper_clothes', 'pants', 'dress', 'skirt'"),
    auto_repair: bool = Form(True, description="Enable LaMa fabric inpainting for occlusions"),
    auto_straighten: bool = Form(True, description="Enable auto-deskewing for tilted garments"),
    include_base64: bool = Form(False, description="Include Base64 data URI in JSON"),
    response_format: str = Form("json", description="'json' for multi-garment metadata or 'image_direct' for single direct PNG"),
    mode_hint: str = Form("auto", description="'auto', 'worn', or 'isolated'"),
) -> Any:
    """Master Unified Pipeline Endpoint (Option 2 Machine-to-Machine API).

    Processes image through the 5-step intelligent pipeline:
    1. Human presence detection.
    2. Branching: SegFormer parsing (worn) or BiRefNet Lite (isolated product).
    3. Occlusion inspection & LaMa Fast Fourier inpainting.
    4. Auto-deskew angle correction.
    5. 1024x1024 transparent PNG canvas normalization.
    """
    image_bytes = await validate_and_read_image(file)
    try:
        async with get_semaphore():
            result = await run_in_threadpool(
                process_garment_pipeline,
                image_bytes=image_bytes,
                target_garment=target_garment,
                auto_repair=auto_repair,
                auto_straighten=auto_straighten,
                include_base64=include_base64,
                mode_hint=mode_hint,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {exc}") from exc
    finally:
        await file.close()

    # Optional direct image return mode if exactly one garment was produced
    if response_format.lower() == "image_direct" and result.get("garments"):
        top_garment = result["garments"][0]
        png_path = OUTPUT_DIR / Path(top_garment["output_png_url"]).name
        if png_path.exists():
            return FileResponse(
                path=png_path,
                media_type="image/png",
                filename=png_path.name,
                headers={
                    "X-Garment-Category": top_garment["category"],
                    "X-Garment-Label": top_garment["recognized_label"],
                    "X-Execution-Time-Ms": str(result["execution_time_ms"]),
                },
            )

    return result


@router.post("/extract-isolated")
async def extract_isolated_endpoint(
    file: UploadFile = File(..., description="Isolated product photo on hanger/flat-lay/mannequin"),
    auto_straighten: bool = Form(True),
) -> Any:
    """Dedicated Flow A endpoint: Extracts product from background and returns 1024x1024 transparent PNG."""
    image_bytes = await validate_and_read_image(file)
    try:
        raw_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        async with get_semaphore():
            cutout_rgba = await run_in_threadpool(extract_isolated_flow_a, raw_img)
            if auto_straighten:
                angle = detect_rotation_angle(cutout_rgba)
                if abs(angle) > 0.5:
                    cutout_rgba = straighten_garment(cutout_rgba, angle)
            norm_1024 = normalize_to_1024(cutout_rgba, target_size=1024)

            buf = BytesIO()
            norm_1024.save(buf, format="PNG", optimize=True)
            png_bytes = buf.getvalue()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Flow A extraction failed: {exc}") from exc
    finally:
        await file.close()

    return Response(content=png_bytes, media_type="image/png")


@router.get("/outputs/{filename}")
async def get_output_image(filename: str) -> FileResponse:
    """Download or view a processed 1024x1024 transparent garment PNG."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Requested garment image does not exist.")
    return FileResponse(path=file_path, media_type="image/png", filename=filename)


@router.get("/download-batch-zip/{job_id}")
async def download_batch_zip(job_id: str) -> Response:
    """Download all extracted 1024x1024 transparent garment PNGs for a given job in a single ZIP archive."""
    png_files = sorted(list(OUTPUT_DIR.glob(f"{job_id}_*.png")))
    if not png_files:
        raise HTTPException(status_code=404, detail="No extracted garments found for this job ID.")

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in png_files:
            zf.write(p, arcname=p.name)

    zip_bytes = zip_buf.getvalue()
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="arangtik_{job_id}_garments.zip"'},
    )


@router.post("/forward-downstream")
async def forward_downstream_endpoint(
    job_id: str = Form(..., description="Job ID of processed garments"),
    target_api_url: str = Form(..., description="Destination API URL"),
) -> Any:
    """Forward all extracted garment PNGs in ONE single multipart request to a downstream API."""
    import httpx
    png_files = sorted(list(OUTPUT_DIR.glob(f"{job_id}_*.png")))
    if not png_files:
        raise HTTPException(status_code=404, detail="No extracted garments found for this job.")

    files_payload = []
    for p in png_files:
        files_payload.append(("files[]", (p.name, p.read_bytes(), "image/png")))

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                target_api_url,
                files=files_payload,
                data={"job_id": job_id, "total_garments": str(len(png_files))},
            )
            return {
                "success": resp.is_success,
                "status_code": resp.status_code,
                "forwarded_files_count": len(png_files),
                "downstream_response": resp.json() if "application/json" in resp.headers.get("content-type", "") else resp.text,
            }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Downstream forward failed: {exc}") from exc


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health check and loaded AI model status."""
    seg_status = are_models_loaded()
    lama_status = is_inpainter_loaded()
    return HealthResponse(
        status="ok",
        service="arangtik-garment-ai",
        models_loaded={
            **seg_status,
            "lama_inpainter": lama_status,
        },
        hardware={
            "cpu_cores": os.cpu_count(),
            "device": "cpu",
        },
    )

