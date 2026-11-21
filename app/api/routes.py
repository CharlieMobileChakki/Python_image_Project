"""FastAPI router defining all endpoints for Arangtik Garment AI."""
from __future__ import annotations

import asyncio
from io import BytesIO
import os
from pathlib import Path
import time
from typing import Any
import zipfile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.schemas import HealthResponse, ProcessGarmentResponse
from app.core.pipeline import OUTPUT_DIR, process_garment_pipeline

router = APIRouter()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _semaphore is None or _semaphore_loop != current_loop:
        _semaphore = asyncio.Semaphore(max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "4"))))
        _semaphore_loop = current_loop
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


@router.post(
    "/background-removal",
    response_model=ProcessGarmentResponse,
    response_model_exclude_none=True,
    summary="Primary Background Removal & Garment Processing API",
)
async def background_removal_endpoint(
    request: Request,
    file: UploadFile = File(None, description="Primary input image (garment, model, flat-lay, or jewelry)"),
    files: list[UploadFile] = File(None, description="Optional multi-angle input images (up to 4 photos for jewelry/rings)"),
    person_mode: str = Form("with_person", description="Execution mode: 'with_person', 'without_person', 'jewelry'"),
    has_human: str | bool | None = Form(None, description="Optional flag"),
    target_garment: str = Form("all", description="Target filter: 'all', 'upper_clothes', 'pants', 'dress', 'necklace', 'ring', 'earrings', 'bangles'"),
    auto_repair: bool = Form(True, description="Enable fabric inpainting"),
    auto_straighten: bool = Form(False, description="Enable auto-deskewing"),
    include_base64: bool = Form(False, description="Include Base64 data URI in JSON"),
    response_format: str = Form("json", description="'json' (default for Node.js / UI) or 'image_direct' for binary PNG stream"),
    mode_hint: str | None = Form(None, description="Legacy mode hint"),
    custom_prompt: str | None = Form(None, description="Optional custom prompt override"),
    presentation_mode: str = Form("mode_c_display_stand", description="Presentation mode: 'mode_a_product_only', 'mode_b_studio_presentation', 'mode_c_display_stand'"),
    item_count_override: int | None = Form(None, description="Optional manual quantity override (e.g. 1 for 1 kada)"),
    output_resolution: str = Form("8k", description="Output canvas resolution: '8k' (8192x8192), '4k' (4096x4096), '2k' (2048x2048), or '1k' (1024x1024)"),
) -> Any:
    """Unified AI 3D Garment & Jewelry Studio API.

    Single unified production workflow:
    - Apparel: ChatGPT Vision -> Qwen GPU -> BiRefNet Lite 1024x1024.
    - Jewelry: Supports 1 to 4 multi-angle photos (top, sides, perspective) -> Qwen GPU -> Specialized luxury catalog stand.

    Supports both Node.js caller formats:
    - JSON Response (default): { status: "success", garments: [{ url, category, stand_type, ... }] }
    - Direct PNG stream: Set response_format='image_direct' or Accept: image/png
    """
    upload_list: list[UploadFile] = []
    if file:
        upload_list.append(file)
    if files:
        upload_list.extend(files)
    if not upload_list:
        raise HTTPException(status_code=400, detail="No image file provided. Please upload at least one image.")

    image_bytes_list: list[bytes] = []
    for f in upload_list[:4]:
        b = await validate_and_read_image(f)
        image_bytes_list.append(b)

    payload_bytes: bytes | list[bytes] = image_bytes_list[0] if len(image_bytes_list) == 1 else image_bytes_list

    actual_has_human: bool | None = None
    if has_human is not None:
        if isinstance(has_human, bool):
            actual_has_human = has_human
        elif str(has_human).strip().lower() in ("false", "0", "no"):
            actual_has_human = False
        elif str(has_human).strip().lower() in ("true", "1", "yes"):
            actual_has_human = True

    try:
        async with get_semaphore():
            result = await run_in_threadpool(
                process_garment_pipeline,
                image_bytes=payload_bytes,
                person_mode=person_mode,
                has_human=actual_has_human,
                mode_hint=mode_hint,
                target_garment=target_garment,
                auto_repair=auto_repair,
                auto_straighten=auto_straighten,
                include_base64=include_base64,
                custom_prompt=custom_prompt,
                presentation_mode=presentation_mode,
                item_count_override=item_count_override,
                output_resolution=output_resolution,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {exc}") from exc
    finally:
        for f in upload_list:
            await f.close()

    # Check if client requested direct PNG image stream (Option 1)
    accept_hdr = request.headers.get("accept", "").lower()
    wants_image = (
        response_format.lower() in ("image_direct", "direct", "image", "stream", "png")
        or ("image/" in accept_hdr and "application/json" not in accept_hdr)
    )

    if wants_image and result.get("garments") and len(result["garments"]) == 1:
        top_garment = result["garments"][0]
        png_path = OUTPUT_DIR / Path(top_garment["output_png_url"]).name
        if png_path.exists():
            return FileResponse(
                path=png_path,
                media_type="image/png",
                filename=png_path.name,
                headers={
                    "X-Flow-Mode": result.get("mode", "without_person"),
                    "X-Garment-Category": top_garment["category"],
                    "X-Garment-Label": top_garment["recognized_label"],
                    "X-Execution-Time-Ms": str(result["execution_time_ms"]),
                    "X-Canvas-Size": top_garment.get("resolution", "8192x8192"),
                },
            )

    # Format clean production JSON response for Node.js and UI
    base_url = str(request.base_url).rstrip("/")
    forwarded_proto = request.headers.get("x-forwarded-proto", "").lower()
    if forwarded_proto == "https" and base_url.startswith("http://"):
        base_url = base_url.replace("http://", "https://", 1)

    clean_garments = []
    for g in result.get("garments", []):
        raw_url = g.get("url") or g.get("output_png_url") or ""
        if raw_url.startswith("/"):
            full_url = f"{base_url}{raw_url}"
        else:
            full_url = raw_url

        title = g.get("title") or g.get("recognized_label") or "Garment Cutout"
        clean_garments.append({
            "category": g.get("category", "product"),
            "title": title,
            "url": full_url,
            "resolution": g.get("resolution", "8192x8192"),
            "output_png_url": full_url,
            "recognized_label": title,
            "garment_id": g.get("garment_id"),
            "output_png_base64": g.get("output_png_base64") if include_base64 else None,
            "stand_type": g.get("stand_type"),
            "presentation_mode": g.get("presentation_mode", "ghost_mannequin"),
            "detected_item_count": g.get("detected_item_count", 1),
            "validation_metrics": g.get("validation_metrics"),
            "views_analyzed": g.get("views_analyzed", 1),
        })

    total_time_sec = round(result.get("execution_time_ms", 0) / 1000.0, 1)

    return {
        "status": "success",
        "total_garments": len(clean_garments),
        "garments": clean_garments,
        "processing_time_sec": total_time_sec,
        "execution_time_ms": result.get("execution_time_ms", int(total_time_sec * 1000)),
        "mode": result.get("mode", "with_person"),
        "diagnostics": result.get("diagnostics", {}),
        "job_id": result.get("job_id"),
        "success": True,
    }


# ==============================================================================
# JEWELRY & PRECIOUS ORNAMENTS DEDICATED ENDPOINTS
# ------------------------------------------------------------------------------
@router.post(
    "/jewelry-studio",
    response_model=ProcessGarmentResponse,
    response_model_exclude_none=True,
    summary="Dedicated 3D Luxury Jewelry Studio (Single or Multi-Angle)",
)
async def jewelry_studio_endpoint(
    request: Request,
    file: UploadFile = File(None, description="Primary jewelry image"),
    files: list[UploadFile] = File(None, description="Up to 4 multi-angle views of the jewelry piece (top, sides, angle)"),
    jewelry_type: str = Form("auto", description="Category: 'auto', 'necklace', 'ring', 'earrings', 'bangles'"),
    custom_prompt: str | None = Form(None, description="Optional custom prompt override"),
    include_base64: bool = Form(False, description="Include Base64 data URI in JSON"),
    output_resolution: str = Form("8k", description="Output canvas resolution: '8k' (8192x8192), '4k' (4096x4096), '2k' (2048x2048), or '1k' (1024x1024)"),
) -> Any:
    """Dedicated 3D Luxury Jewelry Studio Endpoint.

    Takes 1 to 4 multi-angle photographs of the same jewelry piece:
    - Rings: Analyzes top crown AND side shank/filigree; generates 30-45 degree hero angle on slotted ring block.
    - Necklaces: Drapes symmetrically around luxury white neck bust mannequin stand.
    - Earrings: Symmetrical pair hanging on minimalist T-bar stand.
    - Bangles: Mounted on horizontal cylindrical jewelry display bar.
    """
    upload_list: list[UploadFile] = []
    if file:
        upload_list.append(file)
    if files:
        upload_list.extend(files)
    if not upload_list:
        raise HTTPException(status_code=400, detail="No jewelry image provided. Please upload at least one image.")

    image_bytes_list: list[bytes] = []
    for f in upload_list[:4]:
        b = await validate_and_read_image(f)
        image_bytes_list.append(b)

    payload_bytes: bytes | list[bytes] = image_bytes_list[0] if len(image_bytes_list) == 1 else image_bytes_list

    try:
        async with get_semaphore():
            result = await run_in_threadpool(
                process_garment_pipeline,
                image_bytes=payload_bytes,
                person_mode="jewelry",
                target_garment=jewelry_type,
                include_base64=include_base64,
                custom_prompt=custom_prompt,
                output_resolution=output_resolution,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Jewelry pipeline error: {exc}") from exc
    finally:
        for f in upload_list:
            await f.close()

    base_url = str(request.base_url).rstrip("/")
    forwarded_proto = request.headers.get("x-forwarded-proto", "").lower()
    if forwarded_proto == "https" and base_url.startswith("http://"):
        base_url = base_url.replace("http://", "https://", 1)

    clean_items = []
    for g in result.get("garments", []):
        raw_url = g.get("url") or g.get("output_png_url") or ""
        full_url = f"{base_url}{raw_url}" if raw_url.startswith("/") else raw_url
        title = g.get("title") or g.get("recognized_label") or "Luxury Jewelry Studio"
        clean_items.append({
            "category": g.get("category", "jewelry"),
            "title": title,
            "url": full_url,
            "resolution": g.get("resolution", "8192x8192"),
            "output_png_url": full_url,
            "recognized_label": title,
            "garment_id": g.get("garment_id"),
            "output_png_base64": g.get("output_png_base64") if include_base64 else None,
            "stand_type": g.get("stand_type"),
            "views_analyzed": g.get("views_analyzed", len(upload_list)),
        })

    total_time_sec = round(result.get("execution_time_ms", 0) / 1000.0, 1)

    return {
        "status": "success",
        "total_garments": len(clean_items),
        "garments": clean_items,
        "processing_time_sec": total_time_sec,
        "job_id": result.get("job_id"),
        "success": True,
    }


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


@router.post("/set-gpu-url")
async def set_gpu_url_endpoint(
    url: str = Form(..., description="New Cloudflare GPU Worker URL"),
) -> Any:
    """Dynamically update the remote GPU worker URL in-memory and in .env without server restart."""
    from app.core.flow_b import check_gpu_worker_health
    cleaned_url = url.strip().rstrip("/")
    if not (cleaned_url.startswith("http://") or cleaned_url.startswith("https://")):
        raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")

    os.environ["QWEN_REMOTE_URL"] = cleaned_url

    # Persist into .env file
    env_file = Path(".env")
    if env_file.exists():
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
            new_lines = []
            found = False
            for line in lines:
                if line.startswith("QWEN_REMOTE_URL="):
                    new_lines.append(f"QWEN_REMOTE_URL={cleaned_url}")
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.append(f"QWEN_REMOTE_URL={cleaned_url}")
            env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    health = await run_in_threadpool(check_gpu_worker_health)
    return {
        "status": "ok",
        "configured_url": cleaned_url,
        "online": health.get("online", False),
        "gpu_health": health,
    }
