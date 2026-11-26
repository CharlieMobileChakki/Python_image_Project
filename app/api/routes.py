"""FastAPI router defining all endpoints for Arangtik Garment AI."""
from __future__ import annotations

import asyncio
from io import BytesIO
import os
from pathlib import Path
from typing import Any
import zipfile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from app.core.pipeline import OUTPUT_DIR, process_garment_pipeline
from app.schemas import ProcessGarmentResponse

router = APIRouter()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None
_semaphore_limit: int = 0


def _resolve_concurrency_limit() -> int:
    raw = os.getenv("MAX_CONCURRENT_JOBS", "auto").strip().lower()
    if raw and raw != "auto":
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    from app.core.flow_b import get_worker_capacity
    return max(4, get_worker_capacity() * 2)


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop, _semaphore_limit
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    target_limit = _resolve_concurrency_limit()
    if _semaphore is None or _semaphore_loop != current_loop:
        _semaphore = asyncio.Semaphore(target_limit)
        _semaphore_loop = current_loop
        _semaphore_limit = target_limit
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


async def _collect_uploaded_images(
    file: UploadFile | None,
    files: list[UploadFile] | None,
    empty_detail: str,
) -> tuple[bytes | list[bytes], list[UploadFile]]:
    """Collect and validate up to 4 uploaded images, returning payload bytes and file handles."""
    upload_list: list[UploadFile] = []
    if file:
        upload_list.append(file)
    if files:
        upload_list.extend(files)
    if not upload_list:
        raise HTTPException(status_code=400, detail=empty_detail)

    image_bytes_list: list[bytes] = []
    for f in upload_list[:4]:
        image_bytes_list.append(await validate_and_read_image(f))

    payload: bytes | list[bytes] = image_bytes_list[0] if len(image_bytes_list) == 1 else image_bytes_list
    return payload, upload_list


def _resolve_base_url(request: Request) -> str:
    base_url = str(request.base_url).rstrip("/")
    if request.headers.get("x-forwarded-proto", "").lower() == "https" and base_url.startswith("http://"):
        return base_url.replace("http://", "https://", 1)
    return base_url


def _format_garment_items(
    result: dict[str, Any],
    base_url: str,
    include_base64: bool,
    default_title: str = "Studio Garment",
) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for g in result.get("garments", []):
        raw_url = g.get("url") or g.get("output_png_url") or ""
        full_url = f"{base_url}{raw_url}" if raw_url.startswith("/") else raw_url
        title = g.get("title") or g.get("recognized_label") or default_title
        formatted.append({
            "category": g.get("category", "product"),
            "title": title,
            "url": full_url,
            "resolution": g.get("resolution") or ("2048x2048" if g.get("stand_type") else "1024x1024"),
            "output_png_url": full_url,
            "recognized_label": title,
            "garment_id": g.get("garment_id"),
            "output_png_base64": g.get("output_png_base64") if include_base64 else None,
            "stand_type": g.get("stand_type"),
            "presentation_mode": g.get("presentation_mode", "ghost_mannequin"),
            "detected_item_count": g.get("detected_item_count", 1),
            "validation_metrics": g.get("validation_metrics"),
            "engine": g.get("engine"),
            "gpu_name": g.get("gpu_name"),
            "views_analyzed": g.get("views_analyzed", 1),
        })
    return formatted


@router.post(
    "/background-removal",
    response_model=ProcessGarmentResponse,
    response_model_exclude_none=True,
    summary="Direct 1024x1024 Studio Garment & Jewelry Generator API",
)
@router.post(
    "/generate-garment",
    response_model=ProcessGarmentResponse,
    response_model_exclude_none=True,
    summary="Direct 1024x1024 Studio Garment & Jewelry Generator API (Alias)",
    include_in_schema=False,
)
async def generate_garment_endpoint(
    request: Request,
    file: UploadFile = File(None, description="Primary input image (apparel, model-worn, flat-lay, or jewelry)"),
    files: list[UploadFile] = File(None, description="Optional multi-angle input images (up to 4 photos for jewelry)"),
    target_garment: str = Form("all", description="Target filter: 'all', 'upper_clothes', 'pants', 'dress', 'saree', 'necklace', 'ring', 'earrings', 'bangles'"),
    include_base64: bool = Form(False, description="Include Base64 data URI in JSON"),
    response_format: str = Form("json", description="'json' (default for API / UI) or 'image_direct' for binary PNG stream"),
    custom_prompt: str | None = Form(None, description="Optional custom prompt override"),
    item_count_override: int | None = Form(None, description="Optional manual quantity override (e.g. 1 for 1 kada)"),
) -> Any:
    """Direct AI Studio Product Generation API (`1024x1024` white-background PNG)."""
    payload_bytes, upload_list = await _collect_uploaded_images(
        file, files, "No image file provided. Please upload at least one image."
    )

    try:
        async with get_semaphore():
            result = await run_in_threadpool(
                process_garment_pipeline,
                image_bytes=payload_bytes,
                target_garment=target_garment,
                include_base64=include_base64,
                custom_prompt=custom_prompt,
                item_count_override=item_count_override,
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
                    "X-Flow-Mode": result.get("mode", "studio_generation"),
                    "X-Garment-Category": top_garment["category"],
                    "X-Garment-Label": top_garment["recognized_label"],
                    "X-Execution-Time-Ms": str(result["execution_time_ms"]),
                    "X-Canvas-Size": "1024x1024",
                },
            )

    clean_garments = _format_garment_items(result, _resolve_base_url(request), include_base64)
    total_time_sec = round(result.get("execution_time_ms", 0) / 1000.0, 1)

    return {
        "status": "success",
        "total_garments": len(clean_garments),
        "garments": clean_garments,
        "processing_time_sec": total_time_sec,
        "execution_time_ms": result.get("execution_time_ms", int(total_time_sec * 1000)),
        "mode": result.get("mode", "studio_generation"),
        "diagnostics": result.get("diagnostics", {}),
        "job_id": result.get("job_id"),
        "success": True,
    }


@router.post(
    "/jewelry-studio",
    response_model=ProcessGarmentResponse,
    response_model_exclude_none=True,
    summary="Dedicated 3D Luxury Jewelry Studio (Single or Multi-Angle)",
)
async def jewelry_studio_endpoint(
    request: Request,
    file: UploadFile = File(None, description="Primary jewelry image"),
    files: list[UploadFile] = File(None, description="Up to 4 multi-angle views of the jewelry piece"),
    jewelry_type: str = Form("auto", description="Category: 'auto', 'necklace', 'ring', 'earrings', 'bangles'"),
    custom_prompt: str | None = Form(None, description="Optional custom prompt override"),
    include_base64: bool = Form(False, description="Include Base64 data URI in JSON"),
) -> Any:
    """Dedicated 3D Luxury Jewelry Studio Endpoint."""
    payload_bytes, upload_list = await _collect_uploaded_images(
        file, files, "No jewelry image provided. Please upload at least one image."
    )

    try:
        async with get_semaphore():
            result = await run_in_threadpool(
                process_garment_pipeline,
                image_bytes=payload_bytes,
                target_garment=jewelry_type,
                include_base64=include_base64,
                custom_prompt=custom_prompt,
                force_jewelry=True,
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

    clean_items = _format_garment_items(
        result, _resolve_base_url(request), include_base64, default_title="Luxury Jewelry Studio"
    )
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
    """Download or view a generated studio PNG."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Requested image does not exist.")
    return FileResponse(path=file_path, media_type="image/png", filename=filename)


@router.get("/download-batch-zip/{job_id}")
async def download_batch_zip(job_id: str) -> Response:
    """Download all generated studio PNGs for a given job in a single ZIP archive."""
    png_files = sorted(list(OUTPUT_DIR.glob(f"{job_id}_*.png")))
    if not png_files:
        raise HTTPException(status_code=404, detail="No generated images found for this job ID.")

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in png_files:
            zf.write(p, arcname=p.name)

    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="arangtik_{job_id}_studio.zip"'},
    )


@router.post("/forward-downstream")
async def forward_downstream_endpoint(
    job_id: str = Form(..., description="Job ID of generated images"),
    target_api_url: str = Form(..., description="Destination API URL"),
) -> Any:
    """Forward all generated studio PNGs in one multipart request to a downstream API."""
    import httpx
    png_files = sorted(list(OUTPUT_DIR.glob(f"{job_id}_*.png")))
    if not png_files:
        raise HTTPException(status_code=404, detail="No generated images found for this job.")

    files_payload = [("files[]", (p.name, p.read_bytes(), "image/png")) for p in png_files]

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
    url: str = Form(..., description="Remote GPU endpoint URL (RunPod https://api.runpod.ai/v2/<ID> or HTTP URL(s))"),
    api_key: str | None = Form(None, description="Optional updated API key (RunPod or X-API-Key)"),
) -> Any:
    """Dynamically update remote GPU endpoint URL(s) in-memory and in .env without restart."""
    from app.core.flow_b import check_gpu_worker_health

    raw_urls = [u.strip().rstrip("/") for u in url.split(",") if u.strip()]
    if not raw_urls or any(not (u.startswith("http://") or u.startswith("https://")) for u in raw_urls):
        raise HTTPException(status_code=400, detail="Invalid URL. Each endpoint must start with http:// or https://")

    cleaned_url = ",".join(raw_urls)
    os.environ["QWEN_REMOTE_URL"] = cleaned_url
    if api_key is not None and api_key.strip():
        cleaned_key = api_key.strip()
        if "api.runpod.ai" in cleaned_url.lower():
            os.environ["RUNPOD_API_KEY"] = cleaned_key
        else:
            os.environ["QWEN_API_KEY"] = cleaned_key

    env_file = Path(".env")
    updates: dict[str, str] = {"QWEN_REMOTE_URL": f'"{cleaned_url}"'}
    if api_key is not None and api_key.strip():
        key_var = "RUNPOD_API_KEY" if "api.runpod.ai" in cleaned_url.lower() else "QWEN_API_KEY"
        updates[key_var] = f'"{api_key.strip()}"'

    if env_file.exists():
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
            seen: set[str] = set()
            new_lines = []
            for line in lines:
                matched_key = next((k for k in updates if line.startswith(f"{k}=")), None)
                if matched_key:
                    new_lines.append(f"{matched_key}={updates[matched_key]}")
                    seen.add(matched_key)
                else:
                    new_lines.append(line)
            for k, v in updates.items():
                if k not in seen:
                    new_lines.append(f"{k}={v}")
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
