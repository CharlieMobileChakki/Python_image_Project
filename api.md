# Arangtik Garment AI — Production Headless API

This document provides complete specification for the Arangtik Garment AI Machine-to-Machine (M2M) API backend.

**Base Server URL:** `http://ai.arangtik.com` (or `http://127.0.0.1:8000`)  
**Prefix:** `/api` (No `/v1` prefix)  
**Standard Output Canvas:** $1024 \times 1024$ pixels, transparent PNG (RGBA)

---

## 1. Primary Garment Processing Endpoint (Single Unified API)

All garment extraction, background removal, auto-straightening, and model-garment separation are consolidated into **one single endpoint**:

```http
POST /api/process-garment
Content-Type: multipart/form-data
```

### Request Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | `File` (**Required**) | — | Input image (`.jpg`, `.png`, `.webp`, up to 15 MB) |
| `person_mode` | `string` | `"without_person"` | **`"without_person"`** (Flow A: 80% workload, flat-lay/hanger background removal, ~1.2s)<br>**`"with_person"`** (Flow B: model parsing, human removal, fabric inpainting) |
| `response_format` | `string` | `"json"` | **`"json"`**: Returns full JSON metadata & download URLs<br>**`"image_direct"`**: Streams transparent $1024 \times 1024$ PNG directly |
| `target_garment` | `string` | `"all"` | Filter: `"all"`, `"upper_clothes"`, `"pants"`, `"dress"`, `"skirt"` |
| `auto_straighten` | `boolean` | `true` | Automatically detects tilt and deskews garments |
| `auto_repair` | `boolean` | `true` | Enables fabric repair for occlusions when in `with_person` mode |
| `include_base64` | `boolean` | `false` | Include Base64 data URI directly in JSON response |

---

### Use Case A: Pure Background Removal (Flat-Lay / Hanger / Mannequin — 80% Workload)

Simply post the image. No extra parameters required!

#### cURL Request:
```bash
curl -X POST "http://ai.arangtik.com/api/process-garment" \
  -F "file=@kurta.jpg"
```

#### JSON Response (HTTP 200):
```json
{
  "success": true,
  "job_id": "8d3e9114-1234-4567-89ab-cdef01234567",
  "mode": "without_person",
  "has_human": false,
  "total_garments_found": 1,
  "execution_time_ms": 1184.5,
  "scene_type": "flat_lay",
  "outfit_mode": "mode_b_single_piece",
  "garments": [
    {
      "garment_id": "garment_001",
      "category": "product",
      "recognized_label": "Garment Cutout",
      "image": "8d3e9114-1234-4567-89ab-cdef01234567_garment_001.png",
      "confidence": 0.985,
      "coverage_percentage": 100.0,
      "width": 1024,
      "height": 1024,
      "image_dimensions": [1024, 1024],
      "rotation_angle_deg": -1.2,
      "repaired": false,
      "output_png_url": "/api/outputs/8d3e9114-1234-4567-89ab-cdef01234567_garment_001.png",
      "background_removal_status": "clean_isolated",
      "background_removal_confidence": 0.99
    }
  ]
}
```

---

### Use Case B: Direct Image Stream (Instant PNG Download)

Returns binary image stream directly (`Content-Type: image/png`):

```bash
curl -X POST "http://ai.arangtik.com/api/process-garment" \
  -F "file=@kurta.jpg" \
  -F "response_format=image_direct" \
  --output clean_kurta_1024.png
```

---

### Use Case C: Model Photo (Person Parsing & Human Removal — Flow B)

Set `person_mode="with_person"`:

```bash
curl -X POST "http://ai.arangtik.com/api/process-garment" \
  -F "file=@model_wearing_suit.jpg" \
  -F "person_mode=with_person"
```

---

## 2. Fabric Reconstruction Endpoint (Qwen-Image-2.1)

For deep generative reconstruction of missing fabric, folds, and arm-occluded regions:

```http
POST /api/reconstruct-garment
Content-Type: multipart/form-data
```

### Request Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `original_image` | `File` | Yes | High-resolution photo of person wearing the garment |
| `damaged_garment`| `File` | No | Garment cutout with arms removed (auto-computed if omitted) |
| `repair_mask` | `File` | No | Occlusion mask (auto-computed if omitted) |
| `prompt` | `string` | No | Optional fabric/garment preservation instructions |
| `steps` | `integer` | No | Denoising steps (default: 28) |

---

## 3. Asset & Utility Endpoints

### Fetch Processed Garment Image
```http
GET /api/outputs/{filename}
```
Returns: Binary $1024 \times 1024$ PNG image.

### Download Batch ZIP Archive
```http
GET /api/download-batch-zip/{job_id}
```
Returns: Single ZIP archive containing all extracted garment pieces for a job.

### System Health & Hardware Status
```http
GET /health
GET /api/health
```
Returns: Loaded AI model status, GPU VRAM, CUDA capability, and active workers.
