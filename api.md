# Arangtik Garment AI — Production Headless API Specification

Complete machine-to-machine API reference for Arangtik Garment AI.

- **Base URL:** `http://ai.arangtik.com` (or `http://127.0.0.1:8000`)
- **Prefix:** `/api` (No `/v1` prefix)
- **Standard Output Canvas:** $1024 \times 1024$ pixels, transparent PNG (RGBA)

---

## 1. Primary Background Removal & Garment Extraction API

This single API handles background removal, multi-garment splitting, auto-deskewing, and model human removal:

```http
POST /api/background-removal
Content-Type: multipart/form-data
```

### Request Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | `File` (**Required**) | — | Input image (`.jpg`, `.png`, `.webp`, up to 15 MB) |
| `person_mode` | `string` | `"without_person"` | **`"without_person"`** (Flow A: 80% Workload, pure background removal, ~1.2s)<br>**`"with_person"`** (Flow B: model parsing, human removal, fabric inpainting) |
| `response_format` | `string` | `"json"` | **`"json"`**: Returns JSON metadata & download URLs<br>**`"image_direct"`**: Streams transparent $1024 \times 1024$ PNG directly |
| `target_garment` | `string` | `"all"` | Filter: `"all"`, `"upper_clothes"`, `"pants"`, `"dress"`, `"skirt"` |
| `auto_straighten` | `boolean` | `true` | Automatically detects tilt and straightens garment |
| `auto_repair` | `boolean` | `true` | Enables fabric repair for occlusions in `with_person` mode |
| `include_base64` | `boolean` | `false` | Include Base64 data URI directly in JSON |

---

### Example A: Fast Background Removal (Flat-Lay / Hanger — 80% Workload)

Just send the file! Default settings automatically trigger ultra-fast Flow A (~1.2s):

```bash
curl -X POST "http://ai.arangtik.com/api/background-removal" \
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

### Example B: Direct Image Stream (Instant PNG Download)

Returns binary image stream directly (`Content-Type: image/png`):

```bash
curl -X POST "http://ai.arangtik.com/api/background-removal" \
  -F "file=@kurta.jpg" \
  -F "response_format=image_direct" \
  --output clean_kurta_1024.png
```

---

### Example C: Model-Worn Garment (Person Removal — Flow B)

When a human model is wearing the outfit:

```bash
curl -X POST "http://ai.arangtik.com/api/background-removal" \
  -F "file=@model_wearing_saree.jpg" \
  -F "person_mode=with_person"
```

---

## 2. Qwen-Image Fabric Reconstruction API

Deep generative synthesis and inpainting for occluded fabric and missing areas using Qwen-Image-2.1:

```http
POST /api/qwen-image
Content-Type: multipart/form-data
```

### Request Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `original_image` | `File` | **Yes** | High-resolution photo of person wearing the garment |
| `damaged_garment`| `File` | No | Garment cutout with arms/hands removed (auto-computed if omitted) |
| `repair_mask` | `File` | No | Occlusion mask (auto-computed if omitted) |
| `prompt` | `string` | No | Optional fabric/garment preservation instructions |
| `steps` | `integer` | No | Denoising steps (default: 28) |

#### cURL Request:
```bash
curl -X POST "http://ai.arangtik.com/api/qwen-image" \
  -F "original_image=@model_with_crossed_arms.jpg" \
  --output qwen_reconstructed_garment.png
```

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
