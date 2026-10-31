# Arangtik Garment Extraction API — prototype

This is a **Flow A** starter only: upload one clothing photo, remove its background with a pretrained model, crop/center the extracted foreground on a transparent 1024x1024 PNG, and return that PNG to the browser.

It does **not** yet implement Flow B (garments worn by a person), SDXL inpainting, user accounts, S3, MongoDB, or a job queue.

## Stack

- Python 3.11–3.13
- FastAPI + Uvicorn
- `rembg` as the inference/wrapper library
- Pretrained ONNX weights selected with `REMBG_MODEL`
- Pillow for input validation, crop/resize, alpha handling, and PNG output

## Windows PowerShell setup

Open PowerShell in this folder (`arangtik-garment-backend`):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

If PowerShell blocks activation, run the venv Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

On first startup, `rembg` downloads the selected model weights. This requires internet and extra disk space; let startup finish before testing. Keep one Uvicorn worker during local testing so the model is not loaded multiple times in memory.

## Connect the existing HTML test UI

Open `arangtik-garment-lab.html` in your browser and leave the endpoint as:

`http://127.0.0.1:8000/api/v1/extract-garment`

Use **Check API**, upload one JPG/PNG/WEBP image, then click **Extract garment**. The upload field is `file`. This endpoint returns `image/png` directly, which matches the HTML test UI.

## Endpoints

- `GET /health` — status and configured model name
- `GET /docs` — interactive Swagger UI
- `POST /api/v1/extract-garment` — multipart upload (`file`), returns a transparent PNG

## Comparing models

Change `REMBG_MODEL` in `.env`, stop Uvicorn with `Ctrl+C`, then restart it. Test the **same set of images** with one model at a time. Supported candidate names include `birefnet-general-lite`, `birefnet-general`, `isnet-general-use`, and `u2net` in current `rembg` releases. The selected model weights are downloaded separately on first use. Larger models may require more RAM and take longer.

Do not compare models from just one image. Use at least 30–50 test images covering plain backgrounds, busy backgrounds, dark clothes, patterned fabric, long sleeves, thin straps, and wrinkles. Record which images lose fabric or retain background. This is a benchmark harness, not a guarantee of a particular accuracy level.

## Important limitations

1. This general background-removal pipeline is a baseline for a single garment photographed on its own. It can accidentally keep a hanger/background object or remove thin garment edges.
2. `u2net_cloth_seg` is described by `rembg` as clothing parsing for human portraits (upper/lower/full body); it is not automatically the best option for an isolated flat-lay garment.
3. No inpainting is enabled. This is intentional: generated sleeves/collars can change the original item design. Keep the original image and inspect output.
4. Model weights have their own licenses, separate from the `rembg` Python package license. Review each selected model and its training-data terms before using it commercially. In particular, `bria-rmbg` in `rembg` is documented as requiring a paid agreement for commercial use.
5. CORS currently allows any origin for local testing. Before deployment, set `allow_origins` to the exact web app origin(s), add authentication/rate limits, and validate resource limits.
6. Images are processed in memory and are not saved to disk, S3, or MongoDB.

## Next engineering phase

After Flow A is benchmarked, add Flow B as a separate pipeline using a human-clothes parsing model such as SegFormer B2 Clothes or SCHP. Then compare garment masks on your own images. Add SAM 2 only where prompt/refinement is useful. Consider SDXL inpainting later, only behind an explicit reconstruction option.
