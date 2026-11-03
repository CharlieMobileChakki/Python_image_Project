# Arangtik Garment Reconstruction Engine

A standalone FastAPI service that uses **SDXL Inpainting + IP-Adapter Plus** to reconstruct missing garment fabric from an original reference photo and a damaged, preferably transparent, garment image.

It does not run human detection, person removal, SegFormer, BiRefNet, or LAMA. Those remain upstream stages in your existing Arangtik pipeline. This service starts after person removal.

## 1. Input contract

The API requires three images:

1. `original_image`: the original photo containing the garment. This is the fallback visual reference.
2. `damaged_garment`: the output from your human-removal pipeline, preferably an RGBA transparent PNG.
3. `repair_mask`: a grayscale or RGB mask with exactly the same canvas geometry as `damaged_garment`.

Optional:

4. `garment_reference`: a tight crop showing only the visible garment from the original photo. This is strongly recommended because a full photo also contains a face, hair, skin, background and accessories.

### Mask convention — critical

- WHITE (255): generate/reconstruct garment fabric here.
- BLACK (0): preserve existing pixels here.
- The white region must cover only the missing garment shape. Do not mark the transparent background white.
- For a missing neckline/shoulder area, draw a mask that follows the expected garment boundary. If the mask includes the background, the model may generate fabric outside the actual garment.

Create a blank mask matching your damaged garment:

```bash
python scripts/make_mask_template.py samples/damaged_garment.png samples/repair_mask.png
```

Then open `repair_mask.png` in GIMP, Krita, Photoshop or another image editor. Paint the missing garment area white and keep everything else black. Save as PNG.

## 2. Hardware

Recommended for the provided SDXL + Plus ViT-H setup:

- NVIDIA GPU with 24 GB VRAM preferred (RTX 3090 24GB is a suitable test server).
- 32 GB system RAM recommended.
- Ubuntu 22.04/24.04, current NVIDIA driver and CUDA-compatible PyTorch wheel.
- Allow at least 20–30 GB free disk space for checkpoints, cache and environment.

12 GB VRAM may work with CPU offload and reduced resolution, but is not guaranteed. CPU inference is extremely slow and disabled by default.

## 3. Server installation

Use Python 3.10 or 3.11 in a fresh virtual environment.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
mkdir -p ~/arangtik-garment-reconstruction
cd ~/arangtik-garment-reconstruction
```

Copy this project folder to the server, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

Install a CUDA-enabled PyTorch build. Choose the command from the official PyTorch selector that matches the server's driver/CUDA support. Example for CUDA 12.6 wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify CUDA before installing the rest:

```bash
nvidia-smi
python scripts/check_gpu.py
```

Install application dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

The default configuration requires CUDA. Do not set `ALLOW_CPU_FALLBACK=true` on the production server unless you deliberately want slow CPU inference.

## 4. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On first startup, Hugging Face downloads the SDXL inpainting checkpoint, IP-Adapter Plus weights and image encoder. This is a large download; wait for startup to complete. The model is loaded once and reused for requests.

Check:

```bash
curl http://127.0.0.1:8000/health
```

Open Swagger UI:

```text
http://SERVER_IP:8000/docs
```

Do not expose port 8000 publicly without firewall rules, authentication and HTTPS. For initial testing, use SSH port forwarding:

```bash
ssh -L 8000:127.0.0.1:8000 username@SERVER_IP
```

Then open `http://127.0.0.1:8000/docs` on your local machine.

## 5. Test with cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/reconstruct-garment" \
  -F "original_image=@samples/original_photo.jpg" \
  -F "damaged_garment=@samples/damaged_garment.png" \
  -F "repair_mask=@samples/repair_mask.png" \
  -F "garment_reference=@samples/garment_crop.png" \
  -F "steps=30" \
  -F "guidance_scale=5.5" \
  -F "strength=0.95" \
  -F "ip_adapter_scale=0.75" \
  -F "seed=12345" \
  --output reconstructed_garment.png
```

`garment_reference` is optional. If omitted, `original_image` is used as the IP-Adapter reference. A garment-only crop usually provides a cleaner reference than the full person photo.

Response is a transparent RGBA PNG. Response headers include:

- `X-Inference-Seconds`
- `X-Model-Device`
- `X-Seed`
- `X-Output-Width`
- `X-Output-Height`

## 6. Python client example

```python
import requests

url = "http://127.0.0.1:8000/api/v1/reconstruct-garment"
files = {
    "original_image": ("original.jpg", open("samples/original_photo.jpg", "rb"), "image/jpeg"),
    "damaged_garment": ("damaged.png", open("samples/damaged_garment.png", "rb"), "image/png"),
    "repair_mask": ("mask.png", open("samples/repair_mask.png", "rb"), "image/png"),
    "garment_reference": ("garment_crop.png", open("samples/garment_crop.png", "rb"), "image/png"),
}
data = {
    "steps": 30,
    "guidance_scale": 5.5,
    "strength": 0.95,
    "ip_adapter_scale": 0.75,
    "seed": 12345,
}
response = requests.post(url, files=files, data=data, timeout=900)
response.raise_for_status()
with open("reconstructed_garment.png", "wb") as output:
    output.write(response.content)
print("Saved reconstructed_garment.png")
print("Inference seconds:", response.headers.get("X-Inference-Seconds"))
```

Install `requests` on the client if needed:

```bash
pip install requests
```

## 7. Integrating into your existing Arangtik pipeline

Call this service only after your current human-removal/garment extraction stage has produced:

- original photo;
- damaged transparent garment cutout;
- a repair mask that marks missing garment pixels only.

Do not send the human segmentation mask as the repair mask without converting it. The repair mask must identify the missing garment region, not the person region generally.

## 8. Parameters to tune during experiments

| Parameter | Starting value | Meaning |
|---|---:|---|
| `steps` | 30 | More steps can improve detail but increase time |
| `guidance_scale` | 5.5 | Prompt influence |
| `strength` | 0.95 | How strongly masked region can be regenerated |
| `ip_adapter_scale` | 0.75 | Influence of garment reference image |
| `MAX_IMAGE_SIDE` | 1024 | Maximum inference image side |
| `MASK_BLUR_RADIUS` | 1.5 | Softens mask boundary slightly |

Try adapter scales 0.55, 0.70, 0.85 and 1.0 on the same test set. Too high a reference scale can overemphasize the crop and may not align folds with the damaged garment. Keep the seed fixed while comparing parameter changes.

## 9. Important limitations

- The model estimates hidden fabric. It cannot recover exact embroidery or folds that were never visible in the original photo.
- A reference crop with only the visible garment is recommended. If hair/skin covers the garment, the model can only infer the hidden portion.
- Mask quality is decisive. A bad mask can create fabric outside the garment silhouette or erase valid fabric.
- The output is an image cutout, not a real 3D mesh or physically reconstructed garment.
- This is a generative restoration model, not a guaranteed exact pixel recovery system. Evaluate it on a held-out test set before production use.
- The SDXL Inpainting checkpoint has its own model license. Review the checkpoint and adapter licenses and your company's intended commercial use before deployment.

## 10. Official references

- SDXL Inpainting model: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
- IP-Adapter Plus weights: https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models
- Diffusers IP-Adapter guide: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter
- Diffusers inpainting guide: https://huggingface.co/docs/diffusers/using-diffusers/inpaint
