# Flow A — Pure Garment Background Removal (Process Guide)

Yeh document **Flow A (Without Person — 80% Workload)** ka complete step-by-step process guide hai. Isme explain kiya gaya hai ki aap kapde ki photo kaise bhejenge, server par background kaise remove hoga, aur aage downstream system ya app me transparent image kaise return hogi.

---

## 1. Flow A Kaunsi API Hai?

Aapko **Flow A (Without Person)** chalane ke liye sirf is single API par photo bhejni hai:

```http
POST /api/background-removal
Content-Type: multipart/form-data
```

- **Live URL:** `http://ai.arangtik.com/api/background-removal`
- **Local URL:** `http://127.0.0.1:8000/api/background-removal`
- **Kaam:** Flat-lay, Hanger, ya Mannequin par rakhe kapde ka pure background remove karke **$1024 \times 1024$ transparent PNG** return karna.
- **Speed:** ~1.0 se 1.5 seconds (BiRefNet Lite engine).
- **Default Behavior:** Is API me `person_mode` by default **`without_person`** set hai. Isliye aapko koi extra parameter dene ki zaroorat nahi hai, bas photo bhejni hai!

---

## 2. Server Par Process Kaise Hota Hai? (Step-by-Step Flow)

Jab aap image bhejte hain, backend server par ye 6 steps execute hote hain:

```
[Input Photo] 
      │
      ▼
1. Validation (File size, format check: JPG/PNG/WEBP)
      │
      ▼
2. BiRefNet Lite Segmentation (Fabric edges aur texture preserve karke background remove karta hai)
      │
      ▼
3. Multi-Garment Detection (Agar photo me Kurta + Pajama dono hain toh alag-alag split karta hai)
      │
      ▼
4. Auto-Deskew & Straighten (Agar kapda tedha/tilted hai toh angle calculate karke seedha karta hai)
      │
      ▼
5. 1024x1024 Canvas Normalization (Aspect ratio maintain karke center me 82% fill karta hai)
      │
      ▼
6. Transparent PNG Generation (Clean RGBA PNG output me save hoti hai)
      │
      ▼
[Client Response: JSON URL ya Direct Image Stream]
```

---

## 3. Image Kaise Bhejna Hai? (Code Examples)

### A. cURL (Terminal / Command Line)

#### Option 1: JSON Response Paane Ke Liye (Recommended)
```bash
curl -X POST "http://ai.arangtik.com/api/background-removal" \
  -F "file=@/path/to/your/garment.jpg"
```

#### Option 2: Direct Image File Download Karne Ke Liye
```bash
curl -X POST "http://ai.arangtik.com/api/background-removal" \
  -F "file=@/path/to/your/garment.jpg" \
  -F "response_format=image_direct" \
  --output clean_garment_1024.png
```

---

### B. Python Client (requests / httpx)

```python
import requests

url = "http://ai.arangtik.com/api/background-removal"

# 1. Send photo to API
with open("saree.jpg", "rb") as img_file:
    files = {"file": ("saree.jpg", img_file, "image/jpeg")}
    response = requests.post(url, files=files)

data = response.json()

if data.get("success"):
    garment = data["garments"][0]
    output_url = "http://ai.arangtik.com" + garment["output_png_url"]
    print(f"Success! Execution Time: {data['execution_time_ms']} ms")
    print(f"Transparent Image URL: {output_url}")

    # 2. Download the transparent PNG for your next step
    img_resp = requests.get(output_url)
    with open("downloaded_clean_saree.png", "wb") as f:
        f.write(img_resp.content)
    print("Saved clean 1024x1024 transparent PNG successfully!")
```

---

### C. Node.js / JavaScript (fetch)

```javascript
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

async function removeGarmentBackground(imagePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(imagePath));

  const response = await axios.post('http://ai.arangtik.com/api/background-removal', form, {
    headers: form.getHeaders(),
  });

  const result = response.data;
  console.log('Processed in:', result.execution_time_ms, 'ms');
  console.log('Garment Label:', result.garments[0].recognized_label);
  console.log('Output Image URL:', 'http://ai.arangtik.com' + result.garments[0].output_png_url);
  
  return result.garments[0].output_png_url;
}

removeGarmentBackground('./kurta.png');
```

---

### D. Postman Setup

1. **Method:** `POST`
2. **URL:** `http://ai.arangtik.com/api/background-removal`
3. **Body Tab:**
   - Select `form-data`
   - Key: `file` (Dropdown me `Text` ko badal kar `File` select karein)
   - Value: Apne computer se kapde ki photo choose karein
4. **Send** button dabayein.

---

## 4. API Ka Response Format

### Standard JSON Output:
```json
{
  "success": true,
  "job_id": "7f8b91a2-3456-4abc-9def-0123456789ab",
  "mode": "without_person",
  "has_human": false,
  "total_garments_found": 1,
  "execution_time_ms": 1140.25,
  "scene_type": "flat_lay",
  "outfit_mode": "mode_b_single_piece",
  "garments": [
    {
      "garment_id": "garment_001",
      "category": "product",
      "recognized_label": "Garment Cutout",
      "image": "7f8b91a2-3456-4abc-9def-0123456789ab_garment_001.png",
      "confidence": 0.99,
      "coverage_percentage": 100.0,
      "width": 1024,
      "height": 1024,
      "image_dimensions": [1024, 1024],
      "rotation_angle_deg": 0.0,
      "repaired": false,
      "output_png_url": "/api/outputs/7f8b91a2-3456-4abc-9def-0123456789ab_garment_001.png",
      "background_removal_status": "clean_isolated",
      "background_removal_confidence": 0.99
    }
  ]
}
```

---

## 5. Returned Image Ko Aage Kaise Use Karein?

Response me mile `output_png_url` ko complete URL banane ke liye server domain jod lein:

```text
http://ai.arangtik.com/api/outputs/7f8b91a2-3456-4abc-9def-0123456789ab_garment_001.png
```

Is image ko aap:
1. **Frontend / Mobile App** me direct `<img src="...">` me show kar sakte hain.
2. **Database** me iska URL save kar sakte hain.
3. **Downstream AI Pipeline** (e.g. Virtual Try-On ya E-commerce catalog generator) me input ki tarah pass kar sakte hain.

---

## 6. Multi-Garment Support (Agar photo me Kurta aur Pajama dono hain)

Agar photo me 2 alag kapde hain (jaise suit set ya kurta + pajama):
- API unhe automatically detect karke `garments` array me alag-alag split kar degi (`garment_001`, `garment_002`).
- Aap dono ka alag-alag transparent PNG URL ek hi request se praapt kar sakte hain.
- Poore set ko ek sath ZIP archive me download karne ke liye:
  ```http
  GET /api/download-batch-zip/{job_id}
  ```
