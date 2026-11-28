# 👗 AI Virtual Try-On API — Integration Specification

Yeh document aapke Node.js backend developer ke liye **direct and clear** specification hai.

---

## 1. API Details

- **Method:** `POST`
- **Endpoint:** `/api/virtual-tryon`
- **Base URL:** `http://localhost:8000` (Local) ya Production Server URL
- **Content-Type:** `multipart/form-data`

---

## 2. Request (Humein Kya Bhejna Hai)

Form-Data me sirf yeh fields bhejni hain:

| Field Name | Type | Zaroori Hai? (Required) | Description |
| :--- | :--- | :--- | :--- |
| **`avatar`** | `File` (Image: jpg/png) | **हाँ (Mandatory)** | User ka Avatar ya clear face photo (iska face preserve hoga). |
| **`garments`** | `File` ya `File[]` | **हाँ (Mandatory)** | Kapde ki image(s) — Shirt, Coat, T-shirt, Jeans, Dress आदि. Ek sath multiple files bhi bhej sakte hain. |
| **`occasion`** | `String` | *Optional* | Default: `"smart_casual"`. Options: `"smart_casual"`, `"formal"`, `"party"`, `"ethnic"`. |
| **`max_combinations`** | `Number` | *Optional* | Default: `1`. Kitne looks chahiye (1 to 4). |

---

## 3. Response (Hum Kya Return Karenge)

API se seedha yeh clean JSON milega:

```json
{
  "success": true,
  "status": "success",
  "job_id": "tryon_9f821a4e21",
  "image_url": "/outputs/tryon_9f821a4e21_look_0.png",
  "output_png_url": "/outputs/tryon_9f821a4e21_look_0.png",
  "combinations": [
    {
      "combination_index": 0,
      "title": "Smart Casual Sporty Crisp",
      "url": "/outputs/tryon_9f821a4e21_look_0.png",
      "style_reason": "Black tailored blazer layered neatly over white dress shirt."
    }
  ],
  "processing_time_sec": 14.5
}
```

> **Direct Output Image URL:** `http://localhost:8000` + `data.image_url`

---

## 4. Node.js Code Snippet (Axios)

Developer is code ko direct copy-paste kar sakta hai:

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function runVirtualTryOn(avatarFilePath, garmentFilePaths) {
  const form = new FormData();

  // 1. Avatar (User Face)
  form.append('avatar', fs.createReadStream(avatarFilePath));

  // 2. Garments (Kapde - single ya multiple)
  garmentFilePaths.forEach((filePath) => {
    form.append('garments', fs.createReadStream(filePath));
  });

  // 3. Optional
  form.append('occasion', 'smart_casual');
  form.append('max_combinations', 1);

  const API_URL = 'http://localhost:8000/api/virtual-tryon';

  const res = await axios.post(API_URL, form, {
    headers: { ...form.getHeaders() },
    timeout: 300000, // 5 min GPU timeout
  });

  // Direct Output Image URL:
  console.log('Result Image URL:', 'http://localhost:8000' + res.data.image_url);
  return res.data;
}

// Example usage:
// runVirtualTryOn('./avatar.png', ['./blazer.jpg', './shirt.jpg']);
```
