# AI Stylized Avatar Studio — Node.js Backend Integration Guide

This guide explains how to integrate the **AI Stylized Avatar Profile Picture Generator API** into any Node.js / Express backend.

---

## 1. Endpoint Overview

| Parameter | Value |
| :--- | :--- |
| **Endpoint URL** | `POST /api/generate-avatar` |
| **Alias URL** | `POST /api/avatar-studio` |
| **Content-Type** | `multipart/form-data` |
| **Output Canvas** | `1024x1024` Crisp Square PNG |
| **Authentication** | Standard Session / Bearer Token (if configured) |

---

## 2. Request Form-Data Fields

| Field Name | Type | Required | Default | Description / Allowed Values |
| :--- | :--- | :---: | :---: | :--- |
| `file` | `Binary File` | **Yes** | — | User photo (Selfie, portrait, or casual photo). Formats: `.jpg`, `.jpeg`, `.png`, `.webp`. |
| `style` | `String` | No | `"vector_art"` | **`"vector_art"`**: Modern 2D Vector Art illustration (Default).<br>**`"digital_illustration"`**: Rich painterly digital art.<br>**`"anime_portrait"`**: Japanese Anime / Webtoon hero aesthetic. |
| `halo_color` | `String` | No | `"auto"` | Circular background badge color:<br>• `"auto"` (AI chooses best complementary color)<br>• `"soft_blue"` (`#D4E6FA`)<br>• `"periwinkle"` (`#DFE3F8`)<br>• `"lavender"` (`#E7DFF7`)<br>• `"blush_pink"` (`#FCDFE4`)<br>• `"mint_green"` (`#D8F1E5`)<br>• `"warm_amber"` (`#FCECD7`)<br>• `"soft_grey"` (`#E8EAED`) |
| `doodle_accent` | `String` | No | `"auto"` | Hand-drawn doodle accent:<br>• `"auto"` (👑 Crown for men, ♡ Heart for women)<br>• `"crown"` (Floating King/Queen crown)<br>• `"heart"` (Floating outlined heart)<br>• `"sparks"` (Radiating action sparks & dashes)<br>• `"stars"` (Twinkling star doodles)<br>• `"none"` (Clean circular halo without doodles) |
| `quality` | `String` | No | `"standard"` | `"standard"` (14-step high-speed) or `"high"` (16-step ultra-definition). |
| `response_format` | `String` | No | `"json"` | `"json"` (returns full metadata and URL) or `"image_direct"` (directly streams binary PNG). |
| `include_base64` | `Boolean` | No | `false` | If `true`, includes Base64 data URI string (`data:image/png;base64,...`) in JSON response. |
| `custom_prompt` | `String` | No | `null` | Optional creative directions (e.g. `"gold wireframe glasses, brighter smile"`). |

---

## 3. JSON Response Schema

### HTTP 200 OK
```json
{
  "status": "success",
  "success": true,
  "job_id": "avatar_a8ac2e6f75c4",
  "avatar_url": "http://your-python-server:8000/outputs/avatar_a8ac2e6f75c4_avatar.png",
  "output_png_url": "http://your-python-server:8000/outputs/avatar_a8ac2e6f75c4_avatar.png",
  "avatar_base64": null,
  "style": "vector_art",
  "halo_color": "soft_blue",
  "doodle_accent": "crown",
  "identity_features": {
    "gender": "male",
    "expression": "confident subtle smile",
    "hair_details": "thick voluminous dark styled textured hair with fringe",
    "eyewear": "thin rectangular wireframe eyeglasses",
    "facial_hair": "neat dark moustache and light chin stubble",
    "upper_garment": "black tailored suit blazer, white collared shirt, vibrant red necktie, and pocket square",
    "distinctive_features": null
  },
  "resolution": "1024x1024",
  "processing_time_sec": 11.4,
  "execution_time_ms": 11400.0,
  "engine": "Qwen/Qwen-Image-2.1-Turbo",
  "gpu_name": "NVIDIA A40"
}
```

---

## 4. Node.js Code Examples

### Option A: Using `axios` + `form-data` (Recommended for Node.js 14+)

Install dependencies:
```bash
npm install axios form-data
```

Implementation (`avatarService.js`):
```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const PYTHON_API_BASE = process.env.PYTHON_API_URL || 'http://127.0.0.1:8000';

/**
 * Generate a stylized 2D avatar profile picture.
 *
 * @param {string|Buffer} imageInput - File path or image Buffer.
 * @param {Object} options - Customization options.
 * @param {string} [options.style='vector_art'] - 'vector_art' | 'digital_illustration' | 'anime_portrait'
 * @param {string} [options.haloColor='auto'] - 'auto' | 'soft_blue' | 'periwinkle' | 'lavender' | 'blush_pink' | 'mint_green'
 * @param {string} [options.doodleAccent='auto'] - 'auto' | 'crown' | 'heart' | 'sparks' | 'none'
 * @param {string} [options.customPrompt] - Optional extra note
 * @returns {Promise<Object>} API Response JSON
 */
async function generateStylizedAvatar(imageInput, options = {}) {
  const form = new FormData();

  // Handle either file path or buffer
  if (typeof imageInput === 'string') {
    form.append('file', fs.createReadStream(imageInput));
  } else if (Buffer.isBuffer(imageInput)) {
    form.append('file', imageInput, { filename: 'user_photo.jpg', contentType: 'image/jpeg' });
  } else {
    throw new Error('imageInput must be a file path string or Buffer');
  }

  // Append optional parameters
  form.append('style', options.style || 'vector_art');
  form.append('halo_color', options.haloColor || 'auto');
  form.append('doodle_accent', options.doodleAccent || 'auto');
  form.append('response_format', 'json');

  if (options.customPrompt) {
    form.append('custom_prompt', options.customPrompt);
  }

  const response = await axios.post(`${PYTHON_API_BASE}/api/generate-avatar`, form, {
    headers: {
      ...form.getHeaders(),
    },
    timeout: 180000, // 3 minutes timeout for cold starts / GPU queue
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
  });

  return response.data;
}

// Example Usage:
async function main() {
  try {
    const result = await generateStylizedAvatar('./test_selfie.jpg', {
      style: 'vector_art',
      haloColor: 'soft_blue',
      doodleAccent: 'crown',
    });
    console.log('Avatar generated successfully!');
    console.log('Avatar URL:', result.avatar_url);
    console.log('Detected Features:', result.identity_features);
  } catch (error) {
    console.error('Avatar generation error:', error.response?.data || error.message);
  }
}

module.exports = { generateStylizedAvatar };
```

---

### Option B: Native `fetch` + `FormData` (Node.js 18+ Built-in)

Zero external dependencies required on Node 18+:

```javascript
import fs from 'node:fs/promises';

const PYTHON_API_BASE = process.env.PYTHON_API_URL || 'http://127.0.0.1:8000';

export async function createAvatar(filePath, style = 'vector_art') {
  const fileBytes = await fs.readFile(filePath);
  const blob = new Blob([fileBytes], { type: 'image/jpeg' });

  const formData = new FormData();
  formData.append('file', blob, 'avatar.jpg');
  formData.append('style', style);
  formData.append('halo_color', 'auto');
  formData.append('doodle_accent', 'auto');

  const res = await fetch(`${PYTHON_API_BASE}/api/generate-avatar`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return await res.json();
}
```

---

### Option C: Express.js Route (User Upload Forwarding with `multer`)

When a client uploads a selfie to your Node.js backend, forward it directly to Python AI service:

```javascript
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });
const router = express.Router();
const PYTHON_API_BASE = process.env.PYTHON_API_URL || 'http://127.0.0.1:8000';

router.post('/user-avatar', upload.single('photo'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Please upload a photo' });
    }

    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname || 'photo.jpg',
      contentType: req.file.mimetype,
    });
    form.append('style', req.body.style || 'vector_art');
    form.append('halo_color', req.body.halo_color || 'auto');
    form.append('doodle_accent', req.body.doodle_accent || 'auto');

    const pyRes = await axios.post(`${PYTHON_API_BASE}/api/generate-avatar`, form, {
      headers: form.getHeaders(),
      timeout: 180000,
    });

    // Return the generated avatar metadata to client
    return res.json({
      success: true,
      avatar_url: pyRes.data.avatar_url,
      features: pyRes.data.identity_features,
      style: pyRes.data.style,
      halo: pyRes.data.halo_color,
      doodle: pyRes.data.doodle_accent,
    });
  } catch (err) {
    const status = err.response?.status || 500;
    const detail = err.response?.data?.detail || err.message;
    return res.status(status).json({ success: false, error: detail });
  }
});

module.exports = router;
```

---

## 5. HTTP Status Codes & Error Handling

| Status Code | Meaning | Handling in Node.js |
| :---: | :--- | :--- |
| **`200 OK`** | Success. | Return `res.data.avatar_url` to your frontend. |
| **`400 Bad Request`** | Empty file uploaded or invalid form-data. | Prompt user to upload a valid photo. |
| **`413 Payload Too Large`** | Image exceeds 15 MB. | Ask user to compress or upload smaller photo. |
| **`415 Unsupported Media`** | Non-image file uploaded (must be JPG/PNG/WEBP). | Reject file type on frontend/multer. |
| **`503 Service Unavailable`** | Remote GPU Worker offline or unreachable. | Check RunPod worker health / retry request. |
| **`500 Internal Error`** | Unhandled pipeline exception. | Log error details and notify admin. |

---

## 6. Pro-Tips for Production
1. **Timeout Setting:** Set axios/fetch timeout to **at least 180 seconds (`180000ms`)**. In case the GPU worker is cold-starting or queuing, it takes 15–30s.
2. **Direct Image Streaming:** If you want your Node backend to serve the image binary directly (without URL redirection), pass `response_format: 'image_direct'` and set `responseType: 'stream'` in Axios.
3. **URL Resolution:** `avatar_url` will contain the full public or internal address of your Python server (e.g. `http://ai-service:8000/outputs/...`). If you use an Nginx reverse proxy or CDN, pass `X-Forwarded-Proto` and `Host` headers so URLs are generated with your public domain name.
