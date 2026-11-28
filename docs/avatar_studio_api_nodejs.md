# AI Stylized Avatar — Node.js Quick Integration Guide

## 1. Single API Endpoint (No Aliases)

- **URL:** `POST http://<SERVER_IP>:8000/api/generate-avatar`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

---

## 2. Request Fields (Kya Bhejna Hai?)

### ✅ Only 1 Mandatory Field:
| Field Name | Type | Description |
| :--- | :--- | :--- |
| **`file`** | `File / Buffer` | **Required.** User ki photo (`.jpg`, `.jpeg`, `.png`, `.webp`). Selfie, portrait ya normal photo. |

### ⚙️ Optional Fields (Agar customise karna ho tabhi bhejo, varna blank chhod do):
| Field Name | Default | Options |
| :--- | :--- | :--- |
| `style` | `"vector_art"` | `"vector_art"` (Default 2D Vector) \| `"digital_illustration"` \| `"anime_portrait"` |
| `halo_color` | `"auto"` | `"auto"` (AI best match) \| `"soft_blue"` \| `"lavender"` \| `"blush_pink"` \| `"mint_green"` |
| `doodle_accent`| `"auto"` | `"auto"` (👑 Men / ♡ Women) \| `"crown"` \| `"heart"` \| `"sparks"` \| `"none"` |

---

## 3. Response Structure (Output me kya milega?)

Node.js ko standard JSON milega:
```json
{
  "status": "success",
  "success": true,
  "job_id": "avatar_a8ac2e6f75c4",
  "avatar_url": "http://127.0.0.1:8000/outputs/avatar_a8ac2e6f75c4_avatar.png",
  "style": "vector_art",
  "halo_color": "soft_blue",
  "doodle_accent": "crown",
  "resolution": "1024x1024",
  "processing_time_sec": 11.4
}
```

> **Main cheez:** `avatar_url` — is URL ko direct frontend/app me `<img src="...">` me show kar do.

---

## 4. Super Simple Node.js Code (Copy & Paste)

### Option A: Pure Axios (Node.js script / service)

```bash
npm install axios form-data
```

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function createAvatar(photoPath) {
  const form = new FormData();
  
  // Sirf 1 mandatory field bhejna hai: 'file'
  form.append('file', fs.createReadStream(photoPath));

  const response = await axios.post('http://127.0.0.1:8000/api/generate-avatar', form, {
    headers: form.getHeaders(),
    timeout: 180000, // 3 minutes timeout (GPU queue / generation)
  });

  // Direct image URL
  console.log('Generated Avatar URL:', response.data.avatar_url);
  return response.data;
}

// Run:
createAvatar('./user_photo.jpg');
```

---

### Option B: Express.js Controller (User Upload -> Python AI Service -> Response)

```javascript
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

router.post('/generate-avatar', upload.single('photo'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Please upload a photo' });
    }

    // Python AI service ko form-data forward karein
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname || 'photo.jpg',
      contentType: req.file.mimetype,
    });

    const aiResponse = await axios.post('http://127.0.0.1:8000/api/generate-avatar', form, {
      headers: form.getHeaders(),
      timeout: 180000,
    });

    // Frontend ko direct result return karein
    return res.json({
      success: true,
      avatar_url: aiResponse.data.avatar_url,
    });

  } catch (error) {
    const errorMsg = error.response?.data?.detail || error.message;
    return res.status(500).json({ success: false, error: errorMsg });
  }
});

module.exports = router;
```
