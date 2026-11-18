"""ChatGPT Vision prompt generator for Flow B (Model-Worn Garment Extraction).

Analyzes a model-worn photograph via OpenAI Vision (gpt-4o-mini / gpt-4o) and builds
detailed, garment-specific instructions combined with fixed studio background-removal rules
for the Qwen-Image-Edit GPU worker.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

# Fixed backend studio rules appended to every Flow B garment generation prompt
COMMON_STUDIO_RULES = (
    "Create a clean, photorealistic, centered flat-lay ghost-mannequin e-commerce product photograph "
    "of ONLY this exact garment from the reference image on a pure solid white background (#FFFFFF). "
    "Completely remove the person, human body, skin, face, neck, hands, arms, hair, jewelry, bags, "
    "hangers, and original background. "
    "Naturally reconstruct any fabric area, folds, pleats, sleeves, or waistline that were covered "
    "by hands, arms, or hair in the original photo. "
    "Strictly preserve the exact original color, fabric texture, weave, prints, embroidery, borders, "
    "buttons, neckline, collar, and silhouette from the reference photo. "
    "Studio catalog lighting, crisp clean edges, no extra objects, pure solid white background."
)

SYSTEM_VISION_PROMPT = """You are a senior e-commerce fashion catalog analyst.
Inspect the provided photograph of a person wearing clothing and identify the main wearable garment(s).

Rules:
1. If the person wears a single full outfit (dress, gown, jumpsuit, saree, anarkali, kurta-set shown as one piece), return 1 garment item.
2. If the person wears distinct separate garments (e.g., shirt/top + jeans/pants/skirt, or jacket + pants), list each main wearable garment separately (maximum 3 items). Do NOT list shoes, bags, glasses, hats, or jewelry.
3. For each garment, write a precise visual description covering: exact garment type, exact color/shade, fabric material/texture, neckline/collar, sleeve length/style, prints/patterns/embroidery/borders, buttons/zippers, and fit/silhouette.
4. Allowed category values ONLY: "upper_clothes", "pants", "dress", "skirt", "saree", "lehenga".

Return strictly valid JSON in this exact format:
{
  "garments": [
    {
      "category": "upper_clothes",
      "recognized_label": "Navy Blue Cotton Kurta",
      "garment_description": "Navy blue straight-cut cotton kurta with mandarin collar, three-quarter sleeves, white floral thread embroidery along the placket and hem, side slits."
    }
  ]
}"""


def _encode_image_for_vision(image_rgb: Image.Image, max_edge: int = 1024) -> str:
    """Resize and base64-encode PIL RGB image as JPEG for fast Vision API calls."""
    img = image_rgb.convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_final_prompt(garment_description: str, category: str) -> str:
    """Combine garment-specific description with common studio rules."""
    return (
        f"Extract ONLY the following garment worn by the person in the reference image: "
        f"{garment_description.strip()} (category: {category}). "
        f"{COMMON_STUDIO_RULES}"
    )


def _fallback_garment_plan(target_garment: str = "all") -> list[dict[str, str]]:
    """Deterministic fallback prompt if OpenAI API key is absent or unavailable."""
    t = (target_garment or "all").strip().lower()
    mapping = {
        "upper_clothes": ("upper_clothes", "Top / Shirt / Upper Garment", "the upper-body garment (top, shirt, kurta, jacket, or blouse)"),
        "pants": ("pants", "Pants / Jeans / Lower Garment", "the lower-body trousers, pants, or jeans"),
        "dress": ("dress", "Dress / Gown / Full Outfit", "the full-length dress, gown, or outfit"),
        "skirt": ("skirt", "Skirt", "the skirt"),
        "saree": ("saree", "Saree", "the complete draped saree"),
        "lehenga": ("lehenga", "Lehenga / Ethnic Set", "the complete lehenga ensemble"),
    }
    cat, label, desc = mapping.get(
        t,
        ("dress", "Garment Cutout", "the primary garment worn by the person"),
    )
    return [
        {
            "category": cat,
            "recognized_label": label,
            "garment_description": desc,
            "prompt": _build_final_prompt(desc, cat),
            "prompt_source": "default_fallback",
        }
    ]


def analyze_garment_and_build_prompts(
    image_rgb: Image.Image,
    target_garment: str = "all",
    custom_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Analyze a model-worn photo using ChatGPT Vision and build Qwen generation prompts.

    Args:
        image_rgb: Original model-worn PIL RGB image.
        target_garment: Optional category filter ('all', 'upper_clothes', 'pants', 'dress', 'skirt', 'saree', 'lehenga').
        custom_prompt: Optional caller-supplied prompt override.

    Returns:
        List of garment prompt dicts: [{category, recognized_label, garment_description, prompt, prompt_source}].
    """
    if custom_prompt and custom_prompt.strip():
        cat = target_garment if target_garment and target_garment != "all" else "dress"
        return [
            {
                "category": cat,
                "recognized_label": "Custom Garment Extraction",
                "garment_description": custom_prompt.strip(),
                "prompt": f"{custom_prompt.strip()} {COMMON_STUDIO_RULES}",
                "prompt_source": "custom_prompt",
            }
        ]

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; using default garment prompt fallback.")
        return _fallback_garment_plan(target_garment)

    model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    b64_jpeg = _encode_image_for_vision(image_rgb)

    user_instruction = (
        f"Analyze the clothing worn by the person in this image. "
        f"Target garment filter: '{target_garment or 'all'}'. "
        f"Return only the requested garment(s) as JSON."
    )

    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": 0.2,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_VISION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_jpeg}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        raw_content = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)
        items = parsed.get("garments", [])
        if not isinstance(items, list) or not items:
            logger.warning("ChatGPT Vision returned empty garments list; falling back.")
            return _fallback_garment_plan(target_garment)

        target_norm = (target_garment or "all").strip().lower()
        results: list[dict[str, str]] = []
        for item in items[:3]:
            cat = str(item.get("category", "dress")).strip().lower()
            label = str(item.get("recognized_label", "Garment Cutout")).strip()
            desc = str(item.get("garment_description", "")).strip()
            if not desc:
                continue
            if target_norm != "all" and cat != target_norm:
                continue
            results.append(
                {
                    "category": cat,
                    "recognized_label": label,
                    "garment_description": desc,
                    "prompt": _build_final_prompt(desc, cat),
                    "prompt_source": f"openai:{model_name}",
                }
            )

        # If target filter eliminated all items, keep the first detected garment rather than failing
        if not results and items:
            first = items[0]
            cat = str(first.get("category", "dress")).strip().lower()
            label = str(first.get("recognized_label", "Garment Cutout")).strip()
            desc = str(first.get("garment_description", "the primary garment")).strip()
            results.append(
                {
                    "category": cat,
                    "recognized_label": label,
                    "garment_description": desc,
                    "prompt": _build_final_prompt(desc, cat),
                    "prompt_source": f"openai:{model_name}",
                }
            )

        return results or _fallback_garment_plan(target_garment)

    except Exception as exc:
        logger.warning("ChatGPT Vision prompt generation failed (%s); using fallback prompt.", exc)
        return _fallback_garment_plan(target_garment)

