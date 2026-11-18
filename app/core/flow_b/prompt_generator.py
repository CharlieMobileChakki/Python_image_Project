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
    "Transform this into a sharp, ultra-clear, high-resolution ghost-mannequin e-commerce studio product photograph "
    "of ONLY this exact garment centered on a pure solid white background (#FFFFFF). "
    "Remove the woman/person, face, head, hair, skin, hands, arms, feet, necklace, jewelry, and outdoor/room background completely. "
    "Keep the EXACT same color shade, smooth fabric finish, neckline shape, sleeve trims/borders, silhouette, and proportions as worn in the input photo — "
    "do NOT invent new patterns, speckles, or extra textures that are not in the original photo. "
    "Crisp studio lighting, clean sharp fabric edges, photorealistic retail catalog quality."
)

SYSTEM_VISION_PROMPT = """You are a senior e-commerce fashion catalog analyst writing concise image-editing instructions.
Inspect the photograph of the person wearing clothing and describe the main wearable garment(s) with 100% factual accuracy.

Rules:
1. If the person wears a matching ethnic suit set (kurta + matching salwar/pants + matching dupatta), dress, gown, jumpsuit, or saree, return 1 combined garment item so the complete matching outfit stays faithful.
2. Only split into 2 items if the person clearly wears contrasting western separates (e.g., white shirt + blue denim jeans). Never list shoes, bags, necklaces, jewelry, or hats.
3. For each garment, accurately specify:
   - Exact color shade (e.g., solid deep magenta-purple, navy blue)
   - Fabric surface: explicitly say "smooth plain solid fabric" if it has NO print, OR describe the exact print/embroidery if present
   - Exact neckline/collar shape (e.g., V-neck slit, round neck, mandarin collar)
   - Exact sleeves and any borders/trim/stripes (e.g., three-quarter sleeves with thin gold zari stripe border, matching draped dupatta with gold border)
   - Exact lower portion if included (e.g., matching straight salwar/pants)
4. Allowed category values ONLY: "upper_clothes", "pants", "dress", "skirt", "saree", "lehenga".

Return strictly valid JSON:
{
  "garments": [
    {
      "category": "dress",
      "recognized_label": "Magenta Kurta Suit Set with Gold Border",
      "garment_description": "Solid smooth deep magenta-purple straight-cut ethnic kurta with slit V-neckline, three-quarter sleeves featuring a thin gold zari stripe border, matching magenta dupatta with gold striped border and tassels, and matching magenta salwar pants."
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

