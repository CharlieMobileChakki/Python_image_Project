"""ChatGPT Vision prompt generator for Flow B (Model-Worn Garment Extraction).

Analyzes a model-worn photograph via OpenAI Vision (gpt-4o-mini / gpt-4o) and builds
forensic-grade, textile-accurate descriptions combined with fixed 3D ghost-mannequin studio
rules for the Qwen-Image-2.1-Turbo GPU worker.
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

# Fixed backend studio rules (empty 3D invisible ghost-mannequin product shot, Myntra/Meesho style)
COMMON_STUDIO_RULES = (
    "Empty hollow 3D inner neckline aperture, empty open sleeve cuffs, and empty open bottom hems showing the hollow interior with zero human body inside. "
    "Seamlessly reconstruct the continuous repeating fabric pattern and textile weave across the chest, stomach, and sleeves where hands or arms previously overlapped. "
    "Strictly preserve the exact original colors, high-contrast repeating print motifs, embroidery stitches, textile texture, neckline shape, borders, "
    "and authentic garment silhouette from the reference photograph — do NOT smooth out, blur, or wash out intricate prints or motifs. "
    "Centered upright 3D ghost-mannequin studio product photograph, crisp professional e-commerce studio lighting, "
    "pure solid white background (#FFFFFF), completely free of any person, model, face, hair, neck, skin, arms, hands, legs, or shadows."
)

SYSTEM_VISION_PROMPT = """You are a master e-commerce fashion catalog director and senior textile analyst creating image-editing prompts for high-end 3D ghost-mannequin catalog photography (Myntra, Ajio, Meesho, Zara, Amazon Fashion).

Your job is to analyze the clothing worn in the reference photograph and produce a forensic-grade, textile-accurate description of each garment so that an AI diffusion image editor can reconstruct the EXACT standalone 3D garment without any human body parts.

CRITICAL TEXTILE & PATTERN RULES:
1. Fabric Surface & Motif Breakdown (DO NOT GENERALIZE):
   - Never say just "printed" or "floral". You must dissect the exact motifs:
     * Base fabric color and texture (e.g., warm ivory/cream raw slub cotton, textured linen weave, pure silk, georgette).
     * Exact motif type: e.g., miniature traditional Indian floral booti/buta, bandhani dots, geometric jaal, block print, paisley, ikat, or embroidery.
     * Specific motif colors: e.g., miniature brick-red, terracotta, and dark chocolate-brown motifs.
     * Pattern layout & density: e.g., dense repeating miniature bootis arranged in neat parallel vertical linear rows spaced ~1 to 1.5 cm apart across the entire body and sleeves.
     * If the garment is PLAIN SOLID with no pattern, explicitly state: "pure solid smooth fabric with no print, no pattern, and no texture".

2. Silhouette & Construction Anatomy:
   - Neckline / Collar: Exact shape (e.g., Mandarin stand band collar / Nehru collar, V-neck slit, round crew neck, polo collar, sweetheart neckline).
   - Placket & Fastenings: e.g., front chest button placket with small round matching buttons, contrast stitching, or hook closures.
   - Sleeves: Full-length, three-quarter, or half; specify if the pattern continues seamlessly to the open cuffs.
   - Slits & Hemline: Straight knee-length hem, side seam slits, flared A-line hem, or curved shirt hem.

3. Outfit Mode & Separates:
   - For matching ethnic kurta sets, dresses, gowns, or sarees, prioritize the primary hero garment (the kurta/dress).
   - If contrasting separates are present (e.g., printed kurta + plain white trousers/pajama), analyze each separate item accurately.

4. Allowed category values ONLY: "upper_clothes", "pants", "dress", "skirt", "saree", "lehenga".

Return strictly valid JSON format:
{
  "garments": [
    {
      "category": "upper_clothes",
      "recognized_label": "Cream Miniature Booti Print Ethnic Kurta",
      "garment_description": "Men's traditional knee-length straight-cut ethnic kurta crafted from warm ivory-cream textured cotton slub fabric, adorned with all-over dense miniature terracotta-red and chocolate-brown ethnic floral booti motifs arranged in repeating vertical linear stripes. Features a stiff mandarin band collar, front chest placket with 4 small buttons, long straight sleeves with identical all-over booti print, and clean straight side seam slits."
    }
  ]
}"""


def _encode_image_for_vision(image_rgb: Image.Image, max_edge: int = 1536) -> str:
    """Resize and base64-encode PIL RGB image as JPEG for high-detail Vision API calls."""
    img = image_rgb.convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_final_prompt(garment_description: str, category: str) -> str:
    """Put the empty 3D ghost-mannequin person-removal instruction FIRST so Qwen-Image-2.1 obeys 100%."""
    return (
        "Generate an empty 3D ghost-mannequin e-commerce studio product photograph showing ONLY the standalone "
        "garment floating upright in natural 3D volume on a pure solid white background (#FFFFFF), with completely NO person, NO model, "
        "NO head, NO face, NO hair, NO neck, NO skin, NO hands, NO arms, NO legs, NO feet, and NO jewelry anywhere in the image. "
        f"Exact standalone garment and intricate textile pattern to isolate and faithfully reconstruct from the reference photo: {garment_description.strip()} (category: {category}). "
        f"{COMMON_STUDIO_RULES}"
    )


def _fallback_garment_plan(target_garment: str = "all") -> list[dict[str, str]]:
    """Deterministic fallback prompt if OpenAI API key is absent or unavailable."""
    t = (target_garment or "all").strip().lower()
    mapping = {
        "upper_clothes": ("upper_clothes", "Top / Shirt / Upper Garment", "the upper-body garment (top, shirt, kurta, jacket, or blouse) with its exact original colors, fabric weave, and print motifs"),
        "pants": ("pants", "Pants / Jeans / Lower Garment", "the lower-body trousers, pants, or jeans with original color and fit"),
        "dress": ("dress", "Dress / Kurta / Full Outfit", "the full-length dress, kurta, or outfit preserving exact original prints, embroidery, and neckline"),
        "skirt": ("skirt", "Skirt", "the skirt with original pattern and borders"),
        "saree": ("saree", "Saree", "the complete draped saree with borders and pallu print"),
        "lehenga": ("lehenga", "Lehenga / Ethnic Set", "the complete lehenga ensemble with intricate embroidery and borders"),
    }
    cat, label, desc = mapping.get(
        t,
        ("dress", "Garment Cutout", "the primary garment worn by the person preserving exact original colors, textile weave, and print motifs"),
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
        f"Examine the fabric at high resolution: identify the exact base color, exact print/booti motif colors, "
        f"pattern density, vertical or horizontal repetition, collar shape, buttons/placket, and sleeve style so the 3D ghost mannequin replicates the exact authentic design."
    )

    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": 0.1,
        "max_tokens": 1000,
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
        with httpx.Client(timeout=35.0) as client:
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
