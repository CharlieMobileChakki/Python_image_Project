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
     * Base fabric color and texture (e.g., crisp white cotton, textured linen weave, pure raw slub silk, georgette, denim wash).
     * Exact motif type: e.g., botanical leaves and palm fronds, miniature traditional Indian floral booti/buta, bandhani dots, geometric jaal, block print, paisley, ikat, or embroidery stitches.
     * Specific motif colors: e.g., soft sky-blue, royal indigo, terracotta-red, or chocolate-brown motifs.
     * Pattern layout & density: e.g., dense repeating botanical foliage evenly distributed across the entire body and sleeves.
     * If the garment is PLAIN SOLID with no pattern (like denim jeans or plain t-shirt), explicitly state: "pure solid smooth fabric with no print, no pattern, and no motif".

2. Silhouette & Construction Anatomy:
   - Neckline / Collar: Exact shape (e.g., standard turn-down shirt point collar, Mandarin band collar, V-neck slit, round crew neck, polo collar, sweetheart neckline).
   - Placket & Fastenings: e.g., front chest button placket with small round matching buttons, contrast stitching, or hook closures.
   - Sleeves: Full-length, three-quarter, or half/short sleeves; specify if the pattern continues seamlessly to the open cuffs.
   - Slits & Hemline: Straight knee-length hem, side seam slits, flared A-line hem, curved casual shirt hem, or straight-leg ankle cuffs.

3. Outfit Mode & Separates (STRICT MULTI-GARMENT RULE):
   - IF THE PERSON IS WEARING CONTRASTING SEPARATES (e.g. Shirt/T-shirt/Top AND Jeans/Pants/Trousers/Shorts/Skirt):
     YOU MUST ALWAYS OUTPUT SEPARATE ENTRIES IN THE "garments" LIST:
     * One entry for the upper garment (category: "upper_clothes"). NEVER describe or mention the pants/jeans in this entry!
     * One entry for the lower garment (category: "pants" or "skirt"). NEVER describe or mention the shirt/top in this entry!
     Separates of different colors or types (such as a printed shirt paired with dark denim jeans) MUST NEVER be combined into a single entry.
   - IF THE PERSON IS WEARING A SINGLE-PIECE ENSEMBLE (e.g. full-length dress, gown, jumpsuit, romper):
     Output a single entry with category "dress".
   - IF THE PERSON IS WEARING A SAREE OR LEHENGA:
     Output a single entry with category "saree" or "lehenga".
   - IF THE PERSON IS WEARING AN ETHNIC SUIT / KURTA SET:
     If the kurta is paired with contrasting pyjama/pants, split into "upper_clothes" (the kurta) and "pants" (the pyjama/salwar). If matching co-ord, provide the primary kurta as "upper_clothes".

4. Allowed category values ONLY: "upper_clothes", "pants", "dress", "skirt", "saree", "lehenga".

Return strictly valid JSON format:
{
  "garments": [
    {
      "category": "upper_clothes",
      "recognized_label": "Light Blue Leaf Print Short-Sleeve Shirt",
      "garment_description": "Men's casual short-sleeve button-down shirt crafted from crisp white cotton fabric, adorned with all-over botanical leaf and palm frond motifs printed in soft sky-blue. Features a standard turn-down collar, front chest placket with small white buttons, short hemmed sleeves, and a curved casual shirt hem."
    },
    {
      "category": "pants",
      "recognized_label": "Dark Indigo Denim Jeans",
      "garment_description": "Men's classic five-pocket straight-fit denim jeans in a deep indigo-blue wash with subtle natural fading, contrast copper-gold stitching, standard waistband with belt loops, front metal button and zip fly, and clean straight leg hems."
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
    cat = (category or "").strip().lower()
    if cat == "upper_clothes":
        isolation_rule = (
            "STRICT CATEGORY ISOLATION: Generate ONLY the upper-body garment (shirt / top / t-shirt / kurta / jacket). "
            "COMPLETELY EXCLUDE, REMOVE, AND DO NOT GENERATE any pants, jeans, trousers, shorts, skirts, belt, waistline of pants, legs, or bottom garments! "
            "The 3D invisible ghost-mannequin terminates cleanly at the bottom hem of the shirt/top with an open hollow hem aperture, "
            "and the bottom half of the image MUST BE completely empty, clean, pure solid white (#FFFFFF)."
        )
    elif cat in ("pants", "skirt"):
        isolation_rule = (
            "STRICT CATEGORY ISOLATION: Generate ONLY the lower-body garment (pants / jeans / trousers / shorts / skirt). "
            "COMPLETELY EXCLUDE, REMOVE, AND DO NOT GENERATE any upper garments, shirt, t-shirt, top, jacket, collar, sleeves, chest, or torso! "
            "The 3D invisible ghost-mannequin begins cleanly at the waistband of the pants/jeans with an open hollow waistband aperture, "
            "and the upper half of the image MUST BE completely empty, clean, pure solid white (#FFFFFF)."
        )
    elif cat in ("dress", "saree", "lehenga"):
        isolation_rule = (
            "STRICT CATEGORY ISOLATION: Generate the full standalone 3D ghost-mannequin outfit with no human body."
        )
    else:
        isolation_rule = ""

    return (
        "Generate an empty 3D ghost-mannequin e-commerce studio product photograph showing ONLY the standalone "
        "garment floating upright in natural 3D volume on a pure solid white background (#FFFFFF), with completely NO person, NO model, "
        "NO head, NO face, NO hair, NO neck, NO skin, NO hands, NO arms, NO legs, NO feet, and NO jewelry anywhere in the image. "
        f"{isolation_rule} "
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

    target_norm = (target_garment or "all").strip().lower()
    if target_norm == "upper_clothes":
        user_instruction = (
            "Analyze ONLY the UPPER-BODY garment (shirt, t-shirt, top, kurta, jacket) worn by the person in this image. "
            "Completely IGNORE and exclude the pants, jeans, trousers, or bottom wear. "
            "Examine the fabric at high resolution: identify the exact base color, print/booti motifs, "
            "pattern density, collar shape, buttons/placket, and sleeve style so the 3D ghost mannequin replicates the exact authentic design."
        )
    elif target_norm in ("pants", "skirt"):
        user_instruction = (
            "Analyze ONLY the LOWER-BODY garment (pants, jeans, trousers, shorts, skirt) worn by the person in this image. "
            "Completely IGNORE and exclude the upper-body shirt, top, or kurta. "
            "Examine the fabric at high resolution: identify the exact denim/fabric wash, color, fit, pockets, stitching, and hems."
        )
    elif target_norm in ("dress", "saree", "lehenga"):
        user_instruction = (
            f"Analyze the full {target_norm} worn by the person in this image. "
            "Examine the fabric at high resolution: identify the exact colors, intricate borders, prints, embroidery, and drape."
        )
    else:  # "all"
        user_instruction = (
            "Analyze the clothing worn by the person in this image. "
            "CRITICAL: If the person is wearing contrasting separates (such as shirt/t-shirt/kurta + jeans/pants/trousers), "
            "you MUST return TWO separate items in the 'garments' list: one for 'upper_clothes' and one for 'pants'. "
            "Never merge contrasting tops and bottoms into a single item. "
            "If the person is wearing a single-piece dress, saree, or lehenga, return that single item. "
            "For each item, examine fabric at high resolution: base color, motifs, collar, buttons, and cut."
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
