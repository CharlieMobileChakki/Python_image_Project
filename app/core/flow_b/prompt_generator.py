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
    "Strictly preserve the exact original dye colors, saturation levels, darkness, fabric weave, embroidery stitches, textile texture, neckline shape, borders, "
    "and authentic garment silhouette from the reference photograph — do NOT oversaturate, do NOT brighten dark/muted tones, and do NOT alter the authentic color shade. "
    "Centered upright, perfectly straight 0-degree direct frontal shot facing the camera straight-on, perfectly symmetrical shoulders and sleeves, strictly level horizontal hemline, "
    "pure solid white background (#FFFFFF), completely free of any person, model, face, hair, neck, skin, arms, hands, legs, bed, mattress, bedsheet, blanket, carpet, floor, furniture, or shadows."
)

SYSTEM_VISION_PROMPT = """You are a master e-commerce fashion catalog director and senior textile analyst creating image-editing prompts for high-end 3D ghost-mannequin catalog photography (Myntra, Ajio, Meesho, Zara, Amazon Fashion).

Your job is to analyze the clothing in the reference photograph (whether worn by a person OR laid flat on a bed, table, floor, or hanger) and produce a forensic-grade, textile-accurate description of each REAL stitched garment so that an AI diffusion image editor can reconstruct the EXACT standalone 3D garment without any human body parts and without any background surfaces.

CRITICAL RULES:

1. BACKGROUND & SURFACE DISCRIMINATION (BEDSHEETS, FLOORS, RUGS, FURNITURE):
   - The photo may show clothing laid flat on a bed, mattress, floor, table, or sofa.
   - STRICT RULE: The bedsheet, bed cover, blanket, quilt, floor rug, carpet, or wooden bed frame is 100% BACKGROUND FURNITURE.
   - NEVER, UNDER ANY CIRCUMSTANCES, MISTAKE A BEDSHEET OR BED COVER PATTERN FOR A GARMENT!
     * Even if the bedsheet has intricate geometric patterns (e.g., honeycomb hexagons, mandalas, block prints, florals, or decorative borders), it is STRICTLY BACKGROUND.
     * DO NOT extract the bedsheet as a shirt, kurta, top, dupatta, or fabric!
     * DO NOT transfer the bedsheet's pattern, prints, or colors onto any garment description!
   - ONLY analyze genuine, stitched, wearable clothing items (e.g., an olive green kurta laid on the bed).
   - If only ONE garment is present in the photo (e.g., only a kurta lying on a bed), return ONLY THAT SINGLE GARMENT in the "garments" list. DO NOT hallucinate or invent a second garment out of the bedsheet!

2. Precise Color Tone, Shade & Fabric Fidelity (SAME-TO-SAME REPLICATION):
   - Never use broad or generic bright color names like just "green", "olive-green", or "blue".
   - You must forensic-analyze the exact real-world shade, darkness, undertone, and saturation level:
     * If the fabric has a dark, muted, dusty, desaturated, or earthy tone (e.g., dusty army-olive, muted slate-sage, dark khaki-green, charcoal navy, faded terracotta, dusty peach), explicitly specify the EXACT muted shade: "deep muted dusty army-olive with desaturated matte greyish undertones, exactly preserving the dark muted dye tone and saturation of the reference photo without any artificial brightening, neon tint, or oversaturation".
     * If the fabric has a specific weave (e.g., textured cotton slub, raw handloom khadi, matte linen, twill denim, crepe, georgette), describe the exact matte weave and thread texture.
     * If plain solid, explicitly state: "pure solid matte fabric with uniform dye, no prints, no patterns, and no shiny gloss".
   - DENIM JEANS, TROUSERS & BOTTOM WEAR (CRITICAL WASH & COLOR FIDELITY):
     * If the jeans/pants in the reference photograph are dark (e.g. solid raw dark indigo, deep midnight navy, inky blue, or black denim):
       You MUST explicitly describe the wash as: "solid uniform dark unwashed raw midnight indigo denim with zero stonewash fading, no bleached thigh highlights, no whiskering, and no distressed abrasion, strictly matching the deep dark saturated tone of the reference photograph".
     * NEVER label or describe dark indigo/navy jeans simply as "blue denim jeans" or "dark blue jeans", because AI diffusion models automatically add bright faded thighs, acid wash, and light blue highlights when given generic names!
     * Explicitly specify: "solid uniform dark dye from waistband to ankle cuffs, zero thigh fading, perfectly straight symmetrical legs".

3. STRICT DIRECT FRONTAL 0-DEGREE STRAIGHT ALIGNMENT (PERFECTLY "SIDHA", NO 3/4 ANGLE):
   - In e-commerce catalog standards (Myntra, Amazon, Ajio), product hero photos MUST be strictly front-facing and straight.
   - The garment must be oriented directly straight-on at a 0-degree frontal camera angle (perfectly centered, plumb, and symmetrical).
   - Both shoulders must be at the exact same horizontal height (level shoulders).
   - Both sleeves must hang naturally and symmetrically on both sides at equal angles.
   - The center front placket, buttons, zipper, or collar slit must run in a perfectly straight vertical line down the center.
   - STRICTLY FORBID: Any 3/4 perspective turn, side profile, diagonal tilt, rotated mannequin body, or dynamic turned pose. The output must be perfectly straight ("sidha") and squarely facing the viewer.

4. Silhouette & Construction Anatomy:
   - Neckline / Collar: Exact shape (e.g., Mandarin stand band collar / Nehru collar with small front slit, standard turn-down point collar, round crew neck, polo collar).
   - Placket & Fastenings: e.g., front chest button placket with small round matching buttons, contrast stitching, or hook closures.
   - Sleeves: Full-length, three-quarter, or half/short sleeves; specify if pattern/color continues seamlessly to open cuffs.
   - Slits & Hemline: Straight knee-length hem, side seam slits, flared A-line hem, curved casual shirt hem, or straight leg cuffs.

5. Outfit Mode & Separates (STRICT REALITY RULE):
   - IF AND ONLY IF the person is actually wearing contrasting separates (e.g. shirt + jeans) OR two distinct garments are clearly laid out:
     Output separate entries for the upper garment ("upper_clothes") and lower garment ("pants").
   - IF ONLY ONE GARMENT IS VISIBLE OR LAID OUT (e.g., a single kurta, a single dress, a single shirt):
     Output ONLY that one garment! Do NOT invent a second garment.
   - IF THE OUTFIT IS A ONE-PIECE ENSEMBLE (dress, gown, jumpsuit, saree, lehenga):
     Output a single entry with category "dress", "saree", or "lehenga".

6. Allowed category values ONLY: "upper_clothes", "pants", "dress", "skirt", "saree", "lehenga".

Return strictly valid JSON format:
{
  "garments": [
    {
      "category": "upper_clothes",
      "recognized_label": "Light Blue Floral Print Short Sleeve Shirt",
      "garment_description": "Men's casual relaxed-fit short-sleeve resort shirt crafted from breathable cotton fabric in soft pastel powder-blue featuring subtle tonal white botanical floral leafy branch prints. Straight-on 0-degree front view with symmetrical level shoulders and sleeves, standard point spread collar, front button placket with small buttons, and a straight clean hem."
    },
    {
      "category": "pants",
      "recognized_label": "Solid Midnight Dark Indigo Raw Denim Jeans",
      "garment_description": "Men's classic straight-leg denim jeans in uniform solid deep midnight dark raw indigo wash, strictly preserving the authentic dark near-navy saturation of the reference photo with zero stonewash, no bleached thigh fading, and no whiskering. Both legs hang straight down symmetrically side-by-side in a direct 0-degree frontal view, featuring a standard waistband with belt loops, button closure, and matching dark tonal stitching."
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
            "and the upper half of the image MUST BE completely empty, clean, pure solid white (#FFFFFF). "
            "STRICT DENIM & PANTS COLOR/WASH LOCK: If the reference jeans/pants are deep dark indigo, midnight navy, charcoal, or black, they MUST be rendered in that exact uniform dark shade throughout. "
            "STRICTLY FORBIDDEN: DO NOT add stonewash fading, DO NOT add bleached thigh patches, DO NOT add whiskers, DO NOT add distress marks or lighter blue highlights unless clearly visible in the reference photo. "
            "Both trouser legs MUST hang straight down symmetrically side-by-side in a 0-degree direct frontal shot, not angled or stepping forward."
        )
    elif cat in ("dress", "saree", "lehenga"):
        isolation_rule = (
            "STRICT CATEGORY ISOLATION: Generate the full standalone 3D ghost-mannequin outfit with no human body."
        )
    else:
        isolation_rule = ""

    surface_rejection = (
        "STRICT SURFACE REJECTION: If the reference photo shows clothing placed on a bedsheet, mattress, blanket, floor, rug, carpet, or table, "
        "COMPLETELY IGNORE the bedsheet and its background patterns! Do NOT apply any bedsheet print, geometric shapes, or bedding colors to the garment. "
        "Reconstruct ONLY the authentic stitched garment itself."
    )

    color_lock_rule = (
        "STRICT COLOR, SHADE & WASH LOCK: Faithfully preserve the EXACT SAME dye shade, hue, darkness, and muted desaturation level from the reference photo. "
        "DO NOT brighten dark or muted tones, DO NOT oversaturate, and DO NOT turn dusty/earthy/matte fabrics into vibrant or glossy colors. "
        "FOR JEANS & DENIM: Do NOT convert dark raw indigo/navy jeans into light stonewashed blue jeans! Preserve the deep dark midnight navy tone with uniform wash and zero artificial thigh fading. "
        "The generated 3D garment must perfectly replicate the authentic real-life color and textile weave/wash of the reference garment."
    )

    straight_alignment_rule = (
        "STRICT STRAIGHT FRONTAL ALIGNMENT (0-DEGREE DIRECT FRONT VIEW): The 3D ghost mannequin MUST face the camera directly straight-on at a strict 0-degree angle. "
        "Perfect symmetrical front-facing catalog view: both shoulders level at identical horizontal height, both sleeves hanging symmetrically at equal angles, "
        "and the center button placket / zipper / collar slit aligned perfectly vertical down the center line. "
        "STRICTLY FORBIDDEN: NO 3/4 angle, NO angled perspective, NO side profile, NO rotated torso, NO tilted shoulders, NO diagonal skew. "
        "The garment must be positioned completely straight ('sidha'), perfectly centered, and squarely facing the camera."
    )

    return (
        "Generate a perfectly straight 0-degree direct front-facing empty 3D ghost-mannequin e-commerce studio product photograph showing ONLY the standalone "
        "garment floating upright, squarely facing the camera with symmetrical level shoulders and symmetrical sleeves on a pure solid white background (#FFFFFF), with completely NO person, NO model, "
        "NO head, NO face, NO hair, NO neck, NO skin, NO hands, NO arms, NO legs, NO feet, NO bed, NO bedsheet, NO blanket, NO carpet, NO floor, and NO jewelry anywhere in the image. "
        f"{straight_alignment_rule} "
        f"{isolation_rule} "
        f"{surface_rejection} "
        f"{color_lock_rule} "
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
            "Analyze ONLY the UPPER-BODY garment (shirt, t-shirt, top, kurta, jacket) in this image. "
            "Completely IGNORE and exclude any pants, jeans, trousers, or bottom wear. "
            "STRICT BACKGROUND RULE: If the clothing is placed on a bedsheet, mattress, blanket, floor, or table, "
            "the bedsheet / floor is 100% BACKGROUND. Do NOT mistake bedsheet patterns for garment motifs! "
            "CRITICAL COLOR & FABRIC FIDELITY: Identify the EXACT real-world dye shade, darkness, and saturation. If it is dark, muted, dusty, or desaturated (e.g., dusty army-olive, slate-khaki, deep moss), describe that exact deep muted matte shade without brightening or oversaturating. "
            "Examine collar shape, placket, matching buttons, fabric weave, and sleeves so the 3D ghost mannequin is 100% authentic."
        )
    elif target_norm in ("pants", "skirt"):
        user_instruction = (
            "Analyze ONLY the LOWER-BODY garment (pants, jeans, trousers, shorts, skirt) in this image. "
            "Completely IGNORE and exclude any upper-body shirt, top, or kurta. "
            "STRICT BACKGROUND RULE: If placed on a bedsheet, blanket, or floor, IGNORE the bedsheet completely. "
            "CRITICAL COLOR FIDELITY: Identify the EXACT real-world dye wash, darkness, and saturation level. "
            "Examine the actual pants fabric at high resolution: fit, pockets, stitching, and hems."
        )
    elif target_norm in ("dress", "saree", "lehenga"):
        user_instruction = (
            f"Analyze the full {target_norm} in this image. "
            "STRICT BACKGROUND RULE: If placed on a bedsheet, blanket, or floor, IGNORE the background bedsheet completely. "
            "CRITICAL COLOR FIDELITY: Identify exact real-world dye shades, borders, prints, embroidery, and drape without oversaturating."
        )
    else:  # "all"
        user_instruction = (
            "Analyze the clothing in this image for exact 3D ghost-mannequin catalog reconstruction. "
            "CRITICAL COLOR & SURFACE RULES: "
            "1. Color & Fabric Fidelity: Identify the EXACT real-world dye shade, darkness, and saturation level. If the fabric is dark, muted, dusty, or desaturated (e.g., dusty army-olive, slate sage, dull khaki, deep navy), explicitly describe that exact deep muted matte shade and warn against oversaturating or brightening. "
            "2. Surface Rejection: If the clothing is laid flat on a bed, mattress, floor, rug, or table, the bedsheet or floor is 100% BACKGROUND. NEVER treat bedsheet patterns (hexagons, mandalas, geometric prints, borders) as clothing! "
            "3. Actual Garments Only: Only identify genuine stitched wearable clothing items (if only one kurta is on the bed, return only one item). "
            "4. Separates: If contrasting separates (e.g. shirt + jeans) are actually present, return separate items for 'upper_clothes' and 'pants'. "
            "Examine the genuine clothing fabric at high resolution: base color, true motifs, collar, buttons, and cut."
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
