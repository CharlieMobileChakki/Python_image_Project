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
UPPER_STUDIO_RULES = (
    "Pristine e-commerce 3D ghost-mannequin catalog presentation (Zara / Myntra / Amazon Fashion hero shot): "
    "Empty hollow 3D inner neckline aperture, empty open sleeve cuffs, and empty open bottom hem showing the hollow interior with zero human body inside. "
    "Seamlessly reconstruct the continuous repeating fabric pattern and textile weave across the chest, stomach, placket, and sleeves. "
    "MANDATORY POSE STRAIGHTENING: Centered upright, perfectly straight 0-degree direct frontal shot facing the camera straight-on, perfectly level horizontal shoulders, both sleeves hanging straight down along both sides at symmetrical matching angles, and strictly level horizontal hemline. "
    "CRITICAL ISOLATION: COMPLETELY EXCLUDE, REMOVE, AND DO NOT GENERATE any pants, jeans, trousers, belt, legs, or bottom garments! The bottom half of the image MUST BE 100% empty, clean pure solid white (#FFFFFF). "
    "Pure solid white background (#FFFFFF), completely free of any person, model, face, hair, neck, skin, arms, hands, legs, bed, mattress, bedsheet, blanket, carpet, floor, furniture, or shadows."
)

LOWER_STUDIO_RULES = (
    "Pristine e-commerce 3D ghost-mannequin catalog presentation (Zara / Myntra / Amazon Fashion hero shot): "
    "Empty hollow 3D open waistband aperture at top, and empty open bottom ankle leg cuffs showing the hollow interior with zero human body inside. "
    "Seamlessly reconstruct the authentic fabric weave, textile texture, waistband, front fly button/zipper, pockets, and full trouser leg length. "
    "MANDATORY POSE STRAIGHTENING: Centered upright, perfectly straight 0-degree direct frontal shot facing the camera straight-on, horizontally level waistband across the top, both trouser legs hanging completely straight down vertically parallel to each other side-by-side with zero knee bend and zero walking tilt, both ankle leg cuffs aligned at the exact same horizontal height. "
    "CRITICAL ISOLATION: ABSOLUTELY DO NOT GENERATE ANY SHIRT, TOP, T-SHIRT, JACKET, COLLAR, SHOULDERS, SLEEVES, CHEST, TORSO, OR UPPER BODY! "
    "The top half of the canvas MUST BE 100% empty, clean, pure solid white (#FFFFFF). Generate EXCLUSIVELY the standalone pants/trousers/skirt floating by themselves with NO upper garment attached! "
    "Pure solid white background (#FFFFFF), completely free of any person, model, face, hair, neck, skin, arms, hands, legs, bed, mattress, bedsheet, blanket, carpet, floor, furniture, or shadows."
)

FULL_OUTFIT_STUDIO_RULES = (
    "Pristine e-commerce 3D ghost-mannequin catalog presentation (Zara / Myntra / Amazon Fashion hero shot): "
    "Empty hollow 3D inner neckline aperture, empty open sleeve cuffs, and empty open bottom hem showing hollow interior with zero human body inside. "
    "MANDATORY POSE STRAIGHTENING: Centered upright, perfectly straight 0-degree direct frontal shot facing the camera straight-on, perfectly level horizontal shoulders, symmetrical silhouette, and plumb vertical alignment. "
    "Pure solid white background (#FFFFFF), completely free of any person, model, face, hair, neck, skin, arms, hands, legs, bed, mattress, bedsheet, blanket, carpet, floor, furniture, or shadows."
)

COMMON_STUDIO_RULES = UPPER_STUDIO_RULES

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
   - TROUSERS, CHINOS, JEANS & BOTTOM WEAR (CRITICAL REAL-WORLD COLOR & TEXTILE ACCURACY):
     * Forensic-analyze the exact real-world color, wash, and fabric:
       - If the pants are light beige, khaki, cream, or sand (like chino trousers): explicitly specify "men's flat-front straight-leg chino trousers in smooth light beige / warm khaki cotton twill fabric, with uniform solid matte light beige dye, slant front pockets, and clean straight hems".
       - If dark / raw indigo / black jeans: explicitly specify "men's classic straight-leg denim jeans in uniform solid deep dark raw indigo denim wash with zero artificial thigh fading, no bleached thigh highlights, and no whiskering".
       - If olive, grey, white, brown, or navy: describe the exact authentic dye shade and weave without altering it.
     * Always match the real-world fabric weave (cotton twill, chino, denim, linen, khadi, wool blend).
     * Explicitly specify: "solid uniform dye from waistband to ankle cuffs, perfectly straight symmetrical legs".

3. MANDATORY POSE RECTIFICATION & STRAIGHTENING (E-COMMERCE CATALOG GHOST-MANNEQUIN STANDARD):
   - In e-commerce catalog standards (Zara, Myntra, Amazon, Ajio), product hero photos MUST be strictly front-facing, straight, and symmetrical.
   - Reference photos often show human models standing casually, walking, with bent knees, legs apart, tilted waistbands, hands in pockets, or bent arms.
   - STRICT RULE: YOU MUST EXPLICITLY INSTRUCT THE RECONSTRUCTION TO STRAIGHTEN AND RE-ALIGN THE GARMENT INTO A PERFECT 0-DEGREE STRAIGHT E-COMMERCE STUDIO 3D GHOST MANNEQUIN SHOT.
   - FOR LOWER BODY (PANTS, CHINOS, JEANS, TROUSERS, SKIRTS):
     * Explicitly specify: "OVERRIDE HUMAN BODY POSE: Straighten both legs completely! Do NOT copy the model's bent knee, tilted leg, or walking stance from the reference photo. Reconstruct both trouser legs hanging perfectly straight down, plumb, vertical, and parallel to each other side-by-side with zero bend and zero tilt. The waistband must be 100% horizontal and level across the top with an open 3D hollow waistband aperture, and both bottom leg cuffs must align at the exact same horizontal bottom height."
   - FOR UPPER BODY (SHIRTS, TOPS, KURTAS, JACKETS):
     * Explicitly specify: "OVERRIDE HUMAN BODY POSE: Perfectly level horizontal shoulders, both sleeves hanging naturally and straight down along both sides at symmetrical matching angles, front buttons/placket running in a plumb vertical line down the center, and a level horizontal bottom hem."
   - STRICT AUTHENTICITY ("DRESS WAHI REHNA CHAHIYE"):
     * While straightening the pose into a perfect e-commerce studio display, the garment itself (color, dye hue, fabric weave, texture, pockets, buttons, fly, hems) MUST match the reference photo 100% identically!

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
      "garment_description": "Men's casual relaxed-fit short-sleeve resort shirt crafted from breathable cotton fabric in soft pastel powder-blue featuring subtle tonal white botanical floral leafy branch prints. Pristine e-commerce 3D ghost-mannequin catalog presentation: eliminate any tilted posture or bent arm; straight-on 0-degree front view with perfectly level horizontal shoulders, symmetrical hanging sleeves along both sides, standard point spread collar, front button placket with small buttons running straight down the center, and a level horizontal hem."
    },
    {
      "category": "pants",
      "recognized_label": "Light Beige Flat-Front Chino Trousers",
      "garment_description": "Men's flat-front straight-leg chino trousers crafted from smooth cotton twill fabric in authentic uniform light beige / warm khaki dye. Pristine e-commerce 3D ghost-mannequin catalog presentation: eliminate the model's bent knee and walking tilt; both trouser legs hang completely straight down, vertically parallel to each other side-by-side with zero bend and zero tilt, with horizontally level waistband at top, slant front pockets, matching button and fly zipper, and both bottom leg cuffs ending at the exact same horizontal bottom height."
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
    """Put category-specific 3D ghost-mannequin isolation, e-commerce pose straightening, and fabric lock FIRST."""
    cat = (category or "").strip().lower()

    surface_rejection = (
        "STRICT SURFACE REJECTION: If the reference photo shows clothing placed on a bedsheet, mattress, blanket, floor, rug, carpet, or table, "
        "COMPLETELY IGNORE the bedsheet and its background patterns! Do NOT apply any bedsheet print, geometric shapes, or bedding colors to the garment. "
        "Reconstruct ONLY the authentic stitched garment itself."
    )

    if cat == "upper_clothes":
        isolation_header = (
            "Generate an empty 3D ghost-mannequin e-commerce studio catalog product photograph (Zara / Myntra / Amazon fashion hero shot) showing EXCLUSIVELY the standalone "
            "upper-body shirt / top / kurta floating upright in a direct 0-degree straight front-facing view on a pure solid white background (#FFFFFF). "
            "The bottom half of the image MUST BE 100% EMPTY, clean pure solid white. "
            "CRITICAL ISOLATION RULE: COMPLETELY EXCLUDE, REMOVE, AND DO NOT GENERATE any pants, jeans, trousers, belt, waistline, or legs! "
            "No human body, NO head, NO face, NO hair, NO neck, NO skin, NO arms, NO hands, NO legs, NO feet. "
        )
        pose_rectification = (
            "MANDATORY E-COMMERCE POSE STRAIGHTENING: "
            "DO NOT COPY ANY BENT ARM, HAND-IN-POCKET, OR TILTED POSTURE FROM THE REFERENCE PHOTO! "
            "Realign into a pristine symmetrical catalog display: "
            "1. Both shoulders must be at the exact same horizontal height (perfectly level shoulders). "
            "2. Both sleeves must hang straight down symmetrically on both sides along the torso at equal matching angles. "
            "3. Front placket, buttons, zipper, and collar slit must run in a plumb, perfectly vertical straight line down the center. "
            "4. Bottom hemline must be completely straight and horizontally level across. "
            "5. Empty hollow 3D inner neckline aperture, open sleeve cuffs, and open bottom hem showing the hollow interior with zero human body inside. "
        )
        garment_lock = (
            "STRICT AUTHENTICITY & DESIGN CLONING (DRESS WAHI REHNA CHAHIYE): "
            "While straightening into the e-commerce studio pose, you MUST preserve the EXACT garment from the reference photo without altering it! "
            "Faithfully replicate the identical real-world dye color, hue, lightness, textile weave, pattern motifs, collar shape, buttons, placket, and stitching directly from the reference photo."
        )
        return (
            f"{isolation_header} "
            f"{pose_rectification} "
            f"{garment_lock} "
            f"{surface_rejection} "
            f"Exact standalone upper garment to isolate and faithfully reconstruct from reference photo: {garment_description.strip()}. "
            f"{UPPER_STUDIO_RULES}"
        )

    elif cat in ("pants", "skirt"):
        isolation_header = (
            "Generate an empty 3D ghost-mannequin e-commerce studio catalog product photograph (Zara / Myntra / Amazon fashion hero shot) showing EXCLUSIVELY the standalone "
            "lower-body pants / trousers / chinos / jeans / shorts / skirt floating upright in a direct 0-degree straight front-facing view on a pure solid white background (#FFFFFF). "
            "The top half of the image MUST BE 100% EMPTY, clean pure solid white. "
            "CRITICAL ISOLATION RULE: ABSOLUTELY DO NOT GENERATE ANY SHIRT, TOP, T-SHIRT, JACKET, COLLAR, SHOULDERS, SLEEVES, CHEST, TORSO, OR UPPER BODY! "
            "Generate EXCLUSIVELY the standalone lower garment by itself with NO upper body garment attached! "
            "No human body, NO head, NO face, NO hair, NO neck, NO skin, NO arms, NO hands, NO legs, NO feet. "
        )
        pose_rectification = (
            "MANDATORY E-COMMERCE POSE STRAIGHTENING (OVERRIDE HUMAN BODY POSE): "
            "DO NOT COPY THE MODEL'S CROOKED WALKING POSE, BENT KNEE, BOWED LEGS, OR ASYMMETRIC STANCE FROM THE REFERENCE PHOTO! "
            "You MUST completely straighten and realign the trousers into a pristine e-commerce 3D ghost-mannequin studio catalog hero photograph: "
            "1. STRAIGHT PARALLEL LEGS: Both trouser legs MUST hang completely straight down vertically, perfectly parallel to each other side-by-side with zero knee bend, zero curve, and zero walking tilt. "
            "2. HORIZONTALLY LEVEL WAISTBAND: The waistband at top MUST be 100% horizontal, straight, and level across, with an open circular/oval 3D hollow waistband aperture showing the empty interior with zero human body inside. "
            "3. EVEN BOTTOM CUFFS: Both left and right bottom ankle leg cuffs MUST align at the exact same horizontal bottom baseline height (completely even, parallel hems). "
            "4. NATURAL 3D MANNEQUIN VOLUME: Filled out with smooth, natural cylindrical 3D depth as if fitted on an invisible 3D mannequin, smooth steam-pressed ironed fabric without person-walking wrinkles or bent knee folds. "
        )
        garment_lock = (
            "STRICT AUTHENTICITY & DESIGN CLONING (DRESS WAHI REHNA CHAHIYE): "
            "While straightening the pose into the e-commerce studio display, you MUST preserve the EXACT garment from the reference photo without altering it! "
            "Faithfully replicate the identical real-world dye color, hue, lightness, smooth fabric weave, textile texture, seams, front fly, button closure, belt loops, and pockets directly from the reference photo without changing the garment."
        )
        return (
            f"{isolation_header} "
            f"{pose_rectification} "
            f"{garment_lock} "
            f"{surface_rejection} "
            f"Exact standalone lower garment to isolate and faithfully reconstruct from reference photo: {garment_description.strip()}. "
            f"{LOWER_STUDIO_RULES}"
        )

    else:
        isolation_header = (
            "Generate a perfectly straight 0-degree direct front-facing empty 3D ghost-mannequin e-commerce studio product photograph (Zara / Myntra catalog hero shot) showing the standalone "
            "outfit floating upright on a pure solid white background (#FFFFFF), with completely NO person, NO model, NO skin, NO human body. "
        )
        pose_rectification = (
            "MANDATORY E-COMMERCE POSE STRAIGHTENING: "
            "DO NOT COPY ANY WALKING POSE, TILTED HIP, OR BENT ARM FROM THE REFERENCE PHOTO! "
            "Straighten the outfit into a perfectly plumb, symmetrical 0-degree direct frontal e-commerce studio catalog hero photograph with level horizontal shoulders, symmetrical silhouette, and smooth 3D hollow ghost mannequin volume. "
        )
        garment_lock = (
            "STRICT AUTHENTICITY & DESIGN CLONING (DRESS WAHI REHNA CHAHIYE): "
            "While straightening into the e-commerce studio pose, you MUST preserve the EXACT garment from the reference photo without altering it! "
            "Faithfully replicate the identical real-world color, hue, fabric weave, textile texture, embroidery, prints, and construction directly from the reference photo without changing the garment."
        )
        return (
            f"{isolation_header} "
            f"{pose_rectification} "
            f"{garment_lock} "
            f"{surface_rejection} "
            f"Exact standalone garment to reconstruct from reference photo: {garment_description.strip()}. "
            f"{FULL_OUTFIT_STUDIO_RULES}"
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
            "CRITICAL POSE STRAIGHTENING (E-COMMERCE CATALOG STANDARD): Reconstruct with level horizontal shoulders, symmetrical hanging sleeves along both sides at matching angles, and vertical placket/buttons, while preserving the exact original fabric, color, prints, and collar ('dress wahi rehna chahiye')."
        )
    elif target_norm in ("pants", "skirt"):
        user_instruction = (
            "Analyze ONLY the LOWER-BODY garment (pants, jeans, trousers, shorts, skirt) in this image. "
            "Completely IGNORE and exclude any upper-body shirt, top, or kurta. "
            "STRICT BACKGROUND RULE: If placed on a bedsheet, blanket, or floor, IGNORE the bedsheet completely. "
            "CRITICAL COLOR & FABRIC FIDELITY: Identify the EXACT real-world dye wash, darkness, and saturation level without altering the color. "
            "CRITICAL POSE STRAIGHTENING (E-COMMERCE CATALOG STANDARD): In e-commerce ghost mannequin catalog shots, the pant MUST NOT have bent knees or tilted legs! Explicitly command both legs to be straightened out completely to hang vertically parallel to each other side-by-side with zero knee bend and zero walking tilt, with horizontally level waistband across the top and level ankle cuffs, while preserving the exact original fabric, color, pockets, and stitching ('dress wahi rehna chahiye')."
        )
    elif target_norm in ("dress", "saree", "lehenga"):
        user_instruction = (
            f"Analyze the full {target_norm} in this image. "
            "STRICT BACKGROUND RULE: If placed on a bedsheet, blanket, or floor, IGNORE the background bedsheet completely. "
            "CRITICAL COLOR FIDELITY: Identify exact real-world dye shades, borders, prints, embroidery, and drape without oversaturating. "
            "CRITICAL POSE STRAIGHTENING: Reconstruct in a plumb, symmetrical 0-degree straight e-commerce studio catalog display while preserving all authentic garment details."
        )
    else:  # "all"
        user_instruction = (
            "Analyze the clothing in this image for exact 3D ghost-mannequin catalog reconstruction. "
            "CRITICAL RULES: "
            "1. Color & Fabric Fidelity: Identify the EXACT real-world dye shade, darkness, and saturation level. Preserve the exact authentic colors without altering them. "
            "2. Pose Straightening & Rectification (E-Commerce Standard): In e-commerce 3D ghost mannequin catalogs (Zara, Myntra, Amazon), garments are displayed perfectly straight and symmetrical. Command both trouser legs to hang straight down vertically parallel with level waistband and level cuffs (eliminate bent knees or walking poses). For tops, ensure level shoulders, symmetrical sleeves, and vertical plackets. "
            "3. Garment Authenticity ('Dress Wahi Rehna Chahiye'): The dress/garment itself must remain 100% faithful to the reference photo: exact fabric, color, weave, pockets, buttons, and stitching. "
            "4. Surface Rejection: If the clothing is laid flat on a bed, mattress, floor, rug, or table, the bedsheet or floor is 100% BACKGROUND. NEVER treat bedsheet patterns (hexagons, mandalas, geometric prints, borders) as clothing! "
            "5. Actual Garments Only: Only identify genuine stitched wearable clothing items (if only one kurta is on the bed, return only one item). "
            "6. Separates: If contrasting separates (e.g. shirt + jeans) are actually present, return separate items for 'upper_clothes' and 'pants'. "
            "Examine the genuine clothing fabric at high resolution: base color, true motifs, collar, buttons, pockets, and cut."
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
