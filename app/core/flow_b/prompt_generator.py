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

6. JEWELRY & PRECIOUS ORNAMENTS DISCRIMINATION (GOLD, SILVER, DIAMONDS):
   - If the reference photograph shows JEWELRY (such as earrings, jhumkas, necklace, pendant, ring, bangles, or kada):
     ABSOLUTELY NEVER hallucinate or classify it as clothing, dress, shirt, or kurta!
     Identify it strictly with its true jewelry category:
     * "earrings" (Earrings, Jhumkas, Danglers, Studs)
     * "necklace" (Necklace, Haar, Choker, Chain, Pendant)
     * "ring" (Finger Ring, Band)
     * "bangles" (Bangles, Kada, Bracelet)
     Describe the authentic precious metal (22K/24K warm yellow gold), openwork jaali filigree, hanging bead clusters, and gemstones with forensic accuracy.

7. Allowed category values:
   - For Apparel: "upper_clothes", "pants", "dress", "skirt", "saree", "lehenga"
   - For Jewelry: "necklace", "ring", "earrings", "bangles"

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
    },
    {
      "category": "earrings",
      "recognized_label": "22K Yellow Gold Traditional Jhumka Earrings",
      "garment_description": "Pair of traditional Indian temple jhumka earrings crafted from 22K warm yellow gold, featuring floral ear studs with spherical bead borders, bell-shaped jhumka domes with intricate openwork filigree and jaali carvings, and delicate hanging pearl clusters along the bottom rim."
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

    # Route jewelry categories directly to luxury jewelry studio rules
    if is_jewelry_domain(cat):
        stand = (
            "t_bar_earring_stand" if cat in ("earrings", "jhumkas")
            else ("slotted_ring_block" if cat == "ring"
            else ("cylindrical_bangle_bar" if cat in ("bangles", "kada", "bracelet")
            else "white_neck_bust"))
        )
        return _build_final_jewelry_prompt({
            "category": cat,
            "recognized_label": garment_description[:60],
            "metal_type": "22K Warm Yellow Gold",
            "stand_type": stand,
            "jewelry_description": garment_description,
        })

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
            "Analyze the item (apparel clothing OR precious jewelry ornament) in this image for exact 3D studio catalog reconstruction. "
            "CRITICAL DOMAIN DISCRIMINATION (APPAREL vs JEWELRY): "
            "1. If the photo contains JEWELRY (such as earrings, jhumkas, necklace, pendant, ring, bangles, or kada): "
            "   You MUST categorize it strictly as 'earrings', 'necklace', 'ring', or 'bangles'! "
            "   ABSOLUTELY NEVER categorize jewelry as a dress, shirt, top, or clothing! "
            "   Describe the authentic gold/silver metal tone (e.g. 22K warm yellow gold), filigree openwork jaali, hanging beads, and gemstones. "
            "2. If the photo contains APPAREL (shirt, top, pants, dress, kurta, saree): "
            "   Categorize as 'upper_clothes', 'pants', 'dress', 'skirt', 'saree', or 'lehenga'. "
            "3. Surface Rejection: If placed on a bedsheet, blanket, table, or floor, the surface is 100% BACKGROUND. "
            "4. Actual Items Only: Only identify the genuine item(s). If only earrings are present, return ONLY one entry for category 'earrings'."
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


# ==============================================================================
# JEWELRY & PRECIOUS ORNAMENTS 3D STUDIO PIPELINE (MULTI-ANGLE AWARE)
# ------------------------------------------------------------------------------
# Specialized 3D display stand generation for Necklaces, Rings, Earrings, Bangles
# Supports single or multi-angle input fusion (up to 4 reference photos).
# Mounts authentic gold, diamond, and gemstone ornaments on luxury catalog stands:
# - Necklaces/Chokers: Smooth Matte-White Neck Bust Mannequin Stand (Image 2)
# - Rings/Bands: Angled 3D White Velvet / Slotted Acrylic Ring Display Block
# - Earrings/Jhumkas: Symmetrical T-Bar Hanging Earring Stand
# - Bangles/Kadas: Horizontal Cylindrical Leatherette Bangle Display Bar
# ==============================================================================

JEWELRY_NECKLACE_STUDIO_RULES = (
    "Pristine luxury jewelry e-commerce studio catalog photograph (Tanishq / Kalyan Jewellers / Cartier luxury hero shot). "
    "The standalone authentic necklace is mounted and draped elegantly around the neck of a smooth, clean, premium matte-white neck bust mannequin display stand. "
    "PERFECT SYMMETRY & ALIGNMENT: Centered upright, direct 0-degree straight frontal studio camera view. The necklace loops naturally and symmetrically around both sides of the contoured white neck bust, with the central medallion / pendant resting completely flat, centered, and plumb on the front chest of the bust. "
    "PRECIOUS METAL & POLISH LUSTER: Faithfully preserve the authentic 22K/24K warm yellow gold metal, polished luster, intricate openwork jaali filigree, delicate leaf motifs, engraved borders, and hanging bead drops with 100% forensic accuracy directly from the reference photo(s). "
    "8K ULTRA-HD MACRO RESOLUTION (EXTREME ZOOM CLARITY): Razor-sharp microscopic jewelry studio illumination. Every individual filigree thread, openwork jaali wire, embossed leaf vein, polished gold bead drop, and metal reflection must be rendered in ultra-crisp, tack-sharp microscopic focus with ZERO blur and ZERO pixelation even when deeply zoomed in. "
    "SOLID STUDIO WHITE BACKGROUND (STAND INTACT): Pure solid pristine studio white background (#FFFFFF). The entire white neck bust display stand MUST remain 100% intact, fully visible, and seamlessly grounded on the pure solid white background with subtle realistic contact shadows. DO NOT make transparent and DO NOT cut or remove the display stand! Completely free of any human model, skin, neck, face, hands, clothes, table clutter, or props."
)

JEWELRY_RING_STUDIO_RULES = (
    "Pristine luxury jewelry e-commerce studio catalog photograph (Tiffany & Co. / BlueStone / CaratLane luxury hero shot). "
    "The standalone ring is mounted upright on a clean, minimalist, angled matte-white velvet or slotted acrylic ring display block. "
    "OPTIMAL 3D HERO PERSPECTIVE (MULTI-FACET VISIBILITY): Direct 30-degree to 45-degree three-quarter isometric 3D beauty angle. "
    "CRITICAL SHANK & CROWN VISIBILITY: BOTH the top crown / center gemstone / head motif AND the intricate side shank carvings, side filigree, gallery jali, and band engravings must be SIMULTANEOUSLY prominent and clearly visible in this single photograph! "
    "PRECIOUS METAL & GEMSTONE FIDELITY: Reconstruct the exact precious metal alloy (22K/18K warm yellow gold, platinum, or rose gold), micro-pave prong settings, diamond facets, and side band engravings with 100% forensic precision from all reference angles. "
    "8K ULTRA-HD MACRO RESOLUTION (EXTREME ZOOM CLARITY): Razor-sharp macro jewelry studio illumination with crisp facet reflections and tack-sharp microscopic focus. Every prong, facet, and carved metal grain must be crystal clear with zero blur when zoomed in. "
    "SOLID STUDIO WHITE BACKGROUND (STAND INTACT): Pure solid clean studio white background (#FFFFFF). The white ring display block must remain 100% intact and visible on the solid white background. DO NOT make transparent. Zero human fingers, NO hands, NO skin, and NO table props."
)

JEWELRY_EARRINGS_STUDIO_RULES = (
    "Pristine luxury jewelry e-commerce studio catalog photograph (Tanishq / Malabar Gold luxury hero shot). "
    "The pair of matching earrings / jhumkas is displayed hanging elegantly on a minimalist white or matte-metallic studio T-bar jewelry earring display stand. "
    "PERFECT PAIR SYMMETRY: Both left and right earrings hang side-by-side perfectly parallel at the exact same horizontal height, facing directly forward straight to camera. "
    "PRECIOUS CRAFTSMANSHIP: Exact replication of authentic gold filigree, jhumka bell domes, dangling pearl/gold beads, gemstone inlays, and ear-stud motifs from the reference photo(s). "
    "8K ULTRA-HD MACRO RESOLUTION (EXTREME ZOOM CLARITY): Ultra-sharp macro studio focus on every hanging bead and filigree curve with zero blur under deep zoom. "
    "SOLID STUDIO WHITE BACKGROUND (STAND INTACT): Pure solid clean studio white background (#FFFFFF) with the T-bar stand fully intact. DO NOT make transparent. Completely free of any human ears, hair, face, neck, skin, or hands."
)

JEWELRY_BANGLES_STUDIO_RULES = (
    "Pristine luxury jewelry e-commerce studio catalog photograph (Kalyan / Joyalukkas luxury hero shot). "
    "The bangles / kada / bracelet are mounted around a horizontal cylindrical white leatherette or velvet jewelry bangle display bar. "
    "3D CIRCULAR PERSPECTIVE: Elegant three-quarter perspective highlighting the circular 3D wrist profile, outer circumference carvings, side filigree work, screw/clasp fastenings, and hollow inner curvature. "
    "PRECIOUS METAL FIDELITY: Exact replication of the authentic 22K yellow gold tone, antique finish, meenakari enamel, or diamond studs from the reference photo(s). "
    "8K ULTRA-HD MACRO RESOLUTION (EXTREME ZOOM CLARITY): Hyper-detailed macro studio lighting with crisp specular reflections and tack-sharp detail across the entire circumference. "
    "SOLID STUDIO WHITE BACKGROUND (STAND INTACT): Pure solid clean studio white background (#FFFFFF) with the cylindrical display bar fully intact. DO NOT make transparent. Completely free of any human wrist, hands, arms, or skin."
)

JEWELRY_STAND_RULES_MAP = {
    "necklace": JEWELRY_NECKLACE_STUDIO_RULES,
    "pendant": JEWELRY_NECKLACE_STUDIO_RULES,
    "choker": JEWELRY_NECKLACE_STUDIO_RULES,
    "haar": JEWELRY_NECKLACE_STUDIO_RULES,
    "mangalsutra": JEWELRY_NECKLACE_STUDIO_RULES,
    "ring": JEWELRY_RING_STUDIO_RULES,
    "earrings": JEWELRY_EARRINGS_STUDIO_RULES,
    "jhumkas": JEWELRY_EARRINGS_STUDIO_RULES,
    "bangles": JEWELRY_BANGLES_STUDIO_RULES,
    "kada": JEWELRY_BANGLES_STUDIO_RULES,
    "bracelet": JEWELRY_BANGLES_STUDIO_RULES,
}

SYSTEM_JEWELRY_VISION_PROMPT = """You are a master jewelry gemologist, high-jewelry e-commerce catalog director, and 3D precious ornament analyst (Tanishq, Cartier, Tiffany & Co., CaratLane).

Your mission is to analyze 1 to 4 photographs of the SAME piece of jewelry (taken from different angles: Top view, Side profile, 45-degree angle, Close-up) and synthesize a unified, forensic-grade 3D catalog description.

CRITICAL JEWELRY RULES:
1. MULTI-ANGLE DETAIL FUSION (DO NOT MISS SIDE DESIGNS):
   - When multiple images are provided, cross-examine all angles:
     * Image 1 might show the top face/crown (e.g. center gemstone or medallion).
     * Image 2 & 3 might reveal the side shank, gallery openwork, side jali filigree, or band carvings.
   - Synthesize ALL angles into a unified description so that NO side detail or hidden engraving is lost!

2. METALS, POLISH & ALLOY FIDELITY:
   - Identify the exact metal: 22K/24K rich warm yellow gold, 18K rose gold, 950 platinum, sterling silver, or antique oxidized temple gold.
   - Preserve authentic metal texture: high-polish mirror finish, matte brushed gold, hammered gold, antique matte patina.
   - Strictly prohibit changing gold color (e.g. do NOT turn yellow gold into silver or copper!).

3. GEMSTONES & PRECIOUS CRAFTSMANSHIP:
   - Identify gemstone types (solitaire diamond, ruby, emerald, sapphire, cubic zirconia, polki, kundan, south sea pearl).
   - Detail the setting style: prong setting, bezel setting, channel setting, pave setting, openwork jali filigree.

4. CATEGORY & STAND ASSIGNMENT:
   - "necklace": Smooth matte-white neck bust mannequin display stand (like Tanishq catalog).
   - "ring": Angled 3D white velvet/slotted acrylic ring display block (30-45 degree hero angle showing BOTH top crown AND side shank).
   - "earrings": Symmetrical T-bar hanging jewelry earring stand.
   - "bangles": Horizontal cylindrical jewelry bangle display bar.

Return strictly valid JSON:
{
  "jewelry_item": {
    "category": "necklace" | "ring" | "earrings" | "bangles",
    "recognized_label": "22K Yellow Gold Filigree Medallion Necklace",
    "metal_type": "22K Warm Yellow Gold with High-Polish & Filigree Finish",
    "stand_type": "white_neck_bust" | "slotted_ring_block" | "t_bar_earring_stand" | "cylindrical_bangle_bar",
    "optimal_angle": "0-degree frontal" | "30-45 degree three-quarter isometric",
    "jewelry_description": "Forensic, multi-angle description detailing every motif, crown, side profile, and gemstone from all photos."
  }
}"""


def is_jewelry_domain(text: str | None) -> bool:
    """Check whether a given label, category, or query refers to jewelry rather than garments."""
    if not text:
        return False
    t = text.strip().lower()
    jewelry_keywords = {
        "jewelry", "jewellery", "gold", "silver", "diamond", "necklace", "ring",
        "earring", "earrings", "jhumka", "jhumkas", "bangle", "bangles", "kada",
        "bracelet", "pendant", "choker", "mangalsutra", "haar", "chain", "anklet",
        "payal", "polki", "kundan", "solitaire", "gemstone", "ornament",
    }
    tokens = set(t.replace("_", " ").replace("-", " ").split())
    if tokens & jewelry_keywords:
        return True
    return any(kw in t for kw in jewelry_keywords)


def _build_final_jewelry_prompt(jewelry_data: dict[str, Any]) -> str:
    """Build high-fidelity Qwen prompt for 3D jewelry catalog generation on luxury display stands."""
    category = str(jewelry_data.get("category", "necklace")).strip().lower()
    label = str(jewelry_data.get("recognized_label", "Fine Gold Jewelry")).strip()
    metal = str(jewelry_data.get("metal_type", "22K Yellow Gold")).strip()
    desc = str(jewelry_data.get("jewelry_description", "")).strip()
    stand_type = str(jewelry_data.get("stand_type", "white_neck_bust")).strip()

    studio_rules = JEWELRY_STAND_RULES_MAP.get(category, JEWELRY_NECKLACE_STUDIO_RULES)

    if category == "ring":
        header = (
            "Generate an empty luxury jewelry e-commerce studio catalog product photograph showing the standalone "
            f"precious ring mounted upright on a clean angled matte-white velvet or slotted acrylic ring display block. "
            "OPTIMAL 3D HERO PERSPECTIVE: 30-degree to 45-degree three-quarter isometric angle. "
            "CRITICAL MULTI-ANGLE DETAIL RECONSTRUCTION: Reconstruct BOTH the top crown / center gemstone AND the intricate "
            "side shank carvings, side filigree, gallery openwork, and band engravings simultaneously in one single photograph! "
        )
    elif category in ("necklace", "pendant", "choker", "haar", "mangalsutra"):
        header = (
            "Generate an empty luxury jewelry e-commerce studio catalog product photograph showing the standalone "
            f"precious {category} mounted and draped symmetrically on a smooth, clean matte-white neck bust mannequin display stand. "
            "0-degree direct frontal upright camera shot, with the necklace looping naturally around the contoured neck bust and "
            "the central pendant resting flat, centered, and plumb on the front chest of the bust. "
        )
    elif category in ("earrings", "jhumkas"):
        header = (
            "Generate an empty luxury jewelry e-commerce studio catalog product photograph showing the matching pair of "
            f"standalone {category} hanging side-by-side symmetrically on a minimalist matte-white T-bar jewelry earring display stand. "
        )
    else:  # bangles, kada, bracelet
        header = (
            "Generate an empty luxury jewelry e-commerce studio catalog product photograph showing the standalone "
            f"precious {category} mounted around a horizontal cylindrical matte-white leatherette jewelry bangle display bar. "
        )

    metal_lock = (
        f"STRICT PRECIOUS METAL & GEMSTONE FIDELITY: Clone the EXACT real-world metal alloy ({metal}), surface luster, "
        "micro-carvings, prongs, and gemstone colors directly from the reference photo(s). "
        "DO NOT change yellow gold into silver or copper! Preserve authentic gold polish and craftsmanship."
    )

    ultra_hd_zoom_lock = (
        "CRITICAL 8K ULTRA-HD MASTER MACRO RESOLUTION (ZERO BLUR UNDER EXTREME ZOOM): "
        "The final image MUST be rendered in tack-sharp microscopic macro catalog focus! "
        "Every single millimeter of the jewelry piece—every individual filigree wire, embossed jaali texture, prong claw, "
        "faceted diamond edge, gold reflection, and surface engraving—must be razor sharp, crisp, and hyper-detailed. "
        "Zero blur, zero pixelation, zero digital noise, and zero compression fuzziness. Under extreme deep zoom, "
        "every single pixel must remain tack-sharp, clean, and perfectly defined!"
    )

    white_background_lock = (
        "CRITICAL SOLID STUDIO WHITE BACKGROUND (STAND 100% INTACT, NOT TRANSPARENT): "
        "The entire scene MUST be a finished, complete e-commerce studio catalog photograph on a pure solid white background (#FFFFFF). "
        "DO NOT remove the background! DO NOT make transparent! The white luxury display stand (neck bust mannequin / ring block / "
        "T-bar stand / bangle bar) MUST remain 100% fully intact, seamlessly visible, and grounded on the pure solid white studio background."
    )

    return (
        f"{header} "
        f"{ultra_hd_zoom_lock} "
        f"{white_background_lock} "
        f"{metal_lock} "
        f"Exact jewelry piece to faithfully reconstruct from all reference angles: {desc or label}. "
        f"{studio_rules}"
    )


def _fallback_jewelry_plan(jewelry_type: str = "auto") -> dict[str, Any]:
    """Deterministic fallback for jewelry prompt if OpenAI API key is unavailable."""
    jt = (jewelry_type or "necklace").strip().lower()
    if jt not in JEWELRY_STAND_RULES_MAP:
        jt = "necklace"

    labels = {
        "necklace": ("22K Yellow Gold Filigree Medallion Necklace", "white_neck_bust", JEWELRY_NECKLACE_STUDIO_RULES),
        "ring": ("Gold Diamond Solitaire Ring with Filigree Side Shank", "slotted_ring_block", JEWELRY_RING_STUDIO_RULES),
        "earrings": ("22K Gold Traditional Jhumka Earrings Pair", "t_bar_earring_stand", JEWELRY_EARRINGS_STUDIO_RULES),
        "bangles": ("22K Gold Handcrafted Kada Bangle", "cylindrical_bangle_bar", JEWELRY_BANGLES_STUDIO_RULES),
    }
    label, stand, rules = labels.get(jt, labels["necklace"])

    jewelry_data = {
        "category": jt,
        "recognized_label": label,
        "metal_type": "22K Warm Yellow Gold",
        "stand_type": stand,
        "optimal_angle": "0-degree frontal" if jt != "ring" else "30-45 degree isometric",
        "jewelry_description": f"Authentic {label} with intricate traditional hand-carved openwork filigree and high-polish luster.",
    }
    return {
        **jewelry_data,
        "prompt": _build_final_jewelry_prompt(jewelry_data),
        "prompt_source": "default_fallback",
    }


def create_multi_angle_composite(images: list[Image.Image], target_size: int = 1024) -> Image.Image:
    """Combine 1 to 4 multi-angle jewelry photos into a clean composite image for Qwen visual conditioning.

    - 1 image: Resized to fit within target_size square.
    - 2 images: Arranged side-by-side horizontally.
    - 3 or 4 images: Arranged in a clean 2x2 grid.
    """
    if not images:
        return Image.new("RGB", (target_size, target_size), (255, 255, 255))
    if len(images) == 1:
        img = images[0].convert("RGB")
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        paste_x = (target_size - img.width) // 2
        paste_y = (target_size - img.height) // 2
        canvas.paste(img, (paste_x, paste_y))
        return canvas

    if len(images) == 2:
        cell_w, cell_h = target_size // 2, target_size
        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        for idx, src in enumerate(images[:2]):
            im = src.convert("RGB")
            im.thumbnail((cell_w - 10, cell_h - 10), Image.Resampling.LANCZOS)
            px = idx * cell_w + (cell_w - im.width) // 2
            py = (cell_h - im.height) // 2
            canvas.paste(im, (px, py))
        return canvas

    # 3 or 4 images: 2x2 grid
    cell_w, cell_h = target_size // 2, target_size // 2
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    positions = [
        (0, 0),
        (cell_w, 0),
        (0, cell_h),
        (cell_w, cell_h),
    ]
    for idx, src in enumerate(images[:4]):
        im = src.convert("RGB")
        im.thumbnail((cell_w - 10, cell_h - 10), Image.Resampling.LANCZOS)
        base_x, base_y = positions[idx]
        px = base_x + (cell_w - im.width) // 2
        py = base_y + (cell_h - im.height) // 2
        canvas.paste(im, (px, py))
    return canvas


def analyze_jewelry_and_build_prompts(
    images: list[Image.Image] | Image.Image,
    jewelry_type: str = "auto",
    custom_prompt: str | None = None,
) -> dict[str, Any]:
    """Analyze 1 to 4 multi-angle photographs of a jewelry piece and build a unified 3D studio prompt.

    Args:
        images: A single PIL Image or a list of 1 to 4 PIL Images (different angles of the same jewelry item).
        jewelry_type: Hint category ('auto', 'necklace', 'ring', 'earrings', 'bangles').
        custom_prompt: Optional user-supplied prompt override.

    Returns:
        Dictionary containing:
        {
            "category": "necklace" | "ring" | "earrings" | "bangles",
            "recognized_label": str,
            "metal_type": str,
            "stand_type": str,
            "optimal_angle": str,
            "jewelry_description": str,
            "prompt": str,
            "prompt_source": str,
            "views_analyzed": int,
        }
    """
    if isinstance(images, Image.Image):
        img_list = [images]
    else:
        img_list = list(images) if images else []

    if not img_list:
        return _fallback_jewelry_plan(jewelry_type)

    if custom_prompt and custom_prompt.strip():
        cat = jewelry_type if jewelry_type != "auto" else "necklace"
        rules = JEWELRY_STAND_RULES_MAP.get(cat, JEWELRY_NECKLACE_STUDIO_RULES)
        return {
            "category": cat,
            "recognized_label": "Custom Luxury Jewelry Piece",
            "metal_type": "Precious Metal",
            "stand_type": "white_neck_bust" if cat == "necklace" else "slotted_ring_block",
            "optimal_angle": "0-degree frontal",
            "jewelry_description": custom_prompt.strip(),
            "prompt": f"{custom_prompt.strip()} {rules}",
            "prompt_source": "custom_prompt",
            "views_analyzed": len(img_list),
        }

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; using default jewelry fallback plan.")
        plan = _fallback_jewelry_plan(jewelry_type)
        plan["views_analyzed"] = len(img_list)
        return plan

    model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    # Encode all uploaded angles
    encoded_images = [_encode_image_for_vision(im) for im in img_list[:4]]

    user_text = (
        f"Analyze the {len(encoded_images)} photograph(s) provided. All photos show the SAME piece of jewelry from different angles "
        "(e.g. top view, side profile, perspective angle, or close-up). "
        "CRITICAL MULTI-ANGLE FUSION: Synthesize all views into ONE unified forensic jewelry description: "
        "1. Identify the jewelry category ('necklace', 'ring', 'earrings', 'bangles'). "
        "2. Exact metal alloy, karat purity, and polish (22K yellow gold, platinum, etc.). "
        "3. Top / Crown details: center motif, gemstone, setting style. "
        "4. Side profile & Shank details: side filigree, openwork jali, engravings, or accent stones visible in the side photos. "
        "5. Select the appropriate luxury e-commerce catalog stand: "
        "   - Necklaces -> 'white_neck_bust' "
        "   - Rings -> 'slotted_ring_block' (30-45 degree hero angle showing both top crown and side shank) "
        "   - Earrings -> 't_bar_earring_stand' "
        "   - Bangles -> 'cylindrical_bangle_bar' "
        "Strictly preserve the authentic precious metal and craftsmanship without hallucinating different colors."
    )

    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for b64_str in encoded_images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_str}",
                "detail": "high",
            },
        })

    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": 0.1,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_JEWELRY_VISION_PROMPT},
            {"role": "user", "content": user_content},
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
        item = parsed.get("jewelry_item") or parsed

        category = str(item.get("category", "necklace")).strip().lower()
        if category not in JEWELRY_STAND_RULES_MAP:
            category = "necklace"
        label = str(item.get("recognized_label", "Fine Gold Jewelry")).strip()
        metal = str(item.get("metal_type", "22K Warm Yellow Gold")).strip()
        stand = str(item.get("stand_type", "white_neck_bust")).strip()
        angle = str(item.get("optimal_angle", "0-degree frontal")).strip()
        desc = str(item.get("jewelry_description", label)).strip()

        jewelry_data = {
            "category": category,
            "recognized_label": label,
            "metal_type": metal,
            "stand_type": stand,
            "optimal_angle": angle,
            "jewelry_description": desc,
            "views_analyzed": len(encoded_images),
        }
        jewelry_data["prompt"] = _build_final_jewelry_prompt(jewelry_data)
        jewelry_data["prompt_source"] = f"openai:{model_name}"
        return jewelry_data

    except Exception as exc:
        logger.warning("ChatGPT Vision jewelry multi-angle analysis failed (%s); using fallback.", exc)
        plan = _fallback_jewelry_plan(jewelry_type)
        plan["views_analyzed"] = len(img_list)
        return plan
