"""Styling and prompt engineering engine for AI Stylized Avatar Profile Pictures.

Specialized for generating modern 2D digital vector art avatar profile cards
with circular pastel halo backdrops and cute minimalist hand-drawn doodle accents.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# Standard pastel halo color palette
HALO_PALETTE: dict[str, dict[str, str]] = {
    "soft_blue": {
        "name": "soft_blue",
        "label": "soft pastel sky blue",
        "hex": "#D4E6FA",
        "desc": "soft luminous pastel sky blue circular disc badge (#D4E6FA)",
    },
    "periwinkle": {
        "name": "periwinkle",
        "label": "soft pastel periwinkle",
        "hex": "#DFE3F8",
        "desc": "gentle pastel periwinkle circular disc badge (#DFE3F8)",
    },
    "lavender": {
        "name": "lavender",
        "label": "soft pastel lavender",
        "hex": "#E7DFF7",
        "desc": "charming soft pastel lavender circular disc badge (#E7DFF7)",
    },
    "blush_pink": {
        "name": "blush_pink",
        "label": "soft pastel blush pink",
        "hex": "#FCDFE4",
        "desc": "delicate soft pastel blush pink circular disc badge (#FCDFE4)",
    },
    "mint_green": {
        "name": "mint_green",
        "label": "soft pastel mint green",
        "hex": "#D8F1E5",
        "desc": "fresh soft pastel sage mint circular disc badge (#D8F1E5)",
    },
    "warm_amber": {
        "name": "warm_amber",
        "label": "soft pastel buttercream amber",
        "hex": "#FCECD7",
        "desc": "warm soft pastel buttercream amber circular disc badge (#FCECD7)",
    },
    "soft_grey": {
        "name": "soft_grey",
        "label": "soft pastel dove grey",
        "hex": "#E8EAED",
        "desc": "neutral soft pastel dove grey circular disc badge (#E8EAED)",
    },
}

DOODLE_STYLES: dict[str, str] = {
    "crown": (
        "a cute minimalist hand-drawn outlined 3-point king/queen crown doodle floating playfully above the hair "
        "near the top-left, accompanied by two delicate radiating action dashes"
    ),
    "heart": (
        "a cute minimalist hand-drawn outlined heart doodle floating playfully beside the character on the circular badge, "
        "accompanied by two delicate accent dashes"
    ),
    "sparks": (
        "playful minimalist hand-drawn sparkle star doodles and radiating energetic accent dashes around the circular badge"
    ),
    "stars": (
        "minimalist hand-drawn twinkling 4-point star doodles and subtle accent lines"
    ),
    "none": "",
}

AVATAR_VISION_PROMPT = """You are a master digital portrait painter and character artist analyzing ONE user photo to transform it into a premium semi-realistic digital art avatar profile picture (as seen in high-end character illustration lookbooks).

Inspect the person's photo with extreme forensic precision:
1. `gender`: "female" | "male" | "unspecified"
2. `expression`: Exact facial emotion and mouth posture (e.g., "confident subtle smile with lips closed together", "warm broad smile with upper teeth visible", "calm neutral composed closed mouth").
3. `teeth_visible`: boolean.
   - CRITICAL MOUTH & TEETH AUDIT: Inspect the person's mouth and lips with forensic precision!
   - true ONLY if teeth are genuinely parted, exposed, and visibly seen in the photograph!
   - false if the mouth is closed, lips are touching/shut, or if it is a closed-mouth/closed-lip smile with NO teeth visible!
   - ABSOLUTELY NEVER hallucinate or assume teeth if the lips are closed!
4. `hair_details`: Exact hair style, texture, volume, parting, bangs/fringe, color, and maturity (e.g., "cropped salt-and-pepper hair with receding hairline, natural volume on sides and crown, realistic grey and black strands", "thick voluminous textured dark hair with styled fringe").
5. `eyewear`: CRITICAL WORN-ON-FACE RULE: Inspect ONLY the eyes.
   - If eyeglasses or sunglasses are CURRENTLY WORN DIRECTLY OVER THE EYES on the bridge of the nose, describe the frame (e.g., "thin rectangular wireframe silver spectacles with clear lenses").
   - If the person is NOT wearing glasses on their eyes (even if reading glasses are hanging from their shirt collar or pocket), you MUST set `eyewear: null`! Do NOT report eyewear if glasses are only hanging on the shirt!
   - Note `glasses_hanging_on_shirt`: true if glasses are clipped/hanging on shirt collar/button.
6. `facial_hair`: Describe mustache, stubble, or beard (e.g., "neat dark mustache and light groomed chin stubble"), or "clean shaven".
7. `skin_tone_and_complexion`: Exact natural skin tone coordinates (e.g., "natural medium wheatish Indian complexion with warm undertones", "fair ivory with rosy undertone", "warm bronze tan").
8. `upper_garment`: Exact description of upper clothing (e.g., "light sky-blue and white botanical foliage leaf print button-down shirt with spread collar", "dark navy suit jacket with white shirt and red tie").
9. `distinctive_features`: Small bindi, forehead lines, laugh lines, earrings, necklace, or null.
10. `recommended_halo_color`: Best complementary pastel halo color from ("soft_blue", "periwinkle", "lavender", "blush_pink", "mint_green", "warm_amber").
11. `recommended_doodle`: "crown" (for regal/confident/men), "heart" (for warm/charming/women), or "sparks".
12. `face_box_2d`: [ymin, xmin, ymax, xmax] integers scaled 0 to 1000 tightly around the face and hair.

Return STRICTLY valid JSON:
{
  "gender": "male",
  "expression": "subtle confident smile with closed lips",
  "teeth_visible": false,
  "hair_details": "...",
  "eyewear": null,
  "glasses_hanging_on_shirt": false,
  "facial_hair": "neat dark mustache and light groomed chin stubble",
  "skin_tone_and_complexion": "natural medium wheatish Indian skin tone",
  "upper_garment": "...",
  "distinctive_features": null,
  "recommended_halo_color": "soft_blue",
  "recommended_doodle": "crown",
  "face_box_2d": [120, 280, 520, 720]
}"""


def _build_fallback_identity(avatar_img: Image.Image) -> dict[str, Any]:
    """Deterministic fallback identity analysis when OpenAI API is unconfigured."""
    w, h = avatar_img.size
    return {
        "gender": "unspecified",
        "expression": "natural subtle smile with closed lips",
        "teeth_visible": False,
        "hair_details": "naturally styled hair with rich textured strands and authentic volume",
        "eyewear": None,
        "glasses_hanging_on_shirt": False,
        "facial_hair": None,
        "skin_tone_and_complexion": "natural authentic skin tone",
        "upper_garment": "stylish upper garment faithfully matching the uploaded photo",
        "distinctive_features": None,
        "recommended_halo_color": "soft_blue",
        "recommended_doodle": "crown",
        "face_box_2d": [int(100 * h / max(1, h)), int(200 * w / max(1, w)), int(550 * h / max(1, h)), int(800 * w / max(1, w))],
    }


def analyze_avatar_identity(
    avatar_img: Image.Image,
    api_key: str | None = None,
    model_name: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
) -> dict[str, Any]:
    """Perform visual inspection of user portrait to guide stylized avatar generation."""
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key or key == "none":
        logger.info("OPENAI_API_KEY not configured; using deterministic avatar analysis fallback.")
        return _build_fallback_identity(avatar_img)

    try:
        from app.core.flow_b.tryon_stylist import _call_openai_json, _vision_message_for_image

        parsed = _call_openai_json(
            AVATAR_VISION_PROMPT,
            _vision_message_for_image(
                "Analyze this portrait to create an exact semi-realistic digital art avatar portrait. Audit mouth/teeth meticulously.",
                avatar_img,
                max_edge=1024,
            ),
            key,
            model_name,
            base_url,
            max_tokens=600,
        )

        gender = str(parsed.get("gender") or "unspecified").strip().lower()
        if gender not in ("female", "male"):
            gender = "unspecified"

        halo = str(parsed.get("recommended_halo_color") or "soft_blue").strip().lower()
        if halo not in HALO_PALETTE:
            halo = "soft_blue"

        doodle = str(parsed.get("recommended_doodle") or ("heart" if gender == "female" else "crown")).strip().lower()
        if doodle not in DOODLE_STYLES:
            doodle = "heart" if gender == "female" else "crown"

        raw_eyewear = str(parsed.get("eyewear")).strip() if parsed.get("eyewear") else None
        glasses_hanging = bool(parsed.get("glasses_hanging_on_shirt", False))
        if raw_eyewear:
            lower_ew = raw_eyewear.lower()
            if any(k in lower_ew for k in ("shirt", "collar", "hanging", "pocket", "chest", "not on face", "not worn")):
                raw_eyewear = None
                glasses_hanging = True

        raw_teeth = parsed.get("teeth_visible")
        raw_expression = str(parsed.get("expression") or "natural subtle smile").strip()
        lower_expr = raw_expression.lower()

        if raw_teeth is not None:
            teeth_vis = bool(raw_teeth)
        else:
            teeth_vis = "teeth" in lower_expr and not any(k in lower_expr for k in ("no teeth", "zero teeth", "without teeth", "closed", "shut"))

        if not teeth_vis:
            clean_expr = re.sub(r'\b(showing|with|revealing)?\s*(upper\s*)?teeth\b', '', raw_expression, flags=re.IGNORECASE).strip()
            clean_expr = re.sub(r'\s+', ' ', clean_expr).strip(', -')
            if not any(k in clean_expr.lower() for k in ("closed", "lips together", "mouth shut")):
                clean_expr = f"{clean_expr} with closed lips (zero teeth visible)" if clean_expr else "subtle smile with closed lips"
            raw_expression = clean_expr
        else:
            if "teeth" not in raw_expression.lower():
                raw_expression = f"{raw_expression} showing upper teeth"

        return {
            "gender": gender,
            "expression": raw_expression,
            "teeth_visible": teeth_vis,
            "hair_details": str(parsed.get("hair_details") or "authentically styled hair").strip(),
            "eyewear": raw_eyewear,
            "glasses_hanging_on_shirt": glasses_hanging,
            "facial_hair": str(parsed.get("facial_hair")).strip() if parsed.get("facial_hair") else None,
            "skin_tone_and_complexion": str(parsed.get("skin_tone_and_complexion") or "natural complexion matching reference photo").strip(),
            "upper_garment": str(parsed.get("upper_garment") or "upper clothing matching reference photo").strip(),
            "distinctive_features": str(parsed.get("distinctive_features")).strip() if parsed.get("distinctive_features") else None,
            "recommended_halo_color": halo,
            "recommended_doodle": doodle,
            "face_box_2d": parsed.get("face_box_2d"),
            "source": f"openai:{model_name}",
        }
    except Exception as exc:
        logger.warning("Avatar Vision analysis failed (%s); using fallback.", exc)
        return _build_fallback_identity(avatar_img)


def resolve_halo_and_doodle(
    avatar_info: dict[str, Any],
    requested_halo: str = "auto",
    requested_doodle: str = "auto",
) -> tuple[dict[str, str], str]:
    """Resolve halo color palette entry and doodle description."""
    halo_key = (requested_halo or "auto").strip().lower()
    if halo_key == "auto":
        halo_key = avatar_info.get("recommended_halo_color", "soft_blue")
    halo_entry = HALO_PALETTE.get(halo_key, HALO_PALETTE["soft_blue"])

    doodle_key = (requested_doodle or "auto").strip().lower()
    if doodle_key == "auto":
        gender = avatar_info.get("gender", "unspecified")
        doodle_key = avatar_info.get("recommended_doodle") or ("heart" if gender == "female" else "crown")
    doodle_desc = DOODLE_STYLES.get(doodle_key, DOODLE_STYLES["crown"])

    return halo_entry, doodle_desc


def build_avatar_stylized_prompt(
    avatar_info: dict[str, Any],
    style: str = "vector_art",
    halo_color: str = "auto",
    doodle_accent: str = "auto",
    custom_prompt: str | None = None,
) -> tuple[str, str, str]:
    """Construct the specialized Qwen prompt for 2D stylized avatar profile illustration.
    
    Returns (prompt, resolved_halo_name, resolved_doodle_name).
    """
    halo_entry, doodle_desc = resolve_halo_and_doodle(avatar_info, halo_color, doodle_accent)
    resolved_halo_name = halo_entry["name"]

    doodle_key = (doodle_accent or "auto").strip().lower()
    if doodle_key == "auto":
        gender = avatar_info.get("gender", "unspecified")
        doodle_key = avatar_info.get("recommended_doodle") or ("heart" if gender == "female" else "crown")
    resolved_doodle_name = doodle_key

    # Style descriptor - SEMI-REALISTIC DIGITAL ART PORTRAIT (matching high-end character illustration)
    style_norm = (style or "vector_art").strip().lower()
    if style_norm == "digital_illustration":
        style_direction = (
            "Premium semi-realistic digital art character portrait painting, rich digital brushwork, "
            "realistic anatomical lighting and three-dimensional facial volume, soft skin tones, "
            "delicate fine outlines, polished character concept art aesthetic."
        )
    elif style_norm == "anime_portrait":
        style_direction = (
            "Masterpiece high-end webtoon/manhwa hero digital character portrait, semi-realistic anime aesthetics, "
            "expressive realistic facial bone structure, delicate line art, smooth realistic skin shading."
        )
    else:  # default: vector_art / digital art portrait
        style_direction = (
            "Masterpiece semi-realistic stylized digital portrait illustration, high-end digital character art, "
            "smooth dimensional facial shading with soft volumetric lighting, delicate micro-fine contours (NOT heavy cartoon lines), "
            "natural skin tones with lifelike depth, polished modern avatar profile card aesthetic."
        )

    # Gender noun
    gender = avatar_info.get("gender", "unspecified")
    gender_noun = "woman" if gender == "female" else ("man" if gender == "male" else "person")

    # Eyewear handling - STRICT WORN-ON-FACE VERIFICATION
    eyewear = avatar_info.get("eyewear")
    glasses_on_shirt = bool(avatar_info.get("glasses_hanging_on_shirt", False))
    if eyewear:
        lower_ew = eyewear.lower()
        if any(k in lower_ew for k in ("shirt", "collar", "hanging", "pocket", "chest", "not on face", "not worn")):
            eyewear = None
            glasses_on_shirt = True

    if eyewear and "no glass" not in eyewear.lower():
        glasses_clause = (
            f"MANDATORY EYEWEAR LOCK: The character MUST wear identical eyeglasses exactly matching <image1> "
            f"({eyewear}) directly over their eyes. Reproduce the exact frame geometry, thin borders, and clear transparent lenses."
        )
    else:
        hanging_note = " (Reading glasses hanging from the shirt collar should be illustrated resting naturally on the shirt placket, but the eyes, eyelids, and nose bridge MUST remain 100% free of glasses)." if glasses_on_shirt else ""
        glasses_clause = (
            f"STRICT EYEWEAR BAN (ZERO GLASSES ON FACE): The person in <image1> is NOT wearing glasses on their eyes or face. "
            f"CRITICAL: ABSOLUTELY DO NOT put, draw, or add any spectacles, eyeglasses, or sunglasses over the eyes or face! "
            f"Keep the eyes, eyelids, and nose bridge 100% natural, visible, and free of glasses{hanging_note}."
        )

    # Facial hair clause
    facial_hair = avatar_info.get("facial_hair")
    if facial_hair and "clean" not in facial_hair.lower():
        beard_clause = f"FACIAL HAIR LOCK: Render the exact {facial_hair} with clean, subtle, realistic digital painting strokes."
    else:
        beard_clause = "FACIAL HAIR: Clean-shaven face matching <image1> without unprompted beard or stubble."

    # Distinctive marks
    distinctive = avatar_info.get("distinctive_features")
    distinctive_clause = f"DISTINCTIVE MARKS: Faithfully include {distinctive} from <image1>." if distinctive else ""

    # Upper outfit
    outfit = avatar_info.get("upper_garment") or "stylish upper garment from <image1>"

    # Doodle clause
    if doodle_desc:
        doodle_clause = (
            f"MINIMALIST HAND-DRAWN DOODLE ACCENTS: Adorn the avatar card with {doodle_desc}, "
            "drawn in charming, clean, delicate minimalist vector lines."
        )
    else:
        doodle_clause = "DOODLE ACCENTS: Minimalist, clean graphic styling."

    skin_desc = avatar_info.get("skin_tone_and_complexion") or "natural authentic skin tone matching <image1>"
    hair_desc = avatar_info.get("hair_details") or "authentically styled hair"

    # Mouth posture and teeth visibility lock
    raw_teeth = avatar_info.get("teeth_visible")
    expr_text = (avatar_info.get("expression") or "").lower()
    if raw_teeth is not None:
        teeth_vis = bool(raw_teeth)
    else:
        teeth_vis = "teeth" in expr_text and not any(k in expr_text for k in ("no teeth", "zero teeth", "without teeth", "closed", "shut"))

    if teeth_vis:
        mouth_clause = (
            "MOUTH & TEETH: The person in <image1> has an open smile showing teeth. "
            "Faithfully render a natural open smile with realistically visible upper teeth matching <image1>."
        )
        negative_teeth = ""
    else:
        mouth_clause = (
            "STRICT CLOSED-LIP MOUTH LOCK (ABSOLUTELY ZERO TEETH VISIBLE): "
            "The person in <image1> has their lips completely closed together with ZERO teeth showing. "
            "CRITICAL MANDATE: The character's mouth MUST BE CLOSED with lips resting naturally together! "
            "ABSOLUTELY DO NOT open the mouth, DO NOT expose teeth, and DO NOT draw any visible teeth or grinning mouth aperture! "
            "Faithfully replicate their exact closed-lip subtle smile with mouth shut matching <image1>."
        )
        negative_teeth = "NO exposed teeth, NO open grinning mouth, NO visible teeth, "

    sections = [
        (
            f"STYLIZED SEMI-REALISTIC DIGITAL AVATAR PORTRAIT (<image1> EXACT PERSON IDENTITY LOCK): "
            f"Transform the real {gender_noun} in <image1> into an iconic, charming semi-realistic digital art portrait illustration. "
            f"Style: {style_direction} "
            "EXACT FACIAL LIKENESS MANDATE: The illustrated character MUST be unmistakably recognizable as the exact same person from <image1>. "
            f"Preserve their exact facial bone structure, authentic eye shape, irises, eyebrow arch, nose contour, and genuine expression ({avatar_info.get('expression', 'warm smile')}). "
            f"{mouth_clause} "
            f"AUTHENTIC HAIR & MATURITY: Preserve their exact hairline, styled hair volume, texture, and natural color from <image1> ({hair_desc}). "
            "If the person has salt-and-pepper, silver, or grey hair, faithfully render the natural silver-grey and dark strands with fine digital strokes and soft highlights, "
            "preserving their dignified, natural age and genuine character. "
            f"AUTHENTIC SKIN COMPLEXION & ZERO CARICATURE ORANGE LOCK: "
            f"Preserve the EXACT natural skin tone and undertones from <image1>: {skin_desc}. "
            "The skin must look lifelike and natural with soft, smooth three-dimensional digital painting highlights — "
            "CRITICAL: ABSOLUTELY DO NOT make the skin cartoonish yellow or oversaturated orange! "
            "Keep the authentic wheatish/olive/fair complexion 100% faithful to the real person in <image1>. "
            f"{glasses_clause} "
            f"{beard_clause} "
            f"{distinctive_clause} "
            "STRICT PROPORTION & NATURAL CHEEK LOCK (ABSOLUTELY NO ARTIFICIAL FACIAL SLIMMING OR CARICATURE DISTORTION): "
            "Preserve the authentic face width, cheek fullness, and natural jaw contour of <image1>. "
            "DO NOT artificially slim the face, DO NOT narrow the jaw, and DO NOT exaggerate features into an unnatural comic caricature."
        ),
        (
            f"UPPER OUTFIT STYLING: "
            f"Illustrate a faithful stylized digital art rendition of the upper garment worn in <image1>: {outfit}. "
            "Preserve the exact garment colors, print patterns (e.g. botanical floral foliage, stripes, or tailored lapels), "
            "crisp collar lines, and tailored silhouette with clean subtle outlines and smooth dimensional shading."
        ),
        (
            f"CIRCULAR HALO BADGE & COMPOSITION: "
            "Centered bust-up portrait (head, hair volume, neck, shoulders, and upper chest) cleanly centered in the square canvas. "
            f"Directly behind the character's head and upper shoulders is a prominent, clean, large circular disc badge rendered in {halo_entry['desc']}. "
            f"{doodle_clause} "
            "OUTER CANVAS: Pristine, spotless, solid uniform ivory/off-white background canvas (#FAF8F5) surrounding the circular badge."
        ),
        (
            "QUALITY & GRAPHIC RENDERING STANDARDS: "
            "Crisp 1024x1024 semi-realistic digital artwork, ultra-delicate fine contours, smooth painterly shading, perfect graphic balance, "
            "appealing modern social media profile picture (PFP). "
            f"STRICT NEGATIVE EXCLUSIONS: {negative_teeth}ABSOLUTELY NO flat 2D cartoon, NO thick black comic book outlines, NO caricature distortion, "
            "NO exaggerated cartoon eyes or teeth, NO Simpsons-style flat yellow/orange skin, NO cheap vector clipart, "
            "NO photorealistic skin pores, NO waxy 3D plastic mannequin, NO uncanny CGI doll, "
            "NO cluttered photographic background, NO photographic noise, NO distorted fingers, NO watermark, NO logo overlay."
        ),
    ]

    if custom_prompt and custom_prompt.strip():
        sections.append(f"Additional creative direction: {custom_prompt.strip()}")

    full_prompt = " ".join(sections)
    return full_prompt, resolved_halo_name, resolved_doodle_name
