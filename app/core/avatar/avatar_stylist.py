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

AVATAR_VISION_PROMPT = """You are a master digital character designer and art director analyzing ONE user photo to transform it into a premium stylized 2D digital vector art avatar profile picture.

Inspect the person's photo with extreme forensic precision:
1. `gender`: "female" | "male" | "unspecified"
2. `expression`: Exact facial expression (e.g., "warm genuine smile showing upper teeth", "subtle confident closed-lip smile", "neutral composed gaze").
3. `hair_details`: Exact hair style, texture, volume, parting, bangs/fringe, length, and color (e.g. "thick voluminous textured dark hair with styled fringe falling naturally over forehead", "long silky straight-wavy dark hair parted in center cascading over shoulders").
4. `eyewear`: If eyeglasses are present, describe exact frame shape, frame material/color (e.g., "thin square wireframe silver spectacles with clear transparent lenses"), or null if NO glasses.
5. `facial_hair`: Describe mustache, stubble, or beard (e.g. "neat dark mustache and light groomed chin stubble"), or null if clean shaven.
6. `upper_garment`: Exact description of upper clothing (e.g. "dark navy floral printed top with cream-yellow flowers and square neckline", "sharp tailored black blazer, crisp white collared dress shirt, vibrant red necktie, and black pocket square").
7. `distinctive_features`: Small bindi, forehead marks, earrings, necklace, or null.
8. `recommended_halo_color`: Best complementary pastel halo color from ("soft_blue", "periwinkle", "lavender", "blush_pink", "mint_green", "warm_amber").
9. `recommended_doodle`: "crown" (for regal/confident/masculine), "heart" (for charming/sweet/feminine), or "sparks".
10. `face_box_2d`: [ymin, xmin, ymax, xmax] integers scaled 0 to 1000 tightly around the face.

Return STRICTLY valid JSON:
{
  "gender": "female",
  "expression": "warm radiant smile showing teeth",
  "hair_details": "...",
  "eyewear": null,
  "facial_hair": null,
  "upper_garment": "...",
  "distinctive_features": "small dark bindi on center of forehead",
  "recommended_halo_color": "periwinkle",
  "recommended_doodle": "heart",
  "face_box_2d": [120, 280, 520, 720]
}"""


def _build_fallback_identity(avatar_img: Image.Image) -> dict[str, Any]:
    """Deterministic fallback identity analysis when OpenAI API is unconfigured."""
    w, h = avatar_img.size
    return {
        "gender": "unspecified",
        "expression": "natural warm smile",
        "hair_details": "naturally styled hair with rich textured strands and authentic volume",
        "eyewear": None,
        "facial_hair": None,
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
                "Analyze this portrait to create an exact stylized 2D digital vector art avatar.",
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

        return {
            "gender": gender,
            "expression": str(parsed.get("expression") or "warm friendly smile").strip(),
            "hair_details": str(parsed.get("hair_details") or "authentically styled hair").strip(),
            "eyewear": str(parsed.get("eyewear")).strip() if parsed.get("eyewear") else None,
            "facial_hair": str(parsed.get("facial_hair")).strip() if parsed.get("facial_hair") else None,
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

    # Style descriptor
    style_norm = (style or "vector_art").strip().lower()
    if style_norm == "digital_illustration":
        style_direction = (
            "Premium modern digital art character portrait illustration, rich stylized brushwork, "
            "vibrant semi-realistic digital painting, clean outlines, polished studio lighting."
        )
    elif style_norm == "anime_portrait":
        style_direction = (
            "Masterpiece modern anime character portrait, crisp expressive line work, clean cel-shading, "
            "vibrant anime palette, polished manhwa / webtoon hero aesthetic."
        )
    else:  # default: vector_art
        style_direction = (
            "Masterpiece 2D modern digital vector art avatar illustration, crisp clean vector line art, "
            "smooth luminous cel-shading, vibrant stylized colors, polished professional avatar profile card aesthetic."
        )

    # Gender noun
    gender = avatar_info.get("gender", "unspecified")
    gender_noun = "woman" if gender == "female" else ("man" if gender == "male" else "person")

    # Eyewear clause
    eyewear = avatar_info.get("eyewear")
    if eyewear and "no glass" not in eyewear.lower():
        glasses_clause = (
            f"MANDATORY EYEWEAR LOCK: The character MUST wear identical eyeglasses exactly matching <image1> "
            f"({eyewear}). Reproduce the exact frame geometry, thin wireframe borders, and transparent clear lenses."
        )
    else:
        glasses_clause = "EYEWEAR: No eyeglasses worn on the eyes in <image1>; do not add any unprompted glasses or sunglasses."

    # Facial hair clause
    facial_hair = avatar_info.get("facial_hair")
    if facial_hair:
        beard_clause = f"FACIAL HAIR LOCK: Render the exact {facial_hair} with clean stylized vector strokes."
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

    sections = [
        (
            f"STYLIZED 2D AVATAR PROFILE ILLUSTRATION (<image1> EXACT PERSON IDENTITY LOCK): "
            f"Transform the real {gender_noun} in <image1> into an iconic, charming 2D digital vector art avatar illustration. "
            f"Style: {style_direction} "
            "EXACT FACIAL LIKENESS MANDATE: The illustrated character MUST be unmistakably recognizable as the exact same person from <image1>. "
            f"Preserve their exact facial proportions, authentic eye shape, irises, eyebrow arch, nose contour, and genuine expression ({avatar_info.get('expression', 'warm smile')}). "
            f"Preserve their exact hairstyle: {avatar_info.get('hair_details', 'authentically styled hair')}, "
            "rendered with detailed graphic vector strokes, smooth highlights, and stylized hair volume. "
            f"{glasses_clause} "
            f"{beard_clause} "
            f"{distinctive_clause} "
            "STRICT PROPORTION & NATURAL CHEEK LOCK (ABSOLUTELY NO ARTIFICIAL FACIAL SLIMMING): "
            "Preserve the authentic face width, cheek fullness, and natural jaw contour of <image1>. "
            "DO NOT artificially slim the face, DO NOT narrow the jaw, and DO NOT exaggerate into an unnatural V-line."
        ),
        (
            f"UPPER OUTFIT VECTOR STYLING: "
            f"Illustrate a faithful stylized vector rendition of the upper garment worn in <image1>: {outfit}. "
            "Preserve the exact garment colors, print patterns (e.g. floral blossoms, stripes, or tailored lapels and tie), "
            "crisp collar lines, and tailored silhouette with clean vector outlines and smooth shading."
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
            "Crisp 1024x1024 digital artwork, sharp vector linework, vibrant luminous palette, smooth cel-shading, perfect graphic balance, "
            "appealing modern social media profile picture (PFP). "
            "STRICT NEGATIVE EXCLUSIONS: ABSOLUTELY NO photorealistic skin pores, NO waxy 3D plastic mannequin, NO uncanny CGI doll, "
            "NO cluttered photographic background, NO photographic noise, NO distorted fingers, NO watermark, NO logo overlay."
        ),
    ]

    if custom_prompt and custom_prompt.strip():
        sections.append(f"Additional creative direction: {custom_prompt.strip()}")

    full_prompt = " ".join(sections)
    return full_prompt, resolved_halo_name, resolved_doodle_name
