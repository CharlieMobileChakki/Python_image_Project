"""Styling and prompt engineering engine for AI Stylized Avatar Profile Pictures.

Specialized for generating modern 2D digital vector art avatar profile cards
with circular pastel halo backdrops, vibrant luminous skin tones, and cute minimalist doodles.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# Premium pastel halo color palette (vibrant, clean, flattering)
HALO_PALETTE: dict[str, dict[str, str]] = {
    "soft_blue": {
        "name": "soft_blue",
        "label": "luminous pastel sky blue",
        "hex": "#C6DDFB",
        "desc": "large, prominent luminous soft pastel sky blue circular disc badge (#C6DDFB) framing the head and upper torso",
    },
    "periwinkle": {
        "name": "periwinkle",
        "label": "soft pastel periwinkle",
        "hex": "#CCD8FA",
        "desc": "large, prominent soft pastel periwinkle-blue circular disc badge (#CCD8FA) framing the head and upper torso",
    },
    "lavender": {
        "name": "lavender",
        "label": "soft pastel lavender",
        "hex": "#E6DDF8",
        "desc": "large, prominent soft pastel lavender circular disc badge (#E6DDF8) framing the head and upper torso",
    },
    "blush_pink": {
        "name": "blush_pink",
        "label": "soft pastel blush pink",
        "hex": "#FCDDE4",
        "desc": "large, prominent soft pastel blush pink circular disc badge (#FCDDE4) framing the head and upper torso",
    },
    "mint_green": {
        "name": "mint_green",
        "label": "soft pastel mint green",
        "hex": "#D4F0E2",
        "desc": "large, prominent soft pastel sage mint circular disc badge (#D4F0E2) framing the head and upper torso",
    },
    "warm_amber": {
        "name": "warm_amber",
        "label": "soft pastel buttercream amber",
        "hex": "#FCEBD5",
        "desc": "large, prominent soft pastel buttercream amber circular disc badge (#FCEBD5) framing the head and upper torso",
    },
    "soft_grey": {
        "name": "soft_grey",
        "label": "soft pastel dove grey",
        "hex": "#E8EAEF",
        "desc": "large, prominent soft pastel dove grey circular disc badge (#E8EAEF) framing the head and upper torso",
    },
}

DOODLE_STYLES: dict[str, str] = {
    "crown": (
        "an iconic minimalist hand-drawn outlined 3-point king crown doodle with 3 tiny circular dots at the crown tips "
        "floating playfully above the hair on the upper-left, accompanied by two crisp delicate diagonal action dashes on the right"
    ),
    "heart": (
        "a cute minimalist hand-drawn outlined heart doodle (♡) floating playfully beside the character on the circular badge, "
        "accompanied by two delicate radiating accent dashes"
    ),
    "sparks": (
        "playful minimalist hand-drawn sparkle star doodles and radiating energetic accent dashes around the circular badge"
    ),
    "stars": (
        "minimalist hand-drawn twinkling 4-point star doodles and subtle delicate accent lines"
    ),
    "none": "",
}

AVATAR_VISION_PROMPT = """You are a master digital character designer and art director analyzing ONE user photo to transform it into a premium stylized 2D digital vector art avatar profile picture.

Inspect the person's photo with extreme forensic precision:
1. `gender`: "female" | "male" | "unspecified"
2. `expression`: Exact facial expression (e.g., "warm genuine smile showing upper teeth", "subtle confident closed-lip smile", "friendly composed gaze").
3. `skin_tone`: Exact natural skin tone and undertone (e.g. "radiant warm golden-peachy skin", "fair luminous skin with subtle rosy undertones", "rich warm honey-bronze").
4. `hair_details`: Exact hair style, texture, volume, parting, fringe/bangs, and flow (e.g. "thick voluminous textured dark hair with styled fringe swept forward over forehead", "long silky straight-wavy dark hair parted in middle cascading past shoulders").
5. `eyewear`: If eyeglasses are present, describe exact frame shape, frame material/color (e.g., "sleek thin rectangular wireframe silver/black spectacles with crystal-clear non-reflective lenses"), or null if NO glasses.
6. `facial_hair`: Describe mustache, stubble, or beard (e.g. "neat manicured dark mustache and light groomed jawline stubble"), or null if clean shaven.
7. `upper_garment`: Exact description of upper clothing (e.g. "dark navy floral printed top with cream-yellow flowers and square neckline", "sharp tailored black blazer, crisp white collared dress shirt, vibrant crimson red necktie, and black pocket square").
8. `distinctive_features`: Small bindi, forehead marks, earrings, necklace, or null.
9. `recommended_halo_color`: Default to "soft_blue" or "periwinkle" unless garment strongly demands another color.
10. `recommended_doodle`: "crown" (for men/confident styling), "heart" (for women/charming styling), or "sparks".
11. `face_box_2d`: [ymin, xmin, ymax, xmax] integers scaled 0 to 1000 tightly around the face.

Return STRICTLY valid JSON:
{
  "gender": "male",
  "expression": "confident subtle warm smile",
  "skin_tone": "radiant warm golden-peachy skin with healthy glow",
  "hair_details": "thick voluminous dark hair with textured fringe styled over forehead",
  "eyewear": "sleek thin rectangular wireframe spectacles with clear transparent lenses",
  "facial_hair": "neat dark mustache and light groomed chin stubble",
  "upper_garment": "tailored black suit jacket, crisp white collared shirt, vibrant red necktie, and black pocket square",
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
        "expression": "natural warm friendly smile",
        "skin_tone": "radiant healthy warm peachy-golden complexion with soft natural glow",
        "hair_details": "naturally styled dark hair with rich textured layered strands and authentic volume",
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
            "skin_tone": str(parsed.get("skin_tone") or "radiant warm golden-peachy skin").strip(),
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
    
    Guarantees:
    - Radiant peachy-golden luminous skin tone with subtle rosy warmth (NO dull olive/grey/yellow).
    - Bright alert captivating eyes with glossy catchlights.
    - Sleek, symmetrical thin wireframe eyeglasses (if present) with crystal-clear lenses.
    - Volumetric layered dark hair with rich highlights.
    - Crisp vector tailored clothing (suit, collar, vibrant red tie, floral dress).
    - Large circular pastel halo badge framed by crisp minimalist doodles.
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
            "Masterpiece 2D modern digital vector art avatar illustration. Premium polished editorial character portrait, "
            "razor-sharp clean vector line art, luminous smooth cel-shading, vibrant radiant digital color palette, "
            "high-end social media profile picture (PFP) aesthetic."
        )

    # Gender noun
    gender = avatar_info.get("gender", "unspecified")
    gender_noun = "man" if gender == "male" else ("woman" if gender == "female" else "person")

    # Eyewear clause: enforce sleek thin symmetrical wireframe
    eyewear = avatar_info.get("eyewear")
    if eyewear and "no glass" not in eyewear.lower():
        glasses_clause = (
            f"MANDATORY EYEWEAR LOCK: The character MUST wear identical eyeglasses exactly matching <image1> ({eyewear}). "
            "The glasses MUST be rendered with sleek, ultra-thin, perfectly symmetrical metallic wireframe rims, "
            "a crisp straight horizontal bridge, and crystal-clear transparent anti-reflective lenses showing sharp alert eyes behind them. "
            "ABSOLUTELY NO thick wobbly frames, and NO warped or bent glasses."
        )
    else:
        glasses_clause = "EYEWEAR: No eyeglasses worn on the eyes in <image1>; do not add any unprompted glasses or sunglasses."

    # Facial hair clause
    facial_hair = avatar_info.get("facial_hair")
    if facial_hair:
        beard_clause = (
            f"FACIAL HAIR LOCK: Render the exact {facial_hair} with clean, sharp, manicured vector strokes "
            "and natural hair texture along the lip and jawline."
        )
    else:
        beard_clause = "FACIAL HAIR: Clean-shaven face matching <image1> with smooth jawline and zero unprompted stubble."

    # Distinctive marks
    distinctive = avatar_info.get("distinctive_features")
    distinctive_clause = f"DISTINCTIVE MARKS: Faithfully include {distinctive} from <image1>." if distinctive else ""

    # Upper outfit
    outfit = avatar_info.get("upper_garment") or "stylish upper garment from <image1>"
    skin_tone_desc = avatar_info.get("skin_tone") or "radiant healthy warm peachy-golden complexion with natural glow"

    # Doodle clause
    if doodle_desc:
        doodle_clause = (
            f"MINIMALIST HAND-DRAWN DOODLE ACCENTS: Adorn the avatar card with {doodle_desc}, "
            "drawn in charming, crisp, single-stroke black vector lines with zero shakiness or duplicate strokes."
        )
    else:
        doodle_clause = "DOODLE ACCENTS: Clean minimalist graphic badge styling."

    sections = [
        (
            f"STYLIZED 2D AVATAR PROFILE ILLUSTRATION (<image1> EXACT PERSON IDENTITY LOCK): "
            f"Transform the real {gender_noun} in <image1> into an iconic, charming 2D digital vector art avatar illustration. "
            f"Style: {style_direction} "
            "EXACT FACIAL LIKENESS MANDATE: The illustrated character MUST be unmistakably recognizable as the exact same person from <image1>. "
            f"Preserve their exact facial proportions, authentic eye shape, irises, eyebrow arch, nose bridge and tip, and genuine expression ({avatar_info.get('expression', 'warm smile')}). "
            f"VIBRANT LUMINOUS WARM SKIN TONE & GLOWING COMPLEXION: Render {skin_tone_desc} with luminous smooth cel-shaded vector gradients, "
            "subtle soft rosy blush on the cheeks and lips, and bright studio beauty illumination. "
            "STRICT COLOR GRADING BAN: ABSOLUTELY ZERO dull grey shadows, ZERO muddy olive/greenish/yellowish tint, and ZERO dark circles or sunken under-eye bags. "
            "Under-eye area must be clean, smooth, youthful, and naturally lit. "
            "BRIGHT ALERT CAPTIVATING EYES: Eyes must be wide, alert, engaging, and friendly, with pure brilliant white sclera, sharp deep irises, and crisp white specular catchlights. "
            f"{glasses_clause} "
            f"RICH VOLUMETRIC LAYERED HAIR: Render the hairstyle ({avatar_info.get('hair_details', 'styled textured hair')}) "
            "with rich dimensional layered vector brushstrokes, fine individual strand texture, deep dark-espresso tones, "
            "and glossy volumetric highlights across the crest of the hair. Avoid flat solid black clumps. "
            f"{beard_clause} "
            f"{distinctive_clause} "
            "STRICT PROPORTION & NATURAL CHEEK LOCK (ABSOLUTELY NO ARTIFICIAL FACIAL SLIMMING): "
            "Preserve the authentic face width, cheek fullness, and natural jaw contour of <image1>. "
            "DO NOT artificially slim the face, DO NOT narrow the jaw, and DO NOT carve an unnatural V-line."
        ),
        (
            f"UPPER OUTFIT VECTOR STYLING: "
            f"Illustrate a faithful stylized vector rendition of the upper garment worn in <image1>: {outfit}. "
            "Render tailored blazer/suit jackets with crisp vector crease lines and subtle satin lapel sheen, "
            "standing sharp against a bright, clean white dress shirt collar. Red neckties must be a vibrant crimson-red satin with a crisp, dimensional knot. "
            "Preserve exact fabric prints, blossom patterns, collar lines, and silhouette with clean vector outlines and rich smooth shading."
        ),
        (
            f"CIRCULAR HALO BADGE & COMPOSITION: "
            "Centered bust-up portrait (head, hair volume, neck, shoulders, and upper chest) cleanly centered in the square canvas. "
            f"Directly behind the character's head and upper shoulders is a prominent, clean, large circular disc badge rendered in {halo_entry['desc']}. "
            f"{doodle_clause} "
            "OUTER CANVAS: Pristine, spotless, solid uniform ivory/off-white background canvas (#FAF9F6) surrounding the circular badge."
        ),
        (
            "QUALITY & GRAPHIC RENDERING STANDARDS: "
            "Crisp 1024x1024 digital artwork, razor-sharp vector linework, vibrant luminous palette, smooth cel-shading, perfect graphic balance, "
            "appealing modern social media profile picture (PFP). "
            "STRICT NEGATIVE EXCLUSIONS: ABSOLUTELY NO photorealistic skin pores, NO waxy 3D plastic mannequin, NO muddy or dull grey skin, "
            "NO yellowish/olive sallow cast, NO dark under-eye circles, NO warped bent glasses, NO double-line shaky doodles, NO cluttered photographic background, "
            "NO watermark, NO logo overlay."
        ),
    ]

    if custom_prompt and custom_prompt.strip():
        sections.append(f"Additional creative direction: {custom_prompt.strip()}")

    full_prompt = " ".join(sections)
    return full_prompt, resolved_halo_name, resolved_doodle_name
