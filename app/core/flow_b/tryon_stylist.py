"""ChatGPT Vision Virtual Try-On Stylist & Multi-Image Qwen Prompt Builder.

Analyzes `[AVATAR]`, each uploaded `[GARMENT_i]`, and each uploaded `[ACCESSORY_j]` in
strictly isolated parallel single-image Vision calls so clothing worn in the user's
original avatar photo can NEVER bleed into or replace uploaded garment descriptions.
Applies gender-aware styling & accessory filtering, defaults to solid jet-black pants
when no bottom garment is uploaded, and builds precision Qwen-2.1 multi-image prompts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import logging
import os
from typing import Any

import httpx
from PIL import Image

from app.core.flow_b.prompt_generator import _encode_image_for_vision

logger = logging.getLogger(__name__)

MAX_TRYON_CONDITION_IMAGES = 10

AVATAR_ANALYSIS_SYSTEM_PROMPT = """You are a forensic facial & portrait biometric analyst preparing identity specifications for studio fashion generation.

Analyze ONLY the person in this photograph (IGNORE the clothing/shirt/background they are currently wearing — only note clothing in `original_worn_clothes` so the generator discards it).

Determine carefully:
- `avatar_gender`: "male" | "female" — inspect facial bone structure, brow ridge, jawline, facial hair/stubble/mustache, hairstyle, and overall presentation. Never guess blindly; be accurate.
- `age_and_gender`: Apparent age bracket, gender, complexion, and natural body build (e.g., "early-20s Indian man, slim athletic build, medium wheatish complexion").
- `forensic_face_details`: Concise, exact facial landmark description:
  1. Hairline & Hairstyle: hairline pattern, exact hair color, texture, styling/cut (e.g., styled forward messy fringe, textured quiff, side parted, buzzcut, wavy curls).
  2. Eyes & Eyewear: eye shape, eyelids, eyebrows thickness and shape. EYEGLASSES: specify exactly whether the person wears prescription glasses or sunglasses directly on their eyes (describe frame material/shape/color, e.g., thin wireframe rectangular metallic glasses with clear lenses), or explicitly state "no glasses worn on face".
  3. Nose, Cheeks & Mouth: nose bridge width/shape, tip shape, cheek contour, smile lines, lip shape, and natural expression (gentle smile, confident, relaxed).
  4. Facial Hair: clean shaven, light 5 o'clock shadow, trimmed mustache, chin goatee, or full beard.
  5. Skin Complexion: exact natural skin tone and undertone (fair, wheatish, warm tan, deep brown), smooth/natural skin texture.
- `avatar_face_box_2d`: [ymin, xmin, ymax, xmax] integers (0..1000) framing comfortably from TOP OF HAIR down to BASE OF NECK / UPPER COLLARBONE (keep full hair width and both ears inside the box!).

Return STRICTLY valid JSON:
{
  "avatar_gender": "male",
  "avatar_summary": "...",
  "forensic_face_details": "...",
  "original_worn_clothes": "...",
  "avatar_face_box_2d": [60, 260, 480, 740]
}"""

GARMENT_ANALYSIS_SYSTEM_PROMPT = """You are a master e-commerce textile and apparel catalog director performing a deep pixel-by-pixel inspection of ONE standalone garment image.

Classify:
- `role`: "top" | "bottom" | "full_outfit" | "outerwear" (honor caller `role_hint` if valid unless clearly contradicted).
- `target_gender`: "men" | "women" | "unisex" (e.g., sarees, lehengas, female kurtis/crop-tops/gowns -> "women"; standard t-shirts, jeans, chinos, hoodies -> "unisex" unless tailored distinctively for "men" or "women").

Inspect EVERY visual pixel carefully:
- `title`: Specific product title (e.g., "Optic White Crimson-Trimmed Athletic Football Jersey T-Shirt", "Beige Slim-Fit Cotton Chinos", "Stonewashed Indigo Raw Denim Jeans").
- `colors`: Exact dye hues, trim colors, striping colors, and undertones (never vague generic names).
- `fabric_type`: Fabric physical structure (e.g., heavy twill denim with contrast copper stitching, lightweight breathable cotton jersey, crisp poplin cotton, fluid silk/chiffon with natural drape, fine knit wool).
- `patterns_and_design`: Exhaustive pixel-level description covering silhouette/fit, exact neckline or collar construction (e.g., crew-neck round collar vs V-neck vs classic button-down collar, buttoned or strictly pullover/buttonless), sleeve length & cuffs, chest/side graphics, logos/crests (exact side left/right, color, shape), prints, embroidery, weave/knit texture, pockets, waistband/hem.

Return STRICTLY valid JSON:
{
  "role": "top",
  "target_gender": "unisex",
  "title": "...",
  "colors": "...",
  "fabric_type": "...",
  "patterns_and_design": "..."
}"""

ACCESSORY_ANALYSIS_SYSTEM_PROMPT = """You are a luxury accessories catalog director performing a deep pixel-by-pixel inspection of ONE standalone fashion accessory.

Identify carefully:
- `accessory_type`: e.g., "wristwatch", "sneakers", "leather_shoes", "sunglasses", "necklace", "earrings", "jhumkas", "bangles", "kada", "ring", "belt", "handbag", "bracelet", "cap", "heels", "anklet".
- `target_gender`: "men" | "women" | "unisex":
  * Women-only items: jhumkas, dangling/stud women's earrings, bridal necklaces, chokers, mangalsutra, kundan/pearl haar, bridal/ladies bangle stacks, payal/anklets, women's heels/stiletto sandals, ladies clutch purses -> "women".
  * Men's or unisex items: wristwatches, sneakers, loafers/dress shoes, sunglasses, belts, caps, backpacks -> "unisex" (or "men" if distinctly men's oxford/mens kada/necktie).
- `body_placement`: exact natural wearing position (e.g., "left wrist", "worn on both feet", "on face/eyes", "around neck against collarbone", "both earlobes", "around wrist", "around waist", "on ring finger").
- `title`: Specific concise title.
- `design_details`: Exhaustive visual details (case finish, dial color/markers, strap texture/color; shoe upper material, toe box, side stripes/logos, midsole/outsole shape; metal karat luster, stones, lens tint).

Return STRICTLY valid JSON:
{
  "accessory_type": "...",
  "target_gender": "unisex",
  "body_placement": "...",
  "title": "...",
  "design_details": "..."
}"""

COMBO_CURATION_SYSTEM_PROMPT = """You are a celebrity fashion stylist curating complete, modest, head-to-toe outfit combinations from an already-analyzed catalog JSON for the specified `avatar_gender` and `occasion`.

Rules:
- Create up to `max_combinations` distinct, well-coordinated looks matching `avatar_gender`.
- Pair complementary tops with bottoms (and include full_outfit entries). Order upper/full garments before bottoms.
- If a look has ONLY upper garments (no bottom, not full_outfit), dynamically include `"complement_bottom"` describing well-matched solid jet-black/dark trousers or black denim suited to the top's style and gender.
- If a look has ONLY lower garments/pants/jeans (no top, not full_outfit), you MUST dynamically include `"complement_top"` describing a clean, modest, solid neutral upper shirt or fitted crew-neck T-shirt suited to `avatar_gender` so the torso is NEVER shirtless.
- Incorporate ALL provided compatible accessories in every look (only pick between competing items of the same body slot).
- Reference ONLY valid 0-based `garment_index` and `accessory_index` integers from the catalog.

Return STRICTLY valid JSON:
{
  "combinations": [
    {
      "title": "Smart Casual Sporty Crisp",
      "occasion": "smart_casual",
      "style_reason": "Why colors, silhouette, gender styling, and accessories work together.",
      "selected_garment_indices": [0, 1],
      "selected_accessory_indices": [0, 1],
      "complement_top": "",
      "complement_bottom": ""
    }
  ]
}"""

_TOP_HINTS = ("shirt", "tshirt", "t-shirt", "tee", "jersey", "top", "kurta", "jacket", "blazer", "hoodie", "polo", "sweater", "blouse")
_BOTTOM_HINTS = ("pant", "pants", "jean", "jeans", "trouser", "chino", "cargo", "short", "skirt", "joggers", "palazzo")
_OUTFIT_HINTS = ("dress", "gown", "saree", "lehenga", "jumpsuit", "suit")
_WOMEN_ITEM_HINTS = (
    "jhumka", "jhumkas", "earring", "earrings", "mangalsutra", "necklace", "choker", "haar",
    "bangle", "bangles", "payal", "anklet", "stiletto", "heels", "clutch", "saree", "lehenga",
    "blouse", "gown", "kurti", "salwar", "dupatta", "skirt", "nose ring", "maang tikka",
)
_MEN_ITEM_HINTS = ("necktie", "bowtie", "bow tie", "mens kada", "cufflink")
_ROLE_SORT_ORDER = {"full_outfit": 0, "top": 1, "outerwear": 2, "bottom": 3}


def _infer_garment_role(item: dict[str, Any], idx: int, total: int) -> str:
    """Infer garment role from explicit upload hint, filename keywords, or upload order."""
    hint = str(item.get("role_hint") or "auto").strip().lower()
    if hint in ("top", "bottom", "full_outfit", "outerwear"):
        return hint
    fname = str(item.get("filename") or "").lower()
    if any(k in fname for k in _OUTFIT_HINTS):
        return "full_outfit"
    if any(k in fname for k in _BOTTOM_HINTS):
        return "bottom"
    if any(k in fname for k in _TOP_HINTS):
        return "top"
    if total >= 2 and idx >= (total + 1) // 2:
        return "bottom"
    return "top" if total >= 2 else "full_outfit"


def _normalize_role(raw_role: str | None, fallback: str) -> str:
    val = (raw_role or "").strip().lower()
    if val in ("top", "upper", "upper_clothes", "shirt", "t-shirt", "tshirt", "jersey"):
        return "top"
    if val in ("bottom", "pants", "trousers", "lower", "skirt", "jeans"):
        return "bottom"
    if val in ("full_outfit", "dress", "outfit", "one_piece"):
        return "full_outfit"
    if val in ("outerwear", "jacket", "coat", "blazer"):
        return "outerwear"
    return fallback


def _normalize_avatar_gender(raw_gender: str | None) -> str:
    val = (raw_gender or "").strip().lower()
    if val in ("male", "man", "men", "boy", "masculine"):
        return "male"
    if val in ("female", "woman", "women", "girl", "feminine"):
        return "female"
    return "unknown"


def _normalize_target_gender(raw_gender: str | None, text_blob: str = "") -> str:
    val = (raw_gender or "").strip().lower()
    if val in ("women", "woman", "female", "ladies", "girl", "girls"):
        return "women"
    if val in ("men", "man", "male", "boys", "gents"):
        return "men"
    if val == "unisex":
        return "unisex"
    lowered = text_blob.lower()
    if any(k in lowered for k in _WOMEN_ITEM_HINTS):
        return "women"
    if any(k in lowered for k in _MEN_ITEM_HINTS):
        return "men"
    return "unisex"


def _is_item_gender_compatible(target_gender: str, avatar_gender: str) -> bool:
    """Return False when a gendered item clearly belongs to the opposite sex."""
    if avatar_gender == "male" and target_gender == "women":
        return False
    if avatar_gender == "female" and target_gender == "men":
        return False
    return True


def _default_body_placement(accessory_type: str, filename: str = "") -> str:
    text = f"{accessory_type} {filename}".lower()
    if any(k in text for k in ("watch", "smartwatch", "bangle", "kada", "bracelet", "cuff")):
        return "fitted cleanly around wrist"
    if any(k in text for k in ("shoe", "sneaker", "boot", "heel", "loafer", "sandal", "footwear")):
        return "worn on both feet"
    if any(k in text for k in ("glass", "sunglass", "shades", "eyewear", "spectacle")):
        return "seated naturally on nose bridge and face over eyes"
    if any(k in text for k in ("necklace", "chain", "pendant", "choker", "haar", "mangalsutra", "tie", "scarf")):
        return "draped naturally around neck against collarbone"
    if any(k in text for k in ("earring", "jhumka", "stud")):
        return "worn symmetrically on both earlobes"
    if "ring" in text:
        return "fitted cleanly on ring finger"
    if "belt" in text:
        return "buckled cleanly around waistband"
    if any(k in text for k in ("bag", "handbag", "tote", "clutch", "backpack", "purse")):
        return "carried naturally in hand or over shoulder"
    if any(k in text for k in ("hat", "cap", "beanie")):
        return "worn cleanly on head"
    return "styled naturally on the person"


def _validate_face_box_2d(raw_box: Any) -> list[int] | None:
    """Validate optional 0..1000 normalized ``[ymin, xmin, ymax, xmax]`` face box."""
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (int(round(float(v))) for v in raw_box)
    except (TypeError, ValueError):
        return None
    ymin = max(0, min(950, ymin))
    xmin = max(0, min(950, xmin))
    ymax = max(ymin + 40, min(1000, ymax))
    xmax = max(xmin + 40, min(1000, xmax))
    if ymax - ymin < 40 or xmax - xmin < 40:
        return None
    return [ymin, xmin, ymax, xmax]


def _generate_diverse_garment_pairs(garment_catalog: list[dict[str, Any]], max_combinations: int) -> list[list[int]]:
    """Pair tops with bottoms (and include full outfits), prioritizing distinct items first."""
    tops = [g["garment_index"] for g in garment_catalog if g["role"] in ("top", "outerwear")]
    bottoms = [g["garment_index"] for g in garment_catalog if g["role"] == "bottom"]
    fulls = [g["garment_index"] for g in garment_catalog if g["role"] == "full_outfit"]

    candidates: list[list[int]] = []
    if tops and bottoms:
        diag_count = min(len(tops), len(bottoms))
        for i in range(diag_count):
            candidates.append([tops[i], bottoms[i]])
        for t, b in product(tops, bottoms):
            pair = [t, b]
            if pair not in candidates:
                candidates.append(pair)
    elif tops:
        candidates.extend([[t] for t in tops])
    elif bottoms:
        candidates.extend([[b] for b in bottoms])
    for f_idx in fulls:
        if [f_idx] not in candidates:
            candidates.append([f_idx])
    if not candidates:
        candidates = [[g["garment_index"]] for g in garment_catalog]
    return candidates[:max_combinations]


def _build_fallback_garment_item(g: dict[str, Any], idx: int, total: int) -> dict[str, Any]:
    role = _infer_garment_role(g, idx, total)
    raw_name = str(g.get("filename") or f"Garment {idx + 1}")
    stem = os.path.splitext(raw_name)[0].replace("_", " ").replace("-", " ").strip()
    return {
        "garment_index": idx,
        "role": role,
        "target_gender": _normalize_target_gender("auto", stem),
        "title": stem.title() or f"Garment #{idx + 1} ({role})",
        "colors": "authentic original colors from reference image",
        "patterns_and_design": f"exact garment fabric, print patterns, neckline/collar cut, logos, silhouette, and construction visible in Garment #{idx + 1}",
    }


def _build_fallback_accessory_item(acc: dict[str, Any], idx: int) -> dict[str, Any]:
    raw_name = str(acc.get("filename") or f"Accessory {idx + 1}")
    stem = os.path.splitext(raw_name)[0].replace("_", " ").replace("-", " ").strip()
    placement = _default_body_placement(stem, stem)
    return {
        "accessory_index": idx,
        "accessory_type": stem or "accessory",
        "target_gender": _normalize_target_gender("auto", stem),
        "body_placement": placement,
        "title": stem.title() or f"Accessory #{idx + 1}",
        "design_details": f"exact shape, color, strap/material finish, and hardware details visible in Accessory #{idx + 1}",
    }


def _dynamic_missing_garment(
    role_needed: str,
    selected_garments: list[dict[str, Any]],
    avatar_gender: str,
    occasion: str,
    original_worn_clothes: str = "",
) -> tuple[str, str]:
    """Dynamically choose harmonious upper or lower complement matching uploaded clothes, gender, occasion, and avatar style."""
    ctx = " ".join(
        f"{g.get('title', '')} {g.get('colors', '')} {g.get('patterns_and_design', '')}"
        for g in selected_garments
    ).lower()
    occ = (occasion or "smart_casual").lower()
    worn_ctx = (original_worn_clothes or "").lower()

    if role_needed == "bottom":
        is_ethnic = any(k in ctx for k in ("kurta", "sherwani", "ethnic", "nehru", "festive", "saree")) or occ in ("festive", "party")
        is_formal = any(k in ctx for k in ("formal", "blazer", "oxford", "suit", "button-down", "tuxedo")) or occ == "formal"
        is_light_top = any(k in ctx for k in ("white", "cream", "ivory", "beige", "yellow", "light blue", "pink"))

        if is_ethnic:
            if avatar_gender == "female":
                return (
                    "Complementary Silk-Blend Straight Ethnic Trousers / Churidar",
                    "tailored slim silk-blend straight-leg ethnic trousers with natural ankle pleats harmonizing with the upper outfit",
                )
            bottom_color = "rich cream-ivory" if not is_light_top else "tailored deep charcoal"
            return (
                f"Tailored {bottom_color.title()} Ethnic Straight Trousers",
                f"{bottom_color} tailored slim straight-leg ethnic trousers with clean ankle drape, complementing the kurta silhouette",
            )

        if is_formal:
            bottom_color = "tailored charcoal grey" if is_light_top else "tailored slate-navy"
            return (
                f"Tailored Slim {bottom_color.title()} Formal Trousers",
                f"crisp {bottom_color} tailored slim formal trousers extending neatly to the shoes with a sharp front crease and zero patterns",
            )

        # Casual / Streetwear: Check if avatar already wore denim or chinos
        if "chino" in worn_ctx:
            return (
                "Warm Sand Slim-Fit Cotton Chinos",
                "clean warm sand-beige tailored slim cotton chinos extending naturally to the ankles with a modern tapered cut",
            )
        if is_light_top:
            return (
                "Washed Indigo Slim-Fit Denim Jeans",
                "authentic washed medium-indigo tailored slim denim jeans with natural thigh fading and contrast golden topstitching",
            )
        return (
            "Tailored Slim Raw Indigo Denim Jeans / Trousers",
            "clean dark raw-indigo tailored slim denim jeans extending naturally to the shoes with authentic twill texture",
        )

    # role_needed == "top": NEVER leave torso shirtless when user uploads only pants/jeans!
    is_denim = any(k in ctx for k in ("jean", "denim", "indigo"))
    is_formal_bottom = any(k in ctx for k in ("formal", "trouser", "suit", "crease", "pleat")) or occ == "formal"
    is_dark_bottom = any(k in ctx for k in ("black", "dark", "charcoal", "navy", "deep"))

    if avatar_gender == "female":
        if is_formal_bottom:
            return (
                "Modest Tailored Silk-Blend Collared Blouse",
                "crisp ivory-white or soft blush-tinted modest tailored full-coverage shirt tucked neatly into the waistband",
            )
        top_color = "soft heather grey" if is_dark_bottom else "clean crisp white"
        return (
            f"Modest Solid Fitted Short-Sleeve Crew-Neck Top",
            f"{top_color} modest fitted cotton crew-neck short-sleeve T-shirt covering the full torso down to the waistband, unbranded pure fabric",
        )

    if is_formal_bottom:
        shirt_color = "crisp pale mist-blue" if is_dark_bottom else "crisp optic white"
        return (
            f"Modest Solid {shirt_color.title()} Oxford Shirt",
            f"{shirt_color} tailored full-coverage cotton collared Oxford shirt tucked neatly into the waistband so visual focus stays on the trousers",
        )

    top_color = "soft heather-grey" if is_dark_bottom else "clean solid crisp optic white"
    return (
        f"Modest Solid Matte Crew-Neck Cotton T-Shirt",
        f"{top_color} modest crew-neck short-sleeve cotton T-shirt covering the entire chest, shoulders, and torso down to the waistband",
    )


def _resolve_combo_complements(
    sel_meta: list[dict[str, Any]],
    avatar_gender: str,
    occasion: str,
    raw_comp_top: str = "",
    raw_comp_bottom: str = "",
    original_worn_clothes: str = "",
) -> tuple[str, str, str]:
    """Ensure every incomplete combo gets dynamic complementary top or bottom (never bare skin)."""
    has_top = any(g["role"] in ("top", "outerwear", "full_outfit") for g in sel_meta)
    has_bottom = any(g["role"] in ("bottom", "full_outfit") for g in sel_meta)

    comp_top = str(raw_comp_top or "").strip() if not has_top else ""
    comp_bottom = str(raw_comp_bottom or "").strip() if not has_bottom else ""
    note = ""

    if not has_top and not comp_top:
        t_title, comp_top = _dynamic_missing_garment("top", sel_meta, avatar_gender, occasion, original_worn_clothes)
        note = f" styled modestly with {t_title.lower()}"
    elif not has_top:
        note = f" styled modestly with {comp_top}"
    elif not has_bottom and not comp_bottom:
        b_title, comp_bottom = _dynamic_missing_garment("bottom", sel_meta, avatar_gender, occasion, original_worn_clothes)
        note = f" paired with {b_title.lower()}"
    elif not has_bottom:
        note = f" paired with {comp_bottom}"

    return comp_top, comp_bottom, note


def _build_deterministic_combinations(
    eligible_garments: list[dict[str, Any]],
    eligible_accessories: list[dict[str, Any]],
    max_combinations: int,
    occasion: str,
    avatar_gender: str = "unknown",
    original_worn_clothes: str = "",
) -> list[dict[str, Any]]:
    raw_combos = _generate_diverse_garment_pairs(eligible_garments, max_combinations)
    garment_by_idx = {g["garment_index"]: g for g in eligible_garments}
    all_acc_indices = [a["accessory_index"] for a in eligible_accessories][: max(0, MAX_TRYON_CONDITION_IMAGES - 1)]
    combinations: list[dict[str, Any]] = []
    for c_idx, g_indices in enumerate(raw_combos):
        g_sorted = sorted(g_indices, key=lambda idx: (_ROLE_SORT_ORDER.get(garment_by_idx[idx]["role"], 2), idx))
        sel_meta = [garment_by_idx[i] for i in g_sorted]
        titles = [g["title"] for g in sel_meta]
        comp_top, comp_bottom, note = _resolve_combo_complements(
            sel_meta, avatar_gender, occasion, original_worn_clothes=original_worn_clothes
        )
        max_acc = max(0, MAX_TRYON_CONDITION_IMAGES - 1 - len(g_sorted))
        acc_sel = all_acc_indices[:max_acc]
        acc_note = f" + {len(acc_sel)} matching accessory/ies" if acc_sel else ""
        combinations.append({
            "title": " + ".join(titles) or f"Curated Look #{c_idx + 1}",
            "occasion": occasion or "smart_casual",
            "style_reason": f"Balanced coordination showcasing {', '.join(titles)}{note}{acc_note}.",
            "selected_garment_indices": g_sorted,
            "selected_accessory_indices": acc_sel,
            "complement_top": comp_top,
            "complement_bottom": comp_bottom,
        })
    return combinations


def _call_openai_json(
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    api_key: str,
    model_name: str,
    base_url: str,
    max_tokens: int = 800,
    timeout: float = 35.0,
) -> dict[str, Any]:
    """Execute one OpenAI chat completion expecting strict JSON output."""
    payload = {
        "model": model_name,
        "temperature": 0.15,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return json.loads(data["choices"][0]["message"]["content"])


def _vision_message_for_image(instruction: str, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": instruction},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{_encode_image_for_vision(image, max_edge=768)}",
                "detail": "high",
            },
        },
    ]


def _analyze_avatar_single(
    avatar_img: Image.Image,
    api_key: str,
    model_name: str,
    base_url: str,
) -> dict[str, Any]:
    """Analyze the avatar photo in complete isolation for gender, biometric landmarks & head box."""
    default_avatar = {
        "avatar_gender": "unknown",
        "avatar_summary": "Same individual from the reference portrait preserving exact age, facial structure, hairline, skin tone, and expression",
        "forensic_face_details": "exact eye shape, brow thickness, nose bridge width, nasolabial contours, lip shape, jawline, hairline, and natural skin texture",
        "original_worn_clothes": "",
        "avatar_face_box_2d": None,
    }
    try:
        parsed = _call_openai_json(
            AVATAR_ANALYSIS_SYSTEM_PROMPT,
            _vision_message_for_image(
                "[AVATAR] Identify gender accurately and inspect every facial landmark, hairline, age marker, and skin tone deeply; provide tight head box above shirt.",
                avatar_img,
            ),
            api_key,
            model_name,
            base_url,
            max_tokens=650,
        )
        summary = str(parsed.get("avatar_summary") or default_avatar["avatar_summary"]).strip()
        forensic = str(parsed.get("forensic_face_details") or "").strip()
        combined = f"{summary}. Facial landmarks: {forensic}".strip(". ") if forensic else summary
        return {
            "avatar_gender": _normalize_avatar_gender(parsed.get("avatar_gender")),
            "avatar_summary": combined,
            "forensic_face_details": forensic or default_avatar["forensic_face_details"],
            "original_worn_clothes": str(parsed.get("original_worn_clothes") or "").strip(),
            "avatar_face_box_2d": _validate_face_box_2d(parsed.get("avatar_face_box_2d")),
            "source": f"openai:{model_name}",
        }
    except Exception as exc:
        logger.warning("Isolated avatar Vision analysis failed (%s); using fallback.", exc)
        return {**default_avatar, "source": "deterministic_fallback"}


def _analyze_garment_single(
    g_entry: dict[str, Any],
    total_garments: int,
    api_key: str,
    model_name: str,
    base_url: str,
) -> dict[str, Any]:
    """Analyze ONE standalone garment image with zero avatar/other-garment contamination."""
    idx = g_entry["index"]
    fallback = _build_fallback_garment_item(g_entry, idx, total_garments)
    hint = str(g_entry.get("role_hint") or "auto").strip().lower()
    try:
        parsed = _call_openai_json(
            GARMENT_ANALYSIS_SYSTEM_PROMPT,
            _vision_message_for_image(
                f"[GARMENT_{idx}] (caller role_hint={hint}). Analyze ONLY this garment's colors, target_gender, collar/neckline, buttons/pullover front, graphics/logos, prints, and fabric weave in extreme pixel detail.",
                g_entry["image"],
            ),
            api_key,
            model_name,
            base_url,
            max_tokens=550,
        )
        title = str(parsed.get("title") or fallback["title"]).strip()
        patterns = str(parsed.get("patterns_and_design") or fallback["patterns_and_design"]).strip()
        fabric = str(parsed.get("fabric_type") or "").strip()
        if fabric and fabric.lower() not in patterns.lower():
            patterns = f"{patterns}. Fabric: {fabric}".strip(". ")
        return {
            "garment_index": idx,
            "role": _normalize_role(parsed.get("role"), fallback["role"]),
            "target_gender": _normalize_target_gender(parsed.get("target_gender"), f"{title} {patterns}"),
            "title": title,
            "colors": str(parsed.get("colors") or fallback["colors"]).strip(),
            "patterns_and_design": patterns,
        }
    except Exception as exc:
        logger.warning("Isolated garment #%d Vision analysis failed (%s); using fallback.", idx, exc)
        return fallback


def _analyze_accessory_single(
    acc_entry: dict[str, Any],
    api_key: str,
    model_name: str,
    base_url: str,
) -> dict[str, Any]:
    """Analyze ONE standalone accessory image in isolation."""
    idx = acc_entry["index"]
    fallback = _build_fallback_accessory_item(acc_entry, idx)
    try:
        parsed = _call_openai_json(
            ACCESSORY_ANALYSIS_SYSTEM_PROMPT,
            _vision_message_for_image(
                f"[ACCESSORY_{idx}] Analyze ONLY this accessory's exact type, target_gender (men/women/unisex), wearing placement, colors, dials/stripes/soles/gemstones, and materials.",
                acc_entry["image"],
            ),
            api_key,
            model_name,
            base_url,
            max_tokens=450,
        )
        acc_type = str(parsed.get("accessory_type") or fallback["accessory_type"]).strip()
        title = str(parsed.get("title") or fallback["title"]).strip()
        details = str(parsed.get("design_details") or fallback["design_details"]).strip()
        return {
            "accessory_index": idx,
            "accessory_type": acc_type,
            "target_gender": _normalize_target_gender(parsed.get("target_gender"), f"{acc_type} {title} {details}"),
            "body_placement": str(parsed.get("body_placement") or _default_body_placement(acc_type)).strip(),
            "title": title,
            "design_details": details,
        }
    except Exception as exc:
        logger.warning("Isolated accessory #%d Vision analysis failed (%s); using fallback.", idx, exc)
        return fallback


def _filter_catalogs_by_gender(
    avatar_gender: str,
    garment_catalog: list[dict[str, Any]],
    accessory_catalog: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter incompatible gendered garments/accessories cleanly based on detected avatar gender."""
    if avatar_gender not in ("male", "female"):
        return garment_catalog, accessory_catalog, [], []

    compatible_g = [g for g in garment_catalog if _is_item_gender_compatible(g.get("target_gender", "unisex"), avatar_gender)]
    excluded_g = [g for g in garment_catalog if g not in compatible_g]
    # Never leave 0 garments if every uploaded item was tagged opposite gender
    if not compatible_g:
        compatible_g = garment_catalog
        excluded_g = []

    compatible_a = [a for a in accessory_catalog if _is_item_gender_compatible(a.get("target_gender", "unisex"), avatar_gender)]
    excluded_a = [a for a in accessory_catalog if a not in compatible_a]
    return compatible_g, compatible_a, excluded_g, excluded_a


def _sanitize_combinations(
    raw_combos: list[Any],
    eligible_garments: list[dict[str, Any]],
    eligible_accessories: list[dict[str, Any]],
    max_combinations: int,
    occasion: str,
    avatar_gender: str = "unknown",
    original_worn_clothes: str = "",
) -> list[dict[str, Any]]:
    """Validate stylist combination indices, enforce top->bottom ordering, and ensure full modest coverage."""
    garment_by_idx = {g["garment_index"]: g for g in eligible_garments}
    acc_by_idx = {a["accessory_index"]: a for a in eligible_accessories}
    valid_g = set(garment_by_idx.keys())
    valid_a = set(acc_by_idx.keys())

    cleaned: list[dict[str, Any]] = []
    seen_sigs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    for raw_c in raw_combos or []:
        if not isinstance(raw_c, dict):
            continue
        g_sel: list[int] = []
        for idx in (raw_c.get("selected_garment_indices") or []):
            try:
                ival = int(idx)
            except (TypeError, ValueError):
                continue
            if ival in valid_g and ival not in g_sel:
                g_sel.append(ival)
        if not g_sel:
            continue

        g_sel.sort(key=lambda idx: (_ROLE_SORT_ORDER.get(garment_by_idx[idx]["role"], 2), idx))

        a_sel: list[int] = []
        for idx in (raw_c.get("selected_accessory_indices") or []):
            try:
                ival = int(idx)
            except (TypeError, ValueError):
                continue
            if ival in valid_a and ival not in a_sel:
                a_sel.append(ival)

        if valid_a:
            used_slots = {acc_by_idx[i]["body_placement"].lower() for i in a_sel}
            for a_idx in sorted(valid_a):
                slot = acc_by_idx[a_idx]["body_placement"].lower()
                if a_idx not in a_sel and slot not in used_slots:
                    a_sel.append(a_idx)
                    used_slots.add(slot)

        max_acc = max(0, MAX_TRYON_CONDITION_IMAGES - 1 - len(g_sel))
        a_sel = a_sel[:max_acc]

        sig = (tuple(g_sel), tuple(a_sel))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        sel_meta = [garment_by_idx[i] for i in g_sel]
        comp_top, comp_bottom, _ = _resolve_combo_complements(
            sel_meta,
            avatar_gender,
            str(raw_c.get("occasion") or occasion or "smart_casual").strip(),
            raw_comp_top=str(raw_c.get("complement_top") or ""),
            raw_comp_bottom=str(raw_c.get("complement_bottom") or ""),
            original_worn_clothes=original_worn_clothes,
        )

        cleaned.append({
            "title": str(raw_c.get("title") or f"Styled Look #{len(cleaned) + 1}").strip(),
            "occasion": str(raw_c.get("occasion") or occasion or "smart_casual").strip(),
            "style_reason": str(raw_c.get("style_reason") or "Curated wardrobe coordination.").strip(),
            "selected_garment_indices": g_sel,
            "selected_accessory_indices": a_sel,
            "complement_top": comp_top,
            "complement_bottom": comp_bottom,
        })
        if len(cleaned) >= max_combinations:
            break

    return cleaned or _build_deterministic_combinations(
        eligible_garments,
        eligible_accessories,
        max_combinations,
        occasion,
        avatar_gender,
        original_worn_clothes=original_worn_clothes,
    )


def analyze_wardrobe_and_plan_tryon(
    avatar_img: Image.Image,
    garments: list[dict[str, Any]],
    accessories: list[dict[str, Any]] | None = None,
    max_combinations: int = 2,
    occasion: str = "smart_casual",
) -> dict[str, Any]:
    """Analyze avatar (with gender), garments, and accessories in isolated parallel Vision calls, then curate combos."""
    accessories = accessories or []
    max_combinations = max(1, min(int(max_combinations or 2), 4))

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; using deterministic try-on combination fallback.")
        g_cat = [_build_fallback_garment_item(g, g["index"], len(garments)) for g in garments]
        a_cat = [_build_fallback_accessory_item(a, a["index"]) for a in accessories]
        return {
            "avatar_gender": "unknown",
            "avatar_summary": "Same individual from the reference portrait preserving exact age, facial structure, hairline, and skin tone",
            "forensic_face_details": "exact eye shape, brow arch, nose bridge, lips, jawline, and natural skin texture",
            "original_worn_clothes": "",
            "avatar_face_box_2d": None,
            "garments": g_cat,
            "accessories": a_cat,
            "excluded_garments": [],
            "excluded_accessories": [],
            "combinations": _build_deterministic_combinations(g_cat, a_cat, max_combinations, occasion),
            "prompt_source": "deterministic_fallback",
        }

    model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    total_calls = 1 + len(garments) + len(accessories)
    with ThreadPoolExecutor(max_workers=min(8, max(1, total_calls))) as pool:
        avatar_fut = pool.submit(_analyze_avatar_single, avatar_img, api_key, model_name, base_url)
        garment_futs = [
            pool.submit(_analyze_garment_single, g, len(garments), api_key, model_name, base_url)
            for g in garments
        ]
        accessory_futs = [
            pool.submit(_analyze_accessory_single, acc, api_key, model_name, base_url)
            for acc in accessories
        ]

        avatar_info = avatar_fut.result()
        garment_catalog = sorted((f.result() for f in garment_futs), key=lambda x: x["garment_index"])
        accessory_catalog = sorted((f.result() for f in accessory_futs), key=lambda x: x["accessory_index"])

    avatar_gender = avatar_info.get("avatar_gender", "unknown")
    eligible_g, eligible_a, excluded_g, excluded_a = _filter_catalogs_by_gender(
        avatar_gender, garment_catalog, accessory_catalog
    )
    if excluded_a:
        logger.info(
            "Filtered %d gender-incompatible accessory item(s) for %s avatar: %s",
            len(excluded_a),
            avatar_gender,
            [a["title"] for a in excluded_a],
        )

    possible_pairs = _generate_diverse_garment_pairs(eligible_g, max_combinations=max(4, max_combinations))
    has_multiple_garment_choices = len(eligible_g) > 2 or len(possible_pairs) > 1

    combinations: list[dict[str, Any]]
    if has_multiple_garment_choices:
        try:
            combo_req = {
                "avatar_gender": avatar_gender,
                "occasion": occasion,
                "max_combinations": max_combinations,
                "garments": eligible_g,
                "accessories": eligible_a,
            }
            parsed_combos = _call_openai_json(
                COMBO_CURATION_SYSTEM_PROMPT,
                json.dumps(combo_req),
                api_key,
                model_name,
                base_url,
                max_tokens=700,
                timeout=25.0,
            )
            combinations = _sanitize_combinations(
                parsed_combos.get("combinations") or [],
                eligible_g,
                eligible_a,
                max_combinations,
                occasion,
                avatar_gender=avatar_gender,
                original_worn_clothes=avatar_info.get("original_worn_clothes", ""),
            )
        except Exception as exc:
            logger.warning("Combo curation text call failed (%s); using deterministic stylist pairing.", exc)
            combinations = _build_deterministic_combinations(
                eligible_g,
                eligible_a,
                max_combinations,
                occasion,
                avatar_gender,
                original_worn_clothes=avatar_info.get("original_worn_clothes", ""),
            )
    else:
        combinations = _build_deterministic_combinations(
            eligible_g,
            eligible_a,
            max_combinations,
            occasion,
            avatar_gender,
            original_worn_clothes=avatar_info.get("original_worn_clothes", ""),
        )

    return {
        "avatar_gender": avatar_gender,
        "avatar_summary": avatar_info["avatar_summary"],
        "forensic_face_details": avatar_info.get("forensic_face_details", ""),
        "original_worn_clothes": avatar_info.get("original_worn_clothes", ""),
        "avatar_face_box_2d": avatar_info.get("avatar_face_box_2d"),
        "garments": garment_catalog,
        "accessories": accessory_catalog,
        "excluded_garments": excluded_g,
        "excluded_accessories": excluded_a,
        "combinations": combinations[:max_combinations],
        "prompt_source": avatar_info.get("source") or f"openai:{model_name}",
    }


def _describe_model_gender(avatar_gender: str) -> str:
    if avatar_gender == "male":
        return "an adult male fashion model with natural masculine shoulders, flat tailored chest fit, and confident masculine stance"
    if avatar_gender == "female":
        return "an adult female fashion model with graceful feminine silhouette tailoring and natural poised stance"
    return "an adult fashion model standing in clean upright fashion lookbook stance"


def _determine_dynamic_pose(
    avatar_gender: str,
    occasion: str,
    selected_garments: list[dict[str, Any]],
    selected_accessories: list[dict[str, Any]],
    pose_preference: str = "auto",
    combo_index: int = 0,
) -> str:
    """Generate dynamic, natural, relaxed human fashion photography poses instead of stiff mannequins."""
    pref = (pose_preference or "auto").strip().lower()
    occ = (occasion or "smart_casual").strip().lower()

    has_watch = any("watch" in f"{a.get('accessory_type', '')} {a.get('title', '')}".lower() for a in selected_accessories)
    is_saree = any("saree" in f"{g.get('title', '')} {g.get('patterns_and_design', '')}".lower() for g in selected_garments)
    is_traditional = is_saree or any(
        any(k in f"{g.get('title', '')} {g.get('patterns_and_design', '')}".lower() for k in ("kurta", "sherwani", "lehenga", "ethnic", "nehru"))
        for g in selected_garments
    ) or occ in ("festive", "party")
    is_formal = any(
        any(k in f"{g.get('title', '')} {g.get('patterns_and_design', '')}".lower() for k in ("suit", "blazer", "formal", "tuxedo", "oxford"))
        for g in selected_garments
    ) or occ == "formal"

    watch_clause = (
        " The arm with the wristwatch is positioned with the wrist exposed so the watch is clearly and elegantly displayed."
        if has_watch
        else ""
    )

    if is_saree:
        return (
            "Graceful traditional saree posture: body turned at an elegant three-quarter angle to showcase the blouse embroidery and shoulder pleats, "
            "one hand gently holding or guiding the pallu drape near the waist with delicate fingertips, soft fluid pleats flowing naturally to the floor over elegant footwear, "
            "poised expressive head angle looking warmly at camera."
            + watch_clause
        )

    if pref in ("relaxed_casual", "casual") or (pref == "auto" and not is_traditional and not is_formal):
        if combo_index % 2 == 1:
            return (
                "Contemporary relaxed open stance: balanced footing, upright yet fluid shoulders, "
                "one arm naturally bent at waist height with hand resting easily near the belt or wrist, "
                "the other arm resting comfortably at the side with a soft natural elbow bend and relaxed fingers, "
                "natural candid poise with direct engaging eye contact."
                + watch_clause
            )
        return (
            "Natural relaxed contrapposto human stance with body weight comfortably shifted onto one leg, "
            "subtle 10-15 degree three-quarter angle to the camera creating natural depth and relaxed silhouette. "
            "Asymmetric relaxed arm posture: one hand casually resting in the trouser pocket or lightly hooked at the waistband, "
            "the other arm hanging naturally at the side with a soft natural bend at the elbow and relaxed, unclenched fingers. "
            "Shoulders naturally relaxed and fluid, confident engaging eye contact toward camera."
            + watch_clause
        )

    if pref in ("confident_executive", "formal") or (pref == "auto" and is_formal):
        if combo_index % 2 == 1:
            return (
                "Sharp modern editorial fashion stance: natural upright posture, balanced footing with subtle asymmetry, "
                "arms resting comfortably with slight asymmetric bends at elbows, subtle cuff adjustment gesture with one hand, "
                "calm commanding gaze looking directly into the camera lens."
                + watch_clause
            )
        return (
            "Confident modern executive lookbook stance: poised posture with weight subtly on the back leg, "
            "shoulders naturally relaxed and level without military stiffness. "
            "Subtle asymmetric arm placement with one hand resting naturally near the trouser pocket or adjusting a cuff, "
            "the other arm hanging with natural drape. Direct, calm, commanding eye contact with the camera lens."
            + watch_clause
        )

    if pref in ("traditional_graceful", "festive", "traditional") or (pref == "auto" and is_traditional):
        if avatar_gender == "female":
            return (
                "Poised, graceful traditional ethnic stance: body turned at an elegant three-quarter angle, "
                "one hand gently resting near the waist or softly guiding the garment fabric to highlight its natural pleats and drape, "
                "the other arm relaxed at the side with delicate natural hand pose. Poised, warm, expressive facial expression."
                + watch_clause
            )
        return (
            "Regal, stately ethnic lookbook stance: upright dignified posture with natural relaxed shoulders, "
            "subtle contrapposto weight shift, one hand resting naturally at the side and the other hand relaxed near the waistline. "
            "Warm confident gaze directly at the camera."
            + watch_clause
        )

    if pref == "classic_catalog":
        return (
            "Clean balanced fashion lookbook stance: upright natural posture with relaxed shoulders, "
            "arms resting naturally beside the torso with gentle natural bends at elbows and soft relaxed fingers (never stiff or robot-like). "
            "Direct level gaze toward camera."
            + watch_clause
        )

    return (
        "Natural relaxed contrapposto human fashion stance with weight shifted onto one leg, "
        "subtle 10-15 degree torso angle for natural depth, relaxed shoulders, and asymmetric natural arm placement. "
        "Engaging confident eye contact looking directly into the camera lens."
        + watch_clause
    )


def _determine_photography_style(
    photography_style: str = "auto",
    occasion: str = "smart_casual",
    custom_prompt: str | None = None,
) -> tuple[str, str]:
    """Resolve high-end fashion photography lighting, backdrop, and camera direction."""
    style = (photography_style or "auto").strip().lower()
    occ = (occasion or "smart_casual").strip().lower()
    custom_text = (custom_prompt or "").lower()

    # Check for explicit custom background requests
    has_custom_bg = any(k in custom_text for k in ("outdoor", "street", "cafe", "loft", "concrete", "terrace", "interior", "garden", "urban", "grey background", "dark background"))

    if style == "auto":
        if occ in ("formal", "office"):
            style = "editorial_lookbook"
        elif occ in ("festive", "party"):
            style = "traditional_royal"
        elif occ in ("streetwear", "casual"):
            style = "streetwear_lifestyle"
        else:
            style = "premium_studio"

    if has_custom_bg:
        return (
            "Custom Environment Fashion Photography",
            "Professional fashion editorial photography shot with an 85mm portrait prime lens at f/3.2. "
            "Soft directional ambient lighting matching the requested setting, natural grounding contact shadows under both shoes, "
            "crisp edge-to-edge optical clarity, and authentic environmental depth."
        )

    if style in ("editorial_lookbook", "editorial"):
        return (
            "High-End Fashion Editorial",
            "Editorial studio photography shot on Hasselblad H6D-100c with 85mm portrait prime lens at f/3.2. "
            "Sculptural three-point lighting: large diffused key octabox from high 45-degree angle creating flattering facial bone structure definition, "
            "gentle ambient fill light, subtle rim light separating hair from background, and natural soft grounding floor shadows beneath both shoes "
            "against an architectural warm bone studio cyclorama backdrop."
        )

    if style in ("streetwear_lifestyle", "lifestyle"):
        return (
            "Modern Streetwear Lookbook",
            "Contemporary lookbook photography with crisp, punchy fashion lighting: wide diffused beauty dish key light "
            "rendering rich contrast, authentic fabric weave textures, vivid colors, and soft natural ambient floor shadows "
            "against a contemporary minimalist studio wall with subtle architectural texture."
        )

    if style in ("traditional_royal", "royal", "festive"):
        return (
            "Luxury Ethnic Studio Portrait",
            "Luminous luxury studio portraiture with warm golden-hour key light accentuating rich fabric luster, "
            "intricate embroidery reflections, gold zari sheen, and jewelry sparkle, with gentle flattering fill "
            "on face and hair against an elegant warm neutral backdrop."
        )

    return (
        "Premium Studio Catalog Lookbook",
        "Commercial fashion catalog photograph shot with an 85mm portrait lens with edge-to-edge optical sharpness. "
        "Flattering large softbox key lighting providing even, soft illumination across the entire body, "
        "crisp micro-textures on skin pores and textile fibers, realistic floor contact shadows under both shoes, "
        "and clean seamless warm alabaster backdrop (#F4F1EC)."
    )


def build_qwen_tryon_prompt(
    avatar_summary: str,
    selected_garments: list[dict[str, Any]],
    selected_accessories: list[dict[str, Any]],
    occasion: str = "smart_casual",
    custom_prompt: str | None = None,
    original_worn_clothes: str = "",
    avatar_gender: str = "unknown",
    complement_top: str = "",
    complement_bottom: str = "",
    forensic_face_details: str = "",
    photography_style: str = "auto",
    pose_preference: str = "auto",
    combo_index: int = 0,
) -> tuple[str, int]:
    """Build crisp, identity-anchored `<image1>`, `<image2>`, ... multi-image Qwen-2.1 prompt."""
    garment_clauses: list[str] = []
    next_tag_idx = 2
    for g in selected_garments:
        tag = f"<image{next_tag_idx}>"
        next_tag_idx += 1
        garment_clauses.append(
            f"Dress the model's {g['role']} EXCLUSIVELY in the EXACT {g['title']} shown in {tag}: "
            f"colors {g['colors']}; {g['patterns_and_design']}. "
            f"Reproduce every authentic wash, stripe, topstitch, crest, logo, pattern motif, collar/neckline shape, sleeve length, and fabric weave from {tag}."
        )

    accessory_clauses: list[str] = []
    for a in selected_accessories:
        if next_tag_idx > MAX_TRYON_CONDITION_IMAGES:
            break
        tag = f"<image{next_tag_idx}>"
        next_tag_idx += 1
        accessory_clauses.append(
            f"Wear the EXACT {a['title']} ({a['accessory_type']}) from {tag} {a['body_placement']}: {a['design_details']}"
        )

    has_top = any(g["role"] in ("top", "outerwear", "full_outfit") for g in selected_garments)
    has_bottom = any(g["role"] in ("bottom", "full_outfit") for g in selected_garments)
    has_top_and_bottom = (
        any(g["role"] in ("top", "outerwear") for g in selected_garments)
        and any(g["role"] == "bottom" for g in selected_garments)
    )
    has_shoes = any(
        any(k in f"{a.get('accessory_type', '')} {a.get('body_placement', '')}".lower() for k in ("shoe", "sneaker", "boot", "heel", "loafer", "sandal", "feet"))
        for a in selected_accessories
    )

    complement_clauses: list[str] = []
    if not has_top:
        top_desc = complement_top.strip() or _dynamic_missing_garment("top", selected_garments, avatar_gender, occasion)[1]
        complement_clauses.append(
            f"MANDATORY MODEST UPPER COVERAGE (NO SHIRTLESS TORSO): Since only lower wear was provided, dress the entire chest, shoulders, and torso modestly in a {top_desc}, never shirtless or bare-chested."
        )
    if not has_bottom:
        bottom_desc = complement_bottom.strip() or _dynamic_missing_garment("bottom", selected_garments, avatar_gender, occasion)[1]
        complement_clauses.append(
            f"DEFAULT LOWER GARMENT (NO BOTTOM UPLOADED): Dress the lower body cleanly in {bottom_desc}."
        )

    extra_footwear_hint = (
        "Complete the look with clean understated neutral/black footwear matching the outfit."
        if not has_shoes
        else "Keep the exact footwear from its reference accessory image fully visible on both feet planted naturally on the studio floor."
    )
    tuck_note = (
        "Drape the upper garment cleanly and naturally at the waist over the lower trousers."
        if has_top_and_bottom or (not has_top or not has_bottom)
        else ""
    )
    anti_old_shirt = f" Replace and omit the old {original_worn_clothes}." if original_worn_clothes else ""
    model_gender_phrase = _describe_model_gender(avatar_gender)
    gender_person_noun = "man" if avatar_gender == "male" else ("woman" if avatar_gender == "female" else "person")

    # Combine forensic face details into biometric reference block
    forensic_blob = (forensic_face_details or "").strip()
    avatar_desc_blob = (avatar_summary or "").strip()
    identity_details = f"{avatar_desc_blob}. {forensic_blob}".strip(". ") if forensic_blob else avatar_desc_blob

    # Strict eyeglasses directive: match avatar's exact eyewear state
    combined_face_text = f"{avatar_desc_blob} {forensic_blob}".lower()
    has_glasses_on_face = (
        ("glass" in combined_face_text or "spectacle" in combined_face_text)
        and "no glass" not in combined_face_text
        and "without glass" not in combined_face_text
    )
    if has_glasses_on_face:
        glasses_directive = (
            "MANDATORY EYEWEAR LOCK: The model MUST wear identical eyeglasses on their eyes exactly matching <image1> "
            "(reproduce the exact frame shape, frame material/color, and clear transparent prescription lenses). Do NOT omit glasses."
        )
    else:
        glasses_directive = (
            "EYEWEAR: No eyeglasses worn on the eyes in <image1>; do not add unprompted spectacles or sunglasses."
        )

    style_label, photo_direction = _determine_photography_style(photography_style, occasion, custom_prompt)
    dynamic_pose_desc = _determine_dynamic_pose(
        avatar_gender, occasion, selected_garments, selected_accessories, pose_preference, combo_index=combo_index
    )

    sections = [
        (
            "STRICT FACE & IDENTITY PRESERVATION (<image1> IDENTICAL PERSON LOCK): "
            f"The full-length studio fashion model MUST be the EXACT SAME real {gender_person_noun} shown in <image1>. "
            f"Biometric reference: {identity_details}. "
            "Copy and preserve the exact face directly from <image1> — keeping the identical eye shape, irises, eyebrows, "
            "nose shape and width, lips, cheek contours, smile lines, jawline, chin, facial hair/mustache/stubble, exact hairline and styled hair pattern, "
            "true skin tone, and authentic real age 100% faithfully matching <image1>. "
            f"{glasses_directive} "
            "The face must be seamlessly integrated with the neck and shoulders with 100% natural anatomical blending and zero cutout lines or pasting artifacts. "
            "Keep the face sharp, unretouched, and evenly studio-lit so the person is unmistakably recognizable as the exact individual in <image1>."
        ),
        (
            f"PROFESSIONAL FASHION MODEL & NATURAL HUMAN POSTURE: "
            f"Full-length fashion photography of {model_gender_phrase}. "
            f"{dynamic_pose_desc} "
            "The model fills roughly 86-90% of the vertical frame height from top of hair near upper edge down to shoes on the studio floor. "
            "CRITICAL: The model must look like a real, living, breathing human being captured by a world-class fashion photographer — completely avoid stiff mannequin postures, rigid toy doll aesthetics, or symmetrical arms glued to the sides."
        ),
        (
            f"COMPLETE GARMENT DRESSING:{anti_old_shirt} "
            + " ".join(garment_clauses)
            + (" " + " ".join(complement_clauses) if complement_clauses else "")
            + (" " + tuck_note if tuck_note else "")
            + " Preserve exact garment fabric textures, authentic denim twill weave, cotton knit, stitching, prints, and logo placements."
        ),
    ]
    if accessory_clauses:
        sections.append(
            "JEWELRY & ACCESSORY ANATOMICAL FIT (NATURAL SCALE & PLACEMENT): "
            + "; ".join(accessory_clauses)
            + ". Seat jewelry, chains, necklaces, earrings, watches, rings, and glasses naturally in exact physical contact with bare skin/neck/ears/wrists/bridge at realistic delicate scale with authentic metallic luster and zero clipping or duplication."
        )
    sections.append(
        f"PHOTOGRAPHY DIRECTION ({style_label}): "
        f"{photo_direction} "
        f"{extra_footwear_hint} Realistic grounding contact shadows on the clean studio floor under both shoes. Sharp 8K skin pores, authentic individual hair strands, and realistic textile weave. Anatomically accurate hands with natural relaxed fingers."
    )
    if custom_prompt and custom_prompt.strip():
        sections.append(f"Additional styling instruction: {custom_prompt.strip()}")

    return " ".join(sections), next_tag_idx - 1
