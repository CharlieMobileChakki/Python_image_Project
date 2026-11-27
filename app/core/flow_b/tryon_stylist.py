"""ChatGPT Vision Virtual Try-On Stylist & Multi-Image Qwen Prompt Builder.

Analyzes `[AVATAR]`, each uploaded `[GARMENT_i]`, and each uploaded `[ACCESSORY_j]` in
**strictly isolated parallel single-image Vision calls** so clothing worn in the user's
original avatar photo can NEVER bleed into or replace uploaded garment descriptions.
Then curates balanced outfit combinations and constructs pixel-accurate positional
`<image1>`, `<image2>`, ... prompts for Qwen-Image-2.1 virtual try-on synthesis.
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

Examine every facial region deeply:
- `age_and_gender`: Apparent age bracket, gender presentation, ethnic complexion, and natural body build (e.g., "early-50s Indian man, warm medium-olive wheatish skin tone, average build").
- `forensic_face_details`: Detailed geometric description covering:
  1. Forehead & Hairline: exact hairline shape (receding temples, widow's peak, full, etc.), forehead horizontal lines, exact hair color & salt-and-pepper grey ratio, hair length & parting style.
  2. Eyes & Brows: eyebrow thickness/arch/color, upper/lower eyelid shape, eye size & spacing, crow's feet or under-eye bags, exact eyeglasses style ONLY IF actually worn ON the eyes/nose bridge (if glasses hang on shirt, explicitly write "no glasses worn on face").
  3. Nose & Cheeks: nose bridge width, length, bulbous/narrow tip shape, nostril flare, cheek structure, nasolabial smile folds.
  4. Mouth, Jaw & Chin: lip fullness, mustache/beard stubble pattern & grey mix, jawline width, chin shape, visible neck/ear proportions.
- `avatar_face_box_2d`: [ymin, xmin, ymax, xmax] integers (0..1000) framing strictly TOP OF HAIR down to CHIN / UPPER JAWLINE (exclude shoulders and shirt collar below chin!).

Return STRICTLY valid JSON:
{
  "avatar_summary": "...",
  "forensic_face_details": "...",
  "original_worn_clothes": "...",
  "avatar_face_box_2d": [80, 320, 420, 660]
}"""

GARMENT_ANALYSIS_SYSTEM_PROMPT = """You are a master e-commerce textile and apparel catalog director performing a deep pixel-by-pixel inspection of ONE standalone garment image.

Classify the garment into `role`: "top" | "bottom" | "full_outfit" | "outerwear".
Honor the caller's optional `role_hint` if valid unless unmistakably contradicted.

Inspect EVERY visual pixel carefully:
- `title`: Specific product title (e.g., "Optic White Crimson-Trimmed Athletic Football Jersey T-Shirt", "Beige Slim-Fit Cotton Chinos").
- `colors`: Exact dye hues, trim colors, striping colors, and undertones (never vague generic names).
- `patterns_and_design`: Exhaustive pixel-level description covering silhouette/fit, exact neckline or collar construction (e.g., crew-neck round collar vs V-neck vs classic button-down collar, buttoned or strictly pullover/buttonless), sleeve length & cuffs, chest/side graphics, logos/crests (specify exact side left/right, color, shape — e.g. small black sportswear 3-bars logo on wearer's right chest, circular gold-crowned crest on left chest, 3 crimson shoulder stripes), prints, embroidery, weave/knit texture, pockets, waistband/hem.

Return STRICTLY valid JSON:
{
  "role": "top",
  "title": "...",
  "colors": "...",
  "patterns_and_design": "..."
}"""

ACCESSORY_ANALYSIS_SYSTEM_PROMPT = """You are a luxury accessories catalog director performing a deep pixel-level inspection of ONE standalone fashion accessory.

Identify:
- `accessory_type`: e.g., "wristwatch", "sneakers", "leather_shoes", "sunglasses", "necklace", "earrings", "belt", "handbag", "bracelet", "cap".
- `body_placement`: exact natural wearing position (e.g., "left wrist", "worn on both feet", "on face/eyes", "around neck", "both ears", "around waist").
- `title`: Specific concise title.
- `design_details`: Exhaustive visual details (case finish, dial color/markers, strap texture/color; shoe upper material, toe box, side stripes/logos, midsole/outsole shape; metal karat luster, stones, lens tint).

Return STRICTLY valid JSON:
{
  "accessory_type": "...",
  "body_placement": "...",
  "title": "...",
  "design_details": "..."
}"""

COMBO_CURATION_SYSTEM_PROMPT = """You are a celebrity fashion stylist curating outfit combinations from an already-analyzed catalog JSON.

Rules:
- Create up to `max_combinations` distinct, well-coordinated looks for the requested `occasion`.
- Pair complementary tops with bottoms (and include full_outfit entries). Order upper/full garments before bottoms.
- Incorporate ALL uploaded non-competing accessories in every look (only pick between competing items of the same body slot, like 2 pairs of shoes).
- Reference ONLY valid 0-based `garment_index` and `accessory_index` integers from the catalog.

Return STRICTLY valid JSON:
{
  "combinations": [
    {
      "title": "Smart Casual Sporty Crisp",
      "occasion": "smart_casual",
      "style_reason": "Why colors, silhouette, and accessories work together.",
      "selected_garment_indices": [0, 1],
      "selected_accessory_indices": [0, 1]
    }
  ]
}"""

_TOP_HINTS = ("shirt", "tshirt", "t-shirt", "tee", "jersey", "top", "kurta", "jacket", "blazer", "hoodie", "polo", "sweater", "blouse")
_BOTTOM_HINTS = ("pant", "pants", "jean", "jeans", "trouser", "chino", "cargo", "short", "skirt", "joggers", "palazzo")
_OUTFIT_HINTS = ("dress", "gown", "saree", "lehenga", "jumpsuit", "suit")
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


def _default_body_placement(accessory_type: str, filename: str = "") -> str:
    text = f"{accessory_type} {filename}".lower()
    if any(k in text for k in ("watch", "smartwatch", "bangle", "bracelet", "cuff")):
        return "on left wrist"
    if any(k in text for k in ("shoe", "sneaker", "boot", "heel", "loafer", "sandal", "footwear")):
        return "worn on both feet"
    if any(k in text for k in ("glass", "sunglass", "shades", "eyewear", "spectacle")):
        return "worn naturally on face over eyes"
    if any(k in text for k in ("necklace", "chain", "pendant", "choker", "tie", "scarf")):
        return "draped cleanly around neck"
    if any(k in text for k in ("earring", "jhumka", "stud")):
        return "worn on both ears"
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
    for f_idx in fulls:
        if [f_idx] not in candidates:
            candidates.append([f_idx])
    if not candidates:
        candidates = [[g["garment_index"]] for g in garment_catalog]
    return candidates[:max_combinations]


def _build_fallback_garment_item(g: dict[str, Any], idx: int, total: int) -> dict[str, Any]:
    role = _infer_garment_role(g, idx, total)
    stem = os.path.splitext(str(g.get("filename") or f"Garment {idx + 1}"))[0].replace("_", " ").replace("-", " ").strip()
    return {
        "garment_index": idx,
        "role": role,
        "title": stem.title() or f"Garment #{idx + 1} ({role})",
        "colors": "authentic original colors from reference image",
        "patterns_and_design": f"exact garment fabric, print patterns, neckline/collar cut, logos, silhouette, and construction visible in Garment #{idx + 1}",
    }


def _build_fallback_accessory_item(acc: dict[str, Any], idx: int) -> dict[str, Any]:
    stem = os.path.splitext(str(acc.get("filename") or f"Accessory {idx + 1}"))[0].replace("_", " ").replace("-", " ").strip()
    placement = _default_body_placement(stem, stem)
    return {
        "accessory_index": idx,
        "accessory_type": stem or "accessory",
        "body_placement": placement,
        "title": stem.title() or f"Accessory #{idx + 1}",
        "design_details": f"exact shape, color, strap/material finish, and hardware details visible in Accessory #{idx + 1}",
    }


def _build_deterministic_combinations(
    garment_catalog: list[dict[str, Any]],
    accessory_catalog: list[dict[str, Any]],
    max_combinations: int,
    occasion: str,
) -> list[dict[str, Any]]:
    raw_combos = _generate_diverse_garment_pairs(garment_catalog, max_combinations)
    garment_by_idx = {g["garment_index"]: g for g in garment_catalog}
    all_acc_indices = [a["accessory_index"] for a in accessory_catalog][: max(0, MAX_TRYON_CONDITION_IMAGES - 1)]
    combinations: list[dict[str, Any]] = []
    for c_idx, g_indices in enumerate(raw_combos):
        g_sorted = sorted(g_indices, key=lambda idx: (_ROLE_SORT_ORDER.get(garment_by_idx[idx]["role"], 2), idx))
        titles = [garment_by_idx[i]["title"] for i in g_sorted]
        max_acc = max(0, MAX_TRYON_CONDITION_IMAGES - 1 - len(g_sorted))
        acc_sel = all_acc_indices[:max_acc]
        acc_note = f" + {len(acc_sel)} matching accessory/ies" if acc_sel else ""
        combinations.append({
            "title": " + ".join(titles) or f"Curated Look #{c_idx + 1}",
            "occasion": occasion or "smart_casual",
            "style_reason": f"Balanced coordination pairing {', '.join(titles)}{acc_note}.",
            "selected_garment_indices": g_sorted,
            "selected_accessory_indices": acc_sel,
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
    """Analyze the avatar photo in complete isolation for biometric facial geometry & head crop."""
    default_avatar = {
        "avatar_summary": "Same individual from the reference portrait preserving exact age, facial structure, hairline, skin tone, and expression",
        "forensic_face_details": "exact eye shape, brow thickness, nose bridge width, nasolabial contours, lip shape, jawline, hairline, and natural skin texture",
        "original_worn_clothes": "",
        "avatar_face_box_2d": None,
    }
    try:
        parsed = _call_openai_json(
            AVATAR_ANALYSIS_SYSTEM_PROMPT,
            _vision_message_for_image(
                "[AVATAR] Inspect every facial landmark, hairline, age marker, and skin tone deeply; provide tight head box above shirt.",
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
                f"[GARMENT_{idx}] (caller role_hint={hint}). Analyze ONLY this garment's colors, collar/neckline, buttons/pullover front, graphics/logos, prints, and fabric weave in extreme pixel detail.",
                g_entry["image"],
            ),
            api_key,
            model_name,
            base_url,
            max_tokens=550,
        )
        return {
            "garment_index": idx,
            "role": _normalize_role(parsed.get("role"), fallback["role"]),
            "title": str(parsed.get("title") or fallback["title"]).strip(),
            "colors": str(parsed.get("colors") or fallback["colors"]).strip(),
            "patterns_and_design": str(parsed.get("patterns_and_design") or fallback["patterns_and_design"]).strip(),
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
                f"[ACCESSORY_{idx}] Analyze ONLY this accessory's exact type, wearing placement, colors, dials/stripes/soles, and materials.",
                acc_entry["image"],
            ),
            api_key,
            model_name,
            base_url,
            max_tokens=400,
        )
        acc_type = str(parsed.get("accessory_type") or fallback["accessory_type"]).strip()
        return {
            "accessory_index": idx,
            "accessory_type": acc_type,
            "body_placement": str(parsed.get("body_placement") or _default_body_placement(acc_type)).strip(),
            "title": str(parsed.get("title") or fallback["title"]).strip(),
            "design_details": str(parsed.get("design_details") or fallback["design_details"]).strip(),
        }
    except Exception as exc:
        logger.warning("Isolated accessory #%d Vision analysis failed (%s); using fallback.", idx, exc)
        return fallback


def _sanitize_combinations(
    raw_combos: list[Any],
    garment_catalog: list[dict[str, Any]],
    accessory_catalog: list[dict[str, Any]],
    max_combinations: int,
    occasion: str,
) -> list[dict[str, Any]]:
    """Validate stylist combination indices, enforce top->bottom ordering, and retain missed accessories."""
    garment_by_idx = {g["garment_index"]: g for g in garment_catalog}
    acc_by_idx = {a["accessory_index"]: a for a in accessory_catalog}
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

        cleaned.append({
            "title": str(raw_c.get("title") or f"Styled Look #{len(cleaned) + 1}").strip(),
            "occasion": str(raw_c.get("occasion") or occasion or "smart_casual").strip(),
            "style_reason": str(raw_c.get("style_reason") or "Curated wardrobe coordination.").strip(),
            "selected_garment_indices": g_sel,
            "selected_accessory_indices": a_sel,
        })
        if len(cleaned) >= max_combinations:
            break

    return cleaned or _build_deterministic_combinations(garment_catalog, accessory_catalog, max_combinations, occasion)


def analyze_wardrobe_and_plan_tryon(
    avatar_img: Image.Image,
    garments: list[dict[str, Any]],
    accessories: list[dict[str, Any]] | None = None,
    max_combinations: int = 2,
    occasion: str = "smart_casual",
) -> dict[str, Any]:
    """Analyze avatar, every garment, and every accessory in isolated parallel Vision calls, then curate combos."""
    accessories = accessories or []
    max_combinations = max(1, min(int(max_combinations or 2), 4))

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; using deterministic try-on combination fallback.")
        g_cat = [_build_fallback_garment_item(g, g["index"], len(garments)) for g in garments]
        a_cat = [_build_fallback_accessory_item(a, a["index"]) for a in accessories]
        return {
            "avatar_summary": "Same individual from the reference portrait preserving exact age, facial structure, hairline, and skin tone",
            "original_worn_clothes": "",
            "avatar_face_box_2d": None,
            "garments": g_cat,
            "accessories": a_cat,
            "combinations": _build_deterministic_combinations(g_cat, a_cat, max_combinations, occasion),
            "prompt_source": "deterministic_fallback",
        }

    model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    # Execute isolated per-image Vision calls concurrently so Avatar clothing can never bleed into Garments
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

    possible_pairs = _generate_diverse_garment_pairs(garment_catalog, max_combinations=max(4, max_combinations))
    has_multiple_garment_choices = len(garment_catalog) > 2 or len(possible_pairs) > 1

    combinations: list[dict[str, Any]]
    if has_multiple_garment_choices:
        try:
            combo_req = {
                "occasion": occasion,
                "max_combinations": max_combinations,
                "garments": garment_catalog,
                "accessories": accessory_catalog,
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
                garment_catalog,
                accessory_catalog,
                max_combinations,
                occasion,
            )
        except Exception as exc:
            logger.warning("Combo curation text call failed (%s); using deterministic stylist pairing.", exc)
            combinations = _build_deterministic_combinations(garment_catalog, accessory_catalog, max_combinations, occasion)
    else:
        combinations = _build_deterministic_combinations(garment_catalog, accessory_catalog, max_combinations, occasion)

    return {
        "avatar_summary": avatar_info["avatar_summary"],
        "original_worn_clothes": avatar_info.get("original_worn_clothes", ""),
        "avatar_face_box_2d": avatar_info.get("avatar_face_box_2d"),
        "garments": garment_catalog,
        "accessories": accessory_catalog,
        "combinations": combinations[:max_combinations],
        "prompt_source": avatar_info.get("source") or f"openai:{model_name}",
    }


def build_qwen_tryon_prompt(
    avatar_summary: str,
    selected_garments: list[dict[str, Any]],
    selected_accessories: list[dict[str, Any]],
    occasion: str = "smart_casual",
    custom_prompt: str | None = None,
    original_worn_clothes: str = "",
) -> tuple[str, int]:
    """Build positional `<image1>`, `<image2>`, ... multi-image Qwen-2.1 virtual try-on prompt."""
    garment_clauses: list[str] = []
    next_tag_idx = 2
    for g in selected_garments:
        tag = f"<image{next_tag_idx}>"
        next_tag_idx += 1
        garment_clauses.append(
            f"For the {g['role']} garment, dress the model EXCLUSIVELY in the EXACT {g['title']} from {tag}: "
            f"colors {g['colors']}; {g['patterns_and_design']}. "
            f"Replicate every authentic stripe, crest, logo, pattern motif, collar/neckline construction, sleeve length, and fabric texture directly from {tag}"
        )

    accessory_clauses: list[str] = []
    for a in selected_accessories:
        if next_tag_idx > MAX_TRYON_CONDITION_IMAGES:
            break
        tag = f"<image{next_tag_idx}>"
        next_tag_idx += 1
        accessory_clauses.append(
            f"Wear the EXACT {a['title']} ({a['accessory_type']}) from {tag} {a['body_placement']}: "
            f"{a['design_details']}"
        )

    has_top_and_bottom = (
        any(g["role"] in ("top", "outerwear") for g in selected_garments)
        and any(g["role"] == "bottom" for g in selected_garments)
    )
    has_shoes = any(
        any(k in f"{a.get('accessory_type', '')} {a.get('body_placement', '')}".lower() for k in ("shoe", "sneaker", "boot", "heel", "loafer", "sandal", "feet"))
        for a in selected_accessories
    )

    extra_footwear_hint = (
        "Complete the look with clean understated neutral footwear matching the outfit."
        if not has_shoes
        else "Keep the exact footwear from its reference accessory image fully visible on both feet planted naturally on the studio floor."
    )
    tuck_note = (
        "Drape the upper garment cleanly and naturally at the waist over the lower trousers."
        if has_top_and_bottom
        else ""
    )
    anti_old_shirt = (
        f" NEVER draw or keep any element of the old {original_worn_clothes}."
        if original_worn_clothes
        else ""
    )

    sections = [
        (
            "STUDIO E-COMMERCE FASHION CATALOG PHOTOGRAPH (MYNTRA / ZARA LOOKBOOK POSE): "
            "Generate a fresh full-length standing fashion model photograph facing 0-degree straight forward toward the camera "
            "with level horizontal shoulders, upright relaxed posture, and arms hanging naturally along both sides. "
            "Clean, smooth, completely spotless warm alabaster-ivory studio seamless backdrop (#F4F1EC) with subtle floor shadows and zero background clutter or lines."
        ),
        (
            "EXACT BIOMETRIC FACIAL IDENTITY PRESERVATION (SAME PERSON FROM <image1>, NO DE-AGING, NO CUTOUT PASTE): "
            f"The standing model MUST be the EXACT SAME human individual shown in <image1> ({avatar_summary}). "
            "Faithfully preserve their true age, exact hairline & salt-and-pepper hair pattern, exact eye spacing/shape/eyelids, nose bridge & tip width, cheekbone contours, nasolabial folds, mouth corners, jawline, mustache/stubble, and skin undertones from <image1>. "
            "Do NOT replace with a younger generic fashion model face and do NOT airbrush away authentic facial structure. "
            "Freshly align head straight upright facing directly forward at 0-degree eye level with smooth neutral studio daylight key lighting blending head, neck, and shoulders seamlessly."
        ),
        (
            "COMPLETE GARMENT REPLACEMENT (PIXEL-EXACT CLOTHING TRANSFER): "
            f"Dress the model ONLY in the garments specified below.{anti_old_shirt} "
            + " ".join(garment_clauses)
            + (" " + tuck_note if tuck_note else "")
        ),
    ]
    if accessory_clauses:
        sections.append(
            "ACCESSORY STYLING (WEAR ALL SPECIFIED ACCESSORIES): "
            + "; ".join(accessory_clauses)
            + ". Render each accessory at authentic scale without duplication."
        )
    sections.append(
        f"FULL-LENGTH STUDIO CLARITY ({occasion} aesthetic): "
        f"{extra_footwear_hint} Full head-to-toe framing with ample space above the head and below both shoes, crisp 8K skin & textile micro-detail, anatomically accurate hands and fingers."
    )
    if custom_prompt and custom_prompt.strip():
        sections.append(f"Additional styling instruction: {custom_prompt.strip()}")

    return " ".join(sections), next_tag_idx - 1
