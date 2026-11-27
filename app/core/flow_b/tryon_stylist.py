"""ChatGPT Vision Virtual Try-On Stylist & Multi-Image Qwen Prompt Builder.

Analyzes the user avatar, all uploaded wardrobe garments (tops, bottoms, dresses, etc.),
and all uploaded accessories (watches, shoes, eyewear, jewelry, bags, etc.), pairs them into
stylish outfit combinations, and builds explicit positional `<image1>`, `<image2>`, ...
prompts for Qwen-Image-2.1 multi-image try-on synthesis.
"""
from __future__ import annotations

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

TRYON_SYSTEM_PROMPT = """You are a senior celebrity fashion stylist and textile-forensic visual director creating outfit styling plans and multi-image generation descriptions.

You will inspect labeled images:
- [AVATAR]: The user whose EXACT face, facial features, skin tone, hair, expression, and body proportions MUST remain 100% unchanged.
- [GARMENT_0], [GARMENT_1], ...: Candidate garments (shirts, t-shirts, kurtas, tops, jackets, pants, jeans, trousers, skirts, dresses, etc.). Some may include an optional upload role hint.
- [ACCESSORY_0], [ACCESSORY_1], ... (optional): Accessories such as watches, sneakers, dress shoes, heels, sunglasses, necklaces, earrings, belts, handbags, or hats.

YOUR TASKS:
1. AVATAR IDENTITY PROFILE & HEAD BOX: Concisely describe the person in [AVATAR] (face shape, skin tone, hair style/color, eyeglasses if worn, facial hair if any, gender presentation, natural slim/average/athletic build) so a freshly posed model matches the exact same person. ALSO provide `"avatar_face_box_2d": [ymin, xmin, ymax, xmax]` (integers 0..1000) tightly framing ONLY the person's head (top of hair down to chin/jawline, excluding their original torso shirt below the neck).
2. FORENSIC GARMENT ANALYSIS: Inspect EVERY [GARMENT_i] closely. Carefully distinguish T-shirts (crew-neck / round-neck / V-neck / graphic pullover, zero buttons) vs polo shirts vs button-down shirts vs kurtas vs jackets vs trousers/chinos/jeans. Identify:
   - role: "top" | "bottom" | "full_outfit" | "outerwear"
   - title: specific human-friendly title (e.g., "Navy Crew-Neck Graphic Cotton T-Shirt", "Olive Printed Mandarin Collar Shirt")
   - colors: exact shade names & undertones (avoid vague generic colors)
   - patterns_and_design: forensic description of garment silhouette, neckline/collar type, sleeve length, every print motif, floral/geometric/stripe/check scale and placement, chest artwork/logo, pocket, hem, denim wash/whiskers, pleats, and fabric weave texture.
3. FORENSIC ACCESSORY ANALYSIS: Inspect EVERY [ACCESSORY_j] closely. Identify:
   - accessory_type: e.g., "wristwatch", "sneakers", "leather_shoes", "sunglasses", "necklace", "earrings", "handbag", "belt", "bracelet", "cap"
   - body_placement: exact body location where worn (e.g., "left wrist", "both feet", "on face/eyes", "around neck", "both ears", "around waist", "held in hand")
   - title: concise item title
   - design_details: exact dial/strap material & color, shoe upper/sole design & color, metal finish, lens tint, buckle, or clasp details.
4. CURATED OUTFIT COMBINATIONS:
   - Plan up to the requested number of distinct, well-coordinated outfit combinations pairing tops with complementary bottoms (or showcasing full_outfit items). Do not create duplicate garment combinations unless needed to showcase distinct accessory swaps.
   - IMPORTANT ACCESSORY RULE: Whenever accessories are uploaded, incorporate ALL uploaded non-competing accessories into each outfit combination (only choosing between duplicates of the exact same body slot, such as 2 different pairs of shoes).
   - Reference garments ONLY by valid 0-based `garment_index` (`0..N-1`) and accessories ONLY by valid 0-based `accessory_index` (`0..M-1`).

Return STRICTLY valid JSON:
{
  "avatar_summary": "Detailed visual identity summary of the person in [AVATAR] (gender, build, face, skin tone, hair, eyeglasses/facial hair)",
  "avatar_face_box_2d": [80, 320, 420, 660],
  "garments": [
    {
      "garment_index": 0,
      "role": "top",
      "title": "...",
      "colors": "...",
      "patterns_and_design": "..."
    }
  ],
  "accessories": [
    {
      "accessory_index": 0,
      "accessory_type": "wristwatch",
      "body_placement": "left wrist",
      "title": "...",
      "design_details": "..."
    }
  ],
  "combinations": [
    {
      "title": "Smart Casual Crisp Look",
      "occasion": "smart_casual",
      "style_reason": "Why the colors, patterns, silhouette, and accessories harmonize nicely",
      "selected_garment_indices": [0, 2],
      "selected_accessory_indices": [0, 1]
    }
  ]
}"""

_TOP_HINTS = ("shirt", "tshirt", "t-shirt", "tee", "top", "kurta", "jacket", "blazer", "hoodie", "polo", "sweater", "blouse")
_BOTTOM_HINTS = ("pant", "pants", "jean", "jeans", "trouser", "chino", "cargo", "short", "skirt", "joggers", "palazzo")
_OUTFIT_HINTS = ("dress", "gown", "saree", "lehenga", "jumpsuit", "suit")


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
    if val in ("top", "upper", "upper_clothes", "shirt"):
        return "top"
    if val in ("bottom", "pants", "trousers", "lower", "skirt"):
        return "bottom"
    if val in ("full_outfit", "dress", "outfit", "one_piece"):
        return "full_outfit"
    if val in ("outerwear", "jacket", "coat", "blazer"):
        return "outerwear"
    return fallback


def _default_body_placement(accessory_type: str, filename: str = "") -> str:
    text = f"{accessory_type} {filename}".lower()
    if any(k in text for k in ("watch", "smartwatch", "bangle", "bracelet", "cuff")):
        return "on wrist"
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


def _build_fallback_catalog_and_combos(
    garments: list[dict[str, Any]],
    accessories: list[dict[str, Any]],
    max_combinations: int,
    occasion: str,
) -> dict[str, Any]:
    """Build deterministic wardrobe catalog and balanced outfit pairs when Vision API is offline."""
    garment_catalog: list[dict[str, Any]] = []
    for idx, g in enumerate(garments):
        role = _infer_garment_role(g, idx, len(garments))
        name_stem = os.path.splitext(str(g.get("filename") or f"Garment {idx + 1}"))[0].replace("_", " ").replace("-", " ").strip()
        garment_catalog.append({
            "garment_index": idx,
            "role": role,
            "title": name_stem.title() or f"Garment #{idx + 1} ({role})",
            "colors": "authentic original colors from reference image",
            "patterns_and_design": f"exact garment fabric, print patterns, silhouette, collar/waistband details, and construction visible in Garment #{idx + 1}",
        })

    accessory_catalog: list[dict[str, Any]] = []
    for idx, acc in enumerate(accessories):
        name_stem = os.path.splitext(str(acc.get("filename") or f"Accessory {idx + 1}"))[0].replace("_", " ").replace("-", " ").strip()
        placement = _default_body_placement(name_stem, name_stem)
        accessory_catalog.append({
            "accessory_index": idx,
            "accessory_type": name_stem or "accessory",
            "body_placement": placement,
            "title": name_stem.title() or f"Accessory #{idx + 1}",
            "design_details": f"exact shape, color, strap/material finish, and hardware details visible in Accessory #{idx + 1}",
        })

    raw_combos = _generate_diverse_garment_pairs(garment_catalog, max_combinations)
    all_acc_indices = list(range(len(accessory_catalog)))
    combinations: list[dict[str, Any]] = []
    for c_idx, g_indices in enumerate(raw_combos):
        titles = [garment_catalog[i]["title"] for i in g_indices]
        acc_note = f" + {len(all_acc_indices)} matching accessory/ies" if all_acc_indices else ""
        combinations.append({
            "title": " + ".join(titles) or f"Curated Look #{c_idx + 1}",
            "occasion": occasion or "smart_casual",
            "style_reason": f"Balanced pairing of {', '.join(titles)}{acc_note}.",
            "selected_garment_indices": g_indices,
            "selected_accessory_indices": list(all_acc_indices),
        })

    return {
        "avatar_summary": "Person from avatar photo with matching facial features, skin tone, hairstyle, eyewear/facial hair, and natural build",
        "avatar_face_box_2d": None,
        "garments": garment_catalog,
        "accessories": accessory_catalog,
        "combinations": combinations,
        "prompt_source": "deterministic_fallback",
    }


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


_ROLE_SORT_ORDER = {"full_outfit": 0, "top": 1, "outerwear": 2, "bottom": 3}


def _generate_diverse_garment_pairs(garment_catalog: list[dict[str, Any]], max_combinations: int) -> list[list[int]]:
    """Pair tops with bottoms (and include full outfits), prioritizing distinct items first."""
    tops = [g["garment_index"] for g in garment_catalog if g["role"] in ("top", "outerwear")]
    bottoms = [g["garment_index"] for g in garment_catalog if g["role"] == "bottom"]
    fulls = [g["garment_index"] for g in garment_catalog if g["role"] == "full_outfit"]

    candidates: list[list[int]] = []
    if tops and bottoms:
        # Prioritize diagonal pairings first so 2 shirts + 2 pants uses all 4 garments across 2 looks
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


def _normalize_vision_plan(
    parsed: dict[str, Any],
    garments: list[dict[str, Any]],
    accessories: list[dict[str, Any]],
    max_combinations: int,
    occasion: str,
    model_name: str,
) -> dict[str, Any]:
    """Validate and sanitize ChatGPT Vision JSON indices, catalogs, and accessory inclusion."""
    fallback = _build_fallback_catalog_and_combos(garments, accessories, max_combinations, occasion)

    garment_by_idx = {g["garment_index"]: g for g in fallback["garments"]}
    for entry in parsed.get("garments") or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("garment_index"))
        except (TypeError, ValueError):
            continue
        if idx in garment_by_idx:
            base = garment_by_idx[idx]
            garment_by_idx[idx] = {
                "garment_index": idx,
                "role": _normalize_role(entry.get("role"), base["role"]),
                "title": str(entry.get("title") or base["title"]).strip(),
                "colors": str(entry.get("colors") or base["colors"]).strip(),
                "patterns_and_design": str(entry.get("patterns_and_design") or base["patterns_and_design"]).strip(),
            }

    acc_by_idx = {a["accessory_index"]: a for a in fallback["accessories"]}
    for entry in parsed.get("accessories") or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("accessory_index"))
        except (TypeError, ValueError):
            continue
        if idx in acc_by_idx:
            base = acc_by_idx[idx]
            acc_type = str(entry.get("accessory_type") or base["accessory_type"]).strip()
            acc_by_idx[idx] = {
                "accessory_index": idx,
                "accessory_type": acc_type,
                "body_placement": str(entry.get("body_placement") or _default_body_placement(acc_type)).strip(),
                "title": str(entry.get("title") or base["title"]).strip(),
                "design_details": str(entry.get("design_details") or base["design_details"]).strip(),
            }

    garment_catalog = [garment_by_idx[i] for i in range(len(garments))]
    accessory_catalog = [acc_by_idx[i] for i in range(len(accessories))]

    valid_g_range = set(range(len(garments)))
    valid_a_range = set(range(len(accessories)))
    cleaned_combos: list[dict[str, Any]] = []
    seen_signatures: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    for raw_c in (parsed.get("combinations") or []):
        if not isinstance(raw_c, dict):
            continue
        g_sel: list[int] = []
        for idx in (raw_c.get("selected_garment_indices") or []):
            try:
                ival = int(idx)
            except (TypeError, ValueError):
                continue
            if ival in valid_g_range and ival not in g_sel:
                g_sel.append(ival)
        if not g_sel:
            continue

        # Ensure upper / full_outfit items are always ordered first (<image2>) before bottoms (<image3>)
        g_sel.sort(key=lambda idx: (_ROLE_SORT_ORDER.get(garment_by_idx[idx]["role"], 2), idx))

        a_sel: list[int] = []
        for idx in (raw_c.get("selected_accessory_indices") or []):
            try:
                ival = int(idx)
            except (TypeError, ValueError):
                continue
            if ival in valid_a_range and ival not in a_sel:
                a_sel.append(ival)

        # Guarantee all uploaded non-conflicting accessories are worn if Vision missed unique items
        if len(valid_a_range) > 0:
            used_slots = {acc_by_idx[i]["body_placement"].lower() for i in a_sel}
            for a_idx in range(len(accessories)):
                slot = acc_by_idx[a_idx]["body_placement"].lower()
                if a_idx not in a_sel and slot not in used_slots:
                    a_sel.append(a_idx)
                    used_slots.add(slot)

        # Keep condition images within Qwen's 10-image ceiling (1 avatar + garments + accessories)
        max_acc = max(0, MAX_TRYON_CONDITION_IMAGES - 1 - len(g_sel))
        a_sel = a_sel[:max_acc]

        sig = (tuple(g_sel), tuple(a_sel))
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        cleaned_combos.append({
            "title": str(raw_c.get("title") or f"Styled Look #{len(cleaned_combos) + 1}").strip(),
            "occasion": str(raw_c.get("occasion") or occasion or "smart_casual").strip(),
            "style_reason": str(raw_c.get("style_reason") or "Curated wardrobe coordination.").strip(),
            "selected_garment_indices": g_sel,
            "selected_accessory_indices": a_sel,
        })
        if len(cleaned_combos) >= max_combinations:
            break

    if not cleaned_combos:
        repaired = _generate_diverse_garment_pairs(garment_catalog, max_combinations)
        all_a = list(range(len(accessory_catalog)))
        for c_idx, g_sel in enumerate(repaired):
            titles = [garment_by_idx[i]["title"] for i in g_sel]
            cleaned_combos.append({
                "title": " + ".join(titles) or f"Look #{c_idx + 1}",
                "occasion": occasion or "smart_casual",
                "style_reason": "Harmonized outfit pairing from uploaded wardrobe items.",
                "selected_garment_indices": g_sel,
                "selected_accessory_indices": all_a[: max(0, MAX_TRYON_CONDITION_IMAGES - 1 - len(g_sel))],
            })

    return {
        "avatar_summary": str(parsed.get("avatar_summary") or fallback["avatar_summary"]).strip(),
        "avatar_face_box_2d": _validate_face_box_2d(parsed.get("avatar_face_box_2d")),
        "garments": garment_catalog,
        "accessories": accessory_catalog,
        "combinations": cleaned_combos[:max_combinations],
        "prompt_source": f"openai:{model_name}",
    }


def _append_vision_item(
    user_content: list[dict[str, Any]],
    tag: str,
    image: Image.Image,
) -> None:
    user_content.append({"type": "text", "text": tag})
    user_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{_encode_image_for_vision(image, max_edge=768)}",
            "detail": "high",
        },
    })


def analyze_wardrobe_and_plan_tryon(
    avatar_img: Image.Image,
    garments: list[dict[str, Any]],
    accessories: list[dict[str, Any]] | None = None,
    max_combinations: int = 2,
    occasion: str = "smart_casual",
) -> dict[str, Any]:
    """Analyze avatar + wardrobe garments + accessories with ChatGPT Vision and curate outfit combos."""
    accessories = accessories or []
    max_combinations = max(1, min(int(max_combinations or 2), 4))

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; using deterministic try-on combination fallback.")
        return _build_fallback_catalog_and_combos(garments, accessories, max_combinations, occasion)

    model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Plan up to {max_combinations} stylish outfit combination(s) for occasion='{occasion}'. "
                f"There are {len(garments)} garment(s) and {len(accessories)} accessory/accessories. "
                "Analyze every pattern, weave, print motif, color tone, cut, and accessory design in forensic detail, "
                "and include uploaded accessories in their appropriate body placement."
            ),
        }
    ]
    _append_vision_item(user_content, "[AVATAR] User whose exact face and identity must remain identical:", avatar_img)

    for g in garments:
        idx = g["index"]
        hint = g.get("role_hint", "auto")
        _append_vision_item(user_content, f"[GARMENT_{idx}] (role_hint={hint}):", g["image"])

    for acc in accessories:
        idx = acc["index"]
        _append_vision_item(user_content, f"[ACCESSORY_{idx}]:", acc["image"])

    payload = {
        "model": model_name,
        "temperature": 0.2,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": TRYON_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        parsed = json.loads(data["choices"][0]["message"]["content"])
        return _normalize_vision_plan(parsed, garments, accessories, max_combinations, occasion, model_name)
    except Exception as exc:
        logger.warning("ChatGPT Vision try-on stylist failed (%s); falling back to deterministic pairings.", exc)
        return _build_fallback_catalog_and_combos(garments, accessories, max_combinations, occasion)


def build_qwen_tryon_prompt(
    avatar_summary: str,
    selected_garments: list[dict[str, Any]],
    selected_accessories: list[dict[str, Any]],
    occasion: str = "smart_casual",
    custom_prompt: str | None = None,
) -> tuple[str, int]:
    """Build positional `<image1>`, `<image2>`, ... multi-image Qwen-2.1 virtual try-on prompt.

    Image index mapping matches `[face_identity_ref, *garment_imgs, *accessory_imgs]`:
      - `<image1>`: Clean head/face identity reference (strictly no torso clothing or background pose copied)
      - `<image2>..`: Selected garments from `selected_garments` (top/full outfit first, bottom next)
      - `<imageK>..`: Selected accessories from `selected_accessories`
    """
    garment_clauses: list[str] = []
    next_tag_idx = 2
    for g in selected_garments:
        tag = f"<image{next_tag_idx}>"
        next_tag_idx += 1
        garment_clauses.append(
            f"For the {g['role']} garment, dress the model EXCLUSIVELY in the EXACT {g['title']} shown in {tag}: "
            f"colors {g['colors']}; {g['patterns_and_design']}. "
            f"Faithfully transfer every authentic print pattern, graphic, stripe, neckline/collar cut, sleeve length, pocket, and fabric drape directly from {tag}"
        )

    accessory_clauses: list[str] = []
    for a in selected_accessories:
        if next_tag_idx > MAX_TRYON_CONDITION_IMAGES:
            break
        tag = f"<image{next_tag_idx}>"
        next_tag_idx += 1
        accessory_clauses.append(
            f"Include the EXACT {a['title']} ({a['accessory_type']}) from {tag} worn {a['body_placement']}: "
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
        "Fit the upper garment cleanly and naturally over the waistband of the lower trousers with realistic fabric drape."
        if has_top_and_bottom
        else ""
    )

    sections = [
        (
            "STUDIO E-COMMERCE FASHION CATALOG PHOTOGRAPH (MYNTRA / ZARA / AJIO LOOKBOOK POSE): "
            "Generate a completely fresh, full-length standing fashion model photograph facing 0-degree straight forward toward the camera "
            "with level horizontal shoulders, upright spine, relaxed arms hanging naturally along both sides, and balanced head-to-toe framing. "
            "Place the model in a clean, spotless, completely uniform warm alabaster-ivory studio backdrop (#F4F1EC) with soft natural studio shadows on the floor and ZERO background lines, poles, or streaks."
        ),
        (
            "HEAD & IDENTITY RE-GENERATION (NO CUTOUT PASTE, FRESH STRAIGHT FACE MATCH): "
            f"Match ONLY the facial identity of the person in <image1> ({avatar_summary}) — matching their eyes, eyeglasses, nose, lips, jawline, mustache/beard, hair style, and warm skin tone — "
            "while FRESHLY re-generating their head facing straight forward upright at 0-degree level eye alignment looking naturally at the camera! "
            "DO NOT paste <image1> as a raw cutout, DO NOT copy any crooked head tilt, leaning posture, neck shadow, or dim night lighting from <image1>. "
            "Light the face, neck, and jawline smoothly with clean even daylight-balanced studio key light so the head looks naturally part of the standing studio model."
        ),
        (
            "COMPLETE GARMENT REPLACEMENT (ZERO ORIGINAL CLOTHING CARRYOVER): "
            "Discard 100% of any clothing originally worn in the reference photo and dress the freshly generated model in this exact curated outfit: "
            + " ".join(garment_clauses)
            + (" " + tuck_note if tuck_note else "")
        ),
    ]
    if accessory_clauses:
        sections.append(
            "ACCESSORY STYLING (WEAR ALL SPECIFIED ACCESSORIES): "
            + "; ".join(accessory_clauses)
            + ". Place each accessory cleanly and naturally at authentic scale without duplicating items."
        )
    sections.append(
        f"FULL-LENGTH QUALITY & REALISM ({occasion} lookbook): "
        f"{extra_footwear_hint} Ensure ample headroom above the hair and full floor visibility under both shoes. "
        "8K crisp photographic texture, natural skin tones, authentic textile grain, and anatomically accurate hands and fingers."
    )
    if custom_prompt and custom_prompt.strip():
        sections.append(f"Additional user styling instruction: {custom_prompt.strip()}")

    return " ".join(sections), next_tag_idx - 1
