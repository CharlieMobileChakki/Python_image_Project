"""Garment taxonomy, semantic labels, and aliases."""
from __future__ import annotations

# ATR 18 Semantic Classes mapping
ATR_CLASSES = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

# Primary garment classes
GARMENT_CLASSES = {
    4: {"key": "upper_clothes", "label": "T-Shirt / Top / Shirt"},
    5: {"key": "skirt", "label": "Skirt / Lehenga"},
    6: {"key": "pants", "label": "Pants / Jeans / Trousers"},
    7: {"key": "dress", "label": "Dress / Gown / Anarkali"},
}

# Garment search and query aliases
GARMENT_ALIASES = {
    "top": "upper_clothes",
    "shirt": "upper_clothes",
    "tshirt": "upper_clothes",
    "t-shirt": "upper_clothes",
    "kurta": "upper_clothes",
    "kurti": "upper_clothes",
    "bottom": "pants",
    "jeans": "pants",
    "trousers": "pants",
    "pajama": "pants",
    "pyjama": "pants",
    "churidar": "pants",
    "gown": "dress",
    "anarkali": "dress",
    "frock": "dress",
    "one-piece": "dress",
    "lehenga": "skirt",
}

