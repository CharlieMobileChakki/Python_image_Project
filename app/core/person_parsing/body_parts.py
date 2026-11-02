"""Body part class definitions and semantic indices for SegFormer B2 Clothes."""
from __future__ import annotations

# Full 18-class SegFormer semantic taxonomy:
# 0: Background
# 1: Hat
# 2: Hair
# 3: Sunglasses
# 4: Upper-clothes
# 5: Skirt
# 6: Pants
# 7: Dress
# 8: Belt / Bag
# 9: Left-shoe
# 10: Right-shoe
# 11: Face
# 12: Left-leg
# 13: Right-leg
# 14: Left-arm
# 15: Right-arm
# 16: Bag
# 17: Scarf

BODY_PART_CLASSES = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "face": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
}

NON_GARMENT_BODY_CLASSES = [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]

