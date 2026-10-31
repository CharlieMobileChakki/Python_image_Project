"""Unit tests for Flow B multi-class union, alias mapping, and soft-alpha logic.

Can be run via:
    python -m unittest tests/test_flow_b_features.py
or via pytest.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.test_segformer_flow_b import (
    EXPLICIT_ALIASES,
    compute_soft_alpha,
    resolve_single_class,
    resolve_target_classes,
    validate_alpha_thresholds,
)


class TestFlowBFeatures(unittest.TestCase):
    """Test suite covering Flow B class resolution, alias mapping, and alpha thresholds."""

    def setUp(self) -> None:
        # Standard SegFormer ATR 18 classes
        self.mock_id2label = {
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

    # 1. Single class resolution
    def test_resolve_single_class_exact(self) -> None:
        cid, label = resolve_single_class(self.mock_id2label, "pants")
        self.assertEqual(cid, 6)
        self.assertEqual(label, "Pants")

    def test_resolve_single_class_normalized(self) -> None:
        cid, label = resolve_single_class(self.mock_id2label, "upper_clothes")
        self.assertEqual(cid, 4)
        self.assertEqual(label, "Upper-clothes")

        cid, label = resolve_single_class(self.mock_id2label, "upper clothes")
        self.assertEqual(cid, 4)
        self.assertEqual(label, "Upper-clothes")

    def test_resolve_single_class_case_insensitive(self) -> None:
        cid, label = resolve_single_class(self.mock_id2label, "SKIRT")
        self.assertEqual(cid, 5)
        self.assertEqual(label, "Skirt")

    # 2. Multi-class resolution
    def test_resolve_multi_classes(self) -> None:
        resolved, alias = resolve_target_classes(self.mock_id2label, "upper-clothes,skirt")
        self.assertIsNone(alias)
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0], (4, "Upper-clothes"))
        self.assertEqual(resolved[1], (5, "Skirt"))

    def test_resolve_multi_classes_with_whitespace(self) -> None:
        resolved, alias = resolve_target_classes(self.mock_id2label, "  pants ,  upper-clothes  ")
        self.assertIsNone(alias)
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0], (6, "Pants"))
        self.assertEqual(resolved[1], (4, "Upper-clothes"))

    def test_resolve_deduplication(self) -> None:
        resolved, alias = resolve_target_classes(self.mock_id2label, "pants, pants, Pants")
        self.assertIsNone(alias)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0], (6, "Pants"))

    # 3. Explicit alias mapping
    def test_explicit_alias_anarkali(self) -> None:
        self.assertIn("anarkali", EXPLICIT_ALIASES)
        self.assertEqual(EXPLICIT_ALIASES["anarkali"], ["upper-clothes", "skirt"])

        resolved, alias = resolve_target_classes(self.mock_id2label, "anarkali")
        self.assertEqual(alias, "anarkali")
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0], (4, "Upper-clothes"))
        self.assertEqual(resolved[1], (5, "Skirt"))

    def test_explicit_alias_case_insensitive(self) -> None:
        resolved, alias = resolve_target_classes(self.mock_id2label, "  ANARKALI  ")
        self.assertEqual(alias, "anarkali")
        self.assertEqual(resolved, [(4, "Upper-clothes"), (5, "Skirt")])

    # 4. Error handling on invalid/empty inputs
    def test_invalid_class_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_target_classes(self.mock_id2label, "pants,space_suit")
        self.assertIn("Unknown garment class 'space_suit'", str(ctx.exception))

    def test_empty_query_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            resolve_target_classes(self.mock_id2label, "")
        with self.assertRaises(ValueError):
            resolve_target_classes(self.mock_id2label, "   ,   ")

    # 5. Alpha threshold validation
    def test_alpha_thresholds_valid(self) -> None:
        # Should not raise
        validate_alpha_thresholds(0.20, 0.80)
        validate_alpha_thresholds(0.0, 1.0)
        validate_alpha_thresholds(0.49, 0.51)

    def test_alpha_thresholds_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_alpha_thresholds(-0.1, 0.8)
        with self.assertRaises(ValueError):
            validate_alpha_thresholds(0.2, 1.2)
        with self.assertRaises(ValueError):
            validate_alpha_thresholds(0.8, 0.2)  # low > high
        with self.assertRaises(ValueError):
            validate_alpha_thresholds(0.5, 0.5)  # low == high

    # 6. Soft alpha linear ramp mapping
    def test_compute_soft_alpha_mapping(self) -> None:
        probs = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0], dtype=np.float32)
        alpha = compute_soft_alpha(probs, alpha_low=0.20, alpha_high=0.80)

        # Expected:
        # <= 0.20 -> 0
        # >= 0.80 -> 255
        # 0.50 -> midpoint (0.50 - 0.20) / (0.80 - 0.20) = 0.3 / 0.6 = 0.5 * 255 = 127.5 -> 128
        self.assertEqual(alpha[0], 0)
        self.assertEqual(alpha[1], 0)
        self.assertEqual(alpha[2], 0)
        self.assertEqual(alpha[3], 128)
        self.assertEqual(alpha[4], 255)
        self.assertEqual(alpha[5], 255)
        self.assertEqual(alpha[6], 255)
        self.assertEqual(alpha.dtype, np.uint8)

    # 7. Composite mask union logic
    def test_composite_mask_union(self) -> None:
        pred_seg = np.array([
            [0, 4, 4],
            [5, 5, 0],
            [6, 6, 9],
        ], dtype=np.uint8)
        # Select upper-clothes (4) and skirt (5)
        target_ids = [4, 5]
        binary_mask = np.isin(pred_seg, target_ids).astype(np.uint8) * 255

        expected = np.array([
            [0, 255, 255],
            [255, 255, 0],
            [0, 0, 0],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(binary_mask, expected)


if __name__ == "__main__":
    unittest.main()

