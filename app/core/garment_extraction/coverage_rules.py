"""Garment coverage calculation and priority rules."""
from __future__ import annotations

# Configurable Priority Thresholds for Mode A (Independent Garments)
GARMENT_PRIORITY_THRESHOLDS = {
    "ignore_below": 30.0,
    "low_priority": 30.0,
    "second_priority": 50.0,
    "first_priority": 70.0,
}
MALE_GARMENT_THRESHOLDS = GARMENT_PRIORITY_THRESHOLDS

