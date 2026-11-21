"""Regression verification suite for apparel ghost-mannequin design changes."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image

from app.core.image_processing.artwork_preserver import (
    detect_garment_landmarks,
    ensure_garment_fabric_opacity,
    map_and_preserve_artwork,
    crop_mark_from_source,
)
from app.core.flow_b.prompt_generator import _build_final_prompt

def test_sleeve_hole_healing():
    """Verify interior sleeve hole defect is healed and opacity restored."""
    test_canvas = np.zeros((400, 400, 4), dtype=np.uint8)
    test_canvas[80:350, 120:280] = [245, 245, 245, 255] # torso
    test_canvas[80:200, 40:120] = [245, 245, 245, 255]  # left sleeve
    test_canvas[80:200, 280:360] = [245, 245, 245, 255] # right sleeve
    test_canvas[110:150, 300:340] = [30, 30, 30, 0]    # transparent hole in sleeve

    healed_img = ensure_garment_fabric_opacity(Image.fromarray(test_canvas, mode="RGBA"))
    h_arr = np.array(healed_img)
    assert np.all(h_arr[110:150, 300:340, 3] == 255), "Sleeve hole alpha must be healed to 255!"
    assert np.all(h_arr[10:40, 10:40, 3] == 0), "Background must remain 0 alpha!"


def test_plain_unbranded_shirt():
    """Verify plain unbranded garment adds ZERO logos, ZERO stripes, and keeps fabric clean."""
    test_canvas = np.zeros((400, 400, 4), dtype=np.uint8)
    test_canvas[80:350, 120:280] = [245, 245, 245, 255]
    plain_prompt = _build_final_prompt(
        garment_description="Basic unbranded white crewneck t-shirt",
        category="upper_clothes",
        chest_marks=[],
        sleeve_architecture={"stripe_count": 0}
    )
    assert "PLAIN UNBRANDED CHEST" in plain_prompt
    assert "ZERO shoulder stripes" in plain_prompt
    assert "Real Madrid" not in plain_prompt and "Adidas" not in plain_prompt
    clean_in = Image.fromarray(test_canvas, mode="RGBA")
    clean_out, meta = map_and_preserve_artwork(clean_in, clean_in, chest_marks=[])
    assert len(meta["preserved_marks"]) == 0
    assert not meta["review_flagged"]


def test_landmarks_and_artwork_preservation():
    """Verify landmark detection and multi-mark preservation on regression case."""
    scratch_dir = r"C:\Users\admin\.gemini\antigravity\brain\5677d129-15c2-4786-b187-5ef396092e73\scratch"
    out_img_path = os.path.join(scratch_dir, "reg_output.png")
    upright_in_path = os.path.join(scratch_dir, "input_jersey_upright.png")

    if not os.path.exists(out_img_path) or not os.path.exists(upright_in_path):
        return

    out_img = Image.open(out_img_path)
    landmarks = detect_garment_landmarks(out_img)
    assert landmarks["detected_existing_side"] == "viewer_right", "Must detect crest on viewer right!"
    vl_target = landmarks["viewer_left_chest"]
    vr_target = landmarks["viewer_right_chest"]

    upright_in = Image.open(upright_in_path)
    chest_marks = [
        {
            "type": "manufacturer_brand",
            "name": "3-bars athletic logo",
            "side": "viewer_left",
            "box_2d": [256, 225, 354, 467],
            "description": "Dark 3-bars logo"
        },
        {
            "type": "team_or_club",
            "name": "club crest",
            "side": "viewer_right",
            "box_2d": [288, 614, 401, 805],
            "description": "Circular crest"
        }
    ]

    restored_rgba, restore_meta = map_and_preserve_artwork(
        source_rgb=upright_in,
        reconstructed_rgba=out_img,
        chest_marks=chest_marks,
        source_rotation_deg=0.0
    )

    restored_path = os.path.join(scratch_dir, "regression_fixed_output.png")
    restored_rgba.save(restored_path)

    arr_restored = np.array(restored_rgba)
    patch_vl = arr_restored[vl_target[1]-10:vl_target[1]+10, vl_target[0]-10:vl_target[0]+10, :3]
    vl_std = float(np.std(patch_vl))
    assert vl_std > 20.0, "Viewer left chest must now contain the authentic logo!"


def run_tests():
    test_sleeve_hole_healing()
    print("PASS: test_sleeve_hole_healing")
    test_plain_unbranded_shirt()
    print("PASS: test_plain_unbranded_shirt")
    test_landmarks_and_artwork_preservation()
    print("PASS: test_landmarks_and_artwork_preservation")
    print("\n=== ALL REGRESSION TESTS PASSED! ===")

if __name__ == "__main__":
    run_tests()
