"""Unit and integration tests for Arangtik Garment AI Unified Pipeline."""
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.detector import analyze_semantic_map
from app.core.normalizer import detect_rotation_angle, normalize_to_1024, straighten_garment
from app.core.inpainter import repair_garment_cutout


@pytest.fixture
def api_client(mock_load_model):
    """Test client for Arangtik Garment AI FastAPI application."""
    with TestClient(app) as client:
        yield client


def test_detector_detects_human():
    """Human presence detected when body class pixels exceed threshold."""
    # Create mock 200x200 map
    pred_map = np.zeros((200, 200), dtype=np.uint8)
    # Add face (11) and arms (14, 15)
    pred_map[20:60, 80:120] = 11  # Face: 1600 px
    pred_map[60:120, 60:80] = 14  # Left arm: 1200 px
    # Add upper_clothes (4)
    pred_map[60:140, 80:120] = 4  # 3200 px

    analysis = analyze_semantic_map(pred_map)
    assert analysis["has_human"] is True
    assert analysis["human_pixels"] >= 2800
    assert len(analysis["detected_garments"]) >= 1
    assert analysis["detected_garments"][0]["category"] == "upper_clothes"


def test_detector_flat_lay_no_human():
    """Flat-lay product without human pixels correctly flagged has_human=False."""
    pred_map = np.zeros((200, 200), dtype=np.uint8)
    pred_map[50:150, 50:150] = 4  # Garment only, no face/arms/legs

    analysis = analyze_semantic_map(pred_map)
    assert analysis["has_human"] is False
    assert analysis["human_pixels"] == 0
    assert len(analysis["detected_garments"]) == 1


def test_normalize_to_1024_dimensions_and_mode():
    """Normalization strictly produces 1024x1024 RGBA image with transparent borders."""
    img = Image.new("RGBA", (300, 600), (0, 0, 0, 0))
    # Draw mock garment inside
    for y in range(50, 550):
        for x in range(50, 250):
            img.putpixel((x, y), (200, 100, 50, 255))

    normalized = normalize_to_1024(img, target_size=1024, fill_ratio=0.82)
    assert normalized.size == (1024, 1024)
    assert normalized.mode == "RGBA"

    # Borders should remain fully transparent
    alpha = np.array(normalized.split()[3])
    assert alpha[0, 0] == 0
    assert alpha[1023, 1023] == 0
    assert alpha[512, 512] == 255  # Center has garment


def test_straighten_garment_preserves_dimensions():
    """Straightening rotates the garment cleanly."""
    img = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    for y in range(100, 300):
        for x in range(150, 250):
            img.putpixel((x, y), (100, 200, 150, 255))

    rotated = straighten_garment(img, 10.0)
    assert rotated.mode == "RGBA"


def test_repair_handles_image_width_under_512():
    """repair_garment_cutout must not crash with dimensions < 512 (e.g. 365px)."""
    from unittest.mock import patch
    img = Image.new("RGBA", (365, 600), (100, 150, 200, 255))
    mask = np.zeros((600, 365), dtype=np.uint8)
    mask[100:150, 50:100] = 255

    # Mock inpaint_patch_512 to verify padding and crop logic
    dummy_out = np.zeros((512, 512, 3), dtype=np.uint8)
    with patch("app.core.inpainter.inpaint_patch_512", return_value=(dummy_out, 10.0)) as mock_inpaint:
        repaired, _ = repair_garment_cutout(img, mask)
        assert repaired.size == (365, 600)
        # Check that the patch sent to inpaint_patch_512 was strictly 512x512
        call_args = mock_inpaint.call_args[0]
        crop_img, mask_crop = call_args[0], call_args[1]
        assert crop_img.size == (512, 512)
        assert mask_crop.shape == (512, 512)


def test_compute_arm_occlusion_mask_detects_crossed_arms_and_holes():
    """compute_arm_occlusion_mask must cover the actual crossed arms and internal holes."""
    from app.core.pipeline import compute_arm_occlusion_mask
    pred_map = np.zeros((300, 300), dtype=np.uint8)
    # Garment (7: dress)
    pred_map[50:250, 50:250] = 7
    # Arms crossing waist (14)
    pred_map[120:160, 100:200] = 14

    # Cutout alpha where arms were removed (hole in the middle)
    alpha = np.zeros((300, 300), dtype=np.uint8)
    alpha[50:250, 50:250] = 255
    alpha[120:160, 100:200] = 0  # Missing hole

    mask = compute_arm_occlusion_mask(pred_map, garment_class_id=7, cutout_alpha=alpha)
    assert (mask > 0).sum() > 2000
    # Must strictly cover the missing hole
    assert (mask[120:160, 100:200] > 0).all()


def test_health_endpoint(api_client: TestClient):
    """GET /health returns HTTP 200 and system readiness."""
    res = api_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "arangtik" in data["service"]


def test_api_v1_health_endpoint(api_client: TestClient):
    """GET /api/v1/health returns detailed model loaded status."""
    res = api_client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_process_garment_rejects_invalid_mime(api_client: TestClient):
    """Uploading unsupported mime type returns HTTP 415."""
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    res = api_client.post("/api/v1/process-garment", files=files)
    assert res.status_code == 415


def test_process_garment_rejects_empty_file(api_client: TestClient):
    """Uploading empty file returns HTTP 400."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = api_client.post("/api/v1/process-garment", files=files)
    assert res.status_code == 400


def test_classify_outfit_saree_fused_ensemble():
    """Saree with both upper_clothes and dress must be fused into one complete ensemble."""
    from app.core.detector import classify_outfit_structure
    pred_map = np.zeros((800, 500), dtype=np.uint8)
    pred_map[100:400, 150:350] = 4  # Pallu / upper
    pred_map[350:750, 100:400] = 7  # Pleats / dress

    detected = [
        {"class_id": 4, "category": "upper_clothes", "label": "T-Shirt / Top / Shirt", "pixel_count": 60000},
        {"class_id": 7, "category": "dress", "label": "Dress / Gown / Anarkali", "pixel_count": 120000},
    ]

    res = classify_outfit_structure(pred_map, None, detected, scene_type="mannequin", target_filter="all")
    assert res["outfit_mode"] == "mode_c_traditional_ensemble"
    assert res["ensemble_type"] == "saree"
    assert len(res["planned_items"]) == 1
    saree_item = res["planned_items"][0]
    assert saree_item["key"] == "saree"
    assert saree_item["is_fused"] is True
    assert 4 in saree_item["target_class_ids"]
    assert 7 in saree_item["target_class_ids"]
    assert saree_item["coverage_percentage"] == 100.0


def test_classify_outfit_lehenga_fused_ensemble():
    """Lehenga with choli, skirt, and dupatta must be fused into one complete ensemble."""
    from app.core.detector import classify_outfit_structure
    pred_map = np.zeros((800, 500), dtype=np.uint8)
    pred_map[100:300, 180:320] = 4  # Choli
    pred_map[350:750, 100:400] = 5  # Skirt
    pred_map[200:600, 300:450] = 17 # Scarf

    detected = [
        {"class_id": 4, "category": "upper_clothes", "label": "T-Shirt / Top / Shirt", "pixel_count": 28000},
        {"class_id": 5, "category": "skirt", "label": "Skirt / Lehenga", "pixel_count": 120000},
        {"class_id": 17, "category": "scarf", "label": "Scarf / Dupatta / Pallu", "pixel_count": 30000},
    ]

    res = classify_outfit_structure(pred_map, None, detected, scene_type="person", target_filter="all")
    assert res["outfit_mode"] == "mode_c_traditional_ensemble"
    assert res["ensemble_type"] == "lehenga"
    assert len(res["planned_items"]) == 1
    lehenga_item = res["planned_items"][0]
    assert lehenga_item["key"] == "lehenga"
    assert lehenga_item["is_fused"] is True
    assert 4 in lehenga_item["target_class_ids"]
    assert 5 in lehenga_item["target_class_ids"]


def test_classify_outfit_mode_a_independent_priorities():
    """Multiple independent garments (T-Shirt + Jeans) must calculate exact coverage and priorities."""
    from app.core.detector import classify_outfit_structure
    pred_map = np.zeros((800, 500), dtype=np.uint8)
    pred_map[100:400, 150:350] = 4  # T-Shirt (60,000 px)
    pred_map[400:750, 150:350] = 6  # Jeans (70,000 px)

    detected = [
        {"class_id": 4, "category": "upper_clothes", "label": "T-Shirt / Top / Shirt", "pixel_count": 60000},
        {"class_id": 6, "category": "pants", "label": "Pants / Jeans / Trousers", "pixel_count": 70000},
    ]

    res = classify_outfit_structure(pred_map, None, detected, scene_type="person", target_filter="all")
    assert res["outfit_mode"] == "mode_a_independent"
    assert len(res["planned_items"]) == 2

    # Denominator = 130,000
    # Upper: 60,000 / 130,000 = 46.15% -> low_priority (30-49.99%)
    # Pants: 70,000 / 130,000 = 53.85% -> second_priority (50-69.99%)
    upper_item = next(it for it in res["planned_items"] if it["key"] == "upper_clothes")
    pants_item = next(it for it in res["planned_items"] if it["key"] == "pants")

    assert upper_item["coverage_percentage"] == 46.15
    assert upper_item["priority"] == "low_priority"
    assert pants_item["coverage_percentage"] == 53.85
    assert pants_item["priority"] == "second_priority"


def test_scene_type_mannequin_vs_person():
    """analyze_semantic_map distinguishes between real human models, mannequins, and flat-lay."""
    from app.core.detector import analyze_semantic_map

    # 1. Real human model (Face + Arms)
    map_human = np.zeros((300, 300), dtype=np.uint8)
    map_human[20:80, 120:180] = 11  # Face: 3600 px
    map_human[80:200, 80:120] = 14  # Left arm: 4800 px
    map_human[80:220, 120:200] = 4  # Garment: 11200 px
    res_human = analyze_semantic_map(map_human)
    assert res_human["scene_type"] == "person"
    assert res_human["has_human"] is True

    # 2. Mannequin display (no face/hair, only wooden stand / negligible noise)
    map_mannequin = np.zeros((300, 300), dtype=np.uint8)
    map_mannequin[40:260, 80:220] = 4 # Garment only
    map_mannequin[280:290, 145:155] = 9 # Shoe noise (100 px)
    res_mannequin = analyze_semantic_map(map_mannequin)
    assert res_mannequin["scene_type"] == "mannequin"
    assert res_mannequin["has_human"] is False

    # 3. Flat lay (zero body pixels)
    map_flat = np.zeros((300, 300), dtype=np.uint8)
    map_flat[50:250, 50:250] = 6 # Pants only
    res_flat = analyze_semantic_map(map_flat)
    assert res_flat["scene_type"] == "flat_lay"
    assert res_flat["has_human"] is False


def test_arm_occlusion_skips_when_no_arms():
    """compute_arm_occlusion_mask returns empty mask when no arms are present (e.g. mannequin)."""
    from app.core.pipeline import compute_arm_occlusion_mask
    pred_map = np.zeros((300, 300), dtype=np.uint8)
    pred_map[50:250, 50:250] = 4 # Garment, 0 arms
    alpha = np.zeros((300, 300), dtype=np.uint8)
    alpha[50:250, 50:250] = 255

    mask = compute_arm_occlusion_mask(pred_map, garment_class_id=4, cutout_alpha=alpha)
    assert mask.sum() == 0


def test_garment_physical_continuity_layered_kurti_gown():
    """Continuous gown/kurti with high vertical overlap is flagged is_continuous=True."""
    from app.core.detector import check_garment_physical_continuity
    pred_map = np.zeros((1000, 600), dtype=np.uint8)
    # Upper kurti: y in [200, 700], central bodice & overlay
    pred_map[200:700, 250:350] = 4
    # Skirt: y in [450, 950], flanking and lower gown
    pred_map[450:950, 150:250] = 5
    pred_map[450:950, 350:450] = 5

    res = check_garment_physical_continuity(pred_map, upper_cid=4, lower_cids=(5,))
    assert res["is_continuous"] is True
    assert res["overlap_y"] >= 240
    assert res["total_span_ratio"] >= 0.74
    assert res["has_bare_midriff"] is False


def test_garment_physical_continuity_independent_shirt_mini_skirt():
    """Independent shirt and separate mini skirt with low overlap and gap is flagged is_continuous=False."""
    from app.core.detector import check_garment_physical_continuity
    pred_map = np.zeros((1000, 600), dtype=np.uint8)
    # Shirt: y in [200, 400]
    pred_map[200:400, 220:380] = 4
    # Mini skirt: y in [415, 550]
    pred_map[415:550, 200:400] = 5
    # Bare skin in gap (legs/skin 12, 13)
    pred_map[400:415, 250:350] = 12

    res = check_garment_physical_continuity(pred_map, upper_cid=4, lower_cids=(5,))
    assert res["is_continuous"] is False


def test_classify_outfit_continuous_gown_without_scarf():
    """Continuous gown (e.g. ladies-dresses.jpg) without scarf must be classified as Mode B single piece."""
    from app.core.detector import classify_outfit_structure
    pred_map = np.zeros((1200, 800), dtype=np.uint8)
    pred_map[220:990, 240:650] = 4  # Kurti/bodice
    pred_map[430:1180, 100:550] = 5  # Skirt
    detected = [
        {"class_id": 4, "category": "upper_clothes", "label": "T-Shirt / Top / Shirt", "pixel_count": 75000},
        {"class_id": 5, "category": "skirt", "label": "Skirt / Lehenga", "pixel_count": 210000},
    ]

    res = classify_outfit_structure(pred_map, None, detected, scene_type="person", target_filter="all")
    assert res["outfit_mode"] == "mode_b_single_piece"
    assert res["ensemble_type"] == "dress"
    assert len(res["planned_items"]) == 1
    item = res["planned_items"][0]
    assert item["is_fused"] is True
    assert 4 in item["target_class_ids"]
    assert 5 in item["target_class_ids"]
    assert item["coverage_percentage"] == 100.0


def test_arm_occlusion_ignores_arm_on_wall():
    """Arm resting on the wall (outside torso, touching background) must NOT trigger inpainting."""
    from app.core.pipeline import compute_arm_occlusion_mask
    pred_map = np.zeros((500, 500), dtype=np.uint8)
    # Garment in center: y in [100, 400], x in [180, 320]
    pred_map[100:400, 180:320] = 7
    # Arm resting outside on the right wall: y in [120, 200], x in [420, 480]
    pred_map[120:200, 420:480] = 14

    mask = compute_arm_occlusion_mask(pred_map, garment_class_id=7)
    assert mask.sum() == 0


def test_detect_multi_garment_layout():
    """detect_multi_garment_layout must identify multiple independent garments on product catalog images."""
    from app.core.detector import detect_multi_garment_layout
    # Create image with 3 distinct square garments on white background
    img = Image.new("RGB", (600, 600), (255, 255, 255))
    arr = np.array(img)
    # Garment 1: top-left
    arr[50:180, 50:180] = [30, 80, 180]
    # Garment 2: top-right
    arr[50:180, 400:530] = [180, 40, 60]
    # Garment 3: bottom-center
    arr[400:530, 220:380] = [40, 160, 80]

    test_im = Image.fromarray(arr)
    is_multi, comps = detect_multi_garment_layout(test_im)
    assert is_multi is True
    assert len(comps) == 3


def test_validator_detects_clean_vs_under_extracted():
    """validate_garment_background must verify clean transparent 1024x1024 cutouts."""
    from app.core.validator import validate_garment_background
    # Clean 1024x1024 garment
    im = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    for y in range(150, 850):
        for x in range(250, 750):
            im.putpixel((x, y), (50, 100, 200, 255))

    val = validate_garment_background(im)
    assert val["status"] == "success"
    assert val["confidence"] >= 0.95
    assert val["preservation_status"] == "verified"
    assert len(val["validation_errors"]) == 0

    # Under-extracted (tiny specks only)
    tiny_im = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    tiny_im.putpixel((512, 512), (50, 100, 200, 255))
    val_tiny = validate_garment_background(tiny_im)
    assert val_tiny["status"] == "failed"
    assert any("under-extracted" in e for e in val_tiny["validation_errors"])



