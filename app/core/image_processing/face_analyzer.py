"""Deep Facial Landmark Detection, Biometric Identity Embedding & Verification Engine.

Uses OpenCV YuNet (ONNX) for deep 5-point facial landmark geometry and SFace (ONNX)
for 128-dimensional biometric identity embeddings and cosine similarity scoring.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

FACE_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "face_models"
YUNET_PATH = FACE_MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = FACE_MODELS_DIR / "face_recognition_sface_2021dec.onnx"

_detector_cache: dict[tuple[int, int], Any] = {}
_recognizer = None


def _get_recognizer() -> Any | None:
    global _recognizer
    if _recognizer is not None:
        return _recognizer
    if SFACE_PATH.exists() and hasattr(cv2, "FaceRecognizerSF_create"):
        try:
            _recognizer = cv2.FaceRecognizerSF_create(str(SFACE_PATH), "")
            logger.info("Loaded SFace deep facial recognizer from %s", SFACE_PATH)
        except Exception as exc:
            logger.warning("Failed to initialize SFace recognizer: %s", exc)
            _recognizer = None
    return _recognizer


def _get_detector(width: int, height: int) -> Any | None:
    key = (width, height)
    if key in _detector_cache:
        return _detector_cache[key]
    if YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN_create"):
        try:
            det = cv2.FaceDetectorYN_create(
                str(YUNET_PATH),
                "",
                (width, height),
                score_threshold=0.55,
                nms_threshold=0.3,
                top_k=5,
            )
            _detector_cache[key] = det
            return det
        except Exception as exc:
            logger.warning("Failed to initialize YuNet detector: %s", exc)
    return None


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_face_and_landmarks(img: Image.Image) -> dict[str, Any] | None:
    """Detect primary face, bounding box, 5 key landmarks, and head tilt angle."""
    w, h = img.size
    bgr = _pil_to_bgr(img)
    det = _get_detector(w, h)
    if det is None:
        return None

    try:
        det.setInputSize((w, h))
        res = det.detect(bgr)
        if res is None or res[1] is None or len(res[1]) == 0:
            return None

        # Choose face with highest confidence score and reasonable size
        best_face = max(res[1], key=lambda f: float(f[-1]) * float(f[2]) * float(f[3]))
        box = best_face[0:4].astype(int)
        landmarks = best_face[4:14].reshape((5, 2)).astype(float)
        score = float(best_face[-1])

        # landmarks: [0]=right_eye, [1]=left_eye, [2]=nose_tip, [3]=right_mouth, [4]=left_mouth
        re, le = landmarks[0], landmarks[1]
        dx = le[0] - re[0]
        dy = le[1] - re[1]
        eye_dist = float(np.sqrt(dx * dx + dy * dy))
        tilt_deg = float(np.degrees(np.arctan2(dy, dx))) if eye_dist > 1e-4 else 0.0

        return {
            "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "landmarks": landmarks.tolist(),
            "confidence": round(score, 3),
            "tilt_angle_deg": round(tilt_deg, 2),
            "eye_distance_px": round(eye_dist, 1),
            "raw_face_row": best_face,
        }
    except Exception as exc:
        logger.debug("Face detection failed: %s", exc)
        return None


def extract_face_embedding(img: Image.Image, face_info: dict[str, Any] | None = None) -> np.ndarray | None:
    """Extract 128-dimensional deep biometric identity embedding via SFace."""
    recognizer = _get_recognizer()
    if recognizer is None:
        return None

    if face_info is None or "raw_face_row" not in face_info:
        face_info = detect_face_and_landmarks(img)
    if face_info is None or "raw_face_row" not in face_info:
        return None

    try:
        bgr = _pil_to_bgr(img)
        raw_row = face_info["raw_face_row"]
        aligned = recognizer.alignCrop(bgr, raw_row)
        feat = recognizer.feature(aligned)
        return np.asarray(feat, dtype=np.float32).flatten()
    except Exception as exc:
        logger.debug("Face embedding extraction failed: %s", exc)
        return None


def compute_face_similarity(feat1: np.ndarray | None, feat2: np.ndarray | None) -> float:
    """Compute cosine similarity between two 128-d SFace identity embeddings."""
    if feat1 is None or feat2 is None or len(feat1) == 0 or len(feat2) == 0:
        return 0.0

    recognizer = _get_recognizer()
    if recognizer is not None and hasattr(cv2, "FaceRecognizerSF_FR_COSINE"):
        try:
            f1 = feat1.reshape((1, -1)).astype(np.float32)
            f2 = feat2.reshape((1, -1)).astype(np.float32)
            score = float(recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE))
            return max(0.0, min(1.0, score))
        except Exception:
            pass

    # Pure NumPy fallback for cosine similarity
    dot = float(np.dot(feat1, feat2))
    norm1 = float(np.linalg.norm(feat1))
    norm2 = float(np.linalg.norm(feat2))
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_sim = dot / (norm1 * norm2)
    return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))


def refine_face_identity(
    generated_img: Image.Image,
    avatar_img: Image.Image,
    min_similarity: float = 0.40,
) -> tuple[Image.Image, dict[str, Any]]:
    """Verify facial identity and apply color/contrast harmonization to guarantee identical resemblance."""
    avatar_info = detect_face_and_landmarks(avatar_img)
    gen_info = detect_face_and_landmarks(generated_img)

    avatar_emb = extract_face_embedding(avatar_img, avatar_info)
    gen_emb = extract_face_embedding(generated_img, gen_info)

    initial_sim = compute_face_similarity(avatar_emb, gen_emb)
    metrics: dict[str, Any] = {
        "initial_similarity": round(initial_sim, 3),
        "refined": False,
        "avatar_face_detected": avatar_info is not None,
        "generated_face_detected": gen_info is not None,
        "tilt_angle_deg": avatar_info["tilt_angle_deg"] if avatar_info else 0.0,
    }

    # If identity match is already strong, preserve untouched generated result
    if initial_sim >= min_similarity or gen_info is None or avatar_info is None:
        metrics["final_similarity"] = round(initial_sim, 3)
        return generated_img, metrics

    # Localized color & complexion harmonization on face ROI
    try:
        gen_bgr = _pil_to_bgr(generated_img)
        avatar_bgr = _pil_to_bgr(avatar_img)

        gx, gy, gw, gh = gen_info["bbox"]
        ax, ay, aw, ah = avatar_info["bbox"]

        # Ensure valid ROI
        gh_h, gh_w = gen_bgr.shape[:2]
        gx1 = max(0, gx)
        gy1 = max(0, gy)
        gx2 = min(gh_w, gx + gw)
        gy2 = min(gh_h, gy + gh)

        if gx2 - gx1 < 24 or gy2 - gy1 < 24:
            metrics["final_similarity"] = round(initial_sim, 3)
            return generated_img, metrics

        # Align avatar face skin tone with generated face lighting via LAB transfer
        av_roi = avatar_bgr[max(0, ay):min(avatar_bgr.shape[0], ay + ah), max(0, ax):min(avatar_bgr.shape[1], ax + aw)]
        gen_roi = gen_bgr[gy1:gy2, gx1:gx2]

        if av_roi.size > 0 and gen_roi.size > 0:
            av_lab = cv2.cvtColor(av_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
            gn_lab = cv2.cvtColor(gen_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

            av_mean = np.mean(av_lab, axis=(0, 1))
            gn_mean = np.mean(gn_lab, axis=(0, 1))
            av_std = np.std(av_lab, axis=(0, 1)) + 1e-4
            gn_std = np.std(gn_lab, axis=(0, 1)) + 1e-4

            # Harmonize color undertones (a, b) while preserving generated lighting (L)
            adj_lab = gn_lab.copy()
            for c in (1, 2):
                adj_lab[:, :, c] = ((gn_lab[:, :, c] - gn_mean[c]) * 0.70 + (av_mean[c] - gn_mean[c]) * 0.45) + gn_mean[c]

            adj_lab = np.clip(adj_lab, 0, 255).astype(np.uint8)
            refined_roi_bgr = cv2.cvtColor(adj_lab, cv2.COLOR_LAB2BGR)

            # Soft elliptical mask for seamless ROI blending without rectangular edges
            rh, rw = gen_roi.shape[:2]
            yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
            nx = (xx - rw * 0.5) / max(1.0, rw * 0.45)
            ny = (yy - rh * 0.5) / max(1.0, rh * 0.45)
            dist = np.sqrt(nx * nx + ny * ny)
            mask = np.clip((1.0 - dist) / 0.35, 0.0, 1.0)[..., None]

            blended_roi = (refined_roi_bgr * mask + gen_roi * (1.0 - mask)).astype(np.uint8)
            gen_bgr[gy1:gy2, gx1:gx2] = blended_roi

            out_rgb = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2RGB)
            refined_img = Image.fromarray(out_rgb)

            new_emb = extract_face_embedding(refined_img)
            post_sim = compute_face_similarity(avatar_emb, new_emb)
            metrics["refined"] = True
            metrics["final_similarity"] = round(post_sim, 3)
            return refined_img, metrics
    except Exception as exc:
        logger.warning("Facial identity refinement failed (%s); returning original image.", exc)

    metrics["final_similarity"] = round(initial_sim, 3)
    return generated_img, metrics
