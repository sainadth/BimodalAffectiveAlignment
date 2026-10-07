"""Framework-free affect pipeline used by the Streamlit apps and the FastAPI server.

Models are cached process-wide. `analyze_affect` accepts one image or a burst of
images; face probabilities are averaged over frames in which a face is found.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.dissonance import argmax_dissonant, valence_dissonant
from bimodal_empathy.face_crop import crop_largest_face
from bimodal_empathy.fer_finetune_ckpt import load_vision_finetuned
from bimodal_empathy.fusion import fuse
from bimodal_empathy.text_sensor import load_text_model
from bimodal_empathy.vision_sensor import load_vision_model, uniform_face_p_face

FINETUNE_CKPT_PATH = Path(os.getenv("FINETUNE_CKPT", "checkpoints/fer2013_finetune.pt"))


@lru_cache(maxsize=1)
def text_model():
    return load_text_model()


@lru_cache(maxsize=2)
def vision_model(weights_path: str | None = None):
    return load_vision_finetuned(weights_path) if weights_path else load_vision_model()


def warm_up(use_finetune: bool = True) -> None:
    text_model()
    vision_model(str(FINETUNE_CKPT_PATH) if use_finetune and FINETUNE_CKPT_PATH.is_file() else None)


def analyze_affect(
    utterance: str,
    images: Image.Image | Sequence[Image.Image] | None,
    *,
    alpha: float,
    use_finetune: bool = True,
    save_frames_to: Path | None = None,
) -> dict:
    """Run text + face branches (face-cropped, burst-averaged) and fusion.

    `save_frames_to` writes the raw burst and the YuNet crops to that directory for
    offline validation. It is OFF unless a caller passes it; enabling it for real
    participants contradicts the consent text ("frames are never stored"), so it is
    meant for pilot and debugging runs only.
    """
    if images is None:
        frames: list[Image.Image] = []
    elif isinstance(images, Image.Image):
        frames = [images]
    else:
        frames = list(images)

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    p_text, _, _ = text_model().predict_fer7(utterance)
    timings["text_ms"] = (time.perf_counter() - t0) * 1000.0

    weights = str(FINETUNE_CKPT_PATH) if use_finetune and FINETUNE_CKPT_PATH.is_file() else None
    vision_branch = "fine-tuned (FER2013)" if weights else "baseline (AffectNet)"
    faces_found = 0
    per_frame: list[list[float]] = []
    if frames:
        vm = vision_model(weights)
        t0 = time.perf_counter()
        acc = np.zeros(7, dtype=float)
        fallback = None
        per_frame: list[list[float]] = []
        if save_frames_to is not None:
            save_frames_to.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(frames):
            crop = crop_largest_face(im)
            if save_frames_to is not None:
                im.save(save_frames_to / f"raw_{i}.jpg", quality=88)
                if crop is not None:
                    crop.save(save_frames_to / f"crop_{i}.jpg", quality=92)
            if crop is None:
                fallback = im if fallback is None else fallback
                continue
            p, _, _ = vm.predict_fer7(crop)
            pv = np.asarray(p, dtype=float).ravel()
            per_frame.append([round(float(x), 4) for x in pv])
            acc += pv
            faces_found += 1
        if faces_found:
            p_face = acc / faces_found
        else:  # no face in any frame: use the first frame uncropped, flagged below
            p, _, _ = vm.predict_fer7(fallback)
            p_face = np.asarray(p, dtype=float).ravel()
        timings["face_ms"] = (time.perf_counter() - t0) * 1000.0
    else:
        p_face = uniform_face_p_face()
        timings["face_ms"] = 0.0

    t0 = time.perf_counter()
    p_fuse, _, label_star = fuse(p_text, p_face, alpha=alpha)
    timings["fusion_ms"] = (time.perf_counter() - t0) * 1000.0

    p_text = np.asarray(p_text, dtype=float).ravel()
    p_face = np.asarray(p_face, dtype=float).ravel()
    p_fuse = np.asarray(p_fuse, dtype=float).ravel()
    has_face = bool(frames)
    return {
        "p_text": p_text.tolist(),
        "p_face": p_face.tolist(),
        "p_fuse": p_fuse.tolist(),
        "text_label": FER7_LABELS[int(np.argmax(p_text))],
        "face_label": FER7_LABELS[int(np.argmax(p_face))] if has_face else None,
        "fused_label": label_star,
        "had_face_image": has_face,
        "face_detected": faces_found > 0,
        "frames": len(frames),
        "frames_with_face": faces_found,
        "argmax_dissonant": has_face and argmax_dissonant(p_text, p_face),
        "valence_dissonant": has_face and valence_dissonant(p_text, p_face),
        "vision_branch": vision_branch,
        "alpha": float(alpha),
        "timings": timings,
        "per_frame_p_face": per_frame if images is not None else [],
    }
