"""Face detection + cropping for the live path (OpenCV YuNet).

The FER models expect a tight, roughly frontal face crop (FER2013 / AffectNet
style). A raw webcam frame (head, shoulders, background) resized to 224x224 is
out of distribution, so every face-branch number in the paper assumes a crop.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = Path("checkpoints") / "face_detection_yunet_2023mar.onnx"

_DETECTOR = None


def available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        import cv2

        if not YUNET_PATH.is_file():
            YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        _DETECTOR = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.6, 0.3, 5000)
    return _DETECTOR


def crop_largest_face(image: Image.Image, pad: float = 0.15) -> Image.Image | None:
    """Return the largest detected face as a padded RGB crop, or None if no face / no OpenCV."""
    if not available():
        return None
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    det = _detector()
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    x, y, bw, bh = (int(v) for v in max(faces, key=lambda f: f[2] * f[3])[:4])
    p = int(pad * max(bw, bh))
    y0, y1 = max(0, y - p), min(h, y + bh + p)
    x0, x1 = max(0, x - p), min(w, x + bw + p)
    if y1 <= y0 or x1 <= x0:
        return None
    return Image.fromarray(rgb[y0:y1, x0:x1])
