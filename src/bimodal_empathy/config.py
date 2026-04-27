"""Defaults for model IDs, FER7 labels, and paths."""

from __future__ import annotations

import os
from pathlib import Path

# FER2013 / fusion vocabulary (order fixed for P_face, P_text, fusion)
FER7_LABELS: tuple[str, ...] = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)

DEFAULT_ALPHA: float = 0.5

# HuggingFace model IDs
GOEMOTIONS_MODEL_ID: str = "SamLowe/roberta-base-go_emotions"
FLAN_T5_MODEL_ID: str = os.environ.get("BIMODAL_FLAN_T5_MODEL_ID", "google/flan-t5-base")

# Generation
MAX_NEW_TOKENS: int = 128

# Weights directory (FER ResNet-50 checkpoint)
_cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
WEIGHTS_DIR: Path = _cache_root / "bimodal_empathy"
FER_HF_REPO: str = "ElenaRyumina/face_emotion_recognition"
FER_HF_FILE: str = "FER_static_ResNet50_AffectNet.pt"


def device_preference() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
