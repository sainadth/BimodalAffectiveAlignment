"""
Map 28 GoEmotions labels to 7 FER2013-style categories (FER7).

Rationale: FER7 captures broad expression classes used by the face model. We assign
each GoEmotion label to the closest FER7 bucket. Multilabel scores from the text
model are *summed* into the mapped bins and renormalized to a 7-way distribution.

Label names follow the `SamLowe/roberta-base-go_emotions` / GoEmotions convention
(lowercase keys); we match case-insensitively when needed.
"""

from __future__ import annotations

import numpy as np

from bimodal_empathy.config import FER7_LABELS

# GoEmotion label (canonical lowercase) -> FER7 class name
GOEMOTION_TO_FER7: dict[str, str] = {
    "admiration": "Happy",
    "amusement": "Happy",
    "anger": "Angry",
    "annoyance": "Angry",
    "approval": "Happy",
    "caring": "Happy",
    "confusion": "Surprise",
    "curiosity": "Surprise",
    "desire": "Happy",
    "disappointment": "Sad",
    "disapproval": "Angry",
    "disgust": "Disgust",
    "embarrassment": "Fear",
    "excitement": "Happy",
    "fear": "Fear",
    "gratitude": "Happy",
    "grief": "Sad",
    "joy": "Happy",
    "love": "Happy",
    "nervousness": "Fear",
    "optimism": "Happy",
    "pride": "Happy",
    "realization": "Surprise",
    "relief": "Happy",
    "remorse": "Sad",
    "sadness": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral",
}


def _fer7_index(name: str) -> int:
    if name not in FER7_LABELS:
        raise ValueError(f"Unknown FER7 label: {name}")
    return FER7_LABELS.index(name)


def goemotion_name_to_fer7(go_label: str) -> str:
    key = go_label.strip().lower()
    if key not in GOEMOTION_TO_FER7:
        raise KeyError(f"Unknown GoEmotion label: {go_label!r}")
    return GOEMOTION_TO_FER7[key]


COLLAPSE_METHODS = ("sum", "max", "mean")


def collapse_goemotions_to_fer7(
    label_scores: np.ndarray,
    goemotion_id2label: dict[int, str],
    method: str = "sum",
) -> np.ndarray:
    """
    Aggregate multilabel GoEmotion scores (length 28) into FER7 (length 7).

    Args:
        label_scores: shape (28,) or (n_labels,); higher = more present (e.g. sigmoid probs).
        goemotion_id2label: id -> label name from the HF model config.
        method: "sum" (original; 13 labels feed Happy vs 1 Neutral, so Neutral is
            structurally starved on conversational text), "max" (bin = max member
            score) or "mean" (bin = sum / bin size). Output is renormalised to sum to 1.
    """
    if method not in COLLAPSE_METHODS:
        raise ValueError(f"method must be one of {COLLAPSE_METHODS}, got {method!r}")
    s = np.asarray(label_scores, dtype=np.float64).ravel()
    n = s.size
    if n != len(goemotion_id2label):
        m = min(n, len(goemotion_id2label))
        t = np.zeros(len(goemotion_id2label), dtype=np.float64)
        t[:m] = s[:m]
        s = t
    out = np.zeros(7, dtype=np.float64)
    counts = np.zeros(7, dtype=np.float64)
    for i, score in enumerate(s):
        name = goemotion_id2label.get(i)
        if name is None:
            continue
        key = str(name).strip().lower()
        if key not in GOEMOTION_TO_FER7:
            continue
        j = _fer7_index(GOEMOTION_TO_FER7[key])
        v = float(max(score, 0.0))
        counts[j] += 1
        out[j] = max(out[j], v) if method == "max" else out[j] + v
    if method == "mean":
        out = np.where(counts > 0, out / np.maximum(counts, 1), 0.0)
    total = out.sum()
    if total > 0:
        out = out / total
    else:
        out[:] = 1.0 / 7.0
    return out


def goemotion_id2label_28_from_config(config) -> dict[int, str]:
    """Build id2label for GoEmotions from HuggingFace PretrainedConfig."""
    raw = dict(getattr(config, "id2label", {}) or {})
    if not raw:
        raise ValueError("config.id2label is empty")
    return {int(k): v for k, v in raw.items()}
