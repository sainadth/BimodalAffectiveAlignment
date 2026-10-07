"""Affective dissonance flags.

argmax_dissonant  - the original flag: argmax(P_text) != argmax(P_face). With a
                    43%-accurate text branch and a ~20%-accurate face branch this
                    fires on most inputs and mostly reflects classifier noise.
valence_dissonant - both branches are confident (top prob >= tau) AND their
                    valence groups differ AND at least one side is negative.
                    Captures the case the study cares about ("I'm fine" said
                    with a sad face) while ignoring Fear-vs-Sad style noise.
"""

from __future__ import annotations

import numpy as np

from bimodal_empathy.config import FER7_LABELS

VALENCE: dict[str, str] = {
    "Happy": "pos",
    "Angry": "neg",
    "Disgust": "neg",
    "Fear": "neg",
    "Sad": "neg",
    "Neutral": "neu",
    "Surprise": "neu",  # valence-ambiguous without context
}
DEFAULT_TAU = 0.40


def valence_of(label: str) -> str:
    return VALENCE[label]


def argmax_dissonant(p_text: np.ndarray, p_face: np.ndarray) -> bool:
    return int(np.argmax(p_text)) != int(np.argmax(p_face))


def valence_dissonant(p_text: np.ndarray, p_face: np.ndarray, tau: float = DEFAULT_TAU) -> bool:
    pt = np.asarray(p_text, dtype=float).ravel()
    pf = np.asarray(p_face, dtype=float).ravel()
    if pt.max() < tau or pf.max() < tau:
        return False
    vt = VALENCE[FER7_LABELS[int(pt.argmax())]]
    vf = VALENCE[FER7_LABELS[int(pf.argmax())]]
    return vt != vf and "neg" in (vt, vf)


def felt_vs_text_masking(felt_label: str, text_label: str) -> bool | None:
    """Self-report-based masking flag for H2: felt valence differs from the
    valence the words convey, with at least one side negative. None if the
    participant declined to report."""
    if felt_label not in VALENCE:
        return None
    vf, vt = VALENCE[felt_label], VALENCE[text_label]
    return vf != vt and "neg" in (vf, vt)
