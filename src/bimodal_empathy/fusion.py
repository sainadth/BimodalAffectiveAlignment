"""Late fusion: b* = argmax(α * P_text + (1-α) * P_face)."""

from __future__ import annotations

import numpy as np

from bimodal_empathy.config import FER7_LABELS, DEFAULT_ALPHA


def _as_numpy(p: np.ndarray) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    if a.size != 7:
        raise ValueError(f"Expected 7-way distribution, got shape {a.shape}")
    s = a.sum()
    if s > 0:
        a = a / s
    else:
        a = np.ones(7) / 7.0
    return a


def fuse(
    p_text: np.ndarray,
    p_face: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[np.ndarray, int, str]:
    """
    Blend two 7-way probability vectors.

    Returns:
        p_fused: shape (7,) normalized
        argmax_index: 0..6
        argmax_label: FER7 label string
    """
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    pt = _as_numpy(p_text)
    pf = _as_numpy(p_face)
    blended = a * pt + (1.0 - a) * pf
    s = blended.sum()
    if s > 0:
        p_fused = blended / s
    else:
        p_fused = np.ones(7) / 7.0
    j = int(np.argmax(p_fused))
    return p_fused, j, FER7_LABELS[j]
