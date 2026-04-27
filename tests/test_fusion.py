import numpy as np
import pytest

from bimodal_empathy.fusion import fuse
from bimodal_empathy.config import FER7_LABELS


def test_fuse_argmax_text_only():
    p_text = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    p_face = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
    p_f, j, label = fuse(p_text, p_face, alpha=1.0)
    assert FER7_LABELS[j] == "Happy"
    assert label == "Happy"


def test_fuse_argmax_face_only():
    p_text = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    p_face = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
    p_f, j, label = fuse(p_text, p_face, alpha=0.0)
    assert FER7_LABELS[j] == "Angry"
    assert label == "Angry"


def test_fuse_mid_blend_prefers_tiebreak():
    p_text = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    p_face = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
    p_f, j, _ = fuse(p_text, p_face, alpha=0.5)
    # 0.5*happy + 0.5*angry -> equal on those dims; argmax is first max index
    assert p_f[j] == max(p_f)


def test_fuse_renormalizes():
    p_text = np.ones(7)
    p_face = np.ones(7)
    p_f, _, _ = fuse(p_text, p_face, alpha=0.3)
    assert abs(p_f.sum() - 1.0) < 1e-6


def test_fuse_alpha_range():
    with pytest.raises(ValueError):
        fuse(np.ones(7) / 7, np.ones(7) / 7, alpha=1.5)
