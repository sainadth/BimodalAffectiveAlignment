import numpy as np

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.dissonance import argmax_dissonant, felt_vs_text_masking, valence_dissonant


def _p(label, conf=0.8):
    v = np.full(7, (1 - conf) / 6)
    v[FER7_LABELS.index(label)] = conf
    return v


def test_argmax_flag_fires_on_any_disagreement():
    assert argmax_dissonant(_p("Fear"), _p("Sad"))
    assert not argmax_dissonant(_p("Sad"), _p("Sad"))


def test_valence_flag_ignores_same_valence_noise():
    assert not valence_dissonant(_p("Fear"), _p("Sad"))  # both negative
    assert not valence_dissonant(_p("Neutral"), _p("Surprise"))  # both neutral-ish


def test_valence_flag_catches_masking():
    assert valence_dissonant(_p("Happy"), _p("Sad"))
    assert valence_dissonant(_p("Neutral"), _p("Sad"))  # "I'm fine" + sad face
    assert not valence_dissonant(_p("Happy"), _p("Neutral"))  # no negative side


def test_valence_flag_requires_confidence():
    assert not valence_dissonant(_p("Happy", 0.3), _p("Sad", 0.9))
    assert not valence_dissonant(_p("Happy"), np.full(7, 1 / 7))  # uniform face prior


def test_felt_masking():
    assert felt_vs_text_masking("Sad", "Happy") is True
    assert felt_vs_text_masking("Sad", "Neutral") is True
    assert felt_vs_text_masking("Happy", "Happy") is False
    assert felt_vs_text_masking("Prefer not to say", "Happy") is None
