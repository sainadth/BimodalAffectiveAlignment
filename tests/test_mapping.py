import numpy as np

from bimodal_empathy.emotion_mapping import (
    GOEMOTION_TO_FER7,
    collapse_goemotions_to_fer7,
    goemotion_name_to_fer7,
    _fer7_index,
)
from bimodal_empathy.config import FER7_LABELS


def test_count_goemotions_covered():
    assert len(GOEMOTION_TO_FER7) == 28


def test_goemotion_name_happy_anger():
    assert goemotion_name_to_fer7("joy") == "Happy"
    assert goemotion_name_to_fer7("anger") == "Angry"


def test_collapse_single_label():
    labels = list(GOEMOTION_TO_FER7.keys())
    id2label = {i: lab for i, lab in enumerate(labels)}
    vec = np.zeros(28)
    vec[labels.index("joy")] = 1.0
    out = collapse_goemotions_to_fer7(vec, id2label)
    assert out.sum() - 1.0 < 1e-6
    j = int(np.argmax(out))
    assert FER7_LABELS[j] == "Happy"


def test_collapse_sums_to_one():
    labels = list(GOEMOTION_TO_FER7.keys())
    id2label = {i: lab for i, lab in enumerate(labels)}
    vec = np.random.rand(28)
    out = collapse_goemotions_to_fer7(vec, id2label)
    assert abs(out.sum() - 1.0) < 1e-5


def test_fer7_index():
    assert _fer7_index("Neutral") == 6
