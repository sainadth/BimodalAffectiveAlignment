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


def test_collapse_methods_differ_and_normalise():
    import numpy as np
    from bimodal_empathy.emotion_mapping import GOEMOTION_TO_FER7, collapse_goemotions_to_fer7
    from bimodal_empathy.config import FER7_LABELS

    names = sorted(GOEMOTION_TO_FER7)
    id2label = dict(enumerate(names))
    # every Happy-mapped label at 0.3, neutral at 0.6
    scores = np.array([0.3 if GOEMOTION_TO_FER7[n] == "Happy" else (0.6 if n == "neutral" else 0.0) for n in names])
    p_sum = collapse_goemotions_to_fer7(scores, id2label, "sum")
    p_max = collapse_goemotions_to_fer7(scores, id2label, "max")
    p_mean = collapse_goemotions_to_fer7(scores, id2label, "mean")
    for p in (p_sum, p_max, p_mean):
        assert abs(p.sum() - 1) < 1e-9
    h, n = FER7_LABELS.index("Happy"), FER7_LABELS.index("Neutral")
    assert p_sum[h] > p_sum[n]  # 13 x 0.3 beats 0.6 under sum
    assert p_max[n] > p_max[h] and p_mean[n] > p_mean[h]  # a single strong neutral wins otherwise
    import pytest

    with pytest.raises(ValueError):
        collapse_goemotions_to_fer7(scores, id2label, "median")
