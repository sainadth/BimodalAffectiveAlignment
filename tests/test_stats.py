import numpy as np
import pytest

from bimodal_empathy.eval.stats import confusion, macro_f1, mcnemar, wilson_ci


def test_wilson_ci_basic():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson_ci(0, 0) == (0.0, 0.0)
    assert wilson_ci(100, 100)[1] == pytest.approx(1.0)


def test_mcnemar_no_discordant_pairs():
    a = np.array([1, 1, 0, 0], dtype=bool)
    assert mcnemar(a, a) == {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0}


def test_mcnemar_counts_and_direction():
    a = np.array([1] * 30 + [0] * 10 + [1] * 5, dtype=bool)
    b = np.array([0] * 30 + [1] * 10 + [1] * 5, dtype=bool)
    r = mcnemar(a, b)
    assert r["b"] == 30 and r["c"] == 10
    assert r["chi2"] == pytest.approx((abs(30 - 10) - 1) ** 2 / 40)
    assert 0 < r["p"] < 0.01
    with pytest.raises(ValueError):
        mcnemar(a, b[:-1])


def test_confusion_and_macro_f1_perfect_and_degenerate():
    y = np.array([0, 1, 2, 2])
    conf = confusion(y, y, 3)
    assert conf.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    assert macro_f1(conf) == pytest.approx(1.0)
    conf_bad = confusion(y, np.array([1, 2, 0, 0]), 3)
    assert macro_f1(conf_bad) == pytest.approx(0.0)
