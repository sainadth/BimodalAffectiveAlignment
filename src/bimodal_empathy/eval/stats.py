"""Small, dependency-light statistics helpers shared by the evaluation scripts."""

from __future__ import annotations

import math

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, float]:
    """McNemar test with continuity correction on paired 0/1 correctness vectors.

    b = A right & B wrong, c = A wrong & B right. p from chi2(1) survival
    function (scipy if available, else the exact erfc identity for df=1).
    """
    a = np.asarray(a_correct, dtype=bool)
    b_ = np.asarray(b_correct, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("paired vectors must have the same shape")
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore

        p = float(chi2_dist.sf(chi2, df=1))
    except ImportError:  # pragma: no cover
        p = float(math.erfc(math.sqrt(chi2 / 2)))
    return {"b": b, "c": c, "chi2": float(chi2), "p": p}


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for y, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        conf[int(y), int(p)] += 1
    return conf


def macro_f1(conf: np.ndarray) -> float:
    """Macro-averaged F1 over classes that appear in gold or predictions."""
    f1s = []
    for c in range(conf.shape[0]):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0
