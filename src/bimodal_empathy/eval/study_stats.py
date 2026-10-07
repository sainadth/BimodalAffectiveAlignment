"""Pre-specified analysis for the within-subjects study, design v3 (docs/STUDY_PROTOCOL.md).

Factors: label_source {correct, wrong, none} × move {assert, ask}; 12 trials/participant.
Primary DV: per-trial composite = mean of TRIAL_ITEMS (1-5).
  H1  correct > wrong                     (pooled over move)
  H2  asymmetry: penalty (none - wrong) > gain (correct - none), paired per participant
  H3  asking softens the penalty: wrong_ask > wrong_assert
  H3b move main effect: ask vs assert pooled over label_source
Inference: Friedman across the six cells; Holm-corrected Wilcoxon signed-rank on the
pre-specified pairs; matched-pairs rank-biserial r. Manipulation check: proportion of
trials where the participant's perceived label matched the shown label (or "Unsure"/
"none" when no label). Exploratory: sensor labels vs felt report.
"""

from __future__ import annotations

import numpy as np

from bimodal_empathy.dissonance import VALENCE
from bimodal_empathy.study_log import CONDITIONS, LABEL_SOURCES, MOVES, PETS_KEYS, TRIAL_ITEM_KEYS

PRESPECIFIED_PAIRS: tuple[tuple[str, str], ...] = (
    ("correct", "wrong"),            # H1 (pooled label_source)
    ("wrong_ask", "wrong_assert"),   # H3
    ("correct_assert", "none_assert"),
    ("none_assert", "wrong_assert"),
    ("ask", "assert"),               # H3b (pooled move)
)


def cronbach_alpha(items: np.ndarray) -> float:
    items = np.asarray(items, dtype=float)
    if items.ndim != 2 or items.shape[1] < 2 or items.shape[0] < 2:
        return float("nan")
    item_var = items.var(axis=0, ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    k = items.shape[1]
    return float(k / (k - 1) * (1 - item_var / total_var))


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    from scipy.stats import rankdata

    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adj[idx] = running
    return adj.tolist()


def trial_score(row: dict, keys: tuple[str, ...] = TRIAL_ITEM_KEYS) -> float:
    return float(np.mean([float(row[k]) for k in keys]))


def _groups_for(row: dict) -> list[str]:
    """Every grouping label a row belongs to: its cell, its label_source, its move."""
    return [row["condition"], row.get("label_source") or row["condition"].rsplit("_", 1)[0],
            row.get("move") or row["condition"].rsplit("_", 1)[1]]


def per_participant_means(rows: list[dict], groups: tuple[str, ...], keys: tuple[str, ...] = TRIAL_ITEM_KEYS):
    """(pids, matrix[n, len(groups)]) of mean composite per group; complete-case over `groups`."""
    by: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        sc = trial_score(r, keys)
        for g in _groups_for(r):
            by.setdefault(r["participant_id"], {}).setdefault(g, []).append(sc)
    pids = sorted(p for p, d in by.items() if all(g in d for g in groups))
    mat = np.array([[float(np.mean(by[p][g])) for g in groups] for p in pids]) if pids else np.zeros((0, len(groups)))
    return pids, mat


def _wilcoxon_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import wilcoxon

    if np.all(x == y):
        return 0.0, 1.0
    w = wilcoxon(x, y, zero_method="wilcox")
    return float(w.statistic), float(w.pvalue)


def analyze(rows: list[dict], keys: tuple[str, ...] = TRIAL_ITEM_KEYS) -> dict:
    from scipy.stats import friedmanchisquare

    pids, cells = per_participant_means(rows, CONDITIONS, keys)
    n = len(pids)
    out: dict = {
        "n_participants": n, "n_rows": len(rows), "cells": list(CONDITIONS),
        "cell_means": cells.mean(axis=0).tolist() if n else [],
        "cell_sds": cells.std(axis=0, ddof=1).tolist() if n > 1 else [],
    }
    _, ls = per_participant_means(rows, LABEL_SOURCES, keys)
    _, mv = per_participant_means(rows, MOVES, keys)
    out["label_source_means"] = dict(zip(LABEL_SOURCES, ls.mean(axis=0).tolist())) if n else {}
    out["move_means"] = dict(zip(MOVES, mv.mean(axis=0).tolist())) if n else {}
    if n < 5:
        out["note"] = "fewer than 5 complete participants; descriptive only"
        return out
    fr = friedmanchisquare(*[cells[:, j] for j in range(cells.shape[1])])
    out["friedman_cells"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue)}

    pairs, raw = [], []
    all_groups = tuple(CONDITIONS) + LABEL_SOURCES + MOVES
    _, allm = per_participant_means(rows, all_groups, keys)
    idx = {g: i for i, g in enumerate(all_groups)}
    for a, b in PRESPECIFIED_PAIRS:
        x, y = allm[:, idx[a]], allm[:, idx[b]]
        stat, p = _wilcoxon_pair(x, y)
        pairs.append({"a": a, "b": b, "W": stat, "p": p, "rank_biserial": rank_biserial(x, y), "mean_diff": float(np.mean(x - y))})
        raw.append(p)
    # H2 asymmetry: penalty vs gain, per participant
    gain = allm[:, idx["correct"]] - allm[:, idx["none"]]
    penalty = allm[:, idx["none"]] - allm[:, idx["wrong"]]
    stat, p = _wilcoxon_pair(penalty, gain)
    pairs.append({"a": "penalty(none-wrong)", "b": "gain(correct-none)", "W": stat, "p": p,
                  "rank_biserial": rank_biserial(penalty, gain), "mean_diff": float(np.mean(penalty - gain))})
    raw.append(p)
    for pr, adj in zip(pairs, holm(raw)):
        pr["p_holm"] = adj
    out["pairwise"] = pairs
    out["H2_asymmetry"] = {"mean_gain": float(gain.mean()), "mean_penalty": float(penalty.mean())}
    return out


def _is_true(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def manipulation_check(rows: list[dict]) -> dict:
    """Two separate things: was the label *delivered* (objective, from the reply text), and
    was it *perceived* (subjective, from the participant's manipulation-check answer)."""
    out = {}
    for ls in LABEL_SOURCES:
        rs = [r for r in rows if r.get("label_source") == ls]
        if not rs:
            continue
        if ls == "none":
            perceived = np.mean([r["manip_check_label"] in ("Unsure", "It did not say") or
                                 r["manip_check_label"] == r["felt_label"] for r in rs])
            entry = {"n": len(rs), "perceived": float(perceived), "same_valence": None, "delivered": None}
        else:
            perceived = np.mean([r["manip_check_label"] == r["shown_label"] for r in rs])
            val_hit = np.mean([VALENCE.get(r["manip_check_label"]) == VALENCE.get(r["shown_label"]) for r in rs])
            delivered = [_is_true(r.get("label_in_reply", "")) for r in rs if str(r.get("label_in_reply", "")) != ""]
            both = [r for r in rs if _is_true(r.get("label_in_reply", "")) and r["manip_check_label"] == r["shown_label"]]
            entry = {
                "n": len(rs), "perceived": float(perceived), "same_valence": float(val_hit),
                "delivered": float(np.mean(delivered)) if delivered else None,
                "delivered_and_perceived": len(both) / len(rs),
            }
        out[ls] = entry
    return out


def full_report(rows: list[dict], global_rows: list[dict] | None = None) -> dict:
    valid = [r for r in rows if r.get("correctness_effective", "") != "sensor"]  # felt unknown -> excluded from tests
    rep: dict = {"n_rows": len(rows), "n_rows_analysed": len(valid), "primary_composite": analyze(valid)}
    for k in TRIAL_ITEM_KEYS:
        rep[f"item_{k}"] = analyze(valid, (k,))
    by: dict[str, list[list[float]]] = {}
    for r in valid:
        by.setdefault(r["participant_id"], []).append([float(r[k]) for k in TRIAL_ITEM_KEYS])
    pm = np.array([np.mean(v, axis=0) for v in by.values()]) if by else np.zeros((0, len(TRIAL_ITEM_KEYS)))
    rep["composite_cronbach_alpha_participant_means"] = cronbach_alpha(pm)
    rep["manipulation_check"] = manipulation_check(valid)
    # Sensitivity analysis: only trials where the label was both delivered in the reply and
    # perceived by the participant (plus all `none` trials, which have no label to deliver).
    passed = [r for r in valid if r.get("label_source") == "none"
              or (_is_true(r.get("label_in_reply", "")) and r["manip_check_label"] == r["shown_label"])]
    rep["sensitivity_manipulation_passed"] = analyze(passed) if passed else {"n_participants": 0}
    scripted = [r for r in valid if _is_true(r.get("scripted", "0"))]
    rep["scripted_trials"] = analyze(scripted) if scripted else {"n_participants": 0}
    sr = [r for r in valid if r.get("felt_label") in VALENCE]
    rep["sensor_vs_felt"] = {
        "n": len(sr),
        "text_acc": float(np.mean([r["text_argmax"] == r["felt_label"] for r in sr])) if sr else None,
        "face_acc": float(np.mean([r["face_argmax"] == r["felt_label"] for r in sr if r.get("face_argmax") not in ("", "none")])) if sr else None,
        "fused_acc": float(np.mean([r["fused_label"] == r["felt_label"] for r in sr])) if sr else None,
        "valence_dissonant_rate": float(np.mean([_is_true(r.get("valence_dissonant", "0")) for r in sr])) if sr else None,
    }
    if global_rows:
        g = np.array([[float(r[k]) for k in PETS_KEYS] for r in global_rows])
        er, ut = g[:, :6], g[:, 6:]
        rep["pets_global"] = {
            "n": len(global_rows), "total_mean": float(g.mean()), "er_mean": float(er.mean()), "ut_mean": float(ut.mean()),
            "alpha_total": cronbach_alpha(g), "alpha_er": cronbach_alpha(er), "alpha_ut": cronbach_alpha(ut),
            "funnel_noticed_pattern": {v: sum(1 for r in global_rows if str(r.get("funnel_noticed_pattern", "")).lower() == v) for v in ("yes", "no", "unsure")},
        }
    return rep
