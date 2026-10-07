import numpy as np
import pytest

from bimodal_empathy.eval.study_stats import analyze, cronbach_alpha, full_report, holm, manipulation_check, rank_biserial
from bimodal_empathy.study_log import CONDITIONS, PETS_KEYS, TRIAL_ITEM_KEYS, condition_schedule, wrong_label_for


def _fake_rows(n=18, gain=0.4, penalty=1.0, ask_bonus_when_wrong=0.5, seed=0):
    """Plant: correct = base+gain, none = base, wrong = base-penalty (+ask_bonus if ask)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n):
        pid = f"p{p + 1:02d}"
        for t in condition_schedule(pid):
            base = 3.2
            if t["label_source"] == "correct":
                base += gain
            elif t["label_source"] == "wrong":
                base -= penalty
                if t["move"] == "ask":
                    base += ask_bonus_when_wrong
            felt = "Sad"
            shown = {"correct": felt, "wrong": wrong_label_for(felt, f"{pid}::{t['trial']}"), "none": ""}[t["label_source"]]
            rows.append({
                "participant_id": pid, "trial_index": t["trial"], "condition": t["condition"], "label_source": t["label_source"],
                "move": t["move"], "correctness_effective": t["label_source"], "scripted": "1" if t["scripted"] else "0",
                "felt_label": felt, "shown_label": shown, "manip_check_label": shown or "Unsure",
                "text_argmax": "Neutral", "face_argmax": "Sad", "fused_label": "Sad", "valence_dissonant": "1",
                "label_in_reply": "1" if shown else "",
                **{k: int(np.clip(round((base + rng.normal(0, 0.5)) * 20), 0, 100)) for k in TRIAL_ITEM_KEYS},
            })
    return rows


def test_holm_and_rank_biserial():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert rank_biserial(np.array([5, 5]), np.array([1, 1])) == 1.0


def test_cronbach():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(200, 1))
    assert cronbach_alpha(latent + rng.normal(scale=0.3, size=(200, 3))) > 0.9


def test_planted_effects_recovered():
    rep = analyze(_fake_rows(18))
    assert rep["n_participants"] == 18 and rep["cells"] == list(CONDITIONS)
    assert rep["friedman_cells"]["p"] < 0.001
    pair = {(p["a"], p["b"]): p for p in rep["pairwise"]}
    assert pair[("correct", "wrong")]["p_holm"] < 0.01 and pair[("correct", "wrong")]["mean_diff"] > 0.8
    assert pair[("wrong_ask", "wrong_assert")]["p_holm"] < 0.05 and pair[("wrong_ask", "wrong_assert")]["mean_diff"] > 0
    asym = pair[("penalty(none-wrong)", "gain(correct-none)")]
    assert asym["mean_diff"] > 0 and asym["p_holm"] < 0.05  # penalty exceeds gain as planted
    assert rep["label_source_means"]["correct"] > rep["label_source_means"]["none"] > rep["label_source_means"]["wrong"]


def test_null_effects_not_flagged():
    rep = analyze(_fake_rows(18, gain=0.0, penalty=0.0, ask_bonus_when_wrong=0.0, seed=7))
    assert rep["friedman_cells"]["p"] > 0.01
    assert all(p["p_holm"] > 0.05 for p in rep["pairwise"])


def test_manipulation_check_and_report_sections():
    rows = _fake_rows(12)
    mc = manipulation_check(rows)
    assert mc["correct"]["perceived"] == 1.0 and mc["wrong"]["perceived"] == 1.0 and mc["none"]["perceived"] == 1.0
    assert mc["correct"]["delivered"] == 1.0 and mc["wrong"]["delivered"] == 1.0
    assert mc["correct"]["delivered_and_perceived"] == 1.0 and mc["none"]["delivered"] is None
    rep = full_report(rows, [{**{k: 78 for k in PETS_KEYS}, "funnel_noticed_pattern": "no"} for _ in range(12)])
    for key in ("primary_composite", "item_rate_match", "manipulation_check", "scripted_trials", "sensor_vs_felt",
                "pets_global", "sensitivity_manipulation_passed"):
        assert key in rep
    assert rep["sensor_vs_felt"]["face_acc"] == 1.0 and rep["sensor_vs_felt"]["text_acc"] == 0.0


def test_sensor_fallback_rows_excluded():
    rows = _fake_rows(8)
    for r in rows[:6]:
        r["correctness_effective"] = "sensor"
    rep = full_report(rows)
    assert rep["n_rows_analysed"] == len(rows) - 6


def test_small_n_descriptive():
    assert "friedman_cells" not in analyze(_fake_rows(3)) and "note" in analyze(_fake_rows(3))
