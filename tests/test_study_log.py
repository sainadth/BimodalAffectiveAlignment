"""Tests for design-v3 scheduling, label resolution and loggers."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from bimodal_empathy.study_log import (
    BLOCKS, CONDITIONS, FIELDNAMES, GLOBAL_FIELDNAMES, LABEL_SOURCES, MOVES, PETS_KEYS, TRIAL_ITEM_KEYS,
    TRIALS_PER_PARTICIPANT, append_pets_global, append_study_row, completed_trials, condition_schedule,
    normalize_pid, participant_index, resolve_shown_label, split_condition, withdraw_participant, wrong_label_for,
)


def test_conditions_are_the_full_factorial():
    assert len(CONDITIONS) == len(LABEL_SOURCES) * len(MOVES) == 6
    assert TRIALS_PER_PARTICIPANT == 12
    assert split_condition("wrong_ask") == ("wrong", "ask")
    with pytest.raises(ValueError):
        split_condition("random")


def test_normalize_pid():
    assert normalize_pid("p1") == normalize_pid("P01") == "p01" and participant_index("P4") == 3


def test_schedule_within_participant():
    s = condition_schedule("p01")
    assert len(s) == TRIALS_PER_PARTICIPANT
    for b in range(BLOCKS):
        block = [t for t in s if t["block"] == b]
        assert sorted(t["condition"] for t in block) == sorted(CONDITIONS)
        assert sum(t["scripted"] for t in block) == 1
    assert all(t["label_source"] in LABEL_SOURCES and t["move"] in MOVES for t in s)
    assert [t["condition"] for t in s[:6]] != [t["condition"] for t in s[6:]]


def test_schedule_counterbalanced_over_six_participants():
    pids = [f"p{i:02d}" for i in range(1, 13)]
    first = Counter(condition_schedule(p)[0]["condition"] for p in pids)
    assert set(first.values()) == {2}, first
    scripted = Counter(t["condition"] for p in pids for t in condition_schedule(p) if t["scripted"])
    assert set(scripted.values()) == {4}, scripted
    # Williams: each condition immediately follows each other condition equally often within block 1 across 6 pids
    trans = Counter()
    for p in pids[:6]:
        row = [t["condition"] for t in condition_schedule(p)[:6]]
        trans.update(zip(row, row[1:]))
    assert max(trans.values()) == 1 and len(trans) == 30


def test_wrong_label_is_opposite_valence():
    assert wrong_label_for("Happy", "x") in {"Sad", "Angry", "Fear", "Disgust"}
    for neg in ("Sad", "Angry", "Fear", "Disgust"):
        assert wrong_label_for(neg, "x") == "Happy"
    assert wrong_label_for("Neutral", "x") in {"Happy", "Sad"}
    assert wrong_label_for("Happy", "seed-a") == wrong_label_for("Happy", "seed-a")  # deterministic


def test_resolve_shown_label():
    assert resolve_shown_label("correct", "Sad", "Happy", "s") == ("Sad", "correct")
    assert resolve_shown_label("wrong", "Sad", "Happy", "s") == ("Happy", "wrong")
    assert resolve_shown_label("none", "Sad", "Happy", "s") == ("", "none")
    assert resolve_shown_label("correct", "Prefer not to say", "Fear", "s") == ("Fear", "sensor")
    with pytest.raises(ValueError):
        resolve_shown_label("random", "Sad", "Happy", "s")


def _ratings(v=62):
    return {k: v for k in TRIAL_ITEM_KEYS}


def _row(tmp_path: Path, **over):
    kw = dict(
        participant_id="P1", trial_index=0, block=0, condition="wrong_ask", correctness_effective="wrong",
        scripted=False, scripted_prompt="", felt_label="Sad", shown_label="Happy", manip_check_label="Happy",
        ratings=_ratings(), alpha=0.5, vision_branch="fine-tuned (FER2013)", had_face_image=True, face_detected=True,
        argmax_dissonant=True, valence_dissonant=True, text_argmax="Neutral", face_argmax="Sad", fused_label="Sad",
        generator="groq:openai/gpt-oss-120b", attempts=1, e2e_ms=800.0, utterance="I'm fine.",
        response="Are you feeling upbeat today?", path=tmp_path / "s.csv",
    )
    kw.update(over)
    return kw


def test_append_row_and_fields(tmp_path: Path):
    out = append_study_row(**_row(tmp_path))
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(rows[0].keys()) == list(FIELDNAMES)
    assert rows[0]["label_source"] == "wrong" and rows[0]["move"] == "ask" and rows[0]["participant_id"] == "p01"


def test_append_row_validation(tmp_path: Path):
    for bad in (dict(condition="random_ask"), dict(correctness_effective="maybe"), dict(felt_label="Meh"),
                dict(manip_check_label="Meh"), dict(ratings=_ratings(101))):
        with pytest.raises(ValueError):
            append_study_row(**_row(tmp_path, **bad))


def test_completed_and_withdraw(tmp_path: Path):
    p = tmp_path / "s.csv"
    for t in (0, 1, 1, 2):
        append_study_row(**_row(tmp_path, trial_index=t, path=p))
    append_study_row(**_row(tmp_path, participant_id="p02", path=p))
    assert completed_trials("p01", p) == 3 and completed_trials("p02", p) == 1
    g = tmp_path / "g.csv"
    append_pets_global(participant_id="p1", pets={k: 71 for k in PETS_KEYS}, funnel_noticed_pattern="no",
                       funnel_free_text="", consent_version="v3", consent_ts_iso_utc="", path=g)
    assert list(csv.DictReader(g.open()))[0].keys().__len__() == len(GLOBAL_FIELDNAMES)
    assert withdraw_participant("P01", (p, g)) == 5
    assert [r["participant_id"] for r in csv.DictReader(p.open())] == ["p02"]


def test_wrong_label_never_happy_for_ambiguous_valence():
    """Positive surprise would make a 'happy' attribution feel correct, not wrong (pilot 2026-09-14)."""
    for felt in ("Surprise", "Neutral"):
        got = {wrong_label_for(felt, f"s{i}") for i in range(30)}
        assert got <= {"Sad", "Angry"}, (felt, got)
    assert {wrong_label_for("Happy", f"s{i}") for i in range(30)} <= {"Sad", "Angry", "Fear", "Disgust"}
    for neg in ("Sad", "Angry", "Fear", "Disgust"):
        assert wrong_label_for(neg, "s") == "Happy"


def test_label_delivery_check():
    from bimodal_empathy.study_log import label_delivered

    assert label_delivered("You seem happy today.", "Happy") is True
    assert label_delivered("Are you feeling SAD right now?", "Sad") is True
    assert label_delivered("You sound upset.", "Angry") is False       # vaguer word substituted
    assert label_delivered("You seem unhappy.", "Happy") is False      # negation is not delivery
    assert label_delivered("anything", "") is None                      # `none` condition


def test_manip_check_has_explicit_not_said_option():
    from bimodal_empathy.study_log import MANIP_CHECK_TEXT, MANIP_OPTIONS

    assert "It did not say" in MANIP_OPTIONS and "Unsure" in MANIP_OPTIONS
    assert "Not how you actually felt" in MANIP_CHECK_TEXT


def test_label_in_reply_logged(tmp_path: Path):
    p = tmp_path / "s.csv"
    append_study_row(**_row(tmp_path, shown_label="Happy", response="You seem happy today.", path=p))
    append_study_row(**_row(tmp_path, trial_index=1, shown_label="Happy", response="You sound upset.", path=p))
    append_study_row(**_row(tmp_path, trial_index=2, condition="none_ask", correctness_effective="none",
                            shown_label="", response="How are you?", path=p))
    vals = [r["label_in_reply"] for r in csv.DictReader(p.open(encoding="utf-8"))]
    assert vals == ["1", "0", ""]
