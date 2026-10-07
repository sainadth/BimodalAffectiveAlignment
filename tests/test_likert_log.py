"""Tests for HCI Likert CSV logging."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bimodal_empathy.likert_log import append_likert_row


def test_append_likert_creates_file_and_header(tmp_path: Path) -> None:
    p = tmp_path / "l.csv"
    out = append_likert_row(
        path=p,
        participant_id="p1",
        empathy_1_5=4,
        social_presence_1_5=3,
        congruency_1_5=5,
        alpha=0.5,
        vision_branch="baseline",
        dissonant=True,
        text_argmax="Happy",
        face_argmax="Sad",
        fused_label="Happy",
        e2e_ms=900.0,
        utterance="I'm fine.",
        response="I hear you.",
    )
    assert out == p
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert "empathy_1_5" in lines[0]
    assert "4" in lines[1] and "I hear you" in lines[1]


def test_likert_invalid_score(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empathy"):
        append_likert_row(
            path=tmp_path / "n.csv",
            empathy_1_5=0,
            social_presence_1_5=3,
            congruency_1_5=3,
            alpha=0.5,
            vision_branch="x",
            dissonant=False,
            text_argmax="A",
            face_argmax="B",
            fused_label="A",
            e2e_ms=1.0,
            utterance="u",
            response="r",
        )


def test_append_second_row_no_duplicate_header(tmp_path: Path) -> None:
    p = tmp_path / "l2.csv"
    kwargs = dict(
        path=p,
        participant_id="a",
        empathy_1_5=3,
        social_presence_1_5=3,
        congruency_1_5=3,
        alpha=0.5,
        vision_branch="ft",
        dissonant=False,
        text_argmax="N",
        face_argmax="N",
        fused_label="N",
        e2e_ms=100.0,
        utterance="x",
        response="y",
    )
    append_likert_row(**kwargs)
    append_likert_row(**kwargs)
    with p.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3
    assert rows[0][0] == "ts_iso_utc"
