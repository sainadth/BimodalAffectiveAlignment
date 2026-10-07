"""Functional tests of the participant app via streamlit.testing (no models, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "study_app.py")


def _run(monkeypatch, configured=True):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    if configured:
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
    else:
        # .env is loaded with override=False, so an empty value blocks the real key from leaking in.
        monkeypatch.setenv("GROQ_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=30)
    return at.run()


def _texts(at):
    return " ".join(m.value for m in at.markdown) + " ".join(getattr(e, "value", "") for e in at.error) + " ".join(
        getattr(e, "value", "") for e in at.info)


def test_hard_fails_without_generator(monkeypatch):
    at = _run(monkeypatch, configured=False)
    assert at.error and "reply generator is not available" in at.error[0].value
    assert not at.checkbox  # consent never shown


def test_consent_screen_discloses_key_facts(monkeypatch):
    at = _run(monkeypatch)
    text = _texts(at)
    for phrase in ("never saved or transmitted", "Groq", "recall a recent small frustration",
                   "deliberately based on an inaccurate guess", "delete your data with one click",
                   "no effect on any course grade"):
        assert phrase in text, phrase
    assert at.button[0].disabled  # Continue disabled until the box is ticked


def test_consent_then_start_requires_valid_pid(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDY_CSV", str(tmp_path / "s.csv"))
    at = _run(monkeypatch)
    at.checkbox[0].check().run()
    at.button[0].click().run()
    assert any("Start" in m.value for m in at.markdown)
    at.text_input[0].input("???").run()
    assert at.warning and "participant number" in at.warning[0].value
    assert at.button[0].disabled
    at.text_input[0].input("P7").run()
    assert not at.button[0].disabled
