"""FastAPI study server (design v3): full participant flow with the pipeline and LLM stubbed."""

from __future__ import annotations

import base64
import csv
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _png_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (64, 64), (200, 150, 120)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


CALLS: list[tuple] = []


@pytest.fixture()
def client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STUDY_SKIP_WARMUP", "1")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("STUDY_CSV", str(tmp_path / "s.csv"))
    monkeypatch.setenv("STUDY_GLOBAL_CSV", str(tmp_path / "g.csv"))
    tokens = tmp_path / "participants.json"
    tokens.write_text(json.dumps({"tok01": "p01", "tok02": "p02"}))
    monkeypatch.setenv("PARTICIPANTS_JSON", str(tokens))

    import importlib

    import server.app as srv

    importlib.reload(srv)

    def fake_analyze(text, frames, *, alpha, use_finetune=True, save_frames_to=None):
        return {
            "p_text": [0] * 7, "p_face": [0] * 7, "p_fuse": [0] * 7, "text_label": "Neutral", "face_label": "Sad",
            "fused_label": "Sad", "had_face_image": True, "face_detected": True, "frames": len(frames),
            "frames_with_face": len(frames), "argmax_dissonant": True, "valence_dissonant": True,
            "vision_branch": "fine-tuned (FER2013)", "alpha": alpha, "timings": {"text_ms": 1.0, "face_ms": 1.0, "fusion_ms": 0.1},
            "per_frame_p_face": [],
        }

    class R:
        provider = "groq"
        model = "openai/gpt-oss-120b"
        latency_ms = 5.0

        def __init__(self, text):
            self.text = text

    def fake_generate(utterance, move, label, **kw):
        CALLS.append((move, label))
        return R(f"[{move}|{label or 'none'}] I hear you.")

    monkeypatch.setattr(srv.pipeline, "analyze_affect", fake_analyze)
    monkeypatch.setattr(srv, "generate_reply_v3", fake_generate)
    CALLS.clear()
    return TestClient(srv.app)


def test_unknown_token_404(client):
    assert client.get("/api/session/nope").status_code == 404


def test_full_flow_v3(client, tmp_path: Path):
    from bimodal_empathy.study_log import condition_schedule

    s = client.get("/api/session/tok01").json()
    assert s["trials_total"] == 12 and s["trials_done"] == 0 and len(s["scripted"]) == 2
    assert "manip_options" in s and "Unsure" in s["manip_options"]

    assert client.post("/api/session/tok01/trial/0/analyze", json={"text": "hi", "frames": [_png_b64()]}).status_code == 403
    assert client.post("/api/session/tok01/consent", json={"version": s["consent_version"]}).status_code == 200

    sched = condition_schedule("p01")
    rating = {"rate_match": 74, "rate_warm": 88, "rate_mental": 61}
    for t in range(12):
        assert client.post(f"/api/session/tok01/trial/{t + 1}/analyze", json={"text": "hi", "frames": [_png_b64()]}).status_code == 409
        assert client.post(f"/api/session/tok01/trial/{t}/analyze", json={"text": "hi", "frames": []}).status_code == 422
        r = client.post(f"/api/session/tok01/trial/{t}/analyze", json={"text": "I'm fine, really.", "frames": [_png_b64(), _png_b64()]})
        assert r.status_code == 200 and r.json()["face_detected"]
        # no reply exists yet: rating is refused; generation only happens after the felt report
        assert client.post(f"/api/session/tok01/trial/{t}/rate", json={"ratings": rating, "manip_check": "Sad"}).status_code == 409
        assert CALLS == [] or len(CALLS) == t
        r = client.post(f"/api/session/tok01/trial/{t}/felt", json={"felt": "Sad"})
        assert r.status_code == 200
        move, label = CALLS[-1]
        spec = sched[t]
        assert move == spec["move"]
        if spec["label_source"] == "correct":
            assert label == "Sad"
        elif spec["label_source"] == "wrong":
            assert label == "Happy"  # opposite valence of Sad
        else:
            assert label is None
        assert r.json()["reply"].startswith(f"[{move}|")
        assert client.post(f"/api/session/tok01/trial/{t}/rate", json={"ratings": rating, "manip_check": "Nope"}).status_code == 422
        r = client.post(f"/api/session/tok01/trial/{t}/rate", json={"ratings": rating, "manip_check": "Sad"})
        assert r.status_code == 200 and r.json()["trials_done"] == t + 1
    assert r.json()["finished"]

    rows = list(csv.DictReader((tmp_path / "s.csv").open()))
    assert len(rows) == 12
    assert {r["condition"] for r in rows} == set(c for c in __import__("bimodal_empathy.study_log", fromlist=["CONDITIONS"]).CONDITIONS)
    assert all(r["felt_label"] == "Sad" for r in rows)
    wrong = [r for r in rows if r["label_source"] == "wrong"]
    assert wrong and all(r["shown_label"] == "Happy" and r["correctness_effective"] == "wrong" for r in wrong)
    none = [r for r in rows if r["label_source"] == "none"]
    assert none and all(r["shown_label"] == "" and r["correctness_effective"] == "none" for r in none)

    pets = {f"pets_er{i}": 80 for i in range(1, 7)} | {f"pets_ut{i}": 55 for i in range(1, 5)}
    assert client.post("/api/session/tok01/pets", json={"pets": pets, "funnel": "No"}).status_code == 200
    assert client.get("/api/session/tok01").json()["trials_done"] == 12
    r = client.post("/api/session/tok01/withdraw")
    assert r.status_code == 200 and r.json()["rows_removed"] == 13


def test_prefer_not_to_say_falls_back_to_sensor(client):
    s = client.get("/api/session/tok02").json()
    client.post("/api/session/tok02/consent", json={"version": s["consent_version"]})
    client.post("/api/session/tok02/trial/0/analyze", json={"text": "ok", "frames": [_png_b64()]})
    from bimodal_empathy.study_log import condition_schedule

    spec = condition_schedule("p02")[0]
    r = client.post("/api/session/tok02/trial/0/felt", json={"felt": "Prefer not to say"})
    assert r.status_code == 200
    move, label = CALLS[-1]
    assert label == (None if spec["label_source"] == "none" else "Sad")  # sensor fused label


def test_frame_saving_is_off_by_default_and_writes_when_enabled(monkeypatch, tmp_path):
    """SAVE_FRAMES_DIR must be opt-in; when set, frames + meta.json land in a per-trial folder."""
    import importlib

    import server.app as srv

    monkeypatch.setenv("STUDY_SKIP_WARMUP", "1")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("STUDY_CSV", str(tmp_path / "s.csv"))
    monkeypatch.setenv("STUDY_GLOBAL_CSV", str(tmp_path / "g.csv"))
    tok = tmp_path / "participants.json"
    tok.write_text(json.dumps({"tok01": "p01"}))
    monkeypatch.setenv("PARTICIPANTS_JSON", str(tok))

    importlib.reload(srv)
    assert srv.SAVE_FRAMES_DIR == ""  # off unless explicitly enabled

    frames_dir = tmp_path / "frames"
    monkeypatch.setenv("SAVE_FRAMES_DIR", str(frames_dir))
    importlib.reload(srv)
    assert srv.SAVE_FRAMES_DIR == str(frames_dir)

    seen = {}

    def fake_analyze(text, frames, *, alpha, use_finetune=True, save_frames_to=None):
        seen["dir"] = save_frames_to
        if save_frames_to is not None:
            save_frames_to.mkdir(parents=True, exist_ok=True)
            (save_frames_to / "raw_0.jpg").write_bytes(b"x")
        return {
            "p_text": [0] * 7, "p_face": [0] * 7, "p_fuse": [0] * 7, "text_label": "Neutral", "face_label": "Sad",
            "fused_label": "Sad", "had_face_image": True, "face_detected": True, "frames": 1, "frames_with_face": 1,
            "argmax_dissonant": True, "valence_dissonant": True, "vision_branch": "ft", "alpha": alpha,
            "timings": {"text_ms": 1.0, "face_ms": 1.0, "fusion_ms": 0.1}, "per_frame_p_face": [[0.1] * 7],
        }

    class R:
        text, provider, model, latency_ms = "ok", "groq", "m", 1.0

    monkeypatch.setattr(srv.pipeline, "analyze_affect", fake_analyze)
    monkeypatch.setattr(srv, "generate_reply_v3", lambda *a, **k: R())
    c = TestClient(srv.app)
    s = c.get("/api/session/tok01").json()
    c.post("/api/session/tok01/consent", json={"version": s["consent_version"]})
    c.post("/api/session/tok01/trial/0/analyze", json={"text": "hi", "frames": [_png_b64()]})
    c.post("/api/session/tok01/trial/0/felt", json={"felt": "Sad"})
    c.post("/api/session/tok01/trial/0/rate",
           json={"ratings": {"rate_match": 30, "rate_warm": 30, "rate_mental": 30}, "manip_check": "Sad"})

    from bimodal_empathy.study_log import condition_schedule

    d = frames_dir / "p01" / f"t00_{condition_schedule('p01')[0]['condition']}"
    assert seen["dir"] == d and (d / "raw_0.jpg").is_file()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["felt_label"] == "Sad" and meta["manip_check_label"] == "Sad"
    assert meta["condition"] == condition_schedule("p01")[0]["condition"]
    assert meta["sensor"]["fer7_order"][0] == "Angry" and meta["ratings"]["rate_warm"] == 30


def test_unconfigured_generator_blocks_session(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert client.get("/api/session/tok01").status_code == 503


def test_transcribe_requires_consent_and_returns_text(client, monkeypatch):
    """Dictation is optional; audio is transcribed and dropped, never logged."""
    import server.app as srv

    audio = "data:audio/webm;base64," + base64.b64encode(b"x" * 4000).decode()
    assert client.post("/api/session/tok02/transcribe", json={"audio": audio}).status_code == 403

    s = client.get("/api/session/tok02").json()
    assert s["dictation"] is True
    client.post("/api/session/tok02/consent", json={"version": s["consent_version"]})

    class FakeGroq:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kw):
                    assert kw["model"].startswith("whisper")
                    return "  I am doing fine, really.  "

        def __init__(self, *a, **k):
            pass

    monkeypatch.setitem(__import__("sys").modules, "groq", type("m", (), {"Groq": FakeGroq}))
    r = client.post("/api/session/tok02/transcribe", json={"audio": audio})
    assert r.status_code == 200 and r.json()["text"] == "I am doing fine, really."
    assert client.post("/api/session/tok02/transcribe", json={"audio": "data:audio/webm;base64,AAAA"}).status_code == 422


def test_input_method_is_logged(client, tmp_path: Path, monkeypatch):
    s = client.get("/api/session/tok01").json()
    client.post("/api/session/tok01/consent", json={"version": s["consent_version"]})
    client.post("/api/session/tok01/trial/0/analyze",
                json={"text": "hi", "frames": [_png_b64()], "input_method": "spoken+edited"})
    client.post("/api/session/tok01/trial/0/felt", json={"felt": "Sad"})
    client.post("/api/session/tok01/trial/0/rate",
                json={"ratings": {"rate_match": 50, "rate_warm": 50, "rate_mental": 50}, "manip_check": "Sad"})
    rows = list(csv.DictReader((tmp_path / "s.csv").open()))
    assert rows[0]["input_method"] == "spoken+edited"

    client.post("/api/session/tok01/trial/1/analyze",
                json={"text": "hi", "frames": [_png_b64()], "input_method": "nonsense"})
    client.post("/api/session/tok01/trial/1/felt", json={"felt": "Sad"})
    client.post("/api/session/tok01/trial/1/rate",
                json={"ratings": {"rate_match": 50, "rate_warm": 50, "rate_mental": 50}, "manip_check": "Sad"})
    rows = list(csv.DictReader((tmp_path / "s.csv").open()))
    assert rows[1]["input_method"] == "typed"  # unknown values fall back
