"""FastAPI backend for the remote study (design v3: label correctness × conversational move).

Flow per participant token (see docs/STUDY_PROTOCOL.md):
  GET  /api/session/{token}                    -> consent/debrief text, resume index, items
  POST /api/session/{token}/consent            -> log consent (version + timestamp kept server-side)
  POST /api/session/{token}/trial/{t}/analyze  {text, frames[]} -> runs the sensors only; nothing shown
  POST /api/session/{token}/trial/{t}/felt     {felt}          -> resolves the label from the felt report
                                                                  per condition, GENERATES the reply, returns it
  POST /api/session/{token}/trial/{t}/rate     {ratings, manip_check} -> appends the trial row
  POST /api/session/{token}/pets               {pets, funnel, free_text}
  POST /api/session/{token}/withdraw
Photos are decoded in memory and discarded after inference.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from bimodal_empathy import pipeline  # noqa: E402
from bimodal_empathy.config import DEFAULT_ALPHA, FER7_LABELS  # noqa: E402
from bimodal_empathy.llm_generator import generate_reply_v3, resolve_provider  # noqa: E402
from bimodal_empathy.study_log import (  # noqa: E402
    FELT_OPTIONS,
    SCALE_ANCHORS,
    SCALE_MAX,
    SCALE_MIN,
    MANIP_CHECK_TEXT,
    MANIP_OPTIONS,
    PETS_ITEMS,
    PETS_KEYS,
    TRIAL_ITEM_KEYS,
    TRIAL_ITEMS,
    TRIALS_PER_PARTICIPANT,
    append_pets_global,
    append_study_row,
    completed_trials,
    condition_schedule,
    resolve_shown_label,
    withdraw_participant,
)
from server.participants import pid_for  # noqa: E402

CONSENT_VERSION = "v5-remote-correctness-voice-2026-09-14"
# Pilot-only: when SAVE_FRAMES_DIR is set, webcam bursts and their face crops are written
# there for offline validation. MUST stay unset for real participants — the consent text
# states that frames are never stored.
SAVE_FRAMES_DIR = (os.environ.get("SAVE_FRAMES_DIR") or "").strip()
REQUIRED_PROVIDER = os.environ.get("STUDY_LLM_PROVIDER", "groq")
MAX_FRAMES = 5
MIN_AUDIO_BYTES = 1000
MAX_AUDIO_BYTES = 8 * 1024 * 1024
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "whisper-large-v3-turbo")

CONSENT_MD = (ROOT / "server" / "consent_remote.md").read_text(encoding="utf-8")
DEBRIEF_MD = (ROOT / "server" / "debrief_remote.md").read_text(encoding="utf-8")

app = FastAPI(title="Bimodal affect study", docs_url=None, redoc_url=None)

# token -> {"consent_ts": str, "pending": {trial: analysis}, "attempts": {trial: int}}
_SESSIONS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _session(token: str) -> tuple[str, dict]:
    pid = pid_for(token)
    if not pid:
        raise HTTPException(404, "Unknown participant link.")
    with _LOCK:
        s = _SESSIONS.setdefault(token, {"consent_ts": "", "pending": {}, "attempts": {}})
    return pid, s


def _llm_ok() -> bool:
    try:
        provider, _ = resolve_provider()
    except ValueError:
        return False
    key = "GROQ_API_KEY" if provider == "groq" else "ANTHROPIC_API_KEY"
    return provider == REQUIRED_PROVIDER and bool((os.environ.get(key) or "").strip())


@app.on_event("startup")
def _startup() -> None:
    if SAVE_FRAMES_DIR:
        print(f"  WARNING: SAVE_FRAMES_DIR={SAVE_FRAMES_DIR} — webcam frames WILL be written to disk.\n"
              "  This contradicts the consent text. Pilot use only; unset before real participants.", flush=True)
    if os.environ.get("STUDY_SKIP_WARMUP") != "1":
        threading.Thread(target=pipeline.warm_up, kwargs={"use_finetune": True}, daemon=True).start()


class ConsentIn(BaseModel):
    version: str


class AnalyzeIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    frames: list[str] = Field(default_factory=list, max_length=MAX_FRAMES)
    input_method: str = "typed"  # typed | spoken | spoken+edited


class TranscribeIn(BaseModel):
    audio: str = Field(min_length=16)  # base64 webm/ogg/wav, data URL accepted


class FeltIn(BaseModel):
    felt: str


class RateIn(BaseModel):
    ratings: dict[str, int]
    manip_check: str


class PetsIn(BaseModel):
    pets: dict[str, int]
    funnel: str
    free_text: str = ""


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "llm_configured": _llm_ok(), "consent_version": CONSENT_VERSION}


@app.get("/api/session/{token}")
def get_session(token: str) -> dict:
    pid, s = _session(token)
    if not _llm_ok():
        raise HTTPException(503, "The study is not configured (reply generator unavailable). Please contact the researcher.")
    return {
        "consent_md": CONSENT_MD,
        "debrief_md": DEBRIEF_MD,
        "consent_version": CONSENT_VERSION,
        "consented": bool(s["consent_ts"]),
        "trials_total": TRIALS_PER_PARTICIPANT,
        "trials_done": completed_trials(pid),
        "felt_options": list(FELT_OPTIONS),
        "manip_options": list(MANIP_OPTIONS),
        "manip_check_text": MANIP_CHECK_TEXT,
        "scale": {"min": SCALE_MIN, "max": SCALE_MAX, "low": SCALE_ANCHORS[0], "high": SCALE_ANCHORS[1]},
        "dictation": True,
        "trial_items": [{"key": k, "text": t} for k, t in TRIAL_ITEMS],
        "pets_items": [{"key": k, "text": t} for k, t in PETS_ITEMS],
        "scripted": {str(t["trial"]): t["scripted_prompt"] for t in condition_schedule(pid) if t["scripted"]},
    }


@app.post("/api/session/{token}/consent")
def post_consent(token: str, body: ConsentIn) -> dict:
    _, s = _session(token)
    if body.version != CONSENT_VERSION:
        raise HTTPException(409, "Consent version mismatch; please reload.")
    s["consent_ts"] = datetime.now(timezone.utc).isoformat()
    return {"ok": True}


def _decode_frames(frames: list[str]) -> list[Image.Image]:
    out: list[Image.Image] = []
    for f in frames[:MAX_FRAMES]:
        data = f.split(",", 1)[1] if f.startswith("data:") else f
        try:
            out.append(Image.open(BytesIO(base64.b64decode(data))).convert("RGB"))
        except Exception:
            continue
    return out


@app.post("/api/session/{token}/transcribe")
def post_transcribe(token: str, body: TranscribeIn) -> dict:
    """Optional dictation: audio is transcribed and dropped. The text lands in the
    participant's own box, where they can edit it before sending, so the recorded
    utterance is always text they approved. Audio is never stored."""
    _, s = _session(token)
    if not s["consent_ts"]:
        raise HTTPException(403, "Consent required.")
    raw = body.audio.split(",", 1)[1] if body.audio.startswith("data:") else body.audio
    try:
        blob = base64.b64decode(raw)
    except Exception:
        raise HTTPException(422, "Audio could not be read.")
    if not MIN_AUDIO_BYTES <= len(blob) <= MAX_AUDIO_BYTES:
        raise HTTPException(422, "That recording was too short or too long. Try again, or type instead.")
    try:
        from groq import Groq

        text = Groq().audio.transcriptions.create(
            file=("speech.webm", blob), model=TRANSCRIBE_MODEL, response_format="text",
        )
    except Exception as exc:
        raise HTTPException(502, f"Could not transcribe ({type(exc).__name__}). Type your message instead.")
    return {"text": str(text).strip()}


@app.post("/api/session/{token}/trial/{t}/analyze")
def post_analyze(token: str, t: int, body: AnalyzeIn) -> dict:
    """Run the sensors and park the result. The reply is generated only after the felt report."""
    pid, s = _session(token)
    if not s["consent_ts"]:
        raise HTTPException(403, "Consent required.")
    done = completed_trials(pid)
    if t != done:
        raise HTTPException(409, f"Next trial is {done}.")
    frames = _decode_frames(body.frames)
    if not frames:
        raise HTTPException(422, "At least one readable photo is required.")
    spec_now = condition_schedule(pid)[t]
    save_dir = Path(SAVE_FRAMES_DIR) / pid / f"t{t:02d}_{spec_now['condition']}" if SAVE_FRAMES_DIR else None
    a = pipeline.analyze_affect(body.text, frames, alpha=DEFAULT_ALPHA, use_finetune=True, save_frames_to=save_dir)
    del frames
    a.update({"utterance": body.text.strip(), "spec": spec_now, "frames_dir": str(save_dir) if save_dir else "",
              "input_method": body.input_method if body.input_method in ("typed", "spoken", "spoken+edited") else "typed"})
    with _LOCK:
        s["attempts"][t] = s["attempts"].get(t, 0) + 1
        s["pending"][t] = a
    return {"ok": True, "face_detected": a["face_detected"]}


@app.post("/api/session/{token}/trial/{t}/felt")
def post_felt(token: str, t: int, body: FeltIn) -> dict:
    """Felt report -> label per condition -> generate -> return the reply."""
    pid, s = _session(token)
    a = s["pending"].get(t)
    if a is None:
        raise HTTPException(409, "No pending analysis for this trial.")
    if body.felt not in FELT_OPTIONS:
        raise HTTPException(422, "Invalid felt label.")
    spec = a["spec"]
    shown, eff = resolve_shown_label(spec["label_source"], body.felt, a["fused_label"], seed=f"{pid}::{t}")
    t0 = time.perf_counter()
    try:
        rl = generate_reply_v3(a["utterance"], spec["move"], shown or None)
    except Exception as exc:
        raise HTTPException(502, f"Reply generation failed ({type(exc).__name__}). Please try again.")
    if not rl.text.strip():
        raise HTTPException(502, "Empty reply from generator. Please try again.")
    a.update({
        "felt_label": body.felt, "shown_label": shown, "correctness_effective": eff, "reply": rl.text,
        "generator": f"{rl.provider}:{rl.model}",
        "e2e_ms": sum(a["timings"].values()) + (time.perf_counter() - t0) * 1000.0,
    })
    return {"reply": a["reply"]}


@app.post("/api/session/{token}/trial/{t}/rate")
def post_rate(token: str, t: int, body: RateIn) -> dict:
    pid, s = _session(token)
    a = s["pending"].get(t)
    if a is None or "reply" not in a:
        raise HTTPException(409, "Trial not ready to be rated.")
    if set(body.ratings) != set(TRIAL_ITEM_KEYS) or any(not SCALE_MIN <= v <= SCALE_MAX for v in body.ratings.values()):
        raise HTTPException(422, f"Ratings must cover every item with values {SCALE_MIN}-{SCALE_MAX}.")
    if body.manip_check not in MANIP_OPTIONS:
        raise HTTPException(422, "Invalid manipulation-check label.")
    spec = a["spec"]
    append_study_row(
        participant_id=pid, trial_index=t, block=spec["block"], condition=spec["condition"],
        correctness_effective=a["correctness_effective"], scripted=spec["scripted"], scripted_prompt=spec["scripted_prompt"],
        felt_label=a["felt_label"], shown_label=a["shown_label"], manip_check_label=body.manip_check,
        ratings=body.ratings, alpha=a["alpha"], vision_branch=a["vision_branch"], had_face_image=a["had_face_image"],
        face_detected=a["face_detected"], argmax_dissonant=a["argmax_dissonant"], valence_dissonant=a["valence_dissonant"],
        text_argmax=a["text_label"], face_argmax=a["face_label"] or "none", fused_label=a["fused_label"],
        generator=a["generator"], attempts=s["attempts"].get(t, 1), e2e_ms=a["e2e_ms"],
        input_method=a.get("input_method", "typed"), utterance=a["utterance"], response=a["reply"],
    )
    if a.get("frames_dir"):
        _write_frame_meta(Path(a["frames_dir"]), pid, t, a, body)
    with _LOCK:
        s["pending"].pop(t, None)
    done = completed_trials(pid)
    return {"ok": True, "trials_done": done, "finished": done >= TRIALS_PER_PARTICIPANT}


def _write_frame_meta(d: Path, pid: str, trial: int, a: dict, body: "RateIn") -> None:
    """Sidecar so the saved frames can be validated against what the sensors and the participant said."""
    spec = a["spec"]
    meta = {
        "participant_id": pid, "trial_index": trial, "block": spec["block"], "condition": spec["condition"],
        "label_source": spec["label_source"], "move": spec["move"], "scripted": spec["scripted"],
        "felt_label": a["felt_label"], "shown_label": a["shown_label"], "correctness": a["correctness_effective"],
        "manip_check_label": body.manip_check, "ratings": body.ratings,
        "sensor": {"text": a["text_label"], "face": a["face_label"], "fused": a["fused_label"],
                   "face_detected": a["face_detected"], "frames": a["frames"], "frames_with_face": a["frames_with_face"],
                   "per_frame_p_face": a.get("per_frame_p_face", []), "fer7_order": list(FER7_LABELS),
                   "argmax_dissonant": a["argmax_dissonant"], "valence_dissonant": a["valence_dissonant"]},
        "utterance": a["utterance"], "reply": a["reply"], "generator": a["generator"],
        "ts_iso_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass


@app.post("/api/session/{token}/pets")
def post_pets(token: str, body: PetsIn) -> dict:
    pid, s = _session(token)
    if completed_trials(pid) < TRIALS_PER_PARTICIPANT:
        raise HTTPException(409, "Trials not finished.")
    if set(body.pets) != set(PETS_KEYS) or any(not SCALE_MIN <= v <= SCALE_MAX for v in body.pets.values()):
        raise HTTPException(422, f"PETS must cover every item with values {SCALE_MIN}-{SCALE_MAX}.")
    if body.funnel.lower() not in ("yes", "no", "unsure"):
        raise HTTPException(422, "funnel must be yes/no/unsure.")
    append_pets_global(
        participant_id=pid, pets=body.pets, funnel_noticed_pattern=body.funnel.lower(),
        funnel_free_text=body.free_text, consent_version=CONSENT_VERSION, consent_ts_iso_utc=s["consent_ts"],
    )
    return {"ok": True}


@app.post("/api/session/{token}/withdraw")
def post_withdraw(token: str) -> dict:
    pid, _ = _session(token)
    n = withdraw_participant(pid)
    with _LOCK:
        _SESSIONS.pop(token, None)
    return {"ok": True, "rows_removed": n}


DIST = ROOT / "web" / "dist"
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        f = DIST / full_path
        if full_path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
