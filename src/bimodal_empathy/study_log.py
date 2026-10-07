"""Trial-level logging and scheduling for the within-subjects study (design v3).

Design v3 (2026-09-11) — label correctness × conversational move:
  * the participant reports the emotion they feel BEFORE the reply is generated;
    that report defines the "correct" label for the trial
  * 6 conditions = LABEL_SOURCES {correct, wrong, none} × MOVES {assert, ask}
      correct : the reply is conditioned on the felt label
      wrong   : the reply is conditioned on a label of opposite valence
      none    : the reply generator gets no emotion label (LLM reads the text)
      assert  : the agent states how the user seems to feel
      ask     : the agent checks in with a question instead of stating it
  * 2 blocks × 6 = 12 trials; Williams Latin square (6 rows) keyed by
    participant index; one scripted affect-elicitation slot per block
  * per trial: felt label, three 0-100 slider items (TRIAL_ITEMS, primary DV) and a
    manipulation check ("which emotion did the agent think you were feeling?")
  * once per participant: full PETS-10 on the same 0-100 scale (as validated, which uses
    101-point sliders) + blinding funnel (append_pets_global)
Sensor labels (text/face/fused, dissonance flags) are still computed and
logged for exploratory analysis; they do not drive the manipulation.
"""

from __future__ import annotations

import csv
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.dissonance import VALENCE

DEFAULT_CSV = Path("data") / "study_responses.csv"
DEFAULT_GLOBAL_CSV = Path("data") / "study_pets_global.csv"

LABEL_SOURCES: tuple[str, ...] = ("correct", "wrong", "none")
MOVES: tuple[str, ...] = ("assert", "ask")
CONDITIONS: tuple[str, ...] = tuple(f"{ls}_{mv}" for ls in LABEL_SOURCES for mv in MOVES)
BLOCKS = 2
TRIALS_PER_PARTICIPANT = BLOCKS * len(CONDITIONS)


def _williams(n: int) -> tuple[tuple[int, ...], ...]:
    """Williams design for even n: row k is the cyclic shift by k of (0, 1, n-1, 2, n-2, 3, ...)."""
    base: list[int] = []
    lo, hi = 0, n - 1
    for i in range(n):
        base.append(lo if i % 2 == 0 else hi)
        if i % 2 == 0:
            lo += 1
        else:
            hi -= 1
    return tuple(tuple((x + k) % n for x in base) for k in range(n))


_WILLIAMS = _williams(len(CONDITIONS))

SCRIPTED_PROMPTS: tuple[str, ...] = (
    "Think of something small that frustrated or disappointed you this week. While holding it in mind, type a short, polite message saying things are fine.",
    "Think of something you are genuinely looking forward to. While thinking about it, type a short, flat message about what you did today.",
)

FELT_OPTIONS: tuple[str, ...] = (*FER7_LABELS, "Prefer not to say")
MANIP_OPTIONS: tuple[str, ...] = (*FER7_LABELS, "It did not say", "Unsure")

# Statements rated on SCALE_MIN..SCALE_MAX (slider, no numeric readout).
TRIAL_ITEMS: tuple[tuple[str, str], ...] = (
    ("rate_match", "The reply matched how I actually felt."),
    ("rate_warm", "The reply felt warm and understanding."),
    ("rate_mental", "The system considered my mental state."),  # PETS-ER1
)
TRIAL_ITEM_KEYS: tuple[str, ...] = tuple(k for k, _ in TRIAL_ITEMS)
MANIP_CHECK_TEXT = "Which emotion did the agent think you were feeling? (Not how you actually felt.)"

PETS_ITEMS: tuple[tuple[str, str], ...] = (
    ("pets_er1", "The system considered my mental state."),
    ("pets_er2", "The system seemed emotionally intelligent."),
    ("pets_er3", "The system expressed emotions."),
    ("pets_er4", "The system sympathized with me."),
    ("pets_er5", "The system showed interest in me."),
    ("pets_er6", "The system supported me in coping with an emotional situation."),
    ("pets_ut1", "The system understood my goals."),
    ("pets_ut2", "The system understood my needs."),
    ("pets_ut3", "I trusted the system."),
    ("pets_ut4", "The system understood my intentions."),
)
PETS_KEYS: tuple[str, ...] = tuple(k for k, _ in PETS_ITEMS)

FIELDNAMES: tuple[str, ...] = (
    "ts_iso_utc",
    "participant_id",
    "trial_index",
    "block",
    "condition",
    "label_source",
    "move",
    "correctness_effective",
    "scripted",
    "scripted_prompt",
    "felt_label",
    "shown_label",
    "manip_check_label",
    "label_in_reply",
    *TRIAL_ITEM_KEYS,
    "alpha",
    "vision_branch",
    "had_face_image",
    "face_detected",
    "argmax_dissonant",
    "valence_dissonant",
    "text_argmax",
    "face_argmax",
    "fused_label",
    "generator",
    "input_method",
    "attempts",
    "e2e_ms",
    "utterance_excerpt",
    "response_excerpt",
)

GLOBAL_FIELDNAMES: tuple[str, ...] = (
    "ts_iso_utc",
    "participant_id",
    *PETS_KEYS,
    "funnel_noticed_pattern",
    "funnel_free_text",
    "consent_version",
    "consent_ts_iso_utc",
)


def study_csv_path() -> Path:
    p = (os.environ.get("STUDY_CSV") or "").strip()
    return Path(p) if p else DEFAULT_CSV


def global_csv_path() -> Path:
    p = (os.environ.get("STUDY_GLOBAL_CSV") or "").strip()
    return Path(p) if p else DEFAULT_GLOBAL_CSV


def normalize_pid(participant_id: str) -> str:
    """Canonical participant id: lower-case, trimmed, digits zero-padded to 2 (p1 == P01 == p01)."""
    s = (participant_id or "").strip().lower()
    m = re.fullmatch(r"([a-z]*)0*(\d+)", s)
    if m:
        return f"{m.group(1) or 'p'}{int(m.group(2)):02d}"
    return s or "anon"


def participant_index(participant_id: str) -> int:
    pid = normalize_pid(participant_id)
    m = re.search(r"(\d+)$", pid)
    if m:
        return int(m.group(1)) - 1 if int(m.group(1)) > 0 else 0
    return random.Random(f"idx::{pid}").randrange(10**6)


def split_condition(condition: str) -> tuple[str, str]:
    ls, mv = condition.rsplit("_", 1)
    if ls not in LABEL_SOURCES or mv not in MOVES:
        raise ValueError(f"unknown condition {condition!r}")
    return ls, mv


def condition_schedule(participant_id: str) -> list[dict]:
    """TRIALS_PER_PARTICIPANT specs: {trial, block, condition, label_source, move, scripted, scripted_prompt}.

    Block b uses Williams row (k + 3b) mod 6 so the two blocks differ; the
    scripted trial in block b is the one whose condition index is (k + 3b) mod 6,
    so across any 6 consecutive participants every condition starts block 1
    once and is scripted exactly twice.
    """
    k = participant_index(participant_id)
    n = len(CONDITIONS)
    out: list[dict] = []
    t = 0
    for b in range(BLOCKS):
        row = _WILLIAMS[(k + 3 * b) % n]
        scripted_pos = row.index((k + 3 * b) % n)
        for pos, ci in enumerate(row):
            cond = CONDITIONS[ci]
            ls, mv = split_condition(cond)
            out.append({
                "trial": t, "block": b, "condition": cond, "label_source": ls, "move": mv,
                "scripted": pos == scripted_pos,
                "scripted_prompt": SCRIPTED_PROMPTS[b] if pos == scripted_pos else "",
            })
            t += 1
    return out


def wrong_label_for(felt_label: str, seed: str) -> str:
    """A label of opposite valence to the felt one (deterministic per seed).

    pos (Happy)                  -> one of the negative labels
    neg (Angry/Disgust/Fear/Sad) -> Happy
    neu (Neutral/Surprise)       -> Sad or Angry (never Happy: positive surprise would
                                     make a "happy" attribution feel correct, not wrong)
    """
    rng = random.Random(f"wrong::{seed}")
    v = VALENCE.get(felt_label)
    if v == "pos":
        return rng.choice(["Sad", "Angry", "Fear", "Disgust"])
    if v == "neg":
        return "Happy"
    # Neutral and Surprise are valence-ambiguous: a positively surprised participant would
    # not experience "happy" as wrong (pilot 2026-09-14, turn 8), so always go clearly negative.
    return rng.choice(["Sad", "Angry"])


def resolve_shown_label(label_source: str, felt_label: str, sensor_fused_label: str, seed: str) -> tuple[str, str]:
    """Return (shown_label, correctness_effective).

    correctness_effective is what actually happened after accounting for a
    participant who declined to report: "correct", "wrong", "none", or
    "sensor" (felt unknown -> fall back to the sensor's fused label).
    """
    if label_source == "none":
        return "", "none"
    if felt_label not in VALENCE:  # "Prefer not to say"
        return sensor_fused_label, "sensor"
    if label_source == "correct":
        return felt_label, "correct"
    if label_source == "wrong":
        return wrong_label_for(felt_label, seed), "wrong"
    raise ValueError(f"unknown label_source {label_source!r}")


def label_delivered(reply: str, shown_label: str) -> bool | None:
    """Did the generated reply actually contain the emotion word it was told to use?

    Objective check that the manipulation reached the participant, independent of whether
    they noticed it (that is what the manipulation-check item measures). None when no label
    was given (the `none` condition).
    """
    if not shown_label:
        return None
    return re.search(rf"\b{re.escape(shown_label.lower())}", (reply or "").lower()) is not None


SCALE_MIN, SCALE_MAX = 0, 100
SCALE_ANCHORS = ("strongly disagree", "strongly agree")


def _check_scale(name: str, v: Any) -> int:
    """Ratings are continuous 0-100 (slider). Paper-01 reported PETS medians near 80/100
    with IQRs reaching 90, so a 5-point scale compresses exactly where the differences are."""
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not (SCALE_MIN <= v <= SCALE_MAX):
        raise ValueError(f"{name} must be {SCALE_MIN}..{SCALE_MAX}, got {v!r}")
    return int(round(v))


def append_study_row(
    *,
    participant_id: str,
    trial_index: int,
    block: int,
    condition: str,
    correctness_effective: str,
    scripted: bool,
    scripted_prompt: str,
    felt_label: str,
    shown_label: str,
    manip_check_label: str,
    ratings: dict[str, int],
    alpha: float,
    vision_branch: str,
    had_face_image: bool,
    face_detected: bool,
    argmax_dissonant: bool,
    valence_dissonant: bool,
    text_argmax: str,
    face_argmax: str,
    fused_label: str,
    generator: str,
    attempts: int,
    input_method: str = "typed",
    e2e_ms: float,
    utterance: str,
    response: str,
    path: Path | None = None,
) -> Path:
    """Append one trial row. Creates parent dirs and header row on first use."""
    ls, mv = split_condition(condition)
    if correctness_effective not in ("correct", "wrong", "none", "sensor"):
        raise ValueError(f"bad correctness_effective {correctness_effective!r}")
    if felt_label not in FELT_OPTIONS:
        raise ValueError(f"felt_label must be one of {FELT_OPTIONS}, got {felt_label!r}")
    if manip_check_label not in MANIP_OPTIONS:
        raise ValueError(f"manip_check_label must be one of {MANIP_OPTIONS}, got {manip_check_label!r}")
    missing = [k for k in TRIAL_ITEM_KEYS if k not in ratings]
    if missing:
        raise ValueError(f"missing trial items: {missing}")
    out = path or study_csv_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out.is_file() or out.stat().st_size == 0
    row: dict[str, Any] = {
        "ts_iso_utc": datetime.now(timezone.utc).isoformat(),
        "participant_id": normalize_pid(participant_id),
        "trial_index": int(trial_index),
        "block": int(block),
        "condition": condition,
        "label_source": ls,
        "move": mv,
        "correctness_effective": correctness_effective,
        "scripted": bool(scripted),
        "scripted_prompt": scripted_prompt,
        "felt_label": felt_label,
        "shown_label": shown_label,
        "manip_check_label": manip_check_label,
        "label_in_reply": label_delivered(response, shown_label),
        **{k: _check_scale(k, ratings[k]) for k in TRIAL_ITEM_KEYS},
        "alpha": round(float(alpha), 4),
        "vision_branch": vision_branch,
        "had_face_image": bool(had_face_image),
        "face_detected": bool(face_detected),
        "argmax_dissonant": bool(argmax_dissonant),
        "valence_dissonant": bool(valence_dissonant),
        "text_argmax": text_argmax,
        "face_argmax": face_argmax,
        "fused_label": fused_label,
        "generator": generator,
        "input_method": input_method,
        "attempts": int(attempts),
        "e2e_ms": round(float(e2e_ms), 2),
        "utterance_excerpt": _excerpt(utterance, 400),
        "response_excerpt": _excerpt(response, 500),
    }
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: _csv_cell(row.get(k, "")) for k in FIELDNAMES})
    return out


def append_pets_global(
    *,
    participant_id: str,
    pets: dict[str, int],
    funnel_noticed_pattern: str,
    funnel_free_text: str,
    consent_version: str,
    consent_ts_iso_utc: str,
    path: Path | None = None,
) -> Path:
    missing = [k for k in PETS_KEYS if k not in pets]
    if missing:
        raise ValueError(f"missing PETS items: {missing}")
    out = path or global_csv_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out.is_file() or out.stat().st_size == 0
    row = {
        "ts_iso_utc": datetime.now(timezone.utc).isoformat(),
        "participant_id": normalize_pid(participant_id),
        **{k: _check_scale(k, pets[k]) for k in PETS_KEYS},
        "funnel_noticed_pattern": funnel_noticed_pattern,
        "funnel_free_text": _excerpt(funnel_free_text, 500),
        "consent_version": consent_version,
        "consent_ts_iso_utc": consent_ts_iso_utc,
    }
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GLOBAL_FIELDNAMES, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: _csv_cell(row.get(k, "")) for k in GLOBAL_FIELDNAMES})
    return out


def completed_trials(participant_id: str, path: Path | None = None) -> int:
    p = path or study_csv_path()
    if not p.is_file():
        return 0
    pid = normalize_pid(participant_id)
    seen: set[int] = set()
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("participant_id") == pid:
                try:
                    seen.add(int(r["trial_index"]))
                except (KeyError, ValueError):
                    pass
    return len(seen)


def withdraw_participant(participant_id: str, paths: tuple[Path, ...] | None = None) -> int:
    """Delete every row for this participant from the study CSVs. Returns rows removed."""
    pid = normalize_pid(participant_id)
    removed = 0
    for p in paths or (study_csv_path(), global_csv_path()):
        if not p.is_file():
            continue
        with p.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = list(reader)
        keep = [r for r in rows if r.get("participant_id") != pid]
        removed += len(rows) - len(keep)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)
    return removed


def _excerpt(s: str, max_len: int) -> str:
    t = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)
