"""Append HCI Likert study rows to a CSV (Project.pdf §3.2)."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default relative to *current working directory* (repo root when running streamlit)
DEFAULT_CSV = Path("data") / "likert_responses.csv"

FIELDNAMES: tuple[str, ...] = (
    "ts_iso_utc",
    "participant_id",
    "empathy_1_5",
    "social_presence_1_5",
    "congruency_1_5",
    "alpha",
    "vision_branch",
    "dissonant",
    "text_argmax",
    "face_argmax",
    "fused_label",
    "e2e_ms",
    "utterance_excerpt",
    "response_excerpt",
)


def likert_csv_path() -> Path:
    p = (os.environ.get("LIKERT_CSV") or "").strip()
    if p:
        return Path(p)
    return DEFAULT_CSV


def append_likert_row(
    *,
    participant_id: str = "",
    empathy_1_5: int,
    social_presence_1_5: int,
    congruency_1_5: int,
    alpha: float,
    vision_branch: str,
    dissonant: bool,
    text_argmax: str,
    face_argmax: str,
    fused_label: str,
    e2e_ms: float,
    utterance: str,
    response: str,
    path: Path | None = None,
) -> Path:
    """Append one study row. Creates parent dirs and header row on first use."""
    for name, v in [
        ("empathy_1_5", empathy_1_5),
        ("social_presence_1_5", social_presence_1_5),
        ("congruency_1_5", congruency_1_5),
    ]:
        if v not in (1, 2, 3, 4, 5):
            raise ValueError(f"{name} must be 1..5, got {v!r}")
    out = path or likert_csv_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out.is_file() or out.stat().st_size == 0
    row: dict[str, Any] = {
        "ts_iso_utc": datetime.now(timezone.utc).isoformat(),
        "participant_id": (participant_id or "").strip() or "anon",
        "empathy_1_5": empathy_1_5,
        "social_presence_1_5": social_presence_1_5,
        "congruency_1_5": congruency_1_5,
        "alpha": round(float(alpha), 4),
        "vision_branch": vision_branch,
        "dissonant": bool(dissonant),
        "text_argmax": text_argmax,
        "face_argmax": face_argmax,
        "fused_label": fused_label,
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


def _excerpt(s: str, max_len: int) -> str:
    t = (s or "").replace("\n", " ").replace("\r", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _csv_cell(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)
