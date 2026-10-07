"""Participant access tokens for the remote study.

data/participants.json maps an unguessable token -> participant id (p01...).
Generate with scripts/make_participant_links.py. Tokens are the only way in.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

DEFAULT_PATH = Path("data") / "participants.json"


def participants_path() -> Path:
    p = (os.environ.get("PARTICIPANTS_JSON") or "").strip()
    return Path(p) if p else DEFAULT_PATH


def load_tokens(path: Path | None = None) -> dict[str, str]:
    p = path or participants_path()
    if not p.is_file():
        return {}
    return {str(k): str(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}


def make_tokens(n: int, start: int = 1, path: Path | None = None) -> dict[str, str]:
    """Create/extend the token file with participants p{start}..p{start+n-1}."""
    p = path or participants_path()
    tokens = load_tokens(p)
    existing = set(tokens.values())
    for i in range(start, start + n):
        pid = f"p{i:02d}"
        if pid in existing:
            continue
        tokens[secrets.token_urlsafe(12)] = pid
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    return tokens


def pid_for(token: str, path: Path | None = None) -> str | None:
    return load_tokens(path).get(token)
