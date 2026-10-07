#!/usr/bin/env python3
"""Create participant tokens and print the links to hand out.

  python scripts/make_participant_links.py --n 28 --base https://study.example.edu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.participants import make_tokens, participants_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--base", default="http://localhost:8000")
    args = ap.parse_args()
    tokens = make_tokens(args.n, args.start)
    print(f"# {len(tokens)} tokens in {participants_path()} (keep this file private; it is git-ignored)")
    for tok, pid in sorted(tokens.items(), key=lambda kv: kv[1]):
        print(f"{pid}\t{args.base.rstrip('/')}/p/{tok}")


if __name__ == "__main__":
    main()
