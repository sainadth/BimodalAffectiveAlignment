#!/usr/bin/env python3
"""
Build a dissonant evaluation CSV (face = one-hot for true y + conflicting template text).
No real images: fast, but P_face is not from a CNN. For *real* FER2013 face images and ResNet
P_face, use:  python scripts/eval_dissonant_fer2013.py --limit 100

  python scripts/build_dissonant_eval_csv.py --n 100 --out data/dissonant_eval_100.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.eval.benchmarks import CONGRUENT_PHRASES


def _j_dissonant(y: int, k: int) -> int:
    """Pick text class j != y (varies with k)."""
    j = (y + 1 + (k % 6)) % 7
    if j == y:
        j = (y + 2) % 7
    return j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="Number of rows (>=1).")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "dissonant_eval_100.csv",
        help="Output path (e.g. data/dissonant_eval_100.csv).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Base pattern for y selection.")
    args = ap.parse_args()
    if args.n < 1:
        raise SystemExit("--n must be >= 1")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["text", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "note", "intended"]
        )
        for k in range(args.n):
            y = (k * 3 + (args.seed % 7)) % 7
            j = _j_dissonant(y, k)
            face = [0.0] * 7
            face[y] = 1.0
            text = CONGRUENT_PHRASES[j]
            note = f"synthetic k={k} y={y} text_class={j}"
            w.writerow(
                [text, *[str(face[i]) for i in range(7)], note, FER7_LABELS[y]]
            )

    print("Wrote", args.out, f"({args.n} rows). Score with: python scripts/run_ablation.py {args.out} --brief --out-json docs/eval_output/dissonant_results.json")


if __name__ == "__main__":
    main()
