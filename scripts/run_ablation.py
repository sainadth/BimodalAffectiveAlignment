#!/usr/bin/env python3
"""
Ablation: text-only (α=1), vision-only (α=0), fused (α=0.5), plus α sensitivity 0.2–0.8.
Uses a CSV with columns: text, f0..f6 (face distribution over FER7), optional note,
optional `intended` (FER-7 label = human "ground truth" for pilot dissonant set).

If `intended` is present, reports accuracy for α in {0, 0.5, 1.0} and optional --out-json.

  python scripts/run_ablation.py data/dissonant_samples.csv
  python scripts/run_ablation.py data/dissonant_eval_100.csv --brief --out-json docs/eval_output/dissonant_results.json

  Build 100+ synthetic dissonant rows (face one-hot + mismatched template text):

  python scripts/build_dissonant_eval_csv.py --n 100 --out data/dissonant_eval_100.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass
if not (os.environ.get("HF_TOKEN") or "").strip() and not (os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
    mis = (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if mis:
        os.environ["HF_TOKEN"] = mis
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import csv
from typing import Any

import numpy as np
from tqdm import tqdm

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.fusion import fuse
from bimodal_empathy.text_sensor import load_text_model


def _row_face(row: dict) -> np.ndarray:
    p = np.array([float(row[f"f{i}"]) for i in range(7)], dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return np.ones(7) / 7.0
    return p / s


def _normalize_gold(s: str | None) -> str | None:
    if not s or not (s := str(s).strip()):
        return None
    sl = s.lower()
    for lab in FER7_LABELS:
        if sl == lab.lower():
            return lab
    return None


def _pred_a1(p_text: np.ndarray) -> str:
    return FER7_LABELS[int(np.argmax(p_text))]


def _pred_a0(p_face: np.ndarray) -> str:
    return FER7_LABELS[int(np.argmax(p_face))]


def _pred_a05(p_text: np.ndarray, p_face: np.ndarray) -> str:
    return fuse(p_text, p_face, alpha=0.5)[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "data" / "dissonant_samples.csv",
    )
    ap.add_argument("--no-text-model", action="store_true", help="Skip HF text; use uniform P_text")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="If CSV has 'intended' column, write scored summary here (e.g. docs/eval_output/dissonant_results.json).",
    )
    ap.add_argument("--no-score-print", action="store_true", help="Skip accuracy summary (still writes JSON if set).")
    ap.add_argument(
        "--brief",
        action="store_true",
        help="Do not print per-row fusion details (use for large CSVs, e.g. 100+ rows).",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Missing CSV: {args.csv}")

    tm = None
    if not args.no_text_model:
        print("Loading GoEmotions model (first time may download weights)…")
        tm = load_text_model()

    score_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {"n": 0, "a0": 0, "a05": 0, "a1": 0}
    has_intended = False

    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [x.strip() for x in (reader.fieldnames or []) if x]
        has_intended = "intended" in fieldnames

        rows_list = list(reader)
        it = rows_list
        if args.brief:
            it = tqdm(rows_list, desc="dissonant rows", unit="row")

        for row in it:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            note = (row.get("note") or "").strip()
            p_face = _row_face(row)
            if args.no_text_model or tm is None:
                p_text = np.ones(7) / 7.0
            else:
                p_text, _, _ = tm.predict_fer7(text)
            if not args.brief:
                print("-" * 60)
                print("U:", text[:200] + ("…" if len(text) > 200 else ""))
                if note:
                    print("note:", note)
                print("argmax P_text:", FER7_LABELS[int(np.argmax(p_text))])
                print("argmax P_face:", FER7_LABELS[int(np.argmax(p_face))])
                for a in (0.0, 0.5, 1.0):
                    _, _, lab = fuse(p_text, p_face, alpha=a)
                    print(f"  α={a:.1f} -> b* = {lab}")
                print("  sensitivity α in [0.2,0.8] step 0.1:")
                for a in [round(x, 1) for x in np.arange(0.2, 0.81, 0.1)]:
                    _, _, lab = fuse(p_text, p_face, alpha=a)
                    print(f"    {a} -> {lab}")

            if has_intended and not args.no_text_model and tm is not None:
                gold = _normalize_gold(row.get("intended"))
                p1 = _pred_a1(p_text)
                p0 = _pred_a0(p_face)
                p5 = _pred_a05(p_text, p_face)
                r: dict[str, Any] = {
                    "text": text,
                    "intended": gold,
                    "pred_alpha_1_text": p1,
                    "pred_alpha_0_face": p0,
                    "pred_alpha_0.5_fused": p5,
                }
                if gold is not None:
                    counts["n"] += 1
                    for key, pred, akey in (("a1", p1, "α=1 text"), ("a0", p0, "α=0 face"), ("a05", p5, "α=0.5")):
                        if pred == gold:
                            counts[key] += 1
                    r["match_1"] = p1 == gold
                    r["match_0"] = p0 == gold
                    r["match_0.5"] = p5 == gold
                else:
                    r["match_1"] = r["match_0"] = r["match_0.5"] = None
                score_rows.append(r)

    if has_intended and not args.no_text_model and counts["n"] > 0 and not args.no_score_print:
        print("=" * 60)
        nlabel = "n=" + str(counts["n"])
        if counts["n"] >= 30:
            nlabel += " (report: synthetic dissonant set; intended = face one-hot label)"
        else:
            nlabel += " (pilot; justify `intended` in the report)"
        print("Accuracy vs `intended` (" + nlabel + "):")
        n = counts["n"]
        for key, desc in (("a1", "α=1 (text only)"), ("a05", "α=0.5 (fused)"), ("a0", "α=0 (face only)")):
            c = counts[key]
            print(f"  {desc}: {c}/{n} = {c / n:.3f}")
        print(
            "  Note: `intended` is a design choice (which modality encodes the user's true state for each row)."
        )

    if args.out_json and has_intended and not args.no_text_model and score_rows:
        out: dict[str, Any] = {
            "csv": str(args.csv),
            "n_scored": counts["n"],
            "rubric": "intended = FER-7 label; hand rows = human design; build_dissonant_eval_csv = face one-hot as physiology GT.",
            "by_alpha": {
                "1.0_text": (counts["a1"] / counts["n"]) if counts["n"] else None,
                "0.5_fused": (counts["a05"] / counts["n"]) if counts["n"] else None,
                "0.0_face": (counts["a0"] / counts["n"]) if counts["n"] else None,
            },
            "rows": score_rows,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
