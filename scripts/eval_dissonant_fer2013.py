#!/usr/bin/env python3
"""
Dissonant evaluation on *real* FER2013 faces (HuggingFace AutumnQiu/fer2013): run ResNet on each
image to get P_face, pair with a *mismatched* CONGRUENT template as text, compare predictions at
α = 0, 0.5, 1 to the dataset label (intended = true FER2013 class).

This replaces the synthetic CSV setup (one-hot P_face) with actual facial images and CNN output.

  python scripts/eval_dissonant_fer2013.py --limit 100 --out-json docs/eval_output/dissonant_fer2013_results.json
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

import numpy as np
import torch
from tqdm import tqdm

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.eval.benchmarks import CONGRUENT_PHRASES
from bimodal_empathy.eval.fer2013_test_iter import iter_fer2013_test
from bimodal_empathy.fusion import fuse
from bimodal_empathy.text_sensor import load_text_model
from bimodal_empathy.vision_sensor import load_vision_model


def _pick_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return None


def _j_dissonant(y: int, k: int) -> int:
    j = (y + 1 + (k % 6)) % 7
    if j == y:
        j = (y + 2) % 7
    return j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="Number of FER2013 test images to use.")
    ap.add_argument("--split", type=str, default="test", help="AutumnQiu/fer2013 split (e.g. test, train).")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "docs" / "eval_output" / "dissonant_fer2013_results.json",
    )
    ap.add_argument("--no-details", action="store_true", help="Omit per-example rows in JSON (smaller file).")
    ap.add_argument(
        "--finetune-ckpt",
        type=Path,
        default=None,
        help="Optional path to a fine-tuned ResNet50 state_dict (uses fer_finetune_ckpt loader).",
    )
    ap.add_argument("--device", type=str, default=None, help="Force device: cuda, mps, cpu.")
    args = ap.parse_args()

    dev = args.device or _pick_device()
    print("Device:", dev or "default (per model)")

    print("Loading text + vision models…")
    text_m = load_text_model(device=dev)
    if args.finetune_ckpt is not None:
        from bimodal_empathy.fer_finetune_ckpt import load_vision_finetuned

        vis_m = load_vision_finetuned(args.finetune_ckpt, device=dev)
        vis_label = f"fine-tuned ({args.finetune_ckpt.name})"
    else:
        vis_m = load_vision_model(device=dev)
        vis_label = "baseline ElenaRyumina ResNet-50"
    print(f"Vision branch: {vis_label}")

    gold_name = FER7_LABELS  # y index -> name

    correct = {"a0": 0, "a05": 0, "a1": 0}
    rows: list[dict] = []
    n = 0
    it = iter_fer2013_test(limit=args.limit, split=args.split, streaming=True)
    for k, (image, y) in enumerate(tqdm(it, total=args.limit, desc="FER2013 dissonant")):
        j = _j_dissonant(y, k)
        phrase = CONGRUENT_PHRASES[j]
        p_text, _, _ = text_m.predict_fer7(phrase)
        p_face, _, _ = vis_m.predict_fer7(image)
        intended = gold_name[y]

        p1 = FER7_LABELS[int(np.argmax(p_text))]
        p0 = FER7_LABELS[int(np.argmax(p_face))]
        p5 = fuse(p_text, p_face, alpha=0.5)[2]

        for key, pred, akey in (("a1", p1, "1.0"), ("a05", p5, "0.5"), ("a0", p0, "0.0")):
            if pred == intended:
                correct[key] += 1
        n += 1
        if not args.no_details:
            rows.append(
                {
                    "k": k,
                    "y_dataset": y,
                    "intended": intended,
                    "text_class_j": j,
                    "pred_a1_text": p1,
                    "pred_a0_face": p0,
                    "pred_a0.5_fused": p5,
                    "match_1": p1 == intended,
                    "match_0": p0 == intended,
                    "match_0.5": p5 == intended,
                }
            )

    if n == 0:
        raise SystemExit("No examples (check --split / dataset access).")

    by_alpha = {
        "1.0_text": correct["a1"] / n,
        "0.5_fused": correct["a05"] / n,
        "0.0_face": correct["a0"] / n,
    }
    print()
    print(f"Dissonant eval on real images: n={n} | split={args.split}")
    print(f"  α=1 (text / mismatched template):     {correct['a1']}/{n} = {by_alpha['1.0_text']:.3f}")
    print(f"  α=0.5 (fused):                        {correct['a05']}/{n} = {by_alpha['0.5_fused']:.3f}")
    print(f"  α=0 (face / ResNet on image):        {correct['a0']}/{n} = {by_alpha['0.0_face']:.3f}")
    print("  Gold = FER2013 label; text is always mismatched (j != y) on purpose.")
    out = {
        "n": n,
        "split": args.split,
        "limit_requested": args.limit,
        "vision_branch": vis_label,
        "rubric": "P_face from ResNet on real image; P_text from mismatched CONGRUENT template; gold = dataset y.",
        "by_alpha": by_alpha,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
