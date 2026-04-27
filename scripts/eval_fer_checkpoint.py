#!/usr/bin/env python3
"""
Image-only FER7 top-1 on AutumnQiu/fer2013 test: baseline (HF) vs. fine-tuned checkpoint.
Does not change `run_evaluation.py` or `benchmarks.py`.

  python scripts/eval_fer_checkpoint.py --finetune checkpoints/fer2013_finetune.pt --limit 500
  python scripts/eval_fer_checkpoint.py --finetune checkpoints/fer2013_finetune.pt

  Set HF_TOKEN in .env (see run_evaluation.py) to avoid unauthenticated Hub warnings.
"""
from __future__ import annotations

import argparse
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

from bimodal_empathy.eval.fer2013_test_iter import iter_fer2013_test
from bimodal_empathy.fer_finetune_ckpt import load_vision_finetuned
from bimodal_empathy.vision_sensor import load_vision_model


def _pick_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _acc(model, name: str, limit: int | None) -> float:
    correct = 0
    n = 0
    for image, y in tqdm(iter_fer2013_test(limit, split="test", streaming=True), desc=name, total=limit or None):
        p_face, _, _ = model.predict_fer7(image)
        pred = int(np.argmax(p_face))
        n += 1
        if pred == y:
            correct += 1
    return (correct / n) if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--finetune",
        type=Path,
        required=True,
        help="Path to scripts/finetune_fer2013.py output (e.g. checkpoints/fer2013_finetune.pt).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap test examples (default: full test split).")
    ap.add_argument("--device", type=str, default=None, help="e.g. cuda, mps, cpu (default: auto).")
    args = ap.parse_args()

    if not args.finetune.is_file():
        raise SystemExit(f"Not found: {args.finetune}")

    device = args.device or _pick_device_str()
    print("Device:", device)

    # Baseline: existing Elena Ryumina checkpoint
    print("Loading baseline (HF) vision model…")
    base = load_vision_model(device=device)
    b_acc = _acc(base, "baseline", args.limit)
    print(f"Baseline top-1: {b_acc:.6f}")

    print("Loading fine-tuned checkpoint…")
    ft = load_vision_finetuned(args.finetune, device=device)
    f_acc = _acc(ft, "finetuned", args.limit)
    print(f"Finetuned top-1: {f_acc:.6f}")


if __name__ == "__main__":
    main()
