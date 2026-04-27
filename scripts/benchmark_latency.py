#!/usr/bin/env python3
"""
Reproducible pipeline latency (ms) for the report: text + face + fusion + FLAN-T5.
Compare mean end-to-end to the ~400 ms Doherty threshold (discuss in write-up).

  python scripts/benchmark_latency.py
  python scripts/benchmark_latency.py --runs 30 --warmup 5
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
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
from PIL import Image

from bimodal_empathy.config import DEFAULT_ALPHA
from bimodal_empathy.fusion import fuse
from bimodal_empathy.response_synthesizer import load_synthesizer
from bimodal_empathy.text_sensor import load_text_model
from bimodal_empathy.vision_sensor import load_vision_model


def _pick_device_str() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_once(
    utterance: str,
    image: Image.Image,
    alpha: float,
    tm,
    vm,
    syn,
) -> dict[str, float]:
    t0 = time.perf_counter()
    p_text, _, _ = tm.predict_fer7(utterance)
    t1 = time.perf_counter()
    p_face, _, _ = vm.predict_fer7(image)
    t2 = time.perf_counter()
    p_fuse, _i, label_star = fuse(p_text, p_face, alpha=alpha)
    t3 = time.perf_counter()
    _ = syn.generate(utterance, label_star, p_fused=p_fuse)
    t4 = time.perf_counter()
    return {
        "text_ms": (t1 - t0) * 1000.0,
        "face_ms": (t2 - t1) * 1000.0,
        "fusion_ms": (t3 - t2) * 1000.0,
        "t5_ms": (t4 - t3) * 1000.0,
        "e2e_ms": (t4 - t0) * 1000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20, help="Timed iterations (after warmup).")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup runs (not recorded).")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument(
        "--utterance",
        type=str,
        default="I'm doing okay, though today has been a bit much.",
    )
    args = ap.parse_args()

    device = _pick_device_str()
    print("Device:", device)
    print("Loading models (first run may download weights)…")
    tm = load_text_model(device=device)
    vm = load_vision_model(device=device)
    syn = load_synthesizer(device=device)
    # Fixed synthetic face (no webcam); same size as stream pipeline
    image = Image.new("RGB", (224, 224), color=(128, 110, 95))

    for _ in range(args.warmup):
        _run_once(args.utterance, image, args.alpha, tm, vm, syn)

    rows: list[dict[str, float]] = []
    for _ in range(args.runs):
        rows.append(_run_once(args.utterance, image, args.alpha, tm, vm, syn))

    keys = ["text_ms", "face_ms", "fusion_ms", "t5_ms", "e2e_ms"]
    print()
    print(f"Warmup={args.warmup} | Timed runs={args.runs} | utterance length={len(args.utterance)} chars")
    print("-" * 56)
    for k in keys:
        vals = [r[k] for r in rows]
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {k:12s}  mean={m:8.1f} ms   stdev={s:6.2f} ms")
    mean_e2e = statistics.mean([r["e2e_ms"] for r in rows])
    print("-" * 56)
    print(
        f"Mean end-to-end: {mean_e2e:.1f} ms. Doherty threshold often cited ~400 ms for responsiveness; "
        "state your hardware (CPU/MPS/GPU) in the report."
    )


if __name__ == "__main__":
    main()
