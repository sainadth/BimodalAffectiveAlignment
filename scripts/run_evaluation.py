#!/usr/bin/env python3
"""
Run text-only, image-only, and bimodal (fused) top-1 accuracy; write docs/eval_output/results.json and table.tex

Requires: pip install datasets tqdm  (in requirements.txt)

Usage from repo root (optional: cp .env.example .env and set HF_TOKEN):
  python scripts/run_evaluation.py --limit 200
  python scripts/run_evaluation.py --limit 3589 --alpha 0.5 --out docs/eval_output
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src on path if run as script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Optional: load repo-root .env. Use override=True: if the IDE/shell has empty HF_TOKEN, dotenv
# would otherwise be ignored and Hub stays "unauthenticated" (huggingface_hub warning).
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

# huggingface_hub only reads HF_TOKEN or HUGGING_FACE_HUB_TOKEN (not HUGGINGFACE_HUB_TOKEN).
if not (os.environ.get("HF_TOKEN") or "").strip() and not (os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
    mis = (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if mis:
        os.environ["HF_TOKEN"] = mis

# Allow large PIL
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from bimodal_empathy.eval.benchmarks import run_full_benchmark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="Samples per task (text / image / bimodal).")
    ap.add_argument("--alpha", type=float, default=0.5, help="α for bimodal fusion in benchmark.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/eval_output"),
        help="Output directory for results.json and table.tex",
    )
    ap.add_argument(
        "--finetune-ckpt",
        type=Path,
        default=None,
        help="Optional path to a fine-tuned ResNet50 state_dict; bimodal/face metrics use it.",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix appended to results/table filenames (e.g. '_ft' for fine-tuned).",
    )
    ap.add_argument(
        "--table-label",
        type=str,
        default="tab:results",
        help="LaTeX \\label{} for the resulting bimodal results table.",
    )
    args = ap.parse_args()
    os.chdir(ROOT)
    out = (ROOT / args.out).resolve()
    print("Working directory:", ROOT)
    print("Output:", out)

    vision_model = None
    if args.finetune_ckpt is not None:
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        from bimodal_empathy.fer_finetune_ckpt import load_vision_finetuned

        print("Using fine-tuned vision model:", args.finetune_ckpt)
        vision_model = load_vision_finetuned(args.finetune_ckpt)

    run_full_benchmark(
        limit=args.limit,
        alpha=args.alpha,
        out_dir=out,
        vision_model=vision_model,
        table_label=args.table_label,
        file_suffix=args.suffix,
    )
    print(
        "Done. Wrote",
        out / f"results{args.suffix}.json",
        "and",
        out / f"table{args.suffix}.tex",
    )


if __name__ == "__main__":
    main()
