#!/usr/bin/env python3
"""
Build a 10-row dissonant qualitative table for the report (real FER2013 images +
deliberately mismatched template text), in the same visual format as
``build_qualitative_table.py``.

Selection rule (matches scripts/eval_dissonant_fer2013.py exactly):
  * iterate the AutumnQiu/fer2013 ``test`` split sequentially
  * for the k-th image with gold class y, derive a mismatched class
        j = _j_dissonant(y, k)
    and use ``CONGRUENT_PHRASES[j]`` as the user text U
  * intended affect = the dataset's true class y (the face is real)

For each of the first ``--n-rows`` images we save a PNG, run text + face + fusion
and emit:
  * docs/eval_output/qual_dissonant_examples[_ft].json
  * docs/eval_output/qual_dissonant_table[_ft].tex   (LaTeX fragment)
  * docs/eval_output/qual_dissonant_images[_ft]/*.png

Usage:
  # Baseline face branch
  python scripts/build_dissonant_qualitative_table.py --n-rows 10

  # Fine-tuned face branch
  python scripts/build_dissonant_qualitative_table.py \
      --n-rows 10 \
      --finetune-ckpt checkpoints/fer2013_finetune.pt \
      --suffix _ft \
      --image-subdir qual_dissonant_images_ft
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
from PIL import Image

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
    """Same dissonant-class rule as scripts/eval_dissonant_fer2013.py."""
    j = (y + 1 + (k % 6)) % 7
    if j == y:
        j = (y + 2) % 7
    return j


def _save_thumbnail(image: Image.Image, out_path: Path, target_px: int = 192) -> None:
    img = image.convert("L") if image.mode != "L" else image
    w, h = img.size
    if w < target_px or h < target_px:
        scale = max(1, target_px // max(w, h))
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="test", help="AutumnQiu/fer2013 split (test or train).")
    ap.add_argument("--n-rows", type=int, default=10, help="Number of dissonant rows to build.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "eval_output",
        help="Where qual_dissonant_table.tex and qual_dissonant_examples.json are written.",
    )
    ap.add_argument(
        "--image-subdir",
        type=str,
        default="qual_dissonant_images",
        help="Subdirectory (under out-dir) for PNG thumbnails.",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix appended to qual_dissonant_table/json filenames (e.g. '_ft').",
    )
    ap.add_argument(
        "--finetune-ckpt",
        type=Path,
        default=None,
        help="Optional path to a fine-tuned ResNet50 state_dict.",
    )
    ap.add_argument("--device", type=str, default=None, help="Force device: cuda, mps, cpu.")
    args = ap.parse_args()

    dev = args.device or _pick_device()
    print("Device:", dev or "default (per model)")

    print("Loading text + vision models\u2026")
    text_m = load_text_model(device=dev)
    if args.finetune_ckpt is not None:
        from bimodal_empathy.fer_finetune_ckpt import load_vision_finetuned

        vis_m = load_vision_finetuned(args.finetune_ckpt, device=dev)
        vis_label = f"fine-tuned ({args.finetune_ckpt.name})"
    else:
        vis_m = load_vision_model(device=dev)
        vis_label = "baseline ElenaRyumina ResNet-50"
    print(f"Vision branch: {vis_label}")

    img_dir = args.out_dir / args.image_subdir
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Iterating FER2013 {args.split} split for {args.n_rows} dissonant rows\u2026")
    rows = []
    correct = {"a1": 0, "a05": 0, "a0": 0}
    it = iter_fer2013_test(limit=args.n_rows, split=args.split, streaming=True)
    for k, (image, y) in enumerate(it):
        j = _j_dissonant(y, k)
        gold_label = FER7_LABELS[y]
        text_label = FER7_LABELS[j]
        u_text = CONGRUENT_PHRASES[j]

        png_path = img_dir / f"{k:02d}_{gold_label.lower()}_vs_{text_label.lower()}.png"
        _save_thumbnail(image, png_path)

        p_text, _, _ = text_m.predict_fer7(u_text)
        p_face, _, _ = vis_m.predict_fer7(image)
        pred_text = FER7_LABELS[int(np.argmax(p_text))]
        pred_face = FER7_LABELS[int(np.argmax(p_face))]
        _, _, pred_fused = fuse(p_text, p_face, alpha=0.5)

        m1 = pred_text == gold_label
        m05 = pred_fused == gold_label
        m0 = pred_face == gold_label
        if m1:
            correct["a1"] += 1
        if m05:
            correct["a05"] += 1
        if m0:
            correct["a0"] += 1

        rows.append(
            {
                "k": k,
                "y": int(y),
                "gold": gold_label,
                "text_class_j": int(j),
                "text_class_label": text_label,
                "image_relpath": str(png_path.relative_to(args.out_dir.parent)).replace(os.sep, "/"),
                "image_for_latex": f"{args.image_subdir}/{png_path.name}",
                "text": u_text,
                "pred_alpha_1.0_text": pred_text,
                "pred_alpha_0.5_fused": pred_fused,
                "pred_alpha_0.0_face": pred_face,
                "p_text_top": float(np.max(p_text)),
                "p_face_top": float(np.max(p_face)),
                "match_alpha_1.0_text": bool(m1),
                "match_alpha_0.5_fused": bool(m05),
                "match_alpha_0.0_face": bool(m0),
            }
        )
        print(
            f"  k={k:2d} gold={gold_label:8s} text={text_label:8s} | "
            f"a=1: {pred_text:8s} ({'\u2713' if m1 else 'x'}) | "
            f"a=0.5: {pred_fused:8s} ({'\u2713' if m05 else 'x'}) | "
            f"a=0: {pred_face:8s} ({'\u2713' if m0 else 'x'})"
        )

    n = len(rows)
    if n == 0:
        raise SystemExit("No dissonant rows produced (check --split / dataset access).")

    summary = {
        "vision_branch": vis_label,
        "split": args.split,
        "n": n,
        "by_alpha_local": {
            "1.0_text": correct["a1"] / n,
            "0.5_fused": correct["a05"] / n,
            "0.0_face": correct["a0"] / n,
        },
        "rows": rows,
    }
    json_path = args.out_dir / f"qual_dissonant_examples{args.suffix}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", json_path)

    tex_lines: list[str] = []
    tex_lines.append("\\begin{table*}[t]")
    tex_lines.append("  \\small")
    tex_lines.append("  \\centering")
    tex_lines.append(
        "  \\caption{Dissonant qualitative cases. Real FER2013 test image (gold class) paired with "
        "the congruent template text of a deliberately mismatched FER-7 class. We list the predicted "
        "FER-7 class at $\\alpha\\!=\\!1$ (text-only, expected to match the text class), "
        "$\\alpha\\!=\\!0.5$ (fused), and $\\alpha\\!=\\!0$ (face-only, expected to match the gold "
        f"class) using the {_latex_escape(vis_label)} face branch. "
        "A check mark indicates the prediction matches the \\emph{gold} face class. "
        f"Local hit rates over these {n} rows: "
        f"$\\alpha\\!=\\!1$ {correct['a1']}/{n}, "
        f"$\\alpha\\!=\\!0.5$ {correct['a05']}/{n}, "
        f"$\\alpha\\!=\\!0$ {correct['a0']}/{n}.}}"
    )
    label_suffix = args.suffix.replace("_", "")
    tab_label = "tab:qualdissimages" + (label_suffix or "")
    tex_lines.append(f"  \\label{{{tab_label}}}")
    tex_lines.append("  \\setlength{\\tabcolsep}{4pt}")
    tex_lines.append("  \\renewcommand{\\arraystretch}{1.15}")
    tex_lines.append(
        "  \\begin{tabular}{@{}c l l p{0.40\\textwidth} c c c@{}}"
    )
    tex_lines.append("    \\toprule")
    tex_lines.append(
        "    \\textbf{Image} & \\textbf{Gold} & \\textbf{Text class} & \\textbf{Mismatched user text $U$} & "
        "\\textbf{$\\alpha\\!=\\!1$} & \\textbf{$\\alpha\\!=\\!0.5$} & \\textbf{$\\alpha\\!=\\!0$} \\\\"
    )
    tex_lines.append("    \\midrule")
    for row in rows:
        img = row["image_for_latex"]
        gold = row["gold"]
        tcls = row["text_class_label"]
        u_full = _latex_escape(row["text"])

        def _fmt(label: str, ok: bool) -> str:
            tick = "\\checkmark" if ok else ""
            return f"{label}{(' ' + tick) if tick else ''}"

        a1 = _fmt(row["pred_alpha_1.0_text"], row["match_alpha_1.0_text"])
        a5 = _fmt(row["pred_alpha_0.5_fused"], row["match_alpha_0.5_fused"])
        a0 = _fmt(row["pred_alpha_0.0_face"], row["match_alpha_0.0_face"])
        tex_lines.append(
            f"    \\includegraphics[width=1.4cm]{{{img}}} & {gold} & {tcls} & {u_full} & {a1} & {a5} & {a0} \\\\"
        )
    tex_lines.append("    \\bottomrule")
    tex_lines.append("  \\end{tabular}")
    tex_lines.append("\\end{table*}")

    tex_path = args.out_dir / f"qual_dissonant_table{args.suffix}.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print("Wrote", tex_path)
    print(
        "Reminder: when uploading to Overleaf, also upload the PNGs in "
        f"docs/eval_output/{args.image_subdir}/ (referenced by the table)."
    )


if __name__ == "__main__":
    main()
