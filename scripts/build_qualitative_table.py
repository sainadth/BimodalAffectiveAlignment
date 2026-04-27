#!/usr/bin/env python3
"""
Build a per-FER7-label qualitative table for the report.

For each FER-7 class y in {Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral}:
  - Pick the first real FER2013 test image whose label is y.
  - Use CONGRUENT_PHRASES[y] as the user text U (emotion-matched template).
  - Run the text branch (RoBERTa/GoEmotions) and the face branch (ResNet-50).
  - Report predictions at alpha = 1.0 (text-only), 0.5 (fused), 0.0 (face-only).
  - Save the image as a PNG under docs/eval_output/qual_images/.
  - Emit:
      * docs/eval_output/qual_examples.json -- per-row machine-readable record.
      * docs/eval_output/qual_table.tex     -- LaTeX table with \\includegraphics rows.

The script does *not* modify the default app path or any production model file.

Usage:
    python scripts/build_qualitative_table.py --split test
    python scripts/build_qualitative_table.py --finetune-ckpt checkpoints/fer2013_finetuned.pt
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
from bimodal_empathy.response_synthesizer import load_synthesizer
from bimodal_empathy.text_sensor import load_text_model
from bimodal_empathy.vision_sensor import load_vision_model


def _pick_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return None


def _save_thumbnail(image: Image.Image, out_path: Path, target_px: int = 192) -> None:
    """Upscale the 48x48 FER2013 frame to a printable PNG with nearest-neighbor."""
    img = image.convert("L") if image.mode != "L" else image
    w, h = img.size
    if w < target_px or h < target_px:
        scale = max(1, target_px // max(w, h))
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def _shorten(text: str, n: int = 60) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "\u2026"


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
    ap.add_argument(
        "--scan-limit",
        type=int,
        default=2000,
        help="Max images to scan looking for one example of every FER-7 class.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "eval_output",
        help="Where qual_table.tex and qual_examples.json are written.",
    )
    ap.add_argument(
        "--image-subdir",
        type=str,
        default="qual_images",
        help="Subdirectory (under out-dir) for PNG thumbnails.",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix appended to qual_examples/qual_table filenames (e.g. '_ft' for fine-tuned).",
    )
    ap.add_argument(
        "--table-label",
        type=str,
        default="tab:qualimages",
        help="LaTeX \\label{} for the resulting qualitative table.",
    )
    ap.add_argument(
        "--responses-table-label",
        type=str,
        default="tab:qualresponses",
        help="LaTeX \\label{} for the resulting responses table.",
    )
    ap.add_argument(
        "--finetune-ckpt",
        type=Path,
        default=None,
        help="Optional path to a fine-tuned ResNet50 state_dict (uses fer_finetune_ckpt loader).",
    )
    ap.add_argument("--device", type=str, default=None, help="Force device: cuda, mps, cpu.")
    ap.add_argument(
        "--no-response",
        action="store_true",
        help="Skip FLAN-T5 empathetic response generation per row.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260426,
        help="Torch seed used before each FLAN-T5 generation for reproducibility.",
    )
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

    syn = None
    if not args.no_response:
        print("Loading FLAN-T5 response synthesizer\u2026")
        syn = load_synthesizer(device=dev)

    img_dir = args.out_dir / args.image_subdir
    img_dir.mkdir(parents=True, exist_ok=True)

    needed: dict[int, dict] = {}
    print(f"Scanning FER2013 {args.split} split for one image per FER-7 class\u2026")
    for k, (image, y) in enumerate(iter_fer2013_test(limit=args.scan_limit, split=args.split, streaming=True)):
        if y in needed:
            continue
        needed[y] = {"k": k, "image": image, "label": FER7_LABELS[y]}
        if len(needed) == 7:
            break

    missing = [FER7_LABELS[i] for i in range(7) if i not in needed]
    if missing:
        raise SystemExit(
            f"Could not find an example for FER-7 classes: {missing} within --scan-limit={args.scan_limit}. "
            "Increase --scan-limit and rerun."
        )

    print("Running text + face + fusion for each example\u2026")
    rows = []
    for y in range(7):
        rec = needed[y]
        image: Image.Image = rec["image"]
        label = rec["label"]
        u_text = CONGRUENT_PHRASES[y]

        png_path = img_dir / f"{y}_{label.lower()}.png"
        _save_thumbnail(image, png_path)

        p_text, _, _ = text_m.predict_fer7(u_text)
        p_face, _, _ = vis_m.predict_fer7(image)
        pred_text = FER7_LABELS[int(np.argmax(p_text))]
        pred_face = FER7_LABELS[int(np.argmax(p_face))]
        p_fused, _, pred_fused = fuse(p_text, p_face, alpha=0.5)

        response_R = None
        if syn is not None:
            torch.manual_seed(args.seed + y)
            response_R = syn.generate(
                u_text,
                pred_fused,
                p_text=p_text,
                p_face=p_face,
                p_fused=p_fused,
            )

        rows.append(
            {
                "y": y,
                "gold": label,
                "image_relpath": str(png_path.relative_to(args.out_dir.parent)).replace(
                    os.sep, "/"
                ),
                "image_for_latex": f"{args.image_subdir}/{png_path.name}",
                "text": u_text,
                "pred_alpha_1.0_text": pred_text,
                "pred_alpha_0.5_fused": pred_fused,
                "pred_alpha_0.0_face": pred_face,
                "p_text_top": float(np.max(p_text)),
                "p_face_top": float(np.max(p_face)),
                "response_R": response_R,
            }
        )
        r_short = (response_R or "")[:60].replace("\n", " ")
        print(
            f"  y={y} {label:8s} | a=1: {pred_text:8s} | a=0.5: {pred_fused:8s} | a=0: {pred_face:8s}"
            + (f" | R: {r_short}\u2026" if response_R else "")
        )

    json_path = args.out_dir / f"qual_examples{args.suffix}.json"
    json_path.write_text(
        json.dumps(
            {
                "vision_branch": vis_label,
                "split": args.split,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote", json_path)

    tex_lines = []
    tex_lines.append("\\begin{table*}[t]")
    tex_lines.append("  \\small")
    tex_lines.append("  \\centering")
    tex_lines.append(
        "  \\caption{Per-class qualitative cases. One real FER2013 test image per FER-7 gold class, "
        "paired with the emotion-matched congruent template as text. We list the predicted FER-7 class "
        "at $\\alpha\\!=\\!1$ (text-only), $\\alpha\\!=\\!0.5$ (fused), and $\\alpha\\!=\\!0$ (face-only) "
        f"using the {_latex_escape(vis_label)} face branch.}}"
    )
    tex_lines.append(f"  \\label{{{args.table_label}}}")
    tex_lines.append("  \\setlength{\\tabcolsep}{4pt}")
    tex_lines.append("  \\renewcommand{\\arraystretch}{1.15}")
    tex_lines.append(
        "  \\begin{tabular}{@{}c l p{0.46\\textwidth} c c c@{}}"
    )
    tex_lines.append("    \\toprule")
    tex_lines.append(
        "    \\textbf{Image} & \\textbf{Gold} & \\textbf{User text $U$} & "
        "\\textbf{$\\alpha\\!=\\!1$} & \\textbf{$\\alpha\\!=\\!0.5$} & \\textbf{$\\alpha\\!=\\!0$} \\\\"
    )
    tex_lines.append("    \\midrule")
    for row in rows:
        img = row["image_for_latex"]
        gold = row["gold"]
        u_full = _latex_escape(row["text"])
        a1 = row["pred_alpha_1.0_text"]
        a5 = row["pred_alpha_0.5_fused"]
        a0 = row["pred_alpha_0.0_face"]
        tex_lines.append(
            f"    \\includegraphics[width=1.4cm]{{{img}}} & {gold} & {u_full} & {a1} & {a5} & {a0} \\\\"
        )
    tex_lines.append("    \\bottomrule")
    tex_lines.append("  \\end{tabular}")
    tex_lines.append("\\end{table*}")

    if any(r.get("response_R") for r in rows):
        tex_lines.append("")
        tex_lines.append("% --- FLAN-T5 empathetic responses (one per row) ---")
        tex_lines.append("\\begin{table*}[t]")
        tex_lines.append("  \\small")
        tex_lines.append("  \\centering")
        tex_lines.append(
            "  \\caption{FLAN-T5 (small) empathetic responses for the per-class qualitative cases. "
            "The fused FER-7 class drives the prompt; sampling is seeded for reproducibility.}"
        )
        tex_lines.append(f"  \\label{{{args.responses_table_label}}}")
        tex_lines.append("  \\setlength{\\tabcolsep}{4pt}")
        tex_lines.append("  \\renewcommand{\\arraystretch}{1.2}")
        tex_lines.append("  \\begin{tabular}{@{}l l p{0.72\\textwidth}@{}}")
        tex_lines.append("    \\toprule")
        tex_lines.append(
            "    \\textbf{Gold} & \\textbf{Fused $b^{*}$} & \\textbf{Generated empathetic reply $R$} \\\\"
        )
        tex_lines.append("    \\midrule")
        for row in rows:
            r = row.get("response_R") or ""
            tex_lines.append(
                f"    {row['gold']} & {row['pred_alpha_0.5_fused']} & {_latex_escape(r)} \\\\"
            )
        tex_lines.append("    \\bottomrule")
        tex_lines.append("  \\end{tabular}")
        tex_lines.append("\\end{table*}")

    tex_path = args.out_dir / f"qual_table{args.suffix}.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print("Wrote", tex_path)
    print(
        "Reminder: when uploading to Overleaf, also upload the PNGs in "
        f"docs/{args.image_subdir}/ (referenced by the table)."
    )


if __name__ == "__main__":
    main()
