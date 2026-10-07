#!/usr/bin/env python3
"""Build the LaTeX table for the bimodal MELD evaluation from eval_meld.py JSON.

  python scripts/build_meld_bimodal_table.py \
      --ft docs/eval_output/meld_results_bimodal_ft.json \
      --base docs/eval_output/meld_results_bimodal_base.json \
      --out docs/eval_output/meld_bimodal_table.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _row(label: str, r: dict | None) -> str:
    if not r:
        return f"    {label} & -- & -- & -- \\\\"
    lo, hi = r["ci95"]
    return f"    {label} & {100*r['top1']:.1f} & {r['macro_f1']:.3f} & [{100*lo:.1f}, {100*hi:.1f}] \\\\"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", default="docs/eval_output/meld_results_bimodal_ft.json")
    ap.add_argument("--base", default="docs/eval_output/meld_results_bimodal_base.json")
    ap.add_argument("--out", default="docs/eval_output/meld_bimodal_table.tex")
    args = ap.parse_args()

    runs = {}
    for name, p in (("ft", args.ft), ("base", args.base)):
        fp = ROOT / p
        if fp.is_file():
            runs[name] = json.loads(fp.read_text(encoding="utf-8"))
    if not runs:
        raise SystemExit("no MELD bimodal result JSON found")
    any_run = next(iter(runs.values()))
    n_all = any_run["n"]
    majority = any_run["majority_class_floor"]
    bi = any_run["bimodal"]
    n_det = bi["text_on_detected"]["n"] if bi.get("text_on_detected") else 0
    frac = bi["face_detected_frac"]

    lines = [
        "% Auto-generated: scripts/build_meld_bimodal_table.py",
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{MELD test split, utterances with a detected face in the middle frame ($n={n_det}$ of {n_all}, {100*frac:.1f}\\%). Top-1 accuracy, macro-F1, Wilson 95\\% CI. Majority-class floor on the full split: {100*majority:.1f}\\%. Fusion at $\\alpha=0.5$.}}",
        "  \\label{tab:meld_bimodal}",
        "  \\small",
        "  \\begin{tabular}{@{}l r r l@{}}",
        "    \\toprule",
        "    Mode & Top-1 (\\%) & Macro-F1 & 95\\% CI \\\\",
        "    \\midrule",
        _row("Text only", bi.get("text_on_detected")),
    ]
    if "base" in runs:
        b = runs["base"]["bimodal"]
        lines += [_row("Face only (baseline AffectNet)", b.get("face_only_on_detected")),
                  _row("Fused $\\alpha{=}0.5$ (baseline face)", b.get("fused05_on_detected"))]
    if "ft" in runs:
        f = runs["ft"]["bimodal"]
        lines += [_row("Face only (FER2013 fine-tuned)", f.get("face_only_on_detected")),
                  _row("Fused $\\alpha{=}0.5$ (fine-tuned face)", f.get("fused05_on_detected"))]
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    out = ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
