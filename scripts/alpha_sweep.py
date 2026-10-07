#!/usr/bin/env python3
"""
α-sensitivity sweep on the FER2013 congruent benchmark, plus McNemar tests.

Branch probabilities are computed once per image (7 template texts, one vision
pass per image); the α grid is then evaluated arithmetically, so the sweep is
as cheap as a single benchmark run.

  python scripts/alpha_sweep.py --limit 3589 --vision finetuned
  python scripts/alpha_sweep.py --limit 200 --vision baseline

Outputs under docs/eval_output/ (suffix _ft for fine-tuned vision):
  alpha_sweep{suffix}.json   per-α accuracies, McNemar stats, confusion matrix at α=0.5
  alpha_sweep{suffix}.csv    alpha,acc rows (pgfplots-friendly)
  alpha_sweep{suffix}.tex    LaTeX table fragment
  alpha_sweep{suffix}.png    accuracy-vs-α curve
"""
from __future__ import annotations

import argparse
import csv
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from tqdm import tqdm

from bimodal_empathy.eval.stats import confusion, macro_f1

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.eval.benchmarks import CONGRUENT_PHRASES, _iter_fer2013_images
from bimodal_empathy.eval.stats import confusion, mcnemar, wilson_ci
from bimodal_empathy.fer_finetune_ckpt import load_vision_finetuned
from bimodal_empathy.text_sensor import load_text_model
from bimodal_empathy.vision_sensor import load_vision_model


def _pfmt(p: float) -> str:
    """LaTeX-friendly p-value: p<10^{-k} for tiny values, p=0.xxx otherwise."""
    import math

    if p <= 0:
        return "p<10^{-300}"
    if p < 1e-3:
        return f"p<10^{{{math.floor(math.log10(p)) + 1}}}"
    return f"p={p:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3589)
    ap.add_argument("--vision", choices=("baseline", "finetuned"), default="finetuned")
    ap.add_argument("--ckpt", default=os.getenv("FINETUNE_CKPT", "checkpoints/fer2013_finetune.pt"))
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default="docs/eval_output")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_ft" if args.vision == "finetuned" else ""

    tm = load_text_model()
    vm = load_vision_finetuned(args.ckpt) if args.vision == "finetuned" else load_vision_model()

    # 7 template texts -> 7 text distributions, computed once.
    P_TEXT = np.stack([np.asarray(tm.predict_fer7(s)[0], dtype=np.float64).ravel() for s in CONGRUENT_PHRASES])

    gold: list[int] = []
    P_FACE: list[np.ndarray] = []
    for image, y in tqdm(_iter_fer2013_images(args.limit, "test"), total=args.limit, desc="vision"):
        p_face, _, _ = vm.predict_fer7(image)
        P_FACE.append(np.asarray(p_face, dtype=np.float64).ravel())
        gold.append(int(y))
    Y = np.asarray(gold)
    PF = np.stack(P_FACE)
    PT = P_TEXT[Y]  # text distribution for each item's congruent phrase
    n = len(Y)

    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, args.step), 4)
    sweep = []
    preds_by_alpha: dict[float, np.ndarray] = {}
    for a in alphas:
        P = a * PT + (1 - a) * PF
        pred = P.argmax(axis=1)
        preds_by_alpha[float(a)] = pred
        k = int((pred == Y).sum())
        lo, hi = wilson_ci(k, n)
        sweep.append({"alpha": float(a), "acc": k / n, "correct": k, "ci95": [lo, hi],
                      "macro_f1": macro_f1(confusion(Y, pred, 7))})

    def _pred(a: float) -> np.ndarray:
        return preds_by_alpha[float(np.round(a, 4))]

    c_text = _pred(1.0) == Y
    c_face = _pred(0.0) == Y
    c_fuse = _pred(0.5) == Y
    stats = {
        "fused05_vs_text": mcnemar(c_fuse, c_text),
        "fused05_vs_face": mcnemar(c_fuse, c_face),
        "text_vs_face": mcnemar(c_text, c_face),
    }
    best = max(sweep, key=lambda r: r["acc"])
    conf = confusion(Y, _pred(0.5), 7)
    majority = int(np.bincount(Y, minlength=7).max())

    result = {
        "n": n,
        "vision": args.vision,
        "sweep": sweep,
        "best_alpha": best["alpha"],
        "best_acc": best["acc"],
        "majority_class_floor": majority / n,
        "mcnemar": stats,
        "confusion_alpha_0.5": {"labels": list(FER7_LABELS), "matrix": conf.tolist()},
        "confusion_by_alpha": {str(a): confusion(Y, _pred(a), 7).tolist() for a in (0.0, 0.5, 1.0)},
        "macro_f1_by_alpha": {str(a): macro_f1(confusion(Y, _pred(a), 7)) for a in (0.0, 0.5, 1.0)},
    }
    (out_dir / f"alpha_sweep{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / f"alpha_sweep{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "acc", "ci_lo", "ci_hi"])
        for r in sweep:
            w.writerow([r["alpha"], f"{r['acc']:.4f}", f"{r['ci95'][0]:.4f}", f"{r['ci95'][1]:.4f}"])

    rows = "\n".join(
        f"    {r['alpha']:.2f} & {100*r['acc']:.1f} & {r['macro_f1']:.3f} & [{100*r['ci95'][0]:.1f}, {100*r['ci95'][1]:.1f}] \\\\"
        for r in sweep if abs((r["alpha"] * 10) - round(r["alpha"] * 10)) < 1e-6  # every 0.1
    )
    m = stats["fused05_vs_text"]
    tex = f"""% Auto-generated: scripts/alpha_sweep.py --vision {args.vision} --limit {args.limit}
\\begin{{table}}[t]
  \\centering
  \\caption{{$\\alpha$ sensitivity on the FER2013 congruent benchmark ({'fine-tuned' if args.vision == 'finetuned' else 'baseline'} vision branch, $n={n}$). Top-1 accuracy, macro-F1, Wilson 95\\% CI. Majority-class floor {100*majority/n:.1f}\\%. McNemar $\\alpha{{=}}0.5$ vs.\\ $\\alpha{{=}}1$: $b={m['b']}, c={m['c']}, \\chi^2={m['chi2']:.2f}, {_pfmt(m['p'])}$.}}
  \\label{{tab:alpha_sweep{suffix}}}
  \\small
  \\begin{{tabular}}{{@{{}}c r r l@{{}}}}
    \\toprule
    $\\alpha$ & Top-1 (\\%) & Macro-F1 & 95\\% CI \\\\
    \\midrule
{rows}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    (out_dir / f"alpha_sweep{suffix}.tex").write_text(tex, encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["alpha"] for r in sweep]
        ys = [100 * r["acc"] for r in sweep]
        lo = [100 * r["ci95"][0] for r in sweep]
        hi = [100 * r["ci95"][1] for r in sweep]
        fig, ax = plt.subplots(figsize=(5, 3.2), dpi=150)
        ax.fill_between(xs, lo, hi, alpha=0.2)
        ax.plot(xs, ys, marker="o", ms=3)
        ax.axhline(100 * majority / n, ls="--", lw=1, color="gray", label="majority class")
        ax.set_xlabel("α  (1 = text only, 0 = face only)")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_title(f"FER2013 congruent, {args.vision} vision, n={n}")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"alpha_sweep{suffix}.png")
    except ImportError:
        print("matplotlib not installed; skipped PNG", file=sys.stderr)

    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))
    for r in sweep:
        print(f"alpha={r['alpha']:.2f}  acc={100*r['acc']:.2f}%")


if __name__ == "__main__":
    main()
