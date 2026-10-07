#!/usr/bin/env python3
"""Ablate the 28->7 label collapse on MELD test (one text-model pass).

Variants: sum (paper default), max, mean (bin-size normalised), and a logistic
regression from the 28 RoBERTa sigmoids to MELD-7 fitted on MELD *train*
(the honest learned map). Reports top-1 and macro-F1 with Wilson CIs.

  python scripts/collapse_ablation.py            # -> docs/eval_output/collapse_ablation.{json,tex}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
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

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.emotion_mapping import COLLAPSE_METHODS, collapse_goemotions_to_fer7
from bimodal_empathy.eval.stats import confusion, macro_f1, wilson_ci
from bimodal_empathy.text_sensor import load_text_model

MELD_BASE = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/"
MELD_TO_FER7 = {"anger": "Angry", "disgust": "Disgust", "fear": "Fear", "joy": "Happy",
                "sadness": "Sad", "surprise": "Surprise", "neutral": "Neutral"}


def load_split(name: str, limit: int | None) -> tuple[list[str], np.ndarray]:
    path = ROOT / "data" / "meld" / f"{name}_sent_emo.csv"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MELD_BASE + f"{name}_sent_emo.csv", path)
    texts, ys = [], []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            e = r["Emotion"].strip().lower()
            if e in MELD_TO_FER7:
                texts.append(r["Utterance"]); ys.append(FER7_LABELS.index(MELD_TO_FER7[e]))
            if limit and len(texts) >= limit:
                break
    return texts, np.array(ys)


def raw28(tm, texts: list[str], desc: str) -> np.ndarray:
    return np.stack([np.asarray(tm.predict_fer7(t)[1], dtype=np.float64).ravel() for t in tqdm(texts, desc=desc)])


def summarize(Y, P):
    n = len(Y); k = int((P == Y).sum()); lo, hi = wilson_ci(k, n)
    return {"n": n, "top1": k / n, "ci95": [lo, hi], "macro_f1": macro_f1(confusion(Y, P, 7))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--out", default="docs/eval_output")
    args = ap.parse_args()
    out_dir = ROOT / args.out

    tm = load_text_model()
    te_texts, Yte = load_split("test", args.limit)
    tr_texts, Ytr = load_split("train", args.train_limit)
    Rte = raw28(tm, te_texts, "test raw28")
    Rtr = raw28(tm, tr_texts, "train raw28")

    results = {}
    for m in COLLAPSE_METHODS:
        P = np.stack([collapse_goemotions_to_fer7(r, tm.id2label, m) for r in Rte]).argmax(axis=1)
        results[f"collapse_{m}"] = summarize(Yte, P)

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=None).fit(Rtr, Ytr)
    results["logreg_28to7_meld_train"] = summarize(Yte, clf.predict(Rte))
    clf_bal = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Rtr, Ytr)
    results["logreg_28to7_meld_train_balanced"] = summarize(Yte, clf_bal.predict(Rte))
    maj = int(np.bincount(Ytr).argmax())
    results["majority_from_train"] = summarize(Yte, np.full_like(Yte, maj))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "collapse_ablation.json").write_text(json.dumps({"n_test": len(Yte), "n_train": len(Ytr), "results": results}, indent=2))
    labels = {
        "majority_from_train": "Majority class (Neutral)",
        "collapse_sum": "Surjection, sum (paper default)",
        "collapse_max": "Surjection, max",
        "collapse_mean": "Surjection, bin-mean",
        "logreg_28to7_meld_train": "Logistic 28$\\to$7, fit on MELD train",
        "logreg_28to7_meld_train_balanced": "Logistic 28$\\to$7, class-balanced",
    }
    rows = "\n".join(
        f"    {labels[k]} & {100*v['top1']:.1f} & {v['macro_f1']:.3f} & [{100*v['ci95'][0]:.1f}, {100*v['ci95'][1]:.1f}] \\\\"
        for k, v in results.items())
    tex = f"""% Auto-generated: scripts/collapse_ablation.py
\\begin{{table}}[t]
  \\centering
  \\caption{{How the 28$\\to$7 label collapse is computed matters on natural text: RoBERTa-GoEmotions sigmoids on MELD test ($n={len(Yte)}$), same model, different aggregation. The logistic map is fitted on MELD train ($n={len(Ytr)}$), so it is the only variant that has seen dialogue.}}
  \\label{{tab:collapse_ablation}}
  \\small
  \\begin{{tabular}}{{@{{}}l r r l@{{}}}}
    \\toprule
    Aggregation & Top-1 (\\%) & Macro-F1 & 95\\% CI \\\\
    \\midrule
{rows}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    (out_dir / "collapse_ablation.tex").write_text(tex)
    for k, v in results.items():
        print(f"{k:36s} top1={100*v['top1']:.1f} f1={v['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
