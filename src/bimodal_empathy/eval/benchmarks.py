"""
Offline evaluation: top-1 accuracy for text-only, image-only, and bimodal (congruent) modes.

* Text: GoEmotions (simplified) test set, single-label only; gold = mapped FER7 index.
* Image: HuggingFace `AutumnQiu/fer2013` (7-class, standard FER2013 label ids 0--6).
* Bimodal: same FER images with a fixed emotion-congruent sentence per class; fusion at α=0.5
  (or CLI --alpha 0.5; also report α=1 text-branch vs α=0 face-branch for ablation on same data).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from bimodal_empathy.config import FER7_LABELS, DEFAULT_ALPHA
from bimodal_empathy.emotion_mapping import goemotion_name_to_fer7
from bimodal_empathy.fusion import fuse
from bimodal_empathy.text_sensor import load_text_model
from bimodal_empathy.vision_sensor import load_vision_model

# FER2013 / our code: 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral
CONGRUENT_PHRASES: list[str] = [
    "I am absolutely furious and enraged; this is unacceptable and I will not stand for it.",
    "That is completely disgusting, I feel revolted and sickened by what I am seeing.",
    "I am terrified and panicked; I feel deep fear and anxiety about this situation right now.",
    "I am overjoyed and delighted; I could not be happier with this amazing wonderful news today!",
    "I feel heartbroken, deeply sad, and I could cry; everything feels gray and empty to me.",
    "I am so shocked and surprised I can barely process what just happened, this is incredible.",
    "It is a normal quiet day; I feel pretty neutral and nothing in particular is happening to me.",
]


def _fer7_index_from_name(name: str) -> int:
    n = (name or "").strip().lower()
    g = goemotion_name_to_fer7(n)
    return FER7_LABELS.index(g)


@dataclass
class EvalResult:
    mode: str
    n: int
    top1_acc: float
    notes: str = ""


def _gold_fer7_from_go_emotions_row(
    label_ids: list[int], labels_feature: Any
) -> int | None:
    if not label_ids or len(label_ids) != 1:
        return None
    idx = int(label_ids[0])
    # Sequence(ClassLabel) in go_emotions "simplified"
    feat = labels_feature
    if hasattr(feat, "feature"):
        name = feat.feature.int2str(idx)
    else:
        name = feat.int2str(idx)
    return _fer7_index_from_name(name)


def eval_text_only(text_model, limit: int | None) -> EvalResult:
    ds = load_dataset("go_emotions", "simplified", split="test", trust_remote_code=False)
    correct = 0
    n = 0
    lf = ds.features["labels"]
    for row in tqdm(ds, desc="text-only"):
        if limit is not None and n >= limit:
            break
        g = _gold_fer7_from_go_emotions_row(row["labels"], lf)
        if g is None:
            continue
        p_text, _, _ = text_model.predict_fer7(row["text"])
        pred = int(np.argmax(p_text))
        n += 1
        if pred == g:
            correct += 1
    acc = (correct / n) if n else 0.0
    return EvalResult("text-only (GoEmotions test, single-label)", n, acc)


def _iter_fer2013_images(limit: int | None, split: str = "test"):
    """AutumnQiu/fer2013: image (PIL) + label 0--6 (FER2013). Default: official \texttt{test} split."""
    ds = load_dataset("AutumnQiu/fer2013", split=split, trust_remote_code=False, streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield row["image"], int(row["label"])


def eval_bimodal_congruent(
    text_model, vision_model, limit: int | None, alpha: float = DEFAULT_ALPHA
) -> EvalResult:
    correct = 0
    n = 0
    for image, y in tqdm(_iter_fer2013_images(limit, "test"), desc="bimodal (α=0.5)", total=limit):
        phrase = CONGRUENT_PHRASES[y]
        p_text, _, _ = text_model.predict_fer7(phrase)
        p_face, _, _ = vision_model.predict_fer7(image)
        p_f, _, _ = fuse(p_text, p_face, alpha=alpha)
        pred = int(np.argmax(p_f))
        n += 1
        if pred == y:
            correct += 1
    acc = (correct / n) if n else 0.0
    return EvalResult(
        f"bimodal congruent (FER image + template text, α={alpha:.2f})",
        n,
        acc,
    )


def eval_ablation_same_pairs(
    text_model, vision_model, limit: int | None
) -> dict[str, float]:
    """On same bimodal pairs, report text-only and image-only argmax acc vs y (not fusion)."""
    t_ok = f_ok = 0
    n = 0
    for image, y in _iter_fer2013_images(limit, "test"):
        phrase = CONGRUENT_PHRASES[y]
        p_text, _, _ = text_model.predict_fer7(phrase)
        p_face, _, _ = vision_model.predict_fer7(image)
        n += 1
        if int(np.argmax(p_text)) == y:
            t_ok += 1
        if int(np.argmax(p_face)) == y:
            f_ok += 1
    if not n:
        return {"text_branch_acc": 0.0, "face_branch_acc": 0.0, "n": 0}
    return {
        "text_branch_acc": t_ok / n,
        "face_branch_acc": f_ok / n,
        "n": n,
    }


def eval_image_only(vision_model, limit: int | None) -> EvalResult:
    correct = 0
    n = 0
    for image, y in tqdm(_iter_fer2013_images(limit, "test"), desc="image-only", total=limit):
        p_face, _, _ = vision_model.predict_fer7(image)
        pred = int(np.argmax(p_face))
        n += 1
        if pred == y:
            correct += 1
    acc = (correct / n) if n else 0.0
    return EvalResult("image-only (AutumnQiu/fer2013, test split)", n, acc)


def run_full_benchmark(
    limit: int = 200,
    alpha: float = 0.5,
    out_dir: Path | None = None,
    vision_model: Any = None,
    table_label: str = "tab:results",
    file_suffix: str = "",
) -> list[EvalResult]:
    """Run text-only / image-only / bimodal congruent benchmark.

    Pass ``vision_model`` to substitute a fine-tuned face branch (anything with
    a ``predict_fer7(image)`` method). The default ``None`` keeps the prior
    behavior and loads the baseline ElenaRyumina ResNet-50.
    """
    out_dir = out_dir or Path("docs/eval_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    text_model = load_text_model()
    if vision_model is None:
        vision_model = load_vision_model()

    r_text = eval_text_only(text_model, limit)
    r_img = eval_image_only(vision_model, limit)
    r_bio = eval_bimodal_congruent(text_model, vision_model, limit, alpha=alpha)
    abl = eval_ablation_same_pairs(text_model, vision_model, limit)

    results = {
        "limit_per_task": limit,
        "alpha_bimodal": alpha,
        "results": [asdict(r) for r in (r_text, r_img, r_bio)],
        "ablation_on_fer_with_templates": abl,
    }
    (out_dir / f"results{file_suffix}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (out_dir / f"table{file_suffix}.tex").write_text(
        _build_latex_table(r_text, r_img, r_bio, abl, limit, alpha, label=table_label),
        encoding="utf-8",
    )
    return [r_text, r_img, r_bio]


def _build_latex_table(
    r_text: EvalResult,
    r_img: EvalResult,
    r_bio: EvalResult,
    abl: dict[str, float],
    limit: int,
    alpha: float,
    label: str = "tab:results",
) -> str:
    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}"
    t_br = 100.0 * abl.get("text_branch_acc", 0.0)
    f_br = 100.0 * abl.get("face_branch_acc", 0.0)
    n_pair = int(abl.get("n", 0))
    a = float(alpha)
    cpt = (
        f"Top-1 FER$7$ accuracy (run: $n$ limited to {limit} per sub-task where applicable; "
        f"$\\alpha\\!=\\!{a}$ for fusion). "
        r"\emph{Text:} GoEmotions test, single\nobreakdash label (mapped to FER$7$). "
        r"\emph{Image:} \texttt{AutumnQiu/fer2013}. "
        r"\emph{Bimodal:} one emotion\nobreakdash matched template per FER class; fusion per Eq.~\ref{eq:main}. "
        r"Lower two rows: text-only and image-only \emph{branches} on the same bimodal pairs (ablation)."
    )
    return f"""% Auto-generated by scripts/run_evaluation.py
\\begin{{table*}}[t]
  \\centering
  \\small
  \\caption{{{cpt}}}
  \\label{{{label}}}
  \\begin{{tabular}}{{@{{}}lcc@{{}}}}
    \\toprule
    \\textbf{{Setting}} & \\textbf{{$n$}} & \\textbf{{Acc. (\\%)}} \\\\
    \\midrule
    Text--only (GoEmotions) & {r_text.n} & {pct(r_text.top1_acc)} \\\\
    Image--only (FER2013) & {r_img.n} & {pct(r_img.top1_acc)} \\\\
    Text$+$image fused & {r_bio.n} & {pct(r_bio.top1_acc)} \\\\
    \\midrule
    Text branch only (same pairs) & {n_pair} & {t_br:.1f} \\\\
    Image branch only (same pairs) & {n_pair} & {f_br:.1f} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
"""
