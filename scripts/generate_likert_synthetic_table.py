#!/usr/bin/env python3
"""
Generate a synthetic expert-review-style Likert dataset (≥100 rows) and a
comparison table by FER-7 *fused* class label (Project.pdf §3.2).

The scores are *simulated* (seeded RNG + class-dependent means) for report /
slide tables — not real participant data. For human subjects you would replace
this with logged rows from the Streamlit Likert panel.

  python scripts/generate_likert_synthetic_table.py
  python scripts/generate_likert_synthetic_table.py --n 120 --seed 0

Outputs under docs/eval_output/:
  - likert_synthetic_study.csv
  - likert_by_fused_label_table.md
  - likert_by_fused_label_table.tex
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Repo root: scripts/ -> parent
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.likert_log import FIELDNAMES

# ---------------------------------------------------------------------------
# Class-dependent *base* (μ) for 1–5 Likert (empathy, social_presence, congruency)
# Slightly lower for dissonant scenarios (applied when dissonant=True).
# ---------------------------------------------------------------------------
_BASE: dict[str, tuple[float, float, float]] = {
    "Angry": (3.3, 3.1, 3.2),
    "Disgust": (3.1, 2.9, 3.0),
    "Fear": (3.6, 3.2, 3.3),
    "Happy": (4.2, 3.9, 4.0),
    "Sad": (3.8, 3.4, 3.5),
    "Surprise": (3.5, 3.2, 3.4),
    "Neutral": (3.2, 3.0, 3.1),
}
_DIS_PENALTY = (-0.4, -0.3, -0.5)

_UTTER_R: dict[str, tuple[str, str]] = {
    "Angry": (
        "I am furious and I will not stand for this.",
        "I hear that you are angry. What is going on for you right now?",
    ),
    "Disgust": (
        "This is completely disgusting; I feel revolted.",
        "That sounds very upsetting. Can you say more about what you saw?",
    ),
    "Fear": (
        "I am terrified; I feel deep fear about this situation.",
        "I hear that you are scared. What is worrying you the most right now?",
    ),
    "Happy": (
        "I could not be happier with this amazing news today!",
        "I am glad things feel positive for you. What would you like to do next?",
    ),
    "Sad": (
        "I feel heartbroken; everything seems gray to me right now.",
        "I hear that you are feeling low. What has been the hardest part?",
    ),
    "Surprise": (
        "I can barely process what just happened; this is incredible.",
        "That is a lot to take in. What surprised you the most?",
    ),
    "Neutral": (
        "It is a normal quiet day; nothing in particular is happening to me.",
        "Thanks for sharing. Is there anything you would like to talk through?",
    ),
}


def _clip_int_1_5(x: float) -> int:
    return int(max(1, min(5, round(x))))


def _sample_triplet(
    rng, fused: str, dissonant: bool
) -> tuple[int, int, int]:
    import numpy as np

    b = _BASE[fused]
    if dissonant:
        b = (
            b[0] + _DIS_PENALTY[0],
            b[1] + _DIS_PENALTY[1],
            b[2] + _DIS_PENALTY[2],
        )
    e = b[0] + rng.normal(0, 0.55)
    s = b[1] + rng.normal(0, 0.50)
    c = b[2] + rng.normal(0, 0.50)
    return _clip_int_1_5(e), _clip_int_1_5(s), _clip_int_1_5(c)


def _stratify_labels(n: int, k: int) -> list[str]:
    """Return list of n labels, each of k classes as evenly as possible."""
    base = n // k
    rem = n % k
    out: list[str] = []
    for j, lab in enumerate(FER7_LABELS):
        n_j = base + (1 if j < rem else 0)
        out.extend([lab] * n_j)
    assert len(out) == n
    return out


def _write_csv(
    path: Path,
    rows: list[dict[str, str | float | int | bool]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _agg_by_label(
    rows: list[dict[str, object]], key: str = "fused_label"
) -> list[tuple[str, int, float, float, float, float, float, float]]:
    """(label, n, m_emp, sd_emp, m_sp, sd_sp, m_con, sd_con)."""
    import numpy as np

    g: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for r in rows:
        fl = str(r.get(key, ""))
        g[fl].append(
            (int(r["empathy_1_5"]), int(r["social_presence_1_5"]), int(r["congruency_1_5"]))
        )
    out: list[tuple[str, int, float, float, float, float, float, float]] = []
    for lab in FER7_LABELS:
        xs = g.get(lab, [])
        n = len(xs)
        if n == 0:
            out.append((lab, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        a = np.array(xs, dtype=float)
        def sdcol(j: int) -> float:
            v = a[:, j]
            return float(v.std(ddof=1)) if n > 1 else 0.0

        out.append(
            (
                lab,
                n,
                float(a[:, 0].mean()),
                sdcol(0),
                float(a[:, 1].mean()),
                sdcol(1),
                float(a[:, 2].mean()),
                sdcol(2),
            )
        )
    return out


def _md_table(agg: list[tuple[str, int, float, float, float, float, float, float]], total_n: int) -> str:
    lines = [
        "# Likert by fused FER-7 label (synthetic, seeded)",
        "",
        f"Total *n* = **{total_n}** rows. Values are **mean (SD)** on 1–5.",
        "",
        "| FER-7 (fused) | *n* | Empathy | Social presence | Congruency |",
        "|----------------|-----:|---------|-----------------|------------|",
    ]
    for lab, n, m_e, s_e, m_sp, s_sp, m_c, s_c in agg:
        if n == 0:
            continue
        lines.append(
            f"| {lab} | {n} | {m_e:.2f} ({s_e:.2f}) | {m_sp:.2f} ({s_sp:.2f}) | {m_c:.2f} ({s_c:.2f}) |"
        )
    lines.extend(
        [
            "",
            "*Disclaimer: data produced by `scripts/generate_likert_synthetic_table.py` for illustration; "
            "replace with real study logs for publication.*",
        ]
    )
    return "\n".join(lines) + "\n"


def _tex_table(agg: list[tuple[str, int, float, float, float, float, float, float]], total_n: int) -> str:
    lines = [
        r"% Auto-generated: scripts/generate_likert_synthetic_table.py",
        r"% n=" + str(total_n) + " synthetic Likert rows stratified by fused label.",
        r"\begin{center}",  # non-floating: acmart drops small floats queued behind table* pages
        r"  \small",
        r"  \captionof{table}{ Synthetic expert-style Likert means (SD) by fused FER-7 class, $n$ per row stratified. \emph{Illustration only}---replace with real participant data for formal claims.}",
        r"  \label{tab:likert_by_class}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}l r ccc@{}}",
        r"    \toprule",
        r"    \textbf{FER-7 (fused)} & $n$ & \textbf{Empathy} & \textbf{Social pr.} & \textbf{Congruency} \\",
        r"    \midrule",
    ]
    for lab, n, m_e, s_e, m_sp, s_sp, m_c, s_c in agg:
        if n == 0:
            continue
        lines.append(
            rf"    {lab} & {n} & ${m_e:.2f} \pm {s_e:.2f}$ & ${m_sp:.2f} \pm {s_sp:.2f}$ & ${m_c:.2f} \pm {s_c:.2f}$ \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{center}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n",
        type=int,
        default=120,
        help="Number of rows (default 120, ≥100; 7 FER-7 classes balanced as evenly as possible).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "eval_output",
        help="Output directory (CSV + .md + .tex).",
    )
    args = ap.parse_args()
    n = max(1, int(args.n))
    if n < 100:
        print("Warning: rubric / request asked for at least 100; consider --n 105 or 120", file=sys.stderr)

    import numpy as np

    rng = np.random.default_rng(args.seed)
    labels = _stratify_labels(n, len(FER7_LABELS))

    t0 = datetime.now(timezone.utc)
    rows: list[dict] = []
    for i, fused in enumerate(labels):
        dissonant = bool(rng.random() < 0.28)
        if dissonant:
            others = [x for x in FER7_LABELS if x != fused]
            text_arg = str(rng.choice(others))
            face_arg = str(rng.choice(others)) if rng.random() < 0.5 else text_arg
        else:
            text_arg = face_arg = fused
        e, s, c = _sample_triplet(rng, fused, dissonant)
        u, rep = _UTTER_R[fused]
        ts = t0
        if i:
            from datetime import timedelta

            ts = t0 + timedelta(milliseconds=17 * i)
        row = {
            "ts_iso_utc": ts.isoformat(),
            "participant_id": f"syn_p{i+1:04d}",
            "empathy_1_5": e,
            "social_presence_1_5": s,
            "congruency_1_5": c,
            "alpha": 0.5,
            "vision_branch": "synthetic (script)",
            "dissonant": "1" if dissonant else "0",
            "text_argmax": text_arg,
            "face_argmax": face_arg,
            "fused_label": fused,
            "e2e_ms": float(round(700.0 + 180.0 * float(rng.random()), 1)),
            "utterance_excerpt": u[:400],
            "response_excerpt": rep[:500],
        }
        rows.append({k: row[k] for k in FIELDNAMES})

    outdir: Path = args.outdir
    csv_path = outdir / "likert_synthetic_study.csv"
    _write_csv(csv_path, rows)

    rows_num = [
        {
            "fused_label": r["fused_label"],
            "empathy_1_5": int(r["empathy_1_5"]),
            "social_presence_1_5": int(r["social_presence_1_5"]),
            "congruency_1_5": int(r["congruency_1_5"]),
        }
        for r in rows
    ]
    agg = _agg_by_label(rows_num, "fused_label")
    md = _md_table(agg, total_n=n)
    tex = _tex_table(agg, total_n=n)
    (outdir / "likert_by_fused_label_table.md").write_text(md, encoding="utf-8")
    (outdir / "likert_by_fused_label_table.tex").write_text(tex, encoding="utf-8")

    print(f"Wrote {n} rows → {csv_path.resolve()}")
    print(f"Markdown table → {(outdir / 'likert_by_fused_label_table.md').resolve()}")
    print(f"LaTeX table   → {(outdir / 'likert_by_fused_label_table.tex').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
