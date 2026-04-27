# Bimodal Affective Alignment

> **Text (GoEmotions → FER-7) + face (ResNet-50, FER2013) → late fusion (α) → FLAN-T5 empathetic reply.**
> A reproducible course project that fuses linguistic and facial affect cues, exposes the
> trust weight α as an HCI knob, and reports honest baseline / fine-tuned numbers on
> congruent and dissonant inputs.

This document is a clone-and-reproduce guide. Following the steps below from a fresh
clone reproduces every number in `docs/bimodal_report.tex`, the qualitative tables,
the latency benchmark, and the slide deck.

---

## Contents

1. [What you get](#1-what-you-get)
2. [Requirements](#2-requirements)
3. [Clone and set up the environment](#3-clone-and-set-up-the-environment)
4. [Hugging Face token](#4-hugging-face-token)
5. [Run the Streamlit demo](#5-run-the-streamlit-demo)
6. [Reproduce the baseline benchmarks](#6-reproduce-the-baseline-benchmarks)
7. [Fine-tune the visual branch](#7-fine-tune-the-visual-branch)
8. [Reproduce the fine-tuned benchmarks](#8-reproduce-the-fine-tuned-benchmarks)
9. [Dissonance evaluations](#9-dissonance-evaluations)
10. [Per-class qualitative tables](#10-per-class-qualitative-tables)
11. [Latency benchmark](#11-latency-benchmark)
12. [Run the unit tests](#12-run-the-unit-tests)
13. [Build the report](#13-build-the-report)
14. [Build the slide deck](#14-build-the-slide-deck)
15. [Repository layout](#15-repository-layout)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. What you get

Running the steps below produces:

| Artifact | Path | Created by |
|---|---|---|
| Streamlit demo (interactive) | `app/streamlit_app.py` | `streamlit run` |
| Baseline congruent benchmark JSON + LaTeX | `docs/eval_output/results.json`, `table.tex` | `scripts/run_evaluation.py` |
| Fine-tuned congruent benchmark JSON + LaTeX | `docs/eval_output/results_ft.json`, `table_ft.tex` | `scripts/run_evaluation.py --finetune-ckpt` |
| FER2013 fine-tune (≈ 102 MB) | `checkpoints/fer2013_finetune.pt` (+ `.json`) | `scripts/finetune_fer2013.py` |
| Real-image dissonance results (baseline + FT) | `docs/eval_output/dissonant_fer2013_results*.json` | `scripts/eval_dissonant_fer2013.py` |
| Synthetic dissonance results | `docs/eval_output/dissonant_results.json` | `scripts/build_dissonant_eval_csv.py` + `run_ablation.py` |
| Per-class qualitative tables (baseline + FT) | `docs/eval_output/qual_table*.tex`, `qual_images*/` | `scripts/build_qualitative_table.py` |
| 10-row dissonant qualitative tables (baseline + FT) | `docs/eval_output/qual_dissonant_table*.tex`, `qual_dissonant_images*/` | `scripts/build_dissonant_qualitative_table.py` |
| Latency table | `docs/eval_output/latency.txt` | `scripts/benchmark_latency.py` |
| ACM-style report PDF | `docs/bimodal_report.pdf` | `pdflatex` × 2 (or Overleaf) |
| Slide deck (≈ 20 slides + speaker notes) | `docs/Bimodal_Affective_Alignment.pptx` | `scripts/build_slides.py` |

All evaluation scripts share the same `.env`, log to `docs/eval_output/`, and are
deterministic to within run-to-run noise on a single machine.

---

## 2. Requirements

* **Python 3.10 or newer** (3.10 / 3.11 / 3.12 all known good)
* About **8 GB free disk** for Hugging Face model + dataset caches on first run
* About **4 GB RAM**; a CUDA or Apple-MPS GPU is optional but speeds up the bigger
  evaluations (full FER2013 test split, FLAN-T5 generation, fine-tune)
* A working **TeX distribution** (TeX Live / MacTeX) **or an Overleaf account** to
  compile the report
* **PowerPoint, Keynote, or LibreOffice Impress** to view the generated `.pptx`

The fine-tune step (Section 7) needs roughly **6 GB GPU memory** at the default
batch size; on CPU-only machines it is slow but completes. A pre-flight check is
provided as `scripts/check_training_env.py`.

---

## 3. Clone and set up the environment

```bash
git clone <your-fork-url> BimodalAffectiveAlignment
cd BimodalAffectiveAlignment

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Install the package in editable mode together with the test extras.
pip install -e ".[dev]"
```

This installs PyTorch, `transformers`, `datasets`, Streamlit, Pillow, NumPy,
`python-pptx`, and `pytest`, plus the local `bimodal_empathy` package.

---

## 4. Hugging Face token

The text classifier, the face classifier weights, FLAN-T5, the GoEmotions test
split, and the FER2013 dataset are all downloaded from Hugging Face on first use.
Authenticated access avoids rate-limit warnings:

```bash
cp .env.example .env
# Edit .env and set:
#   HF_TOKEN=hf_xxx_your_token_here
```

Every script calls `dotenv.load_dotenv(".env", override=True)`, so this single
file is the only place the token needs to live.

---

## 5. Run the Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Use the sidebar to:

* Type or paste a user utterance.
* Optionally upload a face image (the demo accepts the camera input as well).
* Move the **trust weight α** slider between 0 (face-only) and 1 (text-only).
* Click **Analyze & generate response**.

The page shows three probability bar charts (text, face, fused FER-7), a
dissonance badge, the FLAN-T5 empathetic reply, and a per-stage timing panel.
The "Debug: numeric vectors" expander prints the same probabilities as a
labelled table.

---

## 6. Reproduce the baseline benchmarks

These three commands reproduce **Tables 1, 2 (baseline row), 3, and 5** of the
report.

```bash
# Congruent (text-only, image-only, bimodal fused @ α=0.5) on the full test splits
python scripts/run_evaluation.py --limit 5000 --out docs/eval_output

# Real-image dissonance with the baseline (AffectNet) face branch
python scripts/eval_dissonant_fer2013.py --limit 5000 \
    --no-details \
    --out-json docs/eval_output/dissonant_fer2013_results.json

# Synthetic dissonance (one-hot face vectors, isolates the fusion rule)
python scripts/build_dissonant_eval_csv.py --n 100 --out data/dissonant_eval_100.csv
python scripts/run_ablation.py data/dissonant_eval_100.csv --brief \
    --out-json docs/eval_output/dissonant_results.json
```

Expected results (full test splits, ≈ 6–10 min on Apple MPS):

* Text-only **69.7 %** (`n=4590`), Image-only **23.1 %** (`n=3589`), Fused **81.6 %** (`n=3589`).
* Real-image dissonant fused **4.3 %**, face-only **23.1 %** (`n=3589`).
* Synthetic dissonant fused **100 %**, text-only **3 %** (`n=100`).

---

## 7. Fine-tune the visual branch

This produces the **Table 2 fine-tuned row** (FER2013 visual top-1 23.1 % → 68.5 %)
and unlocks the fine-tuned downstream evaluations.

```bash
# Optional: confirm the machine is suitable
python scripts/check_training_env.py

# Train (default: 3 epochs, batch 32). Saves checkpoints/fer2013_finetune.pt + .json
python scripts/finetune_fer2013.py

# Quick smoke (50 steps, ~1 minute) to confirm the script is wired
python scripts/finetune_fer2013.py --max-steps 50 --batch-size 16

# Independent check of the new checkpoint on the FER2013 test split
python scripts/eval_fer_checkpoint.py \
    --finetune checkpoints/fer2013_finetune.pt --limit 5000
```

The deployed app and `run_evaluation.py` continue to use the original Elena
Ryumina weights unless `--finetune-ckpt` is passed, so the baseline path stays
reproducible regardless of whether the fine-tune was run.

---

## 8. Reproduce the fine-tuned benchmarks

Once `checkpoints/fer2013_finetune.pt` exists, these commands reproduce
**Tables 6 and 7** of the report.

```bash
# Fine-tuned congruent benchmark (Tables 1 / 6 companion)
python scripts/run_evaluation.py --limit 5000 \
    --finetune-ckpt checkpoints/fer2013_finetune.pt \
    --suffix _ft --table-label tab:resultsft \
    --out docs/eval_output

# Fine-tuned real-image dissonance (Table 7 lower row)
python scripts/eval_dissonant_fer2013.py --limit 5000 \
    --finetune-ckpt checkpoints/fer2013_finetune.pt \
    --no-details \
    --out-json docs/eval_output/dissonant_fer2013_results_ft.json
```

Expected: fused congruent **87.8 %** (vs. text-branch 82.6 %), fused dissonant
**41.8 %** (vs. baseline 4.3 %), face-only dissonant **68.5 %** (vs. baseline 23.1 %).

---

## 9. Dissonance evaluations

Already covered in Sections 6 and 8; the underlying scripts are:

* `scripts/eval_dissonant_fer2013.py` — pairs every FER2013 test image with a
  deliberately mismatched template sentence and reports α = 1 / 0.5 / 0 accuracy
  against the FER2013 gold label. Add `--finetune-ckpt …` to swap in the
  fine-tuned face branch.
* `scripts/build_dissonant_eval_csv.py` — emits a CSV of synthetic one-hot face
  rows paired with mismatched text. The companion `scripts/run_ablation.py`
  consumes that CSV and reports α = 1 / 0.5 / 0 accuracy without ever touching a
  ResNet (this is the row that probes the fusion arithmetic itself).

---

## 10. Per-class qualitative tables

These produce **Tables 4, 5, 8, and 9** of the report along with the PNG
thumbnails referenced by `\includegraphics{...}`.

```bash
# 7 congruent rows (one image per FER-7 class) with the baseline face branch
python scripts/build_qualitative_table.py

# Same 7 rows with the fine-tuned face branch
python scripts/build_qualitative_table.py \
    --finetune-ckpt checkpoints/fer2013_finetune.pt \
    --suffix _ft \
    --table-label tab:qualimagesft \
    --responses-table-label tab:qualresponsesft

# 10 dissonant rows (real images + mismatched text) with the baseline face branch
python scripts/build_dissonant_qualitative_table.py

# Same 10 rows with the fine-tuned face branch
python scripts/build_dissonant_qualitative_table.py \
    --finetune-ckpt checkpoints/fer2013_finetune.pt \
    --suffix _ft
```

The `*.tex` fragments under `docs/eval_output/` are already inlined in the
report; only the PNG folders are needed when uploading to Overleaf.

---

## 11. Latency benchmark

```bash
python scripts/benchmark_latency.py
```

Outputs `docs/eval_output/latency.txt` with mean and std-dev (ms) for the text
model, face model, fusion arithmetic, FLAN-T5 generation, and end-to-end. The
benchmark uses 3 warm-up + 15 timed runs on whichever device PyTorch picks
(`cuda` / `mps` / `cpu`).

---

## 12. Run the unit tests

```bash
pytest -q
```

Tests cover the GoEmotions → FER-7 mapping, the fusion rule, the dissonance
flag, and basic Streamlit-app helpers.

---

## 13. Build the report

The report is `docs/bimodal_report.tex` (with `docs/main.tex` kept as a synced
copy for Overleaf upload). The bibliography is inlined (`thebibliography`), so
no BibTeX run is needed.

**Local build** (requires the `acmart` package):

```bash
cd docs
pdflatex bimodal_report
pdflatex bimodal_report          # second pass settles cross-refs
```

**Overleaf build:**

1. Create a new project from the *ACM Conference Template*.
2. Upload `docs/main.tex` (or `docs/bimodal_report.tex`) at the project root.
3. Upload the contents of `docs/eval_output/` so that the `qual_images/`,
   `qual_images_ft/`, `qual_dissonant_images/`, and `qual_dissonant_images_ft/`
   folders sit at the project root or under `eval_output/` (the report's
   `\graphicspath{{eval_output/}{./}}` accepts either layout).
4. Compile.

`docs/REPORT_BUILD.txt` has the same instructions in slightly more detail.

---

## 14. Build the slide deck

```bash
python scripts/build_slides.py
# → writes docs/Bimodal_Affective_Alignment.pptx
```

The script generates ~20 16:9 slides (title, motivation, pipeline, fusion rule,
text/face/synthesizer details, congruent + fine-tuned + dissonance results,
qualitative cases, latency, failure modes, takeaways) with speaker notes
attached to each slide. Open the resulting `.pptx` in PowerPoint, Keynote, or
Google Slides.

---

## 15. Repository layout

```
BimodalAffectiveAlignment/
├── app/                        # Streamlit demo
├── checkpoints/                # Fine-tuned weights (excluded from git; .json kept)
├── data/
│   ├── dissonant_samples.csv   # Hand-authored dissonance probes for run_ablation.py
│   └── dissonant_eval_100.csv  # Generated by build_dissonant_eval_csv.py
├── docs/
│   ├── bimodal_report.tex      # ACM-style report (single source of truth)
│   ├── main.tex                # Overleaf-uploadable copy of bimodal_report.tex
│   ├── REPORT_BUILD.txt        # Long-form report build notes
│   ├── HCI_STUDY_TEMPLATE.md   # Likert-study scaffold
│   └── eval_output/            # Generated JSON, LaTeX fragments, PNG thumbnails
├── scripts/
│   ├── run_evaluation.py                  # §6, §8
│   ├── finetune_fer2013.py                # §7
│   ├── eval_fer_checkpoint.py             # §7
│   ├── eval_dissonant_fer2013.py          # §6, §8, §9
│   ├── build_dissonant_eval_csv.py        # §6, §9
│   ├── run_ablation.py                    # §6, §9
│   ├── build_qualitative_table.py         # §10
│   ├── build_dissonant_qualitative_table.py # §10
│   ├── benchmark_latency.py               # §11
│   ├── build_slides.py                    # §14
│   └── check_training_env.py              # §7 pre-flight
├── src/bimodal_empathy/        # Library: text/vision sensors, fusion, synthesizer
├── tests/                      # Unit tests
├── .env.example                # Template for HF_TOKEN
├── pyproject.toml              # Editable install + pytest config
├── requirements.txt            # Direct dependency list (mirrors pyproject)
└── README.md                   # This file
```

---

## 16. Troubleshooting

* **`Warning: You are sending unauthenticated requests…`** — `HF_TOKEN` is
  missing or unreadable. Confirm `.env` exists, the variable is exactly
  `HF_TOKEN=...` (not `HUGGINGFACE_HUB_TOKEN`), and that the script is run from
  the repository root so the relative `.env` is picked up.
* **`PermissionError: Operation not permitted` while loading datasets** — the
  Hugging Face cache cannot be written from a sandboxed shell. Re-run the
  command from a normal terminal session, or set `HF_HOME` to a writable path
  before invocation.
* **`No module named 'pptx'`** — re-run `pip install -e ".[dev]"` (or
  `pip install python-pptx`) to install the optional slide-deck dependency.
* **LaTeX `Command \Bbbk' already defined`** — your TeX install has a stale
  `amssymb` import; the report does not need it. Make sure no local `.cls`
  files override `acmart`.
* **Long-running benchmarks appear to hang** — the FER2013 image-only stage
  iterates 3589 images silently after the bimodal `tqdm` bar finishes.
  Total wall time on Apple MPS is around 6–10 minutes per `run_evaluation.py`
  call.

---

## License and credits

* Text classifier: `SamLowe/roberta-base-go_emotions` (Apache-2.0).
* Face classifier: `ElenaRyumina/face_emotion_recognition` (Apache-2.0).
* Generator: `google/flan-t5-base` (Apache-2.0).
* Datasets: GoEmotions (CC-BY-4.0) and `AutumnQiu/fer2013` (community redistribution
  of the original FER2013 challenge data).

This repository is course coursework; please credit the upstream model and
dataset authors when reusing.
