# Bimodal Affective Alignment

> **Text (GoEmotions → FER-7) + face (ResNet-50, FER2013) → late fusion (α) → FLAN-T5 empathetic reply.**
> A reproducible course project that fuses linguistic and facial affect cues, exposes the
> trust weight α as an HCI knob, and reports honest baseline / fine-tuned numbers on
> congruent and dissonant inputs.

This document is a clone-and-reproduce guide. Following the steps below from a fresh
clone reproduces every number in `docs/bimodal_report.tex`, the qualitative tables,
the latency benchmark, and the slide deck.

**Repository:** <https://github.com/sainadth/BimodalAffectiveAlignment>

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
git clone https://github.com/sainadth/BimodalAffectiveAlignment.git
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
file is the only place the token needs to live. The same file optionally holds
`ANTHROPIC_API_KEY=` for `scripts/llm_baseline.py` (see 11b); leave it blank otherwise.

---

## 5. Run the apps

Two entrypoints share `app/common.py` (models, YuNet face crop, fusion):

```bash
streamlit run app/study_app.py       # participant study (design v2) — what participants see
streamlit run app/streamlit_app.py   # researcher demo — every internal exposed
```

**Participant study app** (`docs/STUDY_PROTOCOL.md`, design v3): consent
screen (logged with version + timestamp) → participant number → 12 turns
(2 blocks × 6 cells: label source `correct` / `wrong` / `none` × move
`assert` / `ask`, Williams-square counterbalanced, one scripted
affect-elicitation turn per block) → per turn: message + required webcam
photo → felt-emotion self-report → reply generated from the felt report per
cell → three 1–5 ratings + manipulation check → global PETS + blinding
question → in-app debrief with **withdraw my data**. No sidebar, no mode switch, no model output shown. Replies come from
`openai/gpt-oss-120b` on Groq with condition-specific prompts; the app refuses
to start if `GROQ_API_KEY` / `LLM_PROVIDER=groq` are missing. Resumes at the
first unlogged turn after a page refresh. Rows go to
`data/study_responses.csv` and `data/study_pets_global.csv` (`STUDY_CSV`,
`STUDY_GLOBAL_CSV` override). Photos are cropped to the face (OpenCV YuNet),
processed in memory, never stored.

**Researcher demo:** α slider, baseline vs fine-tuned face weights, FLAN-T5 vs
LLM generator with selectable prompt condition, three probability charts,
argmax + valence dissonance badges, timings, debug table, and a quick 3-item
rating logger (`data/likert_responses.csv`). Face frames are YuNet-cropped here
too, so demo numbers match the paper's face-crop assumption.

**Likert (HCI, Project.pdf §3.2):** in Researcher mode, after a run, use the **5-point** radio items
(Empathy, Social presence, Affective congruency) and **Save Likert row to
study log (CSV)**. Rows append to `data/likert_responses.csv` (set **`LIKERT_CSV`**
to override). Optional **Participant ID** in the sidebar is included in the
file. See `data/likert_responses.example.csv` for the column header.

**Synthetic Likert + class comparison table (for slides / report drafts):** to
generate **≥100** stratified (by fused FER-7 label) rows and **mean (SD)**
tables as Markdown and LaTeX, run:

```bash
python scripts/generate_likert_synthetic_table.py
# default: n=120 → docs/eval_output/likert_synthetic_study.csv
#     +  docs/eval_output/likert_by_fused_label_table.md
#     +  docs/eval_output/likert_by_fused_label_table.tex
```

The generator uses a **fixed seed** and class-level priors; it is for
**illustration** only—replace with real participant data for any formal claim.
The LaTeX fragment is meant to be pasted into the report and uses
`\label{tab:likert_by_class}`.

---

## 5b. Remote study (share a link)

For participants on their own computers: a FastAPI backend (`server/app.py`)
serving a React/TypeScript frontend (`web/`), deployed as one Docker container
behind Caddy for HTTPS. Same design v2, same loggers, same analysis.

```bash
pip install -e ".[eval,server]"
cd web && npm install && npm run build && cd ..
python scripts/make_participant_links.py --n 28 --base https://study.example.edu   # token links, data/participants.json
uvicorn server.app:app --host 0.0.0.0 --port 8000                                  # or: docker compose up -d --build
```

Participants open `https://<domain>/p/<token>`. The server runs the sensors on
Send, generates the reply only after the felt-emotion report (which defines
the correct label), enforces trial order, requires at least
one webcam frame (burst of 3, face-cropped, averaged, discarded), resumes after
a closed tab, and deletes all rows on withdrawal. Consent v4 states that frames
are transmitted encrypted and never stored. See `docs/DEPLOY.md`.

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

## 11b. α sweep, MELD, and LLM baseline

```bash
pip install -e ".[eval]"                        # scipy, matplotlib, anthropic

python scripts/alpha_sweep.py --vision finetuned   # docs/eval_output/alpha_sweep_ft.{json,csv,tex,png}
python scripts/alpha_sweep.py --vision baseline    # docs/eval_output/alpha_sweep.{json,csv,tex,png}
```

Sweeps α from 0 to 1 in 0.05 steps on the full FER2013 congruent benchmark,
with Wilson 95 % CIs, McNemar tests (fused-vs-text, fused-vs-face), the
majority-class floor, and the α = 0.5 confusion matrix.

```bash
python scripts/eval_meld.py                       # text branch on MELD test (2,610 natural utterances)
python scripts/eval_meld.py --video-dir data/meld/test   # + face branch + fusion (needs MELD.Raw videos, opencv)
```

Downloads the public MELD test CSV on first run and writes
`docs/eval_output/meld_results.json` + `meld_text_table.tex`. The raw videos
(10.9 GB, <https://affective-meld.github.io/>) are optional and git-ignored;
extract `MELD.Raw/test.tar.gz` and point `--video-dir` at the folder of
`dia*_utt*.mp4` files. Face detection uses OpenCV YuNet (ONNX auto-downloaded
to `checkpoints/`). Run with `--vision finetuned --suffix _bimodal_ft` and
`--vision baseline --suffix _bimodal_base`, then
`python scripts/build_meld_bimodal_table.py` to render `meld_bimodal_table.tex`.

```bash
ANTHROPIC_API_KEY=... python scripts/llm_baseline.py --csv data/meld/test_sent_emo.csv \
    --text-col Utterance --gold-col Emotion --limit 300
```

Add `--provider groq` (default model `openai/gpt-oss-120b`, ~free) for an
open-weight second comparator; output goes to `llm_baseline_groq.*`.

LLM-only comparator: Claude labels each utterance (± image) with a FER-7
emotion and writes an empathetic reply; `--with-pipeline` adds the flan-t5
reply side by side for the user study. `--dry-run` prints prompts only.
Committed runs on all 2,610 MELD utterances: Claude Opus 5
(`llm_baseline_meld.*`) 53.2 % top-1 / macro-F1 0.455, ~$14; gpt-oss-120b via
Groq (`llm_baseline_groq.*`) 54.1 % / 0.410, ~$0.50. Re-running `eval_meld.py`
adds every `llm_baseline_*.csv` as a row in `meld_text_table.tex`.

```bash
python scripts/collapse_ablation.py      # -> docs/eval_output/collapse_ablation.{json,tex}
```

Ablates how the 28 GoEmotions sigmoids are collapsed into FER-7 on MELD test:
`sum` (paper default; 13 labels feed Happy vs 1 Neutral), `max`, `mean`
(bin-size normalised), plus a logistic 28→7 map fitted on MELD train. The
deployed text sensor honours `COLLAPSE_METHOD=sum|max|mean` in `.env` (default
`sum`, so all committed numbers are unchanged unless you opt in).

```bash
python scripts/analyze_study.py            # study_responses.csv + study_pets_global.csv -> docs/eval_output/study_analysis.json
```

Runs the pre-specified design-v2 analysis from `docs/STUDY_PROTOCOL.md`:
per-participant condition means of the 3-item composite (and each item),
Friedman, Holm-corrected Wilcoxon on the four pre-specified pairs with
rank-biserial r, H2 on self-reported masking trials and on scripted trials,
sensor accuracy vs self-report, Cronbach's α on participant means, global PETS
descriptives, and the blinding funnel tabulation.

---

## 12. Run the unit tests

```bash
pytest -q
```

Tests cover the GoEmotions → FER-7 mapping, the fusion rule, the study
schedule (Williams-square balance, scripted-slot coverage, PID normalisation),
both study loggers and withdrawal, valence dissonance, the LLM prompt builder
(condition leakage), the statistics helpers, the study analysis on planted and
null effects, and the participant app via `streamlit.testing` (config guard,
consent content, PID validation).

---

## 13. Build the report

Two papers share `docs/eval_output/` fragments and `docs/bimodal_report.bib`:
`docs/main.tex` (full technical report) and `docs/lbw.tex` (6-page CHI
Late-Breaking-Work draft, reframed around the user study). The canonical
full-report source is `docs/main.tex` (`docs/bimodal_report.tex` is an
older copy that has not received the 2026-08-31 related-work / MELD / α-sweep
edits). The bibliography is BibTeX: `docs/bimodal_report.bib` (35 entries) with
the `ACM-Reference-Format` style that ships with `acmart`.

**Local build** (requires the `acmart` package):

```bash
cd docs
tectonic -X compile main.tex     # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
tectonic -X compile lbw.tex
```

**Overleaf build:**

1. Create a new project from the *ACM Conference Template*.
2. Upload `docs/main.tex` and `docs/bimodal_report.bib` at the project root.
3. Upload the contents of `docs/eval_output/` so that the `qual_images/`,
   `qual_images_ft/`, `qual_dissonant_images/`, and `qual_dissonant_images_ft/`
   folders and the `.tex` fragments (`alpha_sweep*.tex`, `meld_text_table.tex`,
   `likert_by_fused_label_table.tex`) sit under `eval_output/`.
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
