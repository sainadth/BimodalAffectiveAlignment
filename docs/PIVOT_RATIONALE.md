# Why the fusion paper cannot be submitted, and why the correctness study replaces it

Advisor memo. Sainadth Pagadala, 2026-09-28.
All numbers below are reproducible from `docs/eval_output/` in this repository.

---

## Summary

The original framing — *bimodal fusion enhances affect recognition* — is contradicted by
this project's own evaluation artifacts. This is a **validity** problem, not a
weak-results problem, so it cannot be fixed by tuning or by a larger model.

The same evaluation, read correctly, motivates a stronger study. Affect sensors are
wrong a large fraction of the time, and no published work has measured **what a wrong
emotion label costs the user**. Three published studies manipulated the *presence* of
affect sensing and disagree with each other. Presence is confounded with correctness.
The replacement study separates them.

---

## Part 1 — Three reasons the fusion result is not valid

### 1.1 The headline number is measured on fabricated text

The `bimodal congruent` condition pairs a FER2013 face image with **template text
generated from that image's ground-truth label**. The text branch is therefore reading a
sentence written from the answer key.

| Branch, on the FER+template set (n = 3,589) | Top-1 |
|---|---|
| Text branch alone | **0.8256** |
| Face branch alone (base ResNet-50) | 0.2310 |
| Face branch alone (fine-tuned) | 0.6849 |
| Bimodal, base, α = 0.50 | 0.8158 |
| Bimodal, fine-tuned, α = 0.50 | 0.8780 |

Two things follow. First, `0.8256` is not a language-understanding result — it measures
how faithfully the template generator encoded the label. Every fusion number built on top
of it inherits that circularity. Second, **base bimodal (0.8158) is *worse* than the text
branch alone (0.8256)**. Adding the face actively hurt.

The fine-tuned bimodal number, 0.8780, is +5.2 pp over text alone and is the number the
draft leans on. It is measured on the same fabricated pairing, and the face model was
fine-tuned on FER2013 train and scored on FER2013 test. It does not generalise a claim
about conversation.

### 1.2 The α-sweep shows the optimal fusion is "delete the face"

α is the trust weight on the text branch; α = 1.0 means the face is ignored entirely.

Base model, `docs/eval_output/alpha_sweep.csv`:

```
α     0.00   0.25   0.50   0.55   0.60 → 1.00
acc   0.2310 0.4717 0.8158 0.8253 0.8256 (flat, identical to 4 d.p.)
```

Accuracy increases monotonically in α and **plateaus at exactly 0.8256 from α = 0.60
through α = 1.00** — the text-only value, with overlapping 95% CIs throughout. There is no
interior optimum. The best achievable setting of the fusion parameter is the one that
discards the second modality.

This is the cleanest available statement that the face contributes nothing here, and it
was produced by our own sweep, not by a critic.

(The fine-tuned sweep does peak in the interior, α = 0.50 at 0.8780 — but on the
fabricated pairing of §1.1, and it too decays to 0.8256 by α = 0.75.)

### 1.3 On genuinely paired data the pipeline sits at the majority-class floor

MELD (Poria et al., ACL 2019) is real televised dialogue: real faces paired with the real
utterances spoken with them. Test set n = 2,610.

| System | Top-1 | Macro-F1 |
|---|---|---|
| Majority class ("always Neutral") | **0.4812** | 0.093 |
| Text-only (GoEmotions 28 → 7, sum collapse) | 0.4299 | 0.278 |
| Collapse: max | 0.4372 | 0.274 |
| Collapse: mean | 0.5088 | 0.253 |
| Logistic 28 → 7 learned on MELD train | **0.5559** | 0.292 |
| Logistic, class-balanced | 0.4559 | 0.318 |
| Claude Opus 5, zero-shot, text only | 0.5318 | — |

Text-only lands **below** the majority-class floor. The best configuration we have beats
"always say Neutral" by 7.5 pp with macro-F1 0.292, which means it is largely predicting
the majority class. The 0.88 headline and this 0.56 are not one system measured two ways;
the first measures a synthetic construction.

### 1.4 What this means for the draft

`docs/main.tex` is titled *"Enhancing … through Fusion."* Its own α-sweep refutes it for
all α ≥ 0.60, and its own MELD table refutes it outright. Submitting under that title
would assert something the artifacts in the same repository disprove. The title and claim
must change regardless of which study we run next.

---

## Part 2 — Why "label correctness × conversational move" is the right replacement

### 2.1 The negative result reframes the question

The uninteresting reading is "our sensor is weak." The useful reading is: **affect sensors
are wrong a substantial fraction of the time in deployment, and nobody has measured what
that costs the person on the other end.**

### 2.2 The literature manipulated presence, not correctness — and disagrees

| Study | N | Manipulation | Result |
|---|---|---|---|
| Schmidmaier et al. 2025, PACM HCI | 200 | text / placebo camera / real FER / FER + proactive | **Null.** PETS medians 79.2 / 78.6 / 82.5 / 82.2, *H* = 1.72, *p* = .632 |
| *Multimodal Emotional Input*, Springer LNCS 2025 | 28 | text / self-report / facial / both | Perceived empathy slightly **higher without** facial input |
| Brun et al. 2025 | 30 | emotion-sensitive vs insensitive service bot | **Positive.** Trust and competence up; task resolution unchanged |

Schmidmaier 2025 is the decisive one, and it is stronger than a typical null: they
**separately validated that GPT-4 read the faces accurately**, and the facial input
demonstrably changed the *language* of the replies. So "the face reader was too weak"
cannot explain their flat result. That closes off the obvious rescue — and, importantly,
it closes off Path A below.

### 2.3 The confound nobody has separated

In all three studies the sensor was right on some trials and wrong on others, and every
result is averaged across both. If a correct label helps and a wrong label hurts, the
average is approximately zero — **which is exactly the null Schmidmaier reports.**

Separating them requires *assigning* correctness rather than measuring it. That is what
the replacement study does.

### 2.4 The design

3 × 2 within-subjects. Factor 1: what the generator is told about the participant's
emotion — `correct` (their own self-report), `wrong` (opposite valence), `none`.
Factor 2: conversational move — `assert` ("you seem angry") vs `ask` ("are you feeling
angry?"). 12 trials, Williams Latin square, n = 42. Protocol: `docs/STUDY_PROTOCOL.md`;
pre-registration: `docs/OSF_PREREGISTRATION.md`.

### 2.5 Seven reasons this design is defensible

1. **It does not depend on our sensor being accurate** — which is the whole point given
   Part 1. Ground truth is the participant's self-report; the sensor is logged only for
   exploratory analysis. Barrett et al. 2019 argue facial configurations map to emotion
   categories far less reliably than the basic-emotion view assumes, so *any* design that
   requires a face sensor to be right is built on sand.
2. **The placebo is controlled by construction.** Kosch et al. 2022 (TOCHI) showed that
   merely believing an AI adapts to you changes behaviour and ratings. Every trial in
   every cell captures the webcam; only what the generator is told afterwards varies.
   Schmidmaier needed a dedicated placebo arm — our design does not.
3. **Fluency cannot explain an effect.** Liu et al. 2024 found chatbots score *higher*
   on conversational quality yet *lower* on empathy — the two dissociate. We hold the
   generator, prompt template and decoding fixed and vary only the label and the move.
4. **`ask` is theoretically motivated, not an arbitrary contrast.** Inoshita et al. 2026
   (640k responses over GoEmotions and EmoBank) show LLMs reproduce the majority emotion
   label but not human disagreement — they cannot represent their own uncertainty. A
   system that cannot know it is unsure should not assert. Cuadra et al. 2024 (CHI) found
   agents score high on emotional *reaction* but low on *interpretation* and
   *exploration*: asking is the systematically under-used move.
5. **The error-cost mechanism has a published model to extend.** Cao et al. 2024
   (N = 580) show chatbot errors cost via perceived severity and emotional aversion, with
   anthropomorphism buffering. Følstad et al. 2024 show the cost depends on *when* the
   error lands — which is why position is balanced by the Williams square.
6. **Per-turn self-annotated empathy is an established DV.** Suh et al. 2025 (SENSE-7:
   695 conversations, 109 information workers, per-turn self-annotation) found an LLM
   classifier reached only ρ = 0.369 predicting those judgements — you have to ask the
   person, per turn.
7. **The instrument is the validated one.** PETS (Schmidmaier et al. 2024, CHI) on
   101-point sliders, the same scale as the null we are extending, so our numbers are
   directly comparable to it.

### 2.6 The pilot already caught two invalidating bugs

Evidence the instrument is being checked, not assumed:

- **Prompt infidelity.** The `ask` cells silently substituted vaguer words: 4/4 `assert`
  trials used the exact word "angry", 0/4 `ask` trials did — all said "upset". H3 would
  have come out true by *wording* rather than by *move*. Both prompts now demand the exact
  label word; re-verified 3/3 per cell.
- **Valence-ambiguous wrong labels.** Surprise → Happy under the `wrong` condition
  produced ceiling ratings, because a positively surprised participant does not experience
  "happy" as wrong. Wrong labels now always cross valence.
- **Ceiling on 5-point items.** 8 of 12 pilot trials scored 5.0 on every item. Ratings
  moved to continuous 0–100 sliders, matching Schmidmaier's 101-point PETS.

---

## Part 3 — Paths considered and rejected

| Path | Why rejected |
|---|---|
| **A. Improve the fusion** (better face model, tuned α, MLLM vision) | Our own α-sweep puts the optimum at α ≥ 0.60, i.e. at text-only; the ceiling *is* the text branch. And Schmidmaier 2025 showed an *accurate* face reader still moves perceived empathy not at all — so even winning the recognition race buys no HCI contribution. |
| **B. Replicate presence-of-sensing** | Done three times already (Schmidmaier 2025, Springer 2025, Brun 2025) with contradictory outcomes. We would be underpowered against N = 200 and would add a fourth inconsistent data point rather than an explanation. |
| **C. Verbal–visual incongruence detection** (E-THER, Tahir et al. 2025) | Nearest work on masking, and interesting — but it is another *recognition* paper, and the dataset requires author request. Same validity exposure as Path A. |
| **D. Label correctness × conversational move** | **Chosen.** Explains the existing inconsistency rather than adding to it; independent of our sensor's accuracy; placebo and fluency controlled by construction; within-subjects, so n = 42 is adequate. |

---

## Reference list

Full PDFs and DOIs in `LR/README.md`; per-paper relevance notes in `LR/LITERATURE_MAP.md`.

1. Schmidmaier, Rupp, Harrich & Mayer (2025). *Using Nonverbal Cues in Empathic Multi-Modal LLM-Driven Chatbots.* PACM HCI.
2. *Impact of Multimodal Emotional Input on User Experience in Chatbots* (2025). Springer LNCS.
3. Schmidmaier, Rupp, Cvetanova & Mayer (2024). *PETS: Perceived Empathy of Technology Scale.* CHI.
4. Kosch, Welsch, Chuang & Schmidt (2022). *The Placebo Effect of AI in HCI.* TOCHI.
5. Pataranutaporn, Liu, Finn et al. (2023). *Influencing human–AI interaction by priming beliefs.* Nature Machine Intelligence.
6. Cao, Li & Jiang (2024). *When chatbots make errors.* Telematics and Informatics.
7. Inoshita, Zhou, Kawai & Yada (2026). *LLMs Capture Emotion Labels, Not Emotion Uncertainty.* arXiv.
8. Suh, Le, Shayegani et al. (2025). *SENSE-7.* arXiv.
9. Cuadra, Wang, Stein et al. (2024). *The Illusion of Empathy?* CHI.
10. Liu, Giorgi, Aich et al. (2024). *The Illusion of Empathy: How AI Chatbots Shape Conversation Perception.* arXiv.
11. Brun, Liu, Shukla, Watson & Gratch (2025). *Exploring Emotion-Sensitive LLM-Based Conversational AI.* arXiv.
12. Barrett, Adolphs, Marsella, Martinez & Pollak (2019). *Emotional Expressions Reconsidered.* Psychological Science in the Public Interest.
13. Wobbrock, Findlater, Gergle & Higgins (2011). *The Aligned Rank Transform.* CHI.
14. Følstad, Law & van As (2024). *Conversational Breakdown in a Customer Service Chatbot.* TOCHI.
15. Tahir, Johnson, Abu-Khalaf & Shah (2025). *E-THER.* arXiv 2509.02100.
16. Welivita & Pu (2024). *Are LLMs More Empathetic than Humans?* arXiv 2406.05063.
17. Poria, Hazarika, Majumder et al. (2019). *MELD.* ACL.
