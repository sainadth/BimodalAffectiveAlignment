# Study Protocol v3 — When the affect sensor is wrong: correctness, conversational move, and perceived empathy

Status: draft v3 (2026-09-11), supersedes v2. Register on OSF before the first
non-pilot participant (`docs/OSF_PREREGISTRATION.md`). Advisor + IRB sign-off
required before recruiting.

## Why this design

Two prior studies fed an LLM chatbot a facial-emotion label on every turn and
found no significant gain in perceived empathy (Schmidmaier et al. 2025, N=200;
a four-condition N=28 study, 2025). Our own benchmarks show off-the-shelf affect
sensors are wrong on natural input more often than they are right. A null
average is exactly what you would expect if correct reads help a little and
wrong reads hurt a lot. Nobody has manipulated **correctness** directly, and
nobody has tested whether the agent's **conversational move** — asserting the
inferred emotion versus asking about it — changes the cost of being wrong.
The literature review in `docs/research_guide_misattributed-empathy_2026-09-11.docx`
(437 papers screened) found no such study.

## Research questions

1. How much does a correct emotion label add to perceived empathy over an LLM
   reading the message alone, and how much does a wrong one subtract?
2. Is the effect asymmetric (penalty > gain)?
3. Does asking instead of asserting reduce the penalty of a wrong label?

## Hypotheses (pre-specified)

- **H1** composite(correct) > composite(wrong), pooled over move.
- **H2 (asymmetry)** per participant, penalty = none − wrong exceeds gain =
  correct − none.
- **H3** composite(wrong_ask) > composite(wrong_assert).
- **H3b (exploratory)** main effect of move, ask vs assert, pooled.
- **Honest nulls:** correct ≈ none means the sensor adds nothing when right;
  no asymmetry means errors are cheap; ask ≈ assert means the move is a free
  choice. Each is reportable and design-relevant.

## Design

Within-subjects, **2 × 3 factorial**:

| Factor | Levels | What the generator is told |
|---|---|---|
| Label source | `correct` | the emotion the participant just reported |
| | `wrong` | a label of **opposite valence** (Happy ↔ a negative label; Neutral/Surprise → Happy or Sad, alternating) |
| | `none` | nothing; the LLM reads the message |
| Move | `assert` | "state how they seem to feel, then respond" |
| | `ask` | "do not state their emotion as fact; check in with one question, then support" |

- **12 trials** per participant = 2 blocks × 6 cells. Williams Latin square
  (6 rows) keyed by participant number; the two blocks use different rows.
  Across every 6 consecutive participants each cell starts block 1 once.
  Recruit in multiples of 6 (target **n = 42**; 36 minimum).
- **Felt-emotion report** (7 labels + "Prefer not to say") is collected after
  the message is sent and **before** the reply is generated; it defines
  "correct". If a participant declines, the sensor's fused label is used and
  the trial is logged `correctness_effective = sensor` and excluded from the
  inferential tests.
- **Scripted elicitation**: one trial per block (rotating cell) asks the
  participant to recall a small frustration (block 1) or something looked
  forward to (block 2) while writing a flat message. Logged as `scripted`.
- **Generator fixed:** `openai/gpt-oss-120b` via Groq, `reasoning_effort=low`,
  temperature 0.7, one system prompt (`SYSTEM_PROMPT_V3`). The move and label
  are the only things that vary. The app refuses to start otherwise.
- **Sensors still run** (RoBERTa text, YuNet-cropped ResNet-50 face, fusion,
  dissonance flags) and are logged for exploratory analysis; they never drive
  the manipulation in v3.
- **Blinding:** participants see only their message, the reply, and the rating
  items. Consent states replies are generated in several ways and some are
  deliberately based on an inaccurate guess.

## Measures

- **Per trial (primary DV)**, continuous 0–100 slider (no numeric readout, no handle until touched, anchors "strongly disagree"/"strongly agree"): "The reply matched how I actually felt";
  "The reply felt warm and understanding"; "The system considered my mental
  state" (PETS-ER1). Composite = mean.
- **Per trial manipulation check**: "The agent seemed to think I was feeling…"
  (7 labels + "It did not say" + Unsure). Success = matches the shown label
  (wrong/correct) or "It did not say"/own label (none). Delivery is checked
  separately and objectively: `label_in_reply` records whether the generated reply
  literally contained the emotion word, so a failed check separates "the model did
  not say it" from "the participant did not notice".
- **Once per participant**: PETS-10 on the same 0–100 sliders, as validated
  (Schmidmaier et al. 2024 use 101-point sliders; our earlier 5-point version was a
  deviation and is dropped), blinding funnel (No/Unsure/Yes + free text).
- **Logged**: cell, label source, move, effective correctness, felt label,
  shown label, manipulation-check label, sensor labels and dissonance flags,
  generator, attempts, latency, text excerpts. No images.

## Participants and recruitment

Adults 18+, n = 42. Exclusion: project members; self-reported current
treatment for a mood or anxiety disorder (screened on the sign-up form, reason
not recorded). Recruitment by the student researcher via department lists and
posters, never through a course taught by the co-author; consent states
instructors do not see who took part. No compensation; participation is unpaid and voluntary.

## Procedure (~18 min, remote)

1. Participant opens their unique link (`/p/<token>`), reads consent, agrees
   (version + timestamp logged).
2. 12 turns: [scripted instruction if applicable] → write the message, by typing
   or by optional dictation (Whisper on Groq; the transcript lands in the text box
   and stays editable, so the logged utterance is always text the participant
   approved; audio is discarded and `input_method` is logged) → webcam burst →
   Send → felt-emotion report → reply generated and shown → 3 slider ratings +
   manipulation check → Save.
3. PETS-10 + funnel.
4. In-app debrief disclosing the three label sources and two moves, the
   deliberate wrong labels, the contested status of face-based emotion
   inference, counselling contact, and **Keep / Withdraw** (withdraw deletes
   all rows immediately).

A closed tab resumes at the first unlogged turn. Lab fallback:
`app/study_app.py`, same design, same CSVs.

## Power

Paired contrasts, two-sided α = .05, 80%: n ≈ 34 for d = 0.5, n ≈ 42 for
d = 0.45, n ≈ 24 for d = 0.6. Holm across 6 pre-specified tests makes the
smallest effective α ≈ .008. **n = 42 is powered for medium effects on H1 and
H2; H3 (a difference of differences) is powered only for medium-to-large
effects and is reported with CIs regardless.**

## Analysis plan (pre-specified; `scripts/analyze_study.py`)

1. Per participant: mean composite per cell (2 trials each), per label source
   (4 trials), per move (6 trials). Complete-case across the six cells;
   `correctness_effective = sensor` rows excluded.
2. Friedman across the six cells (omnibus).
3. Holm-corrected Wilcoxon signed-rank tests, matched-pairs rank-biserial r and
   mean differences with 95% bootstrap CIs, on exactly: correct–wrong (H1);
   penalty–gain (H2); wrong_ask–wrong_assert (H3); correct_assert–none_assert;
   none_assert–wrong_assert; ask–assert (H3b).
4. Manipulation check: proportion exact and same-valence matches per label
   source; H1–H3 repeated on manipulation-check-passed trials as a sensitivity
   analysis.
5. Same on each item separately and on scripted trials (descriptive if
   incomplete).
6. Exploratory: sensor accuracy vs felt report; whether sensor-correct trials
   in the `none` condition differ from sensor-wrong ones (what a real deployed
   sensor would have done).
7. Cronbach's α of the composite on participant means; PETS descriptives and
   subscale α; funnel tabulation and coding of free text.
8. No analysis decisions after seeing data; anything else is labelled
   exploratory.

## Ethics

- **Deception**: wrong-label replies deliberately misjudge the participant;
  consent discloses that some replies are based on an inaccurate guess without
  saying which; full debrief with immediate withdrawal.
- **Mood induction**: two mild recall prompts; neutral substitution allowed;
  counselling resources in debrief; exclusion criterion above.
- **Data**: frames and any dictation audio processed in RAM and discarded; the
  microphone is requested only if the participant presses the speak button; text to Groq (cite retention
  terms); CSVs on encrypted storage; token→number file kept with consent
  records; excerpts redacted for third-party names before analysis; server
  access logs purged after the study.
- **Biometrics**: transient face geometry for detection only; check Texas CUBI
  applicability with the IRB. Do not assume exempt review.

## Timeline

- Sept 2026: advisor sign-off, IRB submission, OSF registration, 4-person pilot
  (excluded).
- Oct 2026: n = 42 remote.
- Nov 2026: analysis, writing, adversarial review.
- Jan 2027: CHI 2027 Late-Breaking Work.
