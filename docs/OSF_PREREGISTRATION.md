# OSF pre-registration draft v3 (AsPredicted-style)

Register **before the first non-pilot participant**; paste the DOI into the paper.
Full protocol: `docs/STUDY_PROTOCOL.md` v3.

**1. Data collection.** None yet. A 4-person pilot will be run first and excluded.

**2. Hypotheses.**
H1: Replies conditioned on the participant's self-reported emotion (correct
label) are rated higher on a 3-item perceived-empathy composite than replies
conditioned on a label of opposite valence (wrong label), pooled over
conversational move.
H2 (asymmetry): Per participant, the penalty of a wrong label
(none − wrong) exceeds the gain of a correct one (correct − none).
H3: Among wrong-label trials, replies that *ask* about the emotion are rated
higher than replies that *assert* it.
H3b (exploratory): main effect of move (ask > assert) pooled over label source.

**3. Dependent variable.** Per-trial composite = mean of three 0–100 slider items ("The
reply matched how I actually felt", "The reply felt warm and understanding",
"The system considered my mental state"). Secondary: each item; per-trial
manipulation check ("Which emotion did the agent think you were feeling?");
PETS-10 once per participant on the same 0–100 sliders (descriptive).

**4. Conditions.** Within-subjects 3 × 2: label source {correct, wrong, none}
× move {assert, ask}; 2 blocks × 6 = 12 trials; Williams Latin square (6 rows)
keyed by participant number; one scripted affect-elicitation trial per block
with rotating cell. Same LLM (openai/gpt-oss-120b via Groq, reasoning_effort
low, temperature 0.7) and system prompt in every cell.

**5. Analyses.** Per participant × cell means (complete-case over six cells;
trials where the participant declined to report emotion are excluded).
Friedman omnibus across cells. Holm-corrected Wilcoxon signed-rank tests with
matched-pairs rank-biserial r and 95% bootstrap CIs on exactly six contrasts:
correct − wrong; penalty − gain; wrong_ask − wrong_assert;
correct_assert − none_assert; none_assert − wrong_assert; ask − assert.
Sensitivity analysis restricted to trials where the label was both delivered
(`label_in_reply`) and perceived (manipulation check).
Code frozen at registration: `scripts/analyze_study.py`,
`src/bimodal_empathy/eval/study_stats.py` (commit: ______).

**6. Outliers and exclusions.** Exclude pilot participants; participants who
withdraw at debrief; participants with fewer than 12 logged trials from
inferential tests (kept in descriptives). No trial-level exclusions except
`correctness_effective = sensor`. Failed generations are retried; attempts
logged.

**7. Sample size.** n = 42 (multiple of 6). 80% power, two-sided α = .05, for
paired d ≈ 0.45 before correction; H3 is powered only for medium-to-large
effects and is reported with CIs regardless.

**8. Other.** Sensor outputs (text, face, fused labels; dissonance flags) are
logged but do not drive the manipulation; exploratory analyses of sensor
accuracy vs self-report are labelled as such. Known limitation: only the
"assert" prompt names an emotion as fact; "ask" names it inside a question —
the move manipulation is therefore partly a wording manipulation, stated in
the paper.

**9. Study type.** Experiment, within-subjects, single-session, remote (web
app with webcam) or lab (Streamlit app), identical logging.
