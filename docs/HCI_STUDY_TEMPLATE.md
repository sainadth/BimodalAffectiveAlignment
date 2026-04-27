# HCI evaluation template (Project.pdf §3.2)

Use this in your report; fill in or delete sections as your instructor allows.

## Affective congruency (expert or self review)

For each of the 3 hand-written rows in `data/dissonant_samples.csv` (or 3 cases from the Streamlit demo), answer in one line:

- Does the model’s fused label **b\*** and response **R** feel aligned with the user’s *true* state as you defined it in your narrative?

| Example (utterance) | Fused b* | R feels congruent? (Y/N) | Notes |
|---------------------|----------|----------------------------|--------|
|                     |          |                            |        |

## Likert (5 = strongly agree)

If you run a small user study, copy for each participant. If you only have expert review, n=0 is acceptable in some course policies—**confirm with the instructor**.

**Empathy** (1–5): The agent’s responses felt understanding and warm.

| Participant | Score |
|---------------|------:|
| 1             |       |

**Social presence** (1–5): I felt the agent was *there* in the conversation.

| Participant | Score |
|---------------|------:|
| 1             |       |

## Limitations (if no user study)

State explicitly: e.g. no formal participants, qualitative check only on the example table, latency measured on one machine via `scripts/benchmark_latency.py` or the Streamlit ms line.

## Interaction latency (for report)

Run `python scripts/benchmark_latency.py` (or use Streamlit’s end-to-end ms) and state hardware (CPU / MPS / GPU). Compare the mean to the ~**400 ms** Doherty threshold in prose.
