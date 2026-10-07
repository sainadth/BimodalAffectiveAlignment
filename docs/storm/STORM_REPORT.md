# STORM run — 2026-09-11

Topic: *Misattributed emotion in LLM chatbot replies: label correctness, asserting versus asking, and perceived empathy.*
Pipeline: Stanford STORM (`knowledge-storm` 1.1.1) via `scripts/run_storm.py`; LLM = `openai/gpt-oss-120b` on Groq (reasoning_effort low); retrieval = DuckDuckGo via `ddgs`, 4 perspectives × 3 turns, 36 queries, 32 unique URLs. Runtime ≈ 2 min. Output in `misattributed_emotion_llm_replies/` (`storm_gen_outline.txt`, `storm_gen_article_polished.txt`, `url_to_info.json`, `conversation_log.json`).

## Verdict

Useful as a **second, question-driven sweep** after the systematic litreview; not usable as text or as a bibliography. The article invents an established field ("the phenomenon has emerged as a central research focus") and states as fact the very hypothesis we are testing ("an interrogative style ... tends to improve perceived empathy [8][9]") with sources that do not show it. Its outline is a generic Wikipedia skeleton. Value came from five sources the PubMed/OpenAlex lane and our earlier web searches had not surfaced.

## Verified new sources (added to `docs/bimodal_report.bib`)

| Key | What it is | Why it matters here |
|---|---|---|
| `inoshita2026uncertainty` | LLMs reproduce majority emotion labels but not annotator disagreement; zero-shot models diverge; lexical vs pragmatic emotions (GoEmotions, EmoBank; 640k responses) | Direct support for the *ask* move: the model cannot know when its emotion read is uncertain, so asking is the calibrated default. Also explains our text branch's Neutral failures (pragmatic, not lexical). |
| `cao2024forgiveness` | Telematics & Informatics 94 (2024). Forgiveness of chatbot errors via cognitive and affective pathways; anthropomorphism increases forgiveness; context-dependent | The closest published "cost of error" model for chatbots. Frames H2 (asymmetry) and predicts a moderator: anthropomorphic framing may soften the wrong-label penalty. |
| `liu2024illusionconversation` | 155 conversations; GPT chatbots rated higher on conversational quality but *lower* on empathy than humans; GPT-4o judge agrees with users | Fluency ≠ perceived empathy — supports keeping the generator fixed and varying only what it is told. |
| `suh2025sense7` | SENSE-7: 695 real conversations, 109 workers, per-turn user empathy annotations; empathy judgments individual and context-sensitive; breaks when expectations are unmet | Precedent for per-turn empathy ratings by the user (our primary DV) and for expectation violation as the mechanism behind a wrong-label penalty. |
| `brun2025emotionsensitive` | N=30; emotion-sensitive vs insensitive customer-service LLM bot; higher trust/competence, same issue resolution | Another presence-of-sensing study with a positive perception effect and no task effect; contrasts with the Schmidmaier/Springer nulls. |

## Retrieved but not useful
Prompt-engineering blogs (EmotionPrompt, learnprompting), CBT "labeling" self-help pages, a Washington Post survey piece, a CS224N student project, RAG-ConfusionQA (unrelated benchmark), several sentiment-analysis surveys. `Measuring perceived empathy in dialogue systems` (Concannon & Tomalin 2023) and `Empathy by Design` (2025) were already in the litreview.

## Claims in the STORM article to ignore
- "Empirical studies demonstrate that an assertive style ... magnifies the negative impact of misattribution, whereas an interrogative style ... tends to improve perceived empathy" — no source among the 32 URLs tests this. This is our hypothesis, not a finding.
- "Benchmarks such as RAG-ConfusionQA and LongEmotion are being used to evaluate ... the model's ability to choose the appropriate response style" — RAG-ConfusionQA is about confusing questions in RAG, not affect.
- "CES-LCC consistently shows that misattributed emotions lower empathy ratings" — the scale paper validates a scale; it does not manipulate misattribution.

## What to carry into the LBW related work
1. Uncertainty: LLM emotion reads are overconfident by construction (`inoshita2026uncertainty`) → asking as calibrated delivery.
2. Error cost: chatbot errors are forgiven via affective/cognitive routes moderated by anthropomorphism (`cao2024forgiveness`) → predicts asymmetry and a moderator.
3. Fluent ≠ empathic (`liu2024illusionconversation`) and expectation violation breaks perceived empathy (`suh2025sense7`) → mechanism for the wrong-label penalty.
4. Presence-of-sensing results are mixed (`brun2025emotionsensitive` positive; Schmidmaier 2025 and the N=28 study null) → why correctness, not presence, is the variable.

Reproduce: `python scripts/run_storm.py --out docs/storm`. Gotchas: `knowledge-storm` downgrades `datasets` (run `pip install "datasets>=2.16"` after); the old `duckduckgo_search` client returns no results — the script subclasses the retriever onto `ddgs`.
