# Reading list — "When the affect sensor is wrong"

Papers backing `docs/lbw.tex`. Numbering matches the suggested reading order.
Downloaded PDFs are open-access (arXiv / author copies / ACL Anthology). Paywalled
ones are listed with DOIs — pull them through the TAMU-CC library proxy.

Read 01–03 first; they define the gap this project fills. 13 of 18 PDFs are in this folder; the remaining five are paywalled and listed with DOIs.

## Not downloadable here (get via library / DOI)

| # | Paper | Where | Why you need it |
|---|---|---|---|
| 02 | *Impact of Multimodal Emotional Input on User Experience in Chatbots* (2025). Springer LNCS | [10.1007/978-3-032-16454-4_17](https://doi.org/10.1007/978-3-032-16454-4_17) | Second null. N=28, four conditions incl. facial emotion. |
| 05 | Pataranutaporn, Liu, Finn et al. (2023). *Influencing human–AI interaction by priming beliefs about AI…* Nature Machine Intelligence 5, 1076–1086 | [10.1038/s42256-023-00720-7](https://doi.org/10.1038/s42256-023-00720-7) · [code/data](https://github.com/mitmedialab/nmi-ai-2023) | Belief prime alone raises perceived empathy. Constrains our consent wording. |
| 12 | Barrett, Adolphs, Marsella, Martinez & Pollak (2019). *Emotional Expressions Reconsidered.* Psych. Sci. Public Interest 20(1):1–68 | [10.1177/1529100619832930](https://doi.org/10.1177/1529100619832930) | Required caveat on inferring emotion from faces; cited in limitations and debrief. |
| 14 | Følstad, Law & van As (2024). *Conversational Breakdown in a Customer Service Chatbot.* TOCHI | [10.1145/3690383](https://doi.org/10.1145/3690383) | Error cost on trust/emotion by timing and criticality. |
| 18 | Concannon & Tomalin (2023). *Measuring perceived empathy in dialogue systems.* AI & Society | [10.1007/s00146-023-01715-z](https://doi.org/10.1007/s00146-023-01715-z) | Critique of ad-hoc empathy measures; justifies using PETS + per-trial items. |

## Downloaded

| # | File | Paper | Why |
|---|---|---|---|
| 01 | `01_schmidmaier2025_nonverbal_cues_empathic_llm.pdf` | Schmidmaier, Rupp, Harrich & Mayer (2025). *Using Nonverbal Cues in Empathic Multi-Modal LLM-Driven Chatbots for Mental Health Support.* PACM HCI ([10.1145/3743724](https://doi.org/10.1145/3743724)) | **The paper we argue against.** MLLM infers emotion from faces *accurately*; N=200; significant effects on the linguistic dimensions of responses but **no significant effect on perceived empathy**. |
| 03 | `03_schmidmaier2024_pets_scale.pdf` | Schmidmaier, Rupp, Cvetanova & Mayer (2024). *PETS.* CHI ([10.1145/3613904.3642035](https://doi.org/10.1145/3613904.3642035)) | Our measurement instrument. 30 experts + 324 + 396 participants, 22 opposing-empathy scenarios → 10-item, 2-factor scale. |
| 04 | `04_kosch2022_placebo_effect_ai.pdf` | Kosch, Welsch, Chuang & Schmidt (2022). *The Placebo Effect of AI in HCI.* TOCHI (arXiv 2204.04935) | Sham-condition logic; belief in adaptivity raises ratings with no adaptation. |
| 06 | `06_cao2024_chatbot_error_forgiveness.pdf` | Cao, Li & Jiang (2024). *When chatbots make errors.* Telematics and Informatics 94:102189 ([10.1016/j.tele.2024.102189](https://doi.org/10.1016/j.tele.2024.102189)) | Closest cost-of-error model. N=580, 2×2; anthropomorphism raises forgiveness via affective (severity, aversion) and cognitive (self-attribution) routes. Predicts H2 and a moderator. |
| 07 | `07_inoshita2026_llm_emotion_uncertainty.pdf` | Inoshita, Zhou, Kawai & Yada (2026). *LLMs Capture Emotion Labels, Not Emotion Uncertainty.* arXiv 2604.27345 | LLM emotion reads are overconfident → motivates the **ask** move. |
| 08 | `08_suh2025_sense7.pdf` | Suh, Le, Shayegani et al. (2025). *SENSE-7.* arXiv 2509.16437 | Per-turn user empathy annotation precedent; expectation violation as mechanism. |
| 09 | `09_cuadra2024_illusion_of_empathy_chi.pdf` | Cuadra, Wang, Stein et al. (2024). *The Illusion of Empathy?* CHI | LLMs strong on emotional reaction, weak on exploration/interpretation. |
| 10 | `10_liu2024_illusion_of_empathy_conversation.pdf` | Liu, Giorgi, Aich et al. (2024). arXiv 2411.12877 | Chatbots rated more fluent but **less empathic** than humans (155 convs). |
| 11 | `11_brun2025_emotion_sensitive_llm_cai.pdf` | Brun, Liu, Shukla, Watson & Gratch (2025). arXiv 2502.08920 | N=30; a *positive* presence-of-sensing result — contrast with 01/02. |
| 13 | `13_wobbrock2011_aligned_rank_transform.pdf` | Wobbrock, Findlater, Gergle & Higgins (2011). ART. CHI | If we add rank-based factorial ANOVA to the analysis. |
| 15 | `15_tahir2025_ether_incongruence.pdf` | Tahir, Johnson, Abu-Khalaf & Shah (2025). *E-THER.* arXiv 2509.02100 | Verbal–visual incongruence dataset; nearest work on masking. Dataset access: s.tahir@ecu.edu.au |
| 16 | `16_welivita2024_llms_more_empathetic.pdf` | Welivita & Pu (2024). *Are LLMs More Empathetic than Humans?* arXiv 2406.05063 | Baseline for LLM empathy ratings from text alone. |
| 17 | `17_poria2019_meld.pdf` | Poria, Hazarika, Majumder et al. (2019). *MELD.* ACL | The benchmark corpus in our Section 5. |

## Reading order

1. **01, 02, 03** — the nulls we extend and the scale we use.
2. **04, 05** — why belief/placebo forces the consent wording and the design.
3. **06, 07** — why the cost of a wrong label should exceed the gain, and why asking may soften it (H2, H3).
4. **08, 09, 10, 11** — how perceived empathy has been measured and what LLMs already achieve.
5. **12** — the caveat that must appear in limitations and debrief.
6. **13, 15, 16, 17** — method and benchmark support; read as needed.

Full annotated syntheses: `docs/research_guide_misattributed-empathy_2026-09-11.docx` (litreview, 437 papers screened) and `docs/storm/STORM_REPORT.md` (STORM web sweep). BibTeX keys for all of these live in `docs/bimodal_report.bib`.
