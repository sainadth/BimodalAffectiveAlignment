# Advisor email — approval to file IRB

Draft for Dr. Minhua Huang. Assumes 24 participants in the first wave under a 42-person
approval, and that webcam frames are not retained (decided 2026-10-05).

---

**Subject:** Approval to file IRB application for the participant study

Dr. Huang,

I'd like to start collecting participant data for the HCI part of the thesis, which means filing with IRB. Could you look this over and tell me whether you approve?

The fusion results didn't hold up, so rather than keep pushing on recognition accuracy I want to look at what happens when the sensor gets the user wrong. Schmidmaier et al. (2025) ran something close to this with N=200. They gave GPT-4 facial input they had verified was accurate and found no effect on perceived empathy. The thing is, their trials mixed turns where the guess was right with turns where it was wrong and reported the average. Two other studies disagree with each other about whether emotion sensing helps at all, which is about what you'd see if accuracy mattered more than presence.

I also want to test whether phrasing changes the damage. Inoshita et al. showed that language models reproduce the majority emotion label but not the disagreement among human annotators, so they have no way to represent their own uncertainty. Cuadra et al. (CHI 2024) found agents are good at reacting emotionally and bad at exploring, meaning they tell you how you feel far more often than they ask. So the question is whether asking is cheap insurance. If it costs nothing when the guess is right and helps when it's wrong, that's a phrasing rule any system working from an inferred emotion could adopt.

Design is within-subjects, 3 x 2. The model gets told either the emotion the participant just reported, one of opposite valence, or nothing at all, and that's crossed with whether the reply asserts the emotion or asks about it. Twelve trials each, roughly 18 minutes, done remotely on their own computers.

I'd like to run 24 participants first, unpaid, recruited from people I know. That's 288 trials, but the analysis is within-person, so 24 is what sets the power. It's enough for a large effect on the main contrast, which is what I'd expect a wrong emotion label to produce, and not enough for the interaction between accuracy and phrasing. If the first wave looks worth continuing I'd extend to 42, so I'd rather ask IRB for 42 now than file an amendment later.

What we record: the message they write, the emotion they report, their ratings, and the labels our software estimates, all under a participant number with no name attached. Webcam frames and any dictated audio are used to produce one label and then discarded in memory. Nothing goes to disk.

Two points IRB will want to look at. Participants are told upfront that some replies are deliberately based on an inaccurate guess, though not which ones, and the full explanation comes in the debrief with one click to delete their data. Two of the twelve trials ask them to recall a small frustration, which they can decline.

References, if you want to see where this sits:

Schmidmaier et al. (2025), the null we're extending
https://doi.org/10.1145/3743724

Schmidmaier et al. (2024), the PETS scale we measure with
https://doi.org/10.1145/3613904.3642035

Brun et al. (2025), the positive sensing result that contradicts it
https://arxiv.org/abs/2502.08920

Inoshita et al., on model uncertainty, which is what motivates the ask condition
https://arxiv.org/abs/2604.27345

Could you let me know whether you approve, and roughly how long IRB usually takes? I'm watching the calendar against graduation. The protocol, pre-registration, and consent text are ready whenever you want them.

Thanks,
Sainadth

---

## Before sending

- Fill the two placeholders in `server/debrief_remote.md:7`: `[counselling centre contact]`
  and `[email]`. IRB reads the debrief.
- Decide the sample size of record. `docs/OSF_PREREGISTRATION.md` and
  `docs/STUDY_PROTOCOL.md` both still say n = 42; this email proposes 24 first.
- Confirm the Cuadra et al. (2024) citation. `LR/README.md` records no DOI for it, so it
  appears here as a text citation with no link.
- Check whether CITI human-subjects training is current for both of you. Most offices
  will not accept a submission without it, and the course takes several hours.

## Reasoning behind the numbers

Why 24 rather than 20: the Williams square has six rows, so the sample has to be a multiple
of six or condition order goes unbalanced. Minimum detectable effect, paired, 80% power:

| n | α = .05 | after Holm (α ≈ .008) | multiple of 6 |
|---|---|---|---|
| 18 | 0.66 | 0.82 | yes |
| 20 | 0.63 | 0.78 | no |
| 24 | 0.57 | 0.71 | yes |
| 42 | 0.43 | 0.54 | yes |

Full argument for the pivot away from fusion: `docs/PIVOT_RATIONALE.md`.
