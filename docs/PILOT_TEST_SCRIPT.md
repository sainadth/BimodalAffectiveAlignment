# Pilot test script — design v3

Walkthroughs for the researcher (or a pilot participant who has been told the design).
**Real participants must never see this file** — it reveals the manipulation.

## Setup

```bash
# from the repo root, with .env holding GROQ_API_KEY and LLM_PROVIDER=groq
SAVE_FRAMES_DIR=data/frames uvicorn server.app:app --host 0.0.0.0 --port 8000
python scripts/make_participant_links.py --n 4 --base http://localhost:8000
```

`SAVE_FRAMES_DIR` writes each burst and its YuNet crops to
`data/frames/<pid>/t<NN>_<condition>/` with a `meta.json` sidecar (felt label, shown
label, manipulation check, ratings, per-frame face probabilities, reply, generator).
**Unset it before real participants** — the consent text states frames are never stored,
and the server prints a warning at startup while it is on.

Reset between pilot runs:

```bash
rm -rf data/frames data/study_responses.csv data/study_pets_global.csv
```



## How a turn works

1. Type a message, camera takes 3 frames on **Send** (sensors run; nothing is shown).
2. Report **how you actually feel** — this defines the "correct" label for the trial.
3. The reply is generated per the hidden cell for that turn, then shown.
4. Drag three 0–100 sliders and answer the manipulation check. Sliders show no
   number and no handle until touched, so nothing anchors you to the middle.

Cells: label source (`correct` = your reported emotion, `wrong` = opposite valence,
`none` = generator told nothing) × move (`assert` = states the emotion, `ask` = asks
about it using the same word).

### p01 — `/p/<token>`


| Turn | Cell               | Type this                                                  | Report feeling | Expected label in reply |
| ---- | ------------------ | ---------------------------------------------------------- | -------------- | ----------------------- |
| 1    | `correct_assert` ✱ | I'm fine, everything's good.                               | **Angry**      | **angry** (right)       |
| 2    | `none_ask`         | Had a long day of classes, nothing special.                | **Happy**      | *its own read*          |
| 3    | `correct_ask`      | Honestly I could really use a break right now.             | **Sad**        | **sad** (right)         |
| 4    | `none_assert`      | Just finished lunch, back to work soon.                    | **Neutral**    | *its own read*          |
| 5    | `wrong_assert`     | Got some really good news today!                           | **Happy**      | **angry** ← WRONG       |
| 6    | `wrong_ask`        | I keep thinking about something that went wrong last week. | **Sad**        | **happy** ← WRONG       |
| 7    | `wrong_ask` ✱      | Today was completely ordinary.                             | **Neutral**    | **happy** ← WRONG       |
| 8    | `wrong_assert`     | Stay away from me.                                         | **Angry**      | **happy** ← WRONG       |
| 9    | `none_assert`      | I did not see that coming at all.                          | **Surprise**   | *its own read*          |
| 10   | `correct_ask`      | I have a presentation tomorrow and I'm dreading it.        | **Fear**       | **fear** (right)        |
| 11   | `none_ask`         | Weekend plans are finally sorted, can't wait.              | **Happy**      | *its own read*          |
| 12   | `correct_assert`   | That whole situation left a bad taste.                     | **Disgust**    | **disgust** (right)     |




### p02 — `/p/<token>`


| Turn | Cell             | Type this                                                     | Report feeling | Expected label in reply |
| ---- | ---------------- | ------------------------------------------------------------- | -------------- | ----------------------- |
| 1    | `correct_ask` ✱  | Great start to the morning.                                   | **Happy**      | **happy** (right)       |
| 2    | `correct_assert` | Everything feels like a lot right now.                        | **Sad**        | **sad** (right)         |
| 3    | `wrong_assert`   | Nothing much to report today.                                 | **Neutral**    | **happy** ← WRONG       |
| 4    | `none_ask`       | I'm so done with this group project.                          | **Angry**      | *its own read*          |
| 5    | `wrong_ask`      | Results come out tomorrow and I can't stop thinking about it. | **Fear**       | **happy** ← WRONG       |
| 6    | `none_assert`    | Finally finished the thing I was stuck on!                    | **Happy**      | *its own read*          |
| 7    | `none_assert` ✱  | I miss how things used to be.                                 | **Sad**        | *its own read*          |
| 8    | `wrong_ask`      | They announced it out of nowhere.                             | **Surprise**   | **happy** ← WRONG       |
| 9    | `none_ask`       | The whole thing was handled badly.                            | **Disgust**    | *its own read*          |
| 10   | `wrong_assert`   | Same routine as always.                                       | **Neutral**    | **sad** ← WRONG         |
| 11   | `correct_assert` | People keep ignoring what I said.                             | **Angry**      | **angry** (right)       |
| 12   | `correct_ask`    | Looking forward to the weekend.                               | **Happy**      | **happy** (right)       |


✱ = scripted turn. It shows a recall instruction instead of the generic prompt; write
your own short message following that instruction — the table text is only a fallback.

## What to check while testing


| Turn type                     | Question to answer                                                                                       |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| `wrong_assert`                | Does a confident wrong label feel bad, or merely odd? This is the penalty the study is built to measure. |
| `wrong_ask` vs `wrong_assert` | Same wrong label, different speech act. Does asking hurt less? (H3)                                      |
| `correct_*` with Neutral      | "You seem neutral" is awkward English — flag if it reads badly.                                          |
| scripted turns                | Words say "fine / ordinary" while you report something else. Does the agent catch it?                    |
| any turn                      | Do the 1–5 items feel too coarse? Paper 01 saw PETS medians near 80/100, so ceiling is a live risk.      |
| manipulation check            | Could you tell which emotion the agent was working from? If not, the check is not doing its job.         |




## After the run

```bash
python scripts/analyze_study.py            # full pre-specified analysis (needs n >= 5 for tests)
```

Quick look at what was logged, including frame validation:

```bash
python - <<'EOF'
import csv, json, pathlib
for r in csv.DictReader(open("data/study_responses.csv")):
    print(f"t{r['trial_index']:>2} {r['condition']:<15} felt={r['felt_label']:<8} shown={r['shown_label'] or '-':<8} "
          f"manip={r['manip_check_label']:<8} ratings={r['rate_match']},{r['rate_warm']},{r['rate_mental']}")
    print(f"     {r['response_excerpt'][:110]}")
for m in sorted(pathlib.Path("data/frames").rglob("meta.json")):
    d = json.loads(m.read_text())
    print(f"{d['condition']:<15} face_detected={d['sensor']['face_detected']} "
          f"frames_with_face={d['sensor']['frames_with_face']}/{d['sensor']['frames']} "
          f"sensor_face={d['sensor']['face']} vs felt={d['felt_label']}")
EOF
```

Three things that make a pilot run worth keeping:

1. Every cell produced a reply that used the intended emotion word.
2. Face crops in `data/frames/` actually contain a face.
3. The manipulation check matched the shown label on `correct` and `wrong` turns.

