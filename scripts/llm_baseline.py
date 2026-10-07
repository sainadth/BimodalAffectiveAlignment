#!/usr/bin/env python3
"""
LLM-only baseline: give Claude the utterance (and optionally the face image) with
no explicit affect pipeline, and ask for (a) a FER-7 emotion label and (b) a
short empathetic reply. This answers the reviewer question "why not just use a
VLM?" — both as a recognition baseline (label accuracy vs. gold) and as a
reply-quality comparator for the user study.

  # MELD text-only recognition baseline (gold in column Emotion):
  python scripts/llm_baseline.py --csv data/meld/test_sent_emo.csv \
      --text-col Utterance --gold-col Emotion --limit 300

  # Paired replies for study stimuli, plus the pipeline's flan-t5 replies:
  python scripts/llm_baseline.py --csv data/dissonant_samples.csv --text-col text \
      --with-pipeline --out docs/eval_output/llm_baseline_dissonant

  # Inspect prompts without spending tokens:
  python scripts/llm_baseline.py --csv ... --dry-run

Auth: ANTHROPIC_API_KEY in the environment or .env (or an `ant auth login`
profile). Identity-linked keys also need ANTHROPIC_WORKSPACE_ID (Console ->
Settings -> Workspaces, id starts with `wrkspc_`). Model defaults to claude-opus-5.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from bimodal_empathy.config import FER7_LABELS

MELD_TO_FER7 = {
    "anger": "Angry", "disgust": "Disgust", "fear": "Fear", "joy": "Happy",
    "sadness": "Sad", "surprise": "Surprise", "neutral": "Neutral",
}

SYSTEM = (
    "You are a supportive conversational agent. The user sends one short message "
    "and sometimes a photo of their face. Infer the user's dominant emotion as one "
    "of exactly: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. Then write an "
    "empathetic reply of at most two sentences that acknowledges how they feel. If "
    "the words and the face disagree, gently acknowledge both."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "emotion": {"type": "string", "enum": list(FER7_LABELS)},
        "reply": {"type": "string"},
    },
    "required": ["emotion", "reply"],
    "additionalProperties": False,
}


def _normalize_gold(s: str) -> str | None:
    t = (s or "").strip()
    if not t:
        return None
    if t in FER7_LABELS:
        return t
    return MELD_TO_FER7.get(t.lower())


def _image_block(path: Path) -> dict:
    ext = path.suffix.lower()
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[ext.lstrip(".")]
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--gold-col", default=None, help="Column with gold emotion (FER-7 or MELD names).")
    ap.add_argument("--image-col", default=None, help="Column with an image path (relative to --image-dir).")
    ap.add_argument("--image-dir", default=".")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--provider", choices=("anthropic", "groq"), default="anthropic")
    ap.add_argument("--model", default=None, help="Default: claude-opus-5 (anthropic) or llama-3.3-70b-versatile (groq).")
    ap.add_argument("--with-pipeline", action="store_true", help="Also generate flan-t5 replies via the text branch (α=1).")
    ap.add_argument("--out", default=None, help="Default docs/eval_output/llm_baseline_<provider>")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent API requests.")
    args = ap.parse_args()

    rows = []
    with (ROOT / args.csv).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not (r.get(args.text_col) or "").strip():
                continue
            rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"{len(rows)} rows from {args.csv}")

    if args.dry_run:
        print("SYSTEM:", SYSTEM)
        for r in rows[:3]:
            print("USER:", r[args.text_col])
        return

    if args.model is None:
        args.model = "claude-opus-5" if args.provider == "anthropic" else "openai/gpt-oss-120b"
    if args.provider == "anthropic":
        import anthropic

        # Identity-linked keys must name the workspace they act in.
        ws = (os.environ.get("ANTHROPIC_WORKSPACE_ID") or "").strip()
        client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)
    else:
        from groq import Groq

        client = Groq()

    pipeline = None
    if args.with_pipeline:
        from bimodal_empathy.response_synthesizer import load_synthesizer
        from bimodal_empathy.text_sensor import load_text_model

        pipeline = (load_text_model(), load_synthesizer())

    out_base = ROOT / (args.out or f"docs/eval_output/llm_baseline_{args.provider}")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    def one(i: int, r: dict) -> dict | None:
        text = r[args.text_col].strip()
        content: list[dict] = []
        if args.image_col and (r.get(args.image_col) or "").strip():
            pth = ROOT / args.image_dir / r[args.image_col].strip()
            if pth.is_file():
                content.append(_image_block(pth))
        content.append({"type": "text", "text": text})
        if args.provider == "groq":
            try:
                gr = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM + " Respond ONLY with a JSON object {\"emotion\": <one of the seven labels>, \"reply\": <string>}."},
                        {"role": "user", "content": text if not content[:-1] else "[image omitted for this provider] " + text},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=400,
                    temperature=0.2,
                    **({"reasoning_effort": "low"} if "gpt-oss" in args.model else {}),
                )
            except Exception as e:  # groq SDK errors share a base but we only need to log and skip
                print(f"[{i}] groq error: {type(e).__name__}: {e}", file=sys.stderr)
                return None
            txt = gr.choices[0].message.content or "{}"
            try:
                parsed = json.loads(txt)
            except json.JSONDecodeError:
                parsed = {}
            em = str(parsed.get("emotion", "")).strip().title()
            gold = _normalize_gold(r.get(args.gold_col, "")) if args.gold_col else None
            return {
                "idx": i, "utterance": text, "gold": gold or "",
                "llm_emotion": em if em in FER7_LABELS else "",
                "llm_reply": str(parsed.get("reply", "")),
                "refused": 0, "pipeline_label": "", "pipeline_reply": "",
                "model": gr.model, "input_tokens": gr.usage.prompt_tokens, "output_tokens": gr.usage.completion_tokens,
            }
        try:
            resp = client.beta.messages.create(
                model=args.model,
                max_tokens=1024,
                system=SYSTEM,
                messages=[{"role": "user", "content": content}],
                output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            print(f"[{i}] rate limited: {e.message}", file=sys.stderr)
            return None
        except anthropic.APIStatusError as e:
            print(f"[{i}] API error {e.status_code}: {e.message}", file=sys.stderr)
            return None
        except anthropic.APIConnectionError as e:
            print(f"[{i}] connection error: {e}", file=sys.stderr)
            return None

        refused = resp.stop_reason == "refusal"
        llm_emotion = llm_reply = ""
        if not refused:
            txt = next((b.text for b in resp.content if b.type == "text"), "{}")
            try:
                parsed = json.loads(txt)
            except json.JSONDecodeError:
                parsed = {}
            llm_emotion = parsed.get("emotion", "")
            llm_reply = parsed.get("reply", "")

        gold = _normalize_gold(r.get(args.gold_col, "")) if args.gold_col else None
        pipe_label = pipe_reply = ""
        if pipeline is not None:
            import numpy as np

            tm, syn = pipeline
            p_text, _, _ = tm.predict_fer7(text)
            pipe_label = FER7_LABELS[int(np.argmax(p_text))]
            pipe_reply = syn.generate(text, pipe_label, p_fused=np.asarray(p_text))
        return {
            "idx": i,
            "utterance": text,
            "gold": gold or "",
            "llm_emotion": llm_emotion,
            "llm_reply": llm_reply,
            "refused": int(refused),
            "pipeline_label": pipe_label,
            "pipeline_reply": pipe_reply,
            "model": resp.model,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_rows: list[dict] = []
    # The pipeline (torch) is not thread-safe across MPS calls; force serial when used.
    workers = 1 if pipeline is not None else max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, i, r): i for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                out_rows.append(res)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(rows)}", file=sys.stderr)
    out_rows.sort(key=lambda d: d["idx"])
    refusals = sum(r["refused"] for r in out_rows)
    scored = [r for r in out_rows if r["gold"] and r["llm_emotion"]]
    n_gold = len(scored)
    correct = sum(int(r["llm_emotion"] == r["gold"]) for r in scored)

    with (out_base.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["idx"])
        w.writeheader()
        w.writerows(out_rows)

    summary = {
        "model": args.model,
        "n": len(out_rows),
        "refusals": refusals,
        "gold_n": n_gold,
        "llm_top1": (correct / n_gold) if n_gold else None,
        "total_input_tokens": sum(r["input_tokens"] for r in out_rows),
        "total_output_tokens": sum(r["output_tokens"] for r in out_rows),
    }
    (out_base.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
