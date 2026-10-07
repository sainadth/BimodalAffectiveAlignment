"""LLM-backed empathetic reply generation with affect conditioning.

One prompt, four conditioning modes (the study conditions):

  llm_only  - prompt carries no sensor output; the model reads affect itself
  text_only - prompt sees the text label only
  bimodal   - prompt sees text label, face label, fused label and the
              dissonance flag; asked to acknowledge a mismatch when present
  random    - prompt sees a random label presented as the inferred emotion

Providers: "groq" (default model openai/gpt-oss-120b, reasoning_effort=low), "anthropic"
(default claude-opus-5). Provider/model come from LLM_PROVIDER / LLM_MODEL env
vars unless passed explicitly. Network calls are isolated in `_call_*` so the
prompt logic is unit-testable offline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",  # fallback: qwen/qwen3.8-27b (faster, slightly weaker)
    "anthropic": "claude-opus-5",
}

SYSTEM_PROMPT = (
    "You are a warm, supportive conversational companion. The user sends one "
    "short message. You may also be given the output of an affect sensor. Reply in "
    "at most two sentences, in the second person, without repeating the user's "
    "words verbatim and without listing options. Acknowledge how the user seems "
    "to feel, then ask one gentle open question. Do not mention sensors, "
    "models, labels, or confidence scores explicitly."
)


@dataclass(frozen=True)
class AffectContext:
    text_label: str
    face_label: str | None  # None when no face image
    fused_label: str
    dissonant: bool


def build_user_prompt(utterance: str, condition: str, ctx: AffectContext, random_label: str | None = None) -> str:
    """Return the user-turn prompt for the given study condition."""
    u = utterance.strip()
    if condition == "llm_only":
        return f"User message: \"{u}\"\nWrite your reply."
    if condition == "text_only":
        return (
            f"User message: \"{u}\"\n"
            f"Affect sensor (from the words): the user seems {ctx.text_label.lower()}.\n"
            "Write your reply."
        )
    if condition == "random":
        if random_label is None:
            raise ValueError("random condition requires random_label")
        return (
            f"User message: \"{u}\"\n"
            f"Affect sensor: the user seems {random_label.lower()}.\n"
            "Write your reply."
        )
    if condition == "bimodal":
        face = ctx.face_label.lower() if ctx.face_label else None
        lines = [f"User message: \"{u}\"", f"Affect sensor (from the words): the user seems {ctx.text_label.lower()}."]
        if face is None:
            lines.append("No facial expression was captured.")
        else:
            lines.append(f"Affect sensor (from the face): the user looks {face}.")
        if ctx.dissonant and face is not None:
            lines.append(
                "The words and the face disagree. Gently acknowledge both: what they said and what "
                "they seem to show, without accusing them of hiding anything."
            )
        else:
            lines.append(f"Overall the user seems {ctx.fused_label.lower()}.")
        lines.append("Write your reply.")
        return "\n".join(lines)
    raise ValueError(f"unknown condition {condition!r}")


SYSTEM_PROMPT_V3 = (
    "You are a warm, supportive conversational companion. The user sends one "
    "short message. You may be told what an affect sensor read from them. Reply in "
    "at most two sentences, in the second person, without repeating the user's "
    "words verbatim and without listing options. Never mention sensors, models, "
    "labels, cameras, or confidence."
)


def build_user_prompt_v3(utterance: str, move: str, label: str | None) -> str:
    """Design-v3 prompt: `move` is 'assert' or 'ask'; `label` is the emotion the
    generator is told (None/'' for the no-label condition).

    assert -> state how the user seems to feel (the given label, or the model's own
              reading when no label), then respond.
    ask    -> do not state their emotion as fact; gently check in with one question
              about whether they feel the given label (or, with no label, an open
              question about how they feel), then offer brief support.
    """
    if move not in ("assert", "ask"):
        raise ValueError(f"move must be assert|ask, got {move!r}")
    u = utterance.strip()
    lines = [f"User message: \"{u}\""]
    if label:
        lines.append(f"Affect sensor: the user seems {label.lower()}.")
    else:
        lines.append("No affect sensor output is available.")
    if move == "assert":
        lines.append(
            "Briefly state how they seem to be feeling"
            + (f", using the word \"{label.lower()}\" itself (for example: \"you seem {label.lower()}\")"
               if label else " (use your own reading of the message)")
            + ", then respond supportively. Do not substitute a vaguer word."
        )
    else:
        lines.append(
            "Do not state their emotion as a fact. Instead ask them about it in one question, using the word "
            + (f"\"{label.lower()}\" itself (for example: \"are you feeling {label.lower()} right now?\")"
               if label else "an open question about how they are feeling")
            + ", then offer brief support. Do not substitute a vaguer word."
        )
    lines.append("Write your reply.")
    return "\n".join(lines)


def generate_reply_v3(
    utterance: str,
    move: str,
    label: str | None,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 300,
) -> "LLMReply":
    p, m = resolve_provider(provider, model)
    prompt = build_user_prompt_v3(utterance, move, label or None)
    t0 = time.perf_counter()
    text = (_call_groq if p == "groq" else _call_anthropic)(m, SYSTEM_PROMPT_V3, prompt, max_tokens)
    return LLMReply(text=text, provider=p, model=m, latency_ms=(time.perf_counter() - t0) * 1000.0, prompt=prompt)


def resolve_provider(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    p = (provider or os.environ.get("LLM_PROVIDER") or "groq").strip().lower()
    if p not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got {p!r}")
    m = (model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS[p]).strip()
    return p, m


def _call_groq(model: str, system: str, user: str, max_tokens: int) -> str:
    from groq import Groq

    client = Groq(timeout=20.0, max_retries=1)
    kwargs = {}
    if "gpt-oss" in model:
        # Reasoning models spend the budget thinking unless told otherwise.
        kwargs["reasoning_effort"] = "low"
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.7,
        **kwargs,
    )
    return (r.choices[0].message.content or "").strip()


def _call_anthropic(model: str, system: str, user: str, max_tokens: int) -> str:
    import anthropic

    ws = (os.environ.get("ANTHROPIC_WORKSPACE_ID") or "").strip()
    client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None, timeout=30.0, max_retries=1)
    r = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if r.stop_reason == "refusal":
        return ""
    return next((b.text for b in r.content if b.type == "text"), "").strip()


@dataclass
class LLMReply:
    text: str
    provider: str
    model: str
    latency_ms: float
    prompt: str


def generate_reply(
    utterance: str,
    condition: str,
    ctx: AffectContext,
    *,
    random_label: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 300,
) -> LLMReply:
    p, m = resolve_provider(provider, model)
    prompt = build_user_prompt(utterance, condition, ctx, random_label)
    t0 = time.perf_counter()
    text = (_call_groq if p == "groq" else _call_anthropic)(m, SYSTEM_PROMPT, prompt, max_tokens)
    return LLMReply(text=text, provider=p, model=m, latency_ms=(time.perf_counter() - t0) * 1000.0, prompt=prompt)
