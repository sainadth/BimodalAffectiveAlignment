"""FLAN-T5 empathetic response given user utterance U and fused FER-7 label b*.

Per rubric: the synthesizer ingests U and the final fused state b* and returns a
contextually proactive empathetic response R. We pass `fused_label` (the FER-7
class string, e.g. "Angry") and U; optionally we add a single descriptor sentence
about which emotion the fused distribution puts most mass on.

We use a FLAN-style instruction prompt and *deterministic beam search* (no sampling)
because the small/base FLAN-T5 models we ship with this project are fragile under
temperature sampling and frequently devolve into single-token replies (e.g. just
echoing the class label). Beam search with no-repeat n-gram blocking gives stable,
well-formed multi-sentence acknowledgements.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from bimodal_empathy.config import FER7_LABELS, FLAN_T5_MODEL_ID, MAX_NEW_TOKENS, device_preference

EMOTION_DESCRIPTORS: dict[str, str] = {
    "Angry": "frustrated and angry",
    "Disgust": "disgusted and put off",
    "Fear": "scared and anxious",
    "Happy": "happy and excited",
    "Sad": "sad and discouraged",
    "Surprise": "surprised and caught off guard",
    "Neutral": "calm and even-keeled",
}


def _emotion_descriptor(label: str) -> str:
    return EMOTION_DESCRIPTORS.get((label or "").capitalize(), (label or "").lower() or "uncertain")


class EmpatheticSynthesizer:
    def __init__(self, model_id: str = FLAN_T5_MODEL_ID, device: str | None = None):
        self.device = device or device_preference()
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, user_utterance: str, fused_label: str, p_fused: np.ndarray | None) -> str:
        u = (user_utterance or "").strip() or "..."
        descriptor = _emotion_descriptor(fused_label)
        focus = ""
        if p_fused is not None:
            pf = np.asarray(p_fused, dtype=float).ravel()
            if pf.size == 7:
                top_idx = int(np.argmax(pf))
                top_lbl = FER7_LABELS[top_idx]
                if top_lbl.lower() != (fused_label or "").lower():
                    focus = f" Their face also shows traces of {top_lbl.lower()}."
        return (
            "You are an empathetic counsellor talking to a user. Reply to the user "
            "in 2 short sentences using the word \"you\" at least once. "
            "Acknowledge that they feel "
            f"{descriptor}, validate the feeling, then ask one short open question. "
            "Do NOT repeat the user's sentence verbatim and do NOT speak in the first "
            "person; address the user.\n\n"
            f"User said: \"{u}\"\n"
            f"Inferred emotion: {fused_label}.{focus}\n\n"
            "Empathetic reply to the user (begin with \"It sounds like\" or \"I hear\"):"
        )

    @torch.inference_mode()
    def generate(
        self,
        user_utterance: str,
        fused_label: str,
        p_text: np.ndarray | None = None,
        p_face: np.ndarray | None = None,
        p_fused: np.ndarray | None = None,
    ) -> str:
        prompt = self._build_prompt(user_utterance, fused_label, p_fused)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=18,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.6,
            length_penalty=1.0,
            do_sample=False,
            early_stopping=True,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return (text or "").strip()


def load_synthesizer(
    model_id: str = FLAN_T5_MODEL_ID,
    device: str | None = None,
) -> EmpatheticSynthesizer:
    return EmpatheticSynthesizer(model_id=model_id, device=device)
