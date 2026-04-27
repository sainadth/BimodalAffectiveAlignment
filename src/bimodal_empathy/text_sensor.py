"""Linguistic sensor: GoEmotions transformer -> P_text (7-way FER7)."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bimodal_empathy.config import GOEMOTIONS_MODEL_ID, device_preference
from bimodal_empathy.emotion_mapping import (
    collapse_goemotions_to_fer7,
    goemotion_id2label_28_from_config,
)


class TextEmotionModel:
    def __init__(self, model_id: str = GOEMOTIONS_MODEL_ID, device: str | None = None):
        self.device = device or device_preference()
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.id2label = goemotion_id2label_28_from_config(cfg)

    @torch.inference_mode()
    def predict_fer7(self, text: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Returns:
            p_text: (7,) normalized FER7 distribution
            raw28: model-specific GoEmotion scores (sigmoid) length n_labels
            debug: str keys for UI
        """
        if not (text and text.strip()):
            u = np.ones(7) / 7.0
            n = len(self.id2label)
            return u, np.zeros(n), {"note": "empty text"}

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        logits = out.logits[0]
        # Multi-label: sigmoid; single-label: softmax
        p_type = str(getattr(self.model.config, "problem_type", "")).lower()
        if p_type == "multi_label_classification" or logits.numel() == len(self.id2label):
            raw = torch.sigmoid(logits).float().cpu().numpy()
        else:
            raw = torch.softmax(logits, dim=-1).float().cpu().numpy()

        p7 = collapse_goemotions_to_fer7(raw, self.id2label)
        return p7, raw, {"model_id": self.model_id, "n_labels": int(raw.shape[0])}


def load_text_model(
    model_id: str = GOEMOTIONS_MODEL_ID,
    device: str | None = None,
) -> TextEmotionModel:
    return TextEmotionModel(model_id=model_id, device=device)
