"""
Load a fine-tuned FERResNet50 checkpoint (vanilla `state_dict`) for inference.
Separate from `vision_sensor.VisionEmotionModel`, which expects the Elena Ryumina HF key layout.

For training output format, see `scripts/finetune_fer2013.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from bimodal_empathy.vision_sensor import FERResNet50

# Match `vision_sensor.VisionEmotionModel` (FER / AffectNet-style RGB norm)
_FER_RGB_MEAN = (91.49 / 255.0, 103.88 / 255.0, 131.09 / 255.0)
_FER_RGB_STD = (0.5, 0.5, 0.5)


def _pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(_FER_RGB_MEAN, _FER_RGB_STD),
        ]
    )


def load_finetune_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load training bundle: dict with 'state_dict' and optional 'meta' JSON, or a raw state_dict."""
    p = Path(path)
    obj: Any = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj
    if isinstance(obj, dict) and any(k.startswith("r.") for k in obj):
        return {"state_dict": obj, "meta": {}}
    raise TypeError(
        f"Expected checkpoint with 'state_dict' key or a plain ResNet state_dict, got: {type(obj)}"
    )


class VisionEmotionModelFinetuned:
    """
    API-compatible with `VisionEmotionModel.predict_fer7` for evaluation scripts.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str | None = None,
    ):
        self.device = _pick_device(device)
        bundle = load_finetune_checkpoint(ckpt_path)
        self.model = FERResNet50()
        self.model.load_state_dict(bundle["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self._transform = _default_transform()
        self.meta: dict[str, Any] = bundle.get("meta") or {}
        p = Path(ckpt_path)
        meta_json = p.with_suffix(".json")
        if not self.meta and meta_json.is_file():
            self.meta = json.loads(meta_json.read_text(encoding="utf-8"))

    @torch.inference_mode()
    def predict_fer7(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, dict]:
        t = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(t)[0]
        p = F.softmax(logits, dim=-1).float().cpu().numpy()
        return p, logits.float().cpu().numpy(), {}


def load_vision_finetuned(
    ckpt_path: str | Path,
    device: str | None = None,
) -> VisionEmotionModelFinetuned:
    return VisionEmotionModelFinetuned(ckpt_path, device=device)
