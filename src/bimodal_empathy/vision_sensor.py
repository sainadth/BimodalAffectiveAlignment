"""
Visual sensor: ResNet-50 (FER2013) -> P_face (7-way).

Weights: Elena Ryumina / HuggingFace `ElenaRyumina/face_emotion_recognition` checkpoint
`FER_static_ResNet50_AffectNet.pt` (ResNet-50 + fc1 + fc2 head). Stems/keys are remapped
to `torchvision.models.resnet50` naming for the backbone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

from bimodal_empathy.config import FER_HF_FILE, FER_HF_REPO, device_preference

# FER2013 / AffectNet-style normalization (RGB, 0-1 input) from common FER pipelines
_FER_RGB_MEAN = (91.49 / 255.0, 103.88 / 255.0, 131.09 / 255.0)
_FER_RGB_STD = (0.5, 0.5, 0.5)


def _remap_fer_stem_key(k: str) -> str:
    k = k.replace("conv_layer_s2_same", "conv1")
    k = k.replace("i_downsample", "downsample")
    for i in (1, 2, 3):
        k = k.replace(f"batch_norm{i}", f"bn{i}")
    return k


class FERResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = resnet50(weights=None)
        in_f = self.r.fc.in_features
        self.fc1 = nn.Linear(in_f, 512)
        self.fc2 = nn.Linear(512, 7)

    def _features_2048(self, x: torch.Tensor) -> torch.Tensor:
        x = self.r.conv1(x)
        x = self.r.bn1(x)
        x = self.r.relu(x)
        x = self.r.maxpool(x)
        x = self.r.layer1(x)
        x = self.r.layer2(x)
        x = self.r.layer3(x)
        x = self.r.layer4(x)
        x = self.r.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features_2048(x)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


def load_fer_state_dict_from_checkpoint(sd: dict) -> FERResNet50:
    m = FERResNet50()
    bsd = {k: v for k, v in sd.items() if not (k.startswith("fc1") or k.startswith("fc2"))}
    bsd2 = { _remap_fer_stem_key(k): v for k, v in bsd.items() }
    inc = m.r.load_state_dict(bsd2, strict=False)
    # Official checkpoint has no final torchvision fc; random fc weights remain unused.
    if getattr(inc, "unexpected_keys", []):
        raise RuntimeError(f"Unexpected keys: {inc.unexpected_keys}")
    m.fc1.load_state_dict({"weight": sd["fc1.weight"], "bias": sd["fc1.bias"]}, strict=True)
    m.fc2.load_state_dict({"weight": sd["fc2.weight"], "bias": sd["fc2.bias"]}, strict=True)
    return m


def download_fer_weights() -> str:
    return hf_hub_download(FER_HF_REPO, FER_HF_FILE)


def load_vision_model(
    weights_path: str | None = None,
    device: str | None = None,
) -> "VisionEmotionModel":
    path = weights_path or download_fer_weights()
    return VisionEmotionModel(ckpt_path=path, device=device or device_preference())


class VisionEmotionModel:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = device or device_preference()
        # weights_only=True works for this checkpoint (tensors + meta tensors)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict):
            raise TypeError("Unexpected checkpoint format")
        self.model = load_fer_state_dict_from_checkpoint(sd)
        self.model.to(self.device)
        self.model.eval()
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(_FER_RGB_MEAN, _FER_RGB_STD),
            ]
        )

    @torch.inference_mode()
    def predict_fer7(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Args:
            image: PIL Image RGB
        """
        t = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(t)[0]
        p = F.softmax(logits, dim=-1).float().cpu().numpy()
        return p, logits.float().cpu().numpy(), {}


def uniform_face_p_face() -> np.ndarray:
    """Use when no image is available (ablation or demo fallback)."""
    return np.ones(7) / 7.0
