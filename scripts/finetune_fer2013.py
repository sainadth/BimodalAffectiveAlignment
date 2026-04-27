#!/usr/bin/env python3
"""
Fine-tune FERResNet50 on HuggingFace AutumnQiu/fer2013 (train split; val = held-out 10%).

Does not modify `vision_sensor` or the main app. Writes checkpoints/ with vanilla state_dict.

Usage (repo root):
  python scripts/finetune_fer2013.py
  cp .env.example .env   # set HF_TOKEN for Hub (datasets + weight download)
  python scripts/finetune_fer2013.py --epochs 3 --batch-size 24 --max-steps 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Before HuggingFace Hub / datasets: same as scripts/run_evaluation.py
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass
if not (os.environ.get("HF_TOKEN") or "").strip() and not (os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
    mis = (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if mis:
        os.environ["HF_TOKEN"] = mis
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from bimodal_empathy.vision_sensor import (  # noqa: E402
    FERResNet50,
    download_fer_weights,
    load_fer_state_dict_from_checkpoint,
)

# Match VisionEmotionModel
_FER_RGB_MEAN = (91.49 / 255.0, 103.88 / 255.0, 131.09 / 255.0)
_FER_RGB_STD = (0.5, 0.5, 0.5)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FERHFDataset(Dataset):
    """In-memory (indexable) FER2013 from datasets."""

    def __init__(self, hf_ds, transform: transforms.Compose):
        self.ds = hf_ds
        self.t = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.ds[idx]
        im = row["image"]
        if not isinstance(im, Image.Image):
            im = Image.fromarray(np.asarray(im))  # type: ignore[assignment]
        x = self.t(im.convert("RGB"))
        y = int(row["label"])
        return x, y


def train_val_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def main() -> None:
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.1, help="Held-out fraction of train for validation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--output-name", type=str, default="fer2013_finetune")
    ap.add_argument("--max-steps", type=int, default=0, help="If >0, cap total training steps per run (smoke test).")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest for MPS).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device()
    print("Device:", device)
    is_mps = device.type == "mps"
    num_workers = 0 if is_mps else max(0, args.num_workers)

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(_FER_RGB_MEAN, _FER_RGB_STD),
        ]
    )

    print("Loading AutumnQiu/fer2013 train split…")
    train_full = load_dataset("AutumnQiu/fer2013", split="train", trust_remote_code=False, streaming=False)
    n = len(train_full)  # type: ignore[arg-type]
    tr_idx, va_idx = train_val_indices(n, args.val_fraction, args.seed)
    print(f"Train indices: {len(tr_idx)} | Val indices: {len(va_idx)}")

    base_fer = FERHFDataset(train_full, tfm)  # type: ignore[assignment]
    train_set = Subset(base_fer, tr_idx.tolist())
    val_set = Subset(base_fer, va_idx.tolist())

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Loading pretrained FER (Elena Ryumina)…")
    ck = download_fer_weights()
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise TypeError("Unexpected checkpoint")
    model = load_fer_state_dict_from_checkpoint(sd)
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    best_val = 0.0
    saved_this_run = False
    global_step = 0
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / f"{args.output_name}.pt"
    out_json = out_dir / f"{args.output_name}.json"

    stop_training = False
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs} train")
        for x, y in pbar:
            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                break
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", step=global_step)
        if stop_training:
            break

        model.eval()
        correct = tot = 0
        with torch.inference_mode():
            for x, y in tqdm(val_loader, desc="val", leave=False):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
        val_acc = (correct / tot) if tot else 0.0
        print(f"Epoch {epoch+1} val acc: {val_acc:.4f} ({correct}/{tot})")
        if val_acc >= best_val:
            best_val = val_acc
            saved_this_run = True
            payload = {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "meta": {
                    "val_acc": val_acc,
                    "epoch": epoch + 1,
                    "args": vars(args),
                },
            }
            torch.save(payload, out_pt)
            out_json.write_text(
                json.dumps(
                    {
                        "val_acc": val_acc,
                        "best_val_acc": best_val,
                        "epoch": epoch + 1,
                        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        "device": str(device),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("Saved:", out_pt, "| best val so far:", f"{best_val:.4f}")

    if not out_pt.is_file():
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "meta": {"val_acc": None, "args": vars(args), "device": str(device)},
            },
            out_pt,
        )
        print("Wrote final checkpoint (no validation ran this run):", out_pt)
    elif not saved_this_run and stop_training and args.max_steps:
        print(
            "Early exit: --max-steps hit before any validation, so the checkpoint on disk was not updated."
        )
        if out_pt.is_file():
            print("Existing file (unchanged):", out_pt)
    else:
        print("Done. Best val acc:", f"{best_val:.4f}", "|", out_pt)


if __name__ == "__main__":
    main()
