#!/usr/bin/env python3
"""
Print PyTorch / device context for FER ResNet-50 fine-tuning (CUDA / MPS / CPU).
Run from repo root: python scripts/check_training_env.py
"""
from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print("Python:", sys.version.split()[0], "|", platform.system(), platform.machine())
    print("PyTorch:", torch.__version__)
    print()

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            name = p.name
            mem_gb = p.total_memory / (1024**3)
            print(f"CUDA[{i}]: {name} — {mem_gb:.1f} GiB total")
        print()
        print(
            "Guidance: dedicated NVIDIA GPU with 8+ GiB is comfortable for ResNet-50 FER "
            "with batch sizes 32–64; 4–6 GiB: use --batch-size 8–16."
        )
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Device: MPS (Apple GPU)")
        print()
        print(
            "Guidance: MPS is suitable for FER2013 fine-tuning with modest batch (e.g. 16–32) "
            "and a few epochs. Use smaller batch or --max-steps if you hit memory pressure."
        )
    else:
        print("Device: CPU only (no CUDA, no MPS).")
        print()
        print(
            "Guidance: training will run on CPU and can be very slow for the full train split. "
            "Use --max-steps and a small --batch-size for smoke tests, or reduce --epochs first."
        )

    if torch.cuda.is_available() or (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        print(
            "Fine-tune script (scripts/finetune_fer2013.py) picks device as: cuda > mps > cpu, "
            "matching this report."
        )


if __name__ == "__main__":
    main()
