"""
FER2013 images from HuggingFace `AutumnQiu/fer2013` (for standalone eval; benchmarks unchanged).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


def iter_fer2013_test(
    limit: int | None = None, split: str = "test", streaming: bool = True
) -> Iterator[tuple[Any, int]]:
    """
    Yields (PIL image, label 0-6) from FER2013. Stops after `limit` examples when set.
    Matches streaming behavior of `eval.benchmarks._iter_fer2013_images`.
    """
    ds = load_dataset("AutumnQiu/fer2013", split=split, trust_remote_code=False, streaming=streaming)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield row["image"], int(row["label"])
