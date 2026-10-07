"""Smoke test for scripts/generate_likert_synthetic_table.py."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_generator_produces_at_least_100_rows_and_table(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_likert_synthetic_table.py"),
        "--n",
        "100",
        "--seed",
        "7",
        "--outdir",
        str(tmp_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    csv_p = tmp_path / "likert_synthetic_study.csv"
    assert csv_p.is_file()
    with csv_p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    assert (tmp_path / "likert_by_fused_label_table.md").is_file()
    tex = (tmp_path / "likert_by_fused_label_table.tex").read_text(encoding="utf-8")
    assert r"\label{tab:likert_by_class}" in tex
    assert r"\toprule" in tex
