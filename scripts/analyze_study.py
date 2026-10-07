#!/usr/bin/env python3
"""Run the pre-specified study analysis (design v2) on data/study_responses.csv + data/study_pets_global.csv.

  python scripts/analyze_study.py                      # default CSV / STUDY_CSV
  python scripts/analyze_study.py --csv path.csv --out docs/eval_output/study_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bimodal_empathy.eval.study_stats import full_report
from bimodal_empathy.study_log import global_csv_path, study_csv_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--global-csv", default=None, help="Once-per-participant PETS CSV (default STUDY_GLOBAL_CSV).")
    ap.add_argument("--out", default="docs/eval_output/study_analysis.json")
    args = ap.parse_args()
    path = Path(args.csv) if args.csv else study_csv_path()
    if not path.is_file():
        sys.exit(f"no study CSV at {path}")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gpath = Path(args.global_csv) if args.global_csv else global_csv_path()
    grows = list(csv.DictReader(gpath.open(encoding="utf-8"))) if gpath.is_file() else []
    rep = full_report(rows, grows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    t = rep["primary_composite"]
    print(f"rows={rep['n_rows']} complete participants={t['n_participants']}")
    if t.get("means"):
        for c, m in zip(t["conditions"], t["means"]):
            print(f"  {c:10s} composite mean={m:.2f}")
    if "friedman" in t:
        print(f"  Friedman chi2={t['friedman']['chi2']:.2f} p={t['friedman']['p']:.4f}")
        for pr in t["pairwise"]:
            print(f"  {pr['a']} vs {pr['b']}: diff={pr['mean_diff']:+.2f} p_holm={pr['p_holm']:.4f} r_rb={pr['rank_biserial']:+.2f}")
    for key in ("H2_masking_trials", "H2b_scripted_trials"):
        print(f"{key}: complete participants={rep[key].get('n_participants', 0)}")
    if "pets_global" in rep:
        g = rep["pets_global"]
        print(f"PETS global n={g['n']} total={g['total_mean']:.2f} alpha={g['alpha_total']:.2f} funnel={g['funnel_noticed_pattern']}")
    print(f"composite alpha (participant means)={rep['composite_cronbach_alpha_participant_means']:.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
