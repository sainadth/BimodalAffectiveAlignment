#!/usr/bin/env python3
"""
Evaluate on MELD (Poria et al., ACL 2019) — natural multi-party utterances from
*Friends* with 7 emotion labels that map 1:1 onto FER-7.

Text branch (always): RoBERTa-GoEmotions -> FER-7 on the MELD test split
(2,610 utterances). Reports top-1, macro-F1, Wilson CI, majority floor,
per-class confusion. Downloads the public CSV from the declare-lab/MELD repo on
first use.

Bimodal (optional): if --video-dir points at the MELD test mp4s
(dia{D}_utt{U}.mp4, from https://affective-meld.github.io/), the middle frame is
face-cropped with OpenCV's YuNet detector (ONNX, auto-downloaded to checkpoints/) and run through the vision branch; the
α grid is then swept exactly as in scripts/alpha_sweep.py. Requires
`pip install opencv-python-headless`.

  python scripts/eval_meld.py                              # text branch only
  python scripts/eval_meld.py --video-dir data/meld/test   # + face + fusion
  python scripts/eval_meld.py --limit 300                  # quick check
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from tqdm import tqdm

from bimodal_empathy.config import FER7_LABELS
from bimodal_empathy.eval.stats import confusion, macro_f1, wilson_ci
from bimodal_empathy.text_sensor import load_text_model

MELD_TEST_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv"
MELD_TO_FER7 = {
    "anger": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happy",
    "sadness": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral",
}


def ensure_csv(path: Path) -> Path:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MELD test CSV -> {path}")
        urllib.request.urlretrieve(MELD_TEST_URL, path)
    return path


def summarize(Y: np.ndarray, pred: np.ndarray) -> dict:
    n = len(Y)
    k = int((pred == Y).sum())
    conf = confusion(Y, pred, 7)
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "top1": k / n,
        "ci95": [lo, hi],
        "macro_f1": macro_f1(conf),
        "confusion": conf.tolist(),
    }


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = ROOT / "checkpoints" / "face_detection_yunet_2023mar.onnx"


def load_face_detector():
    """OpenCV YuNet face detector (Haar cascades were removed in OpenCV 5)."""
    import cv2  # type: ignore

    if not YUNET_PATH.is_file():
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading YuNet face detector -> {YUNET_PATH}")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.6, 0.3, 5000)


def middle_frame_face(video_path: Path, detector) -> "np.ndarray | None":
    """Largest detected face in the middle frame, padded 15%, as RGB array."""
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])[:4]
    x, y, bw, bh = int(x), int(y), int(bw), int(bh)
    pad = int(0.15 * max(bw, bh))
    y0, y1 = max(0, y - pad), min(h, y + bh + pad)
    x0, x1 = max(0, x - pad), min(w, x + bw + pad)
    if y1 <= y0 or x1 <= x0:
        return None
    return cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/meld/test_sent_emo.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--video-dir", default=None, help="Dir with dia{D}_utt{U}.mp4 (MELD test). Enables bimodal eval.")
    ap.add_argument("--vision", choices=("baseline", "finetuned"), default="finetuned")
    ap.add_argument("--ckpt", default=os.getenv("FINETUNE_CKPT", "checkpoints/fer2013_finetune.pt"))
    ap.add_argument("--out", default="docs/eval_output")
    ap.add_argument("--suffix", default="", help="Appended to output filenames, e.g. _ft or _bimodal_ft.")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ensure_csv(ROOT / args.csv)

    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            emo = r["Emotion"].strip().lower()
            if emo not in MELD_TO_FER7:
                continue
            rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    Y = np.array([FER7_LABELS.index(MELD_TO_FER7[r["Emotion"].strip().lower()]) for r in rows])
    n = len(Y)
    dist = Counter(FER7_LABELS[y] for y in Y)
    majority = max(dist.values()) / n

    tm = load_text_model()
    PT = np.stack([np.asarray(tm.predict_fer7(r["Utterance"])[0], dtype=np.float64).ravel() for r in tqdm(rows, desc="text")])
    text_res = summarize(Y, PT.argmax(axis=1))

    result = {
        "dataset": "MELD test (Poria et al. 2019)",
        "n": n,
        "label_distribution": dict(dist),
        "majority_class_floor": majority,
        "text_only": text_res,
    }

    if args.video_dir:
        try:
            import cv2  # type: ignore
        except ImportError:
            sys.exit("--video-dir needs OpenCV: pip install opencv-python-headless")
        from PIL import Image

        from bimodal_empathy.fer_finetune_ckpt import load_vision_finetuned
        from bimodal_empathy.vision_sensor import load_vision_model

        vm = load_vision_finetuned(args.ckpt) if args.vision == "finetuned" else load_vision_model()
        detector = load_face_detector()
        vdir = ROOT / args.video_dir
        PF = np.full((n, 7), 1.0 / 7)
        has_face = np.zeros(n, dtype=bool)
        for i, r in enumerate(tqdm(rows, desc="faces")):
            vp = vdir / f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4"
            if not vp.is_file():
                continue
            crop = middle_frame_face(vp, detector)
            if crop is None:
                continue
            p_face, _, _ = vm.predict_fer7(Image.fromarray(crop))
            PF[i] = np.asarray(p_face, dtype=np.float64).ravel()
            has_face[i] = True
        sweep = []
        for a in np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 2):
            pred = (a * PT + (1 - a) * PF).argmax(axis=1)
            sweep.append({"alpha": float(a), **summarize(Y, pred)})
        m = has_face
        result["bimodal"] = {
            "vision": args.vision,
            "face_detected_frac": float(m.mean()),
            "sweep_all": sweep,
            "face_only_on_detected": summarize(Y[m], PF[m].argmax(axis=1)) if m.any() else None,
            "fused05_on_detected": summarize(Y[m], (0.5 * PT[m] + 0.5 * PF[m]).argmax(axis=1)) if m.any() else None,
            "text_on_detected": summarize(Y[m], PT[m].argmax(axis=1)) if m.any() else None,
        }

    (out_dir / f"meld_results{args.suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    t = text_res
    tex = f"""% Auto-generated: scripts/eval_meld.py
\\begin{{table}}[t]
  \\centering
  \\caption{{Text branch on natural utterances: MELD test split ($n={n}$), MELD 7-way labels mapped 1:1 to FER-7. Majority-class (Neutral) floor {100*majority:.1f}\\%.}}
  \\label{{tab:meld_text}}
  \\small
  \\begin{{tabular}}{{@{{}}l r r l@{{}}}}
    \\toprule
    Mode & Top-1 (\\%) & Macro-F1 & 95\\% CI \\\\
    \\midrule
    Majority class & {100*majority:.1f} & -- & -- \\\\
    Text only (RoBERTa$\\to$FER-7) & {100*t['top1']:.1f} & {t['macro_f1']:.3f} & [{100*t['ci95'][0]:.1f}, {100*t['ci95'][1]:.1f}] \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    for llm_csv in sorted(out_dir.glob("llm_baseline_*.csv")):  # LLM-only comparator rows (scripts/llm_baseline.py)
        with llm_csv.open(encoding="utf-8") as f:
            lr = [r for r in csv.DictReader(f) if r["gold"] and r["llm_emotion"]]
        if lr:
            ly = np.array([FER7_LABELS.index(r["gold"]) for r in lr])
            lp = np.array([FER7_LABELS.index(r["llm_emotion"]) for r in lr])
            ls = summarize(ly, lp)
            model = lr[0].get("model", "LLM")
            row = (f"    LLM only, {model} & {100*ls['top1']:.1f} & "
                   f"{ls['macro_f1']:.3f} & [{100*ls['ci95'][0]:.1f}, {100*ls['ci95'][1]:.1f}] \\\\\n")
            tex = tex.replace("    \\bottomrule", row + "    \\bottomrule")
    (out_dir / f"meld_text_table{args.suffix}.tex").write_text(tex, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "text_only"}, indent=2))
    print(f"text top-1 = {100*t['top1']:.2f}%  macro-F1 = {t['macro_f1']:.3f}  CI = {t['ci95']}")
    print("confusion (rows gold, cols pred, order", list(FER7_LABELS), ")")
    for lab, row in zip(FER7_LABELS, t["confusion"]):
        print(f"  {lab:9s}", row)


if __name__ == "__main__":
    main()
