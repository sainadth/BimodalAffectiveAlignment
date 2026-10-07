"""
Researcher demo: bimodal affective alignment (GoEmotions + FER ResNet-50 + fusion + reply generator).
Run:  streamlit run app/streamlit_app.py
For the participant study use app/study_app.py instead.
"""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FINETUNE_CKPT_PATH, analyze_affect, cached_synth, llm_configured  # noqa: E402

from bimodal_empathy.config import DEFAULT_ALPHA, FER7_LABELS, device_preference
from bimodal_empathy.likert_log import append_likert_row, likert_csv_path
from bimodal_empathy.llm_generator import AffectContext, generate_reply


def _fer_css():
    st.markdown(
        """
<style>
  .main-title { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;
    background: linear-gradient(120deg, #1e3a5f 0%, #3d6fa8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .subtle { color: #5a6b7a; font-size: 0.95rem; margin-top: 0.2rem; }
  .card { background: rgba(127,127,127,0.10); border: 1px solid rgba(127,127,127,0.25);
    border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.8rem; }
  .badge-dissonance { display: inline-block; padding: 0.35rem 0.7rem; border-radius: 999px; font-size: 0.9rem;
    font-weight: 500; background: #3d2a12; color: #ffc266; border: 1px solid #6a4a1e; }
  .badge-congruent { display: inline-block; padding: 0.35rem 0.7rem; border-radius: 999px; font-size: 0.9rem;
    font-weight: 500; background: #1a2e1c; color: #8fd99a; border: 1px solid #2d5a33; }
  .section-title { font-size: 1.05rem; font-weight: 600; margin: 0.5rem 0 0.35rem; }
  div[data-testid="stMetricValue"] { font-size: 1.1rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _fer7_prob_df(p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "emotion": pd.Categorical(list(FER7_LABELS), categories=list(FER7_LABELS), ordered=True),
        "p": [float(p[i]) for i in range(7)],
    })


def _bar_chart(p: np.ndarray, title: str) -> alt.Chart:
    df = _fer7_prob_df(p)
    return (
        alt.Chart(df, title=alt.TitleParams(text=title, anchor="start", fontSize=13, fontWeight=500))
        .mark_bar(cornerRadius=3, size=18, color="#4a90d9")
        .encode(
            x=alt.X("p:Q", title="Probability", scale=alt.Scale(domain=[0, 1], nice=False)),
            y=alt.Y("emotion:N", title="", sort=list(FER7_LABELS),
                    axis=alt.Axis(labelFontSize=12, labelOverlap=False, labelLimit=140)),
            tooltip=["emotion", alt.Tooltip("p:Q", title="P", format=".4f")],
        )
        .properties(height=alt.Step(36))
        .configure(background="transparent")
        .configure_view(stroke="transparent")
    )


SESSION_KEY = "demo_last_analysis_v2"
LIKERT_SAVED_KEY = "demo_likert_saved_for"
LIKERT_WIDGET_KEYS = ("likert_empathy", "likert_social", "likert_congr")


def main() -> None:
    st.set_page_config(page_title="Bimodal Affective Alignment — researcher demo", page_icon="💬", layout="wide")
    _fer_css()
    st.markdown(
        '<p class="main-title">Bimodal Affective Alignment</p>'
        '<p class="subtle">Researcher demo · text (GoEmotions → FER-7) + face (ResNet-50, YuNet crop) + late fusion + reply generator</p>',
        unsafe_allow_html=True,
    )
    st.caption(f"Device: {device_preference()} · participant study lives in app/study_app.py")

    with st.sidebar:
        st.subheader("Trust weight α")
        alpha = st.slider("α: trust in **text** vs **face**", 0.0, 1.0, DEFAULT_ALPHA, 0.05,
                          help="Fused vector = α·P_text + (1−α)·P_face.")
        st.markdown(f"**{alpha:.2f}** text · **{1 - alpha:.2f}** face")
        st.divider()
        st.subheader("Vision branch")
        ft_available = FINETUNE_CKPT_PATH.is_file()
        if ft_available:
            use_finetune = st.radio("Face classifier weights", ("Baseline (HF AffectNet)", "Fine-tuned (FER2013)"),
                                    index=1).startswith("Fine-tuned")
        else:
            st.caption(f"No fine-tuned checkpoint at `{FINETUNE_CKPT_PATH}`.")
            use_finetune = False
        st.divider()
        st.subheader("Reply generator")
        gen = st.radio("Generator", ("FLAN-T5 (local)", "LLM (LLM_PROVIDER from .env)"),
                       index=1 if llm_configured() else 0)
        use_llm = gen.startswith("LLM")
        condition = "bimodal"
        if use_llm:
            condition = st.selectbox("Prompt condition", ("bimodal", "text_only", "llm_only", "random"), index=0,
                                     help="Same conditions as the study; bimodal names text/face disagreement.")
        st.divider()
        participant_id = st.text_input("Participant ID (quick Likert log)", value="", key="hci_participant_id")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.subheader("What you say (U)")
        utterance = st.text_area("Utterance", height=120, placeholder='e.g. "I\'m fine, really."',
                                 label_visibility="collapsed")
    with c2:
        st.subheader("Face frame")
        cam = st.camera_input("Webcam (or upload below)", label_visibility="collapsed")
        up = st.file_uploader("Or upload a face image", type=["png", "jpg", "jpeg", "webp"])
        image: Image.Image | None = None
        try:
            if cam is not None:
                image = Image.open(BytesIO(cam.getvalue()))
            elif up is not None:
                image = Image.open(up)
        except Exception:
            st.error("Could not read that image.")
    run = st.button("Analyze & generate response", type="primary")

    if run and not (utterance and utterance.strip()):
        st.warning("Please enter some text.")
        return
    if not run and not st.session_state.get(SESSION_KEY):
        st.info("Enter text, optionally add a face image, then press Analyze.")
        return

    if run:
        try:
            with st.spinner("Running text + face models…"):
                a = analyze_affect(utterance, image, alpha=alpha, use_finetune=use_finetune)
            shown = a["fused_label"]
            if use_llm:
                if condition == "text_only":
                    shown = a["text_label"]
                elif condition == "random":
                    shown = str(np.random.default_rng().choice(list(FER7_LABELS)))
                elif condition == "llm_only":
                    shown = ""
                ctx = AffectContext(a["text_label"], a["face_label"], a["fused_label"], bool(a["valence_dissonant"]))
                with st.spinner("Generating reply (LLM)…"):
                    t0 = time.perf_counter()
                    rl = generate_reply(utterance, condition, ctx, random_label=shown if condition == "random" else None)
                    a["gen_ms"] = (time.perf_counter() - t0) * 1000.0
                    a["reply"], a["generator"] = rl.text or "(empty reply)", f"{rl.provider}:{rl.model}"
            else:
                with st.spinner("Generating reply (FLAN-T5)…"):
                    t0 = time.perf_counter()
                    a["reply"] = cached_synth().generate(utterance, shown, p_fused=np.asarray(a["p_fuse"]))
                    a["gen_ms"] = (time.perf_counter() - t0) * 1000.0
                    a["generator"] = "flan-t5-base"
        except Exception as exc:
            st.error("Something went wrong while running the models.")
            st.caption(f"{type(exc).__name__}: {exc}")
            return
        a.update({"utterance": utterance.strip(), "shown_label": shown, "condition": condition if use_llm else "flan",
                  "analysis_id": time.time_ns(), "image": image})
        for k in LIKERT_WIDGET_KEYS:
            st.session_state.pop(k, None)
        st.session_state.pop(LIKERT_SAVED_KEY, None)
        st.session_state[SESSION_KEY] = a
    a = st.session_state[SESSION_KEY]
    p_text, p_face, p_fuse = (np.asarray(a[k]) for k in ("p_text", "p_face", "p_fuse"))

    st.markdown("---")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        if a.get("image") is not None:
            st.image(a["image"], caption=f"{a['vision_branch']} · face detected: {a['face_detected']}", width=260)
        else:
            st.info("No face image — uniform face prior; dissonance not applicable.")
        st.markdown(f"**You said (U):** _{html.escape(a['utterance'])}_")
        if not a["had_face_image"]:
            st.markdown('<span class="badge-congruent">Text-only run</span>', unsafe_allow_html=True)
        elif a["valence_dissonant"]:
            st.markdown(f'<span class="badge-dissonance">Valence dissonance</span> text: <b>{a["text_label"]}</b> · face: <b>{a["face_label"]}</b>',
                        unsafe_allow_html=True)
        elif a["argmax_dissonant"]:
            st.markdown(f'<span class="badge-congruent">Argmax differs, same valence</span> text: <b>{a["text_label"]}</b> · face: <b>{a["face_label"]}</b>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-congruent">Congruent</span>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Text argmax", a["text_label"])
        m2.metric("Face argmax", a["face_label"] or "—")
        m3.metric(f"Fused (α={a['alpha']:.2f})", a["fused_label"])
        st.markdown(f"**Reply (R)** · condition `{a['condition']}` · label given: `{a['shown_label'] or 'none'}` · {a['generator']}")
        st.markdown(f'<div class="card">{html.escape(a["reply"])}</div>', unsafe_allow_html=True)
        t = a["timings"]
        st.caption(f"text {t['text_ms']:.0f} ms · face {t['face_ms']:.0f} ms · fusion {t['fusion_ms']:.1f} ms · "
                   f"generation {a['gen_ms']:.0f} ms")
    with right:
        st.altair_chart(_bar_chart(p_text, "P_text — RoBERTa (GoEmotions → FER-7)"), use_container_width=True)
        st.altair_chart(_bar_chart(p_face, "P_face — ResNet-50 (face crop)"), use_container_width=True)
        st.altair_chart(_bar_chart(p_fuse, f"Fused — α = {a['alpha']:.2f}"), use_container_width=True)
    with st.expander("Debug: numeric vectors"):
        st.dataframe(pd.DataFrame({"Class": list(FER7_LABELS), "P_text": p_text, "P_face": p_face, "P_fused": p_fuse}),
                     hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Quick 3-item rating (researcher log)")
    st.caption(f"Rows append to `{likert_csv_path()}` (override with `LIKERT_CSV`).")
    already = st.session_state.get(LIKERT_SAVED_KEY) == a["analysis_id"]
    l_emp = st.radio("Empathy — the response felt understanding and warm.", [1, 2, 3, 4, 5], index=None, horizontal=True, key="likert_empathy")
    l_sp = st.radio("Social presence — the agent felt present in the conversation.", [1, 2, 3, 4, 5], index=None, horizontal=True, key="likert_social")
    l_cg = st.radio("Affective congruency — the reply fit how the user felt.", [1, 2, 3, 4, 5], index=None, horizontal=True, key="likert_congr")
    if already:
        st.success("Saved for this analysis.")
    elif st.button("Save rating", disabled=None in (l_emp, l_sp, l_cg)):
        append_likert_row(
            path=likert_csv_path(), participant_id=participant_id, empathy_1_5=int(l_emp), social_presence_1_5=int(l_sp),
            congruency_1_5=int(l_cg), alpha=a["alpha"], vision_branch=a["vision_branch"], dissonant=bool(a["valence_dissonant"]),
            text_argmax=a["text_label"], face_argmax=a["face_label"] or "none", fused_label=a["fused_label"],
            e2e_ms=float(sum(a["timings"].values()) + a["gen_ms"]), utterance=a["utterance"], response=a["reply"],
        )
        st.session_state[LIKERT_SAVED_KEY] = a["analysis_id"]
        st.rerun()


if __name__ == "__main__":
    main()
