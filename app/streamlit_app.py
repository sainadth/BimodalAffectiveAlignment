"""
Streamlit demo: bimodal affective alignment (GoEmotions + FER ResNet-50 + fusion + FLAN-T5).
Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from bimodal_empathy.config import DEFAULT_ALPHA, FER7_LABELS, device_preference
from bimodal_empathy.fer_finetune_ckpt import load_vision_finetuned
from bimodal_empathy.fusion import fuse
from bimodal_empathy.response_synthesizer import load_synthesizer
from bimodal_empathy.text_sensor import load_text_model
from bimodal_empathy.vision_sensor import load_vision_model, uniform_face_p_face

# Default location for the fine-tuned ResNet checkpoint. Override with the
# FINETUNE_CKPT env var or via the sidebar selector at runtime.
FINETUNE_CKPT_PATH = Path(
    os.getenv("FINETUNE_CKPT", "checkpoints/fer2013_finetune.pt")
)


def _fer_css():
    st.markdown(
        """
<style>
  .main-title { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;
    background: linear-gradient(120deg, #1e3a5f 0%, #3d6fa8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .subtle { color: #5a6b7a; font-size: 0.95rem; margin-top: 0.2rem; }
  .card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
    color: #eceef1;
  }
  .badge-dissonance {
    display: inline-block; padding: 0.35rem 0.7rem; border-radius: 999px; font-size: 0.9rem; font-weight: 500;
    background: #3d2a12; color: #ffc266; border: 1px solid #6a4a1e;
  }
  .badge-congruent {
    display: inline-block; padding: 0.35rem 0.7rem; border-radius: 999px; font-size: 0.9rem; font-weight: 500;
    background: #1a2e1c; color: #8fd99a; border: 1px solid #2d5a33;
  }
  .section-title { color: #e8eaed; font-size: 1.05rem; font-weight: 600; margin: 0.5rem 0 0.35rem; }
  div[data-testid="stMetricValue"] { font-size: 1.1rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def cached_text():
    return load_text_model()


@st.cache_resource
def cached_vision(weights_path: str | None = None):
    """Cache one vision model per distinct weights_path so the baseline and
    fine-tuned branches can coexist without re-downloading on every toggle.

    The baseline ElenaRyumina checkpoint uses HF-style key naming and is loaded
    by `load_vision_model`. The local fine-tuned bundle written by
    `scripts/finetune_fer2013.py` is `{"state_dict", "meta"}` with pathlib paths
    in `meta`, so it goes through `load_vision_finetuned` (which expects that
    layout and uses `weights_only=False` for the trusted local file).
    Both objects expose the same `.predict_fer7(image)` API.
    """
    if weights_path:
        return load_vision_finetuned(weights_path)
    return load_vision_model()


@st.cache_resource
def cached_synth():
    return load_synthesizer()


def _fer7_prob_df(p: np.ndarray) -> pd.DataFrame:
    """Categorical order fixes Altair y-axis: all 7 FER-7 labels, no label dropping."""
    return pd.DataFrame(
        {
            "emotion": pd.Categorical(
                [FER7_LABELS[i] for i in range(7)],
                categories=list(FER7_LABELS),
                ordered=True,
            ),
            "p": [float(p[i]) for i in range(7)],
        }
    )


def _bar_chart(p: np.ndarray, title: str, *, compact: bool = False) -> alt.Chart:
    """
    Horizontal bars: P on x, emotions on y. All 7 FER-7 labels are forced visible by
    sizing the chart with `alt.Step(...)` (per-band height) and using a nominal
    y-encoding with `labelOverlap=False`. Total chart height is then 7*step plus
    chrome, and Vega renders one tick label per band rather than auto-thinning.
    """
    df = _fer7_prob_df(p)
    step = 36 if compact else 42
    bar_size = 18 if compact else 22
    c = (
        alt.Chart(df, title=alt.TitleParams(text=title, anchor="start", fontSize=13, fontWeight=500))
        .mark_bar(cornerRadius=3, size=bar_size, color="#4a90d9", stroke="#7eb8f0", strokeWidth=0.3)
        .encode(
            x=alt.X("p:Q", title="Probability", scale=alt.Scale(domain=[0, 1], nice=False)),
            y=alt.Y(
                "emotion:N",
                title="",
                sort=list(FER7_LABELS),
                scale=alt.Scale(paddingInner=0.25, paddingOuter=0.15),
                axis=alt.Axis(
                    labelColor="#d0d3dc",
                    labelFontSize=12,
                    labelPadding=6,
                    labelOverlap=False,
                    labelLimit=140,
                ),
            ),
            tooltip=["emotion", alt.Tooltip("p:Q", title="P", format=".4f")],
        )
        .properties(height=alt.Step(step))
        .configure(background="transparent")
        .configure_view(stroke="transparent", cornerRadius=4)
        .configure_axisX(gridColor="#3a3f4d", domainColor="#5c6370", labelColor="#b8bcc8", titleColor="#b8bcc8")
        .configure_axisY(labelOverlap=False, labelLimit=140)
    )
    return c


def main() -> None:
    st.set_page_config(
        page_title="Bimodal Affective Alignment",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _fer_css()
    st.markdown(
        '<p class="main-title">Bimodal Affective Alignment</p>'
        '<p class="subtle">Text (GoEmotions → FER-7) + face (ResNet-50, FER2013) + late fusion + FLAN-T5</p>',
        unsafe_allow_html=True,
    )
    st.caption(f"Device: {device_preference()}")

    with st.sidebar:
        st.subheader("Trust weight α")
        alpha = st.slider(
            "α: trust in **text** vs **face**",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_ALPHA,
            step=0.05,
            help="Fused vector = α·P_text + (1−α)·P_face. Higher α follows language more; lower α follows expressions more.",
        )
        st.markdown(
            f"**{alpha:.2f}** text · **{1 - alpha:.2f}** face"
        )

        st.divider()
        st.subheader("Vision branch")
        ft_available = FINETUNE_CKPT_PATH.is_file()
        if ft_available:
            vision_choice = st.radio(
                "Face classifier weights",
                options=("Baseline (HF AffectNet)", "Fine-tuned (FER2013)"),
                index=1,
                help=(
                    f"Baseline = ElenaRyumina/face_emotion_recognition. "
                    f"Fine-tuned = {FINETUNE_CKPT_PATH} produced by "
                    "scripts/finetune_fer2013.py."
                ),
            )
            use_finetune = vision_choice.startswith("Fine-tuned")
        else:
            st.caption(
                "ℹ️ No fine-tuned checkpoint at "
                f"`{FINETUNE_CKPT_PATH}`. Run "
                "`python scripts/finetune_fer2013.py` to generate one, then reload."
            )
            use_finetune = False

        st.divider()
        run = st.button("Analyze & generate response", type="primary", use_container_width=True)
        st.caption("Models load on first use (may take a minute).")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.subheader("What you say (U)")
        utterance = st.text_area("Utterance", height=120, placeholder='e.g. "I\'m fine, really."', label_visibility="collapsed")
    with c2:
        st.subheader("Face frame")
        cam = st.camera_input("Webcam (or use upload below)", label_visibility="collapsed")
        up = st.file_uploader("Or upload a face image", type=["png", "jpg", "jpeg", "webp"])
        image: Image.Image | None = None
        if cam is not None:
            image = Image.open(BytesIO(cam.getvalue()))
        elif up is not None:
            image = Image.open(up)

    if not run:
        st.info("Enter text, optionally add a face image, then run **Analyze & generate**.")
        return

    if not (utterance and utterance.strip()):
        st.warning("Please enter some text in **What you say**.")
        return

    timings: dict[str, float] = {}
    with st.spinner("Loading / running text model…"):
        tm = cached_text()
        t_text0 = time.perf_counter()
        p_text, _, _ = tm.predict_fer7(utterance)
        timings["text_ms"] = (time.perf_counter() - t_text0) * 1000.0
    vision_weights = str(FINETUNE_CKPT_PATH) if use_finetune else None
    vision_branch_label = "fine-tuned (FER2013)" if use_finetune else "baseline (AffectNet)"
    if image is not None:
        with st.spinner(f"Running face model ({vision_branch_label})…"):
            vm = cached_vision(vision_weights)
            t_face0 = time.perf_counter()
            p_face, _, _ = vm.predict_fer7(image)
            timings["face_ms"] = (time.perf_counter() - t_face0) * 1000.0
    else:
        p_face = uniform_face_p_face()
        timings["face_ms"] = 0.0
        st.info("No image provided: using a **uniform** face prior (1/7 each).")

    t_fu0 = time.perf_counter()
    p_fuse, _idx_star, label_star = fuse(p_text, p_face, alpha=alpha)
    timings["fusion_ms"] = (time.perf_counter() - t_fu0) * 1000.0

    t_arg_text = int(np.argmax(p_text))
    t_arg_face = int(np.argmax(p_face))
    dissonant = t_arg_text != t_arg_face

    t_gen0 = time.perf_counter()
    with st.spinner("Generating empathetic response (FLAN-T5)…"):
        syn = cached_synth()
        response = syn.generate(utterance, label_star, p_fused=p_fuse)
    timings["t5_ms"] = (time.perf_counter() - t_gen0) * 1000.0

    st.markdown("---")
    st.markdown('<p class="section-title">Results</p>', unsafe_allow_html=True)

    res_left, res_right = st.columns([1, 1.15], gap="large")

    with res_left:
        if image is not None:
            st.image(
                image,
                caption=(
                    f"Captured face frame · {vision_branch_label} branch · "
                    f"face argmax: {FER7_LABELS[t_arg_face]}"
                ),
                use_container_width=True,
            )
        else:
            st.info(
                f"No face image provided — using a uniform face prior (1/7 per class). "
                f"Vision branch: {vision_branch_label}."
            )

        st.markdown(f"**You said (U):** _{utterance.strip()}_")

        if dissonant:
            st.markdown(
                f'<p><span class="badge-dissonance">Affective dissonance</span> '
                f"text: <b>{FER7_LABELS[t_arg_text]}</b> · face: <b>{FER7_LABELS[t_arg_face]}</b></p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p><span class="badge-congruent">Congruent signals</span> '
                "(same argmax for text and face)</p>",
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Text argmax", FER7_LABELS[t_arg_text])
        m2.metric("Face argmax", FER7_LABELS[t_arg_face])
        m3.metric(f"Fused (α={alpha:.2f})", label_star)

        st.markdown("**Empathetic response (R)**")
        st.markdown(
            f'<div class="card"><p style="margin:0; font-size:1.05rem; line-height:1.5;">{response}</p></div>',
            unsafe_allow_html=True,
        )

        total_ms = sum(v for v in timings.values())
        st.caption(
            f"text {timings['text_ms']:.0f} ms · face {timings['face_ms']:.0f} ms · "
            f"fusion {timings['fusion_ms']:.1f} ms · FLAN-T5 {timings['t5_ms']:.0f} ms · "
            f"end-to-end ≈ {total_ms:.0f} ms (Doherty threshold ~400 ms)."
        )

    with res_right:
        st.altair_chart(
            _bar_chart(p_text, "P_text — RoBERTa (GoEmotions → FER-7)", compact=True),
            use_container_width=True,
        )
        st.altair_chart(
            _bar_chart(p_face, "P_face — ResNet-50 (FER2013)", compact=True),
            use_container_width=True,
        )
        st.altair_chart(
            _bar_chart(p_fuse, f"Fused — α·P_text + (1−α)·P_face,  α = {alpha:.2f}", compact=True),
            use_container_width=True,
        )

    with st.expander("Debug: numeric vectors"):
        debug_df = pd.DataFrame(
            {
                "Class": list(FER7_LABELS),
                "P_text": np.asarray(p_text, dtype=float).ravel(),
                "P_face": np.asarray(p_face, dtype=float).ravel(),
                "P_fused": np.asarray(p_fuse, dtype=float).ravel(),
            }
        )
        st.dataframe(
            debug_df.style.format(
                {"P_text": "{:.4f}", "P_face": "{:.4f}", "P_fused": "{:.4f}"}
            ).bar(subset=["P_text", "P_face", "P_fused"], color="#3d6fa8"),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"argmax: P_text → **{FER7_LABELS[int(np.argmax(p_text))]}**, "
            f"P_face → **{FER7_LABELS[int(np.argmax(p_face))]}**, "
            f"P_fused (α={alpha:.2f}) → **{FER7_LABELS[int(np.argmax(p_fuse))]}**."
        )


if __name__ == "__main__":
    main()
