"""Shared helpers for the Streamlit apps (thin wrapper over bimodal_empathy.pipeline)."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(Path(".env"), override=False)
except ImportError:
    pass

from bimodal_empathy.llm_generator import resolve_provider
from bimodal_empathy.pipeline import FINETUNE_CKPT_PATH, analyze_affect, warm_up  # noqa: F401  (re-exported)
from bimodal_empathy.response_synthesizer import load_synthesizer


@st.cache_resource
def cached_synth():
    return load_synthesizer()


def llm_configured(required_provider: str | None = None) -> bool:
    try:
        provider, _ = resolve_provider()
    except ValueError:
        return False
    if required_provider and provider != required_provider:
        return False
    key = "GROQ_API_KEY" if provider == "groq" else "ANTHROPIC_API_KEY"
    return bool((os.environ.get(key) or "").strip())
