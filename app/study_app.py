"""Participant-facing study app (design v2). Run:  streamlit run app/study_app.py

No mode switch, no model internals, no researcher controls. Everything the
participant sees is defined here; see docs/STUDY_PROTOCOL.md for the design.
"""

from __future__ import annotations

import html
import random as _random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import analyze_affect, llm_configured, warm_up  # noqa: E402

from bimodal_empathy.config import DEFAULT_ALPHA, FER7_LABELS
from bimodal_empathy.llm_generator import generate_reply_v3
from bimodal_empathy.study_log import (
    FELT_OPTIONS,
    MANIP_CHECK_TEXT,
    MANIP_OPTIONS,
    PETS_ITEMS,
    TRIAL_ITEMS,
    TRIALS_PER_PARTICIPANT,
    append_pets_global,
    append_study_row,
    completed_trials,
    condition_schedule,
    normalize_pid,
    resolve_shown_label,
    withdraw_participant,
)

CONSENT_VERSION = "v5-lab-correctness-2026-09-14"
REQUIRED_PROVIDER = "groq"
S = "study_v2"  # session-state root key

CONSENT_MD = (Path(__file__).resolve().parent.parent / "server" / "consent_remote.md").read_text(encoding="utf-8").replace("(online study)", "(lab study)").replace("Webcam frames are sent encrypted to the research server, used only to estimate a facial-expression label, and discarded immediately; they are never stored, viewed by a person, or shared.", "Webcam frames are processed on this computer only to estimate a facial-expression label and are never saved or transmitted.")

DEBRIEF_MD = (Path(__file__).resolve().parent.parent / "server" / "debrief_remote.md").read_text(encoding="utf-8")


def _state() -> dict:
    return st.session_state.setdefault(S, {"phase": "consent"})


def _css() -> None:
    st.markdown(
        """
<style>
  .card { background: rgba(127,127,127,0.10); border: 1px solid rgba(127,127,127,0.25);
          border-radius: 12px; padding: 1rem 1.1rem; margin: 0.6rem 0 1rem; font-size: 1.1rem; line-height: 1.55; }
  section[data-testid="stSidebar"] { display: none; }
  #MainMenu, footer, header { visibility: hidden; }
</style>
        """,
        unsafe_allow_html=True,
    )


def consent_screen() -> None:
    st.markdown(CONSENT_MD)
    agree = st.checkbox("I have read the information above and agree to take part.", key="consent_cb")
    if st.button("Continue", type="primary", disabled=not agree):
        s = _state()
        s["consent_ts"] = datetime.now(timezone.utc).isoformat()
        s["phase"] = "start"
        st.rerun()


def start_screen() -> None:
    s = _state()
    st.markdown("### Start")
    pid_raw = st.text_input("Participant number (from your study sheet)", key="pid_input")
    pid = normalize_pid(pid_raw) if pid_raw.strip() else ""
    ok = bool(re.fullmatch(r"[a-z]*\d{2,}", pid))  # e.g. p07; rejects free text so schedules stay counterbalanced
    if pid_raw and not ok:
        st.warning("Please enter the participant number exactly as printed, e.g. p07.")
    if st.button("Begin", type="primary", disabled=not ok):
        done = completed_trials(pid)
        s.update({
            "pid": pid,
            "schedule": condition_schedule(pid),
            "trial": done,
            "trial_phase": "input",
            "analysis": None,
            "attempts": 0,
        })
        with st.spinner("Preparing…"):
            warm_up(use_finetune=True)
        if done > 0:
            st.info(f"Resuming from turn {done + 1}.")
        s["phase"] = "trials" if done < TRIALS_PER_PARTICIPANT else "pets"
        st.rerun()


def _reply_card(text: str) -> None:
    st.markdown(f'<div class="card">{html.escape(text)}</div>', unsafe_allow_html=True)


def trial_screen() -> None:
    s = _state()
    t = s["trial"]
    spec = s["schedule"][t]
    st.progress(t / TRIALS_PER_PARTICIPANT, text=f"Turn {t + 1} of {TRIALS_PER_PARTICIPANT}")

    if s["trial_phase"] == "input":
        if spec["scripted"]:
            st.info(f"**For this turn:** {spec['scripted_prompt']}")
        else:
            st.write("Type a short message about how you are feeling right now, or about your day.")
        utterance = st.text_area("Your message", key=f"utt_{t}", height=110, label_visibility="collapsed",
                                 placeholder="Type here…")
        cam = st.camera_input("Take a photo of your face", key=f"cam_{t}")
        image = None
        if cam is not None:
            try:
                image = Image.open(BytesIO(cam.getvalue()))
            except Exception:
                st.error("Could not read the photo. Please retake it.")
        ready = bool(utterance and utterance.strip()) and image is not None
        if not ready:
            st.caption("Both a message and a photo are needed for this turn.")
        if st.button("Send", type="primary", disabled=not ready, key=f"send_{t}"):
            s["attempts"] = s.get("attempts", 0) + 1
            try:
                with st.spinner("One moment…"):
                    a = analyze_affect(utterance, image, alpha=DEFAULT_ALPHA, use_finetune=True)
            except Exception as exc:
                st.error("Something went wrong. Please press Send again.")
                st.caption(f"({type(exc).__name__})")
                return
            a["utterance"] = utterance.strip()
            s["analysis"] = a
            s["trial_phase"] = "felt"
            st.rerun()

    elif s["trial_phase"] == "felt":
        st.write("Before you see the reply: **right now, I mostly feel…**")
        felt = st.radio("felt", options=list(FELT_OPTIONS), index=None, horizontal=True,
                        key=f"felt_{t}", label_visibility="collapsed")
        if st.button("Show the reply", type="primary", disabled=felt is None, key=f"show_{t}"):
            a = s["analysis"]
            shown, eff = resolve_shown_label(spec["label_source"], felt, a["fused_label"], seed=f"{s['pid']}::{t}")
            try:
                with st.spinner("Thinking…"):
                    t0 = time.perf_counter()
                    rl = generate_reply_v3(a["utterance"], spec["move"], shown or None)
                    gen_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                st.error("The agent could not answer this time. Please try again.")
                st.caption(f"({type(exc).__name__})")
                return
            if not rl.text.strip():
                st.error("The agent gave an empty answer. Please try again.")
                return
            a.update({"felt_label": felt, "shown_label": shown, "correctness_effective": eff, "reply": rl.text,
                      "generator": f"{rl.provider}:{rl.model}", "e2e_ms": sum(a["timings"].values()) + gen_ms})
            s["trial_phase"] = "rate"
            st.rerun()

    else:  # rate
        a = s["analysis"]
        st.markdown(f"**You wrote:** _{html.escape(a['utterance'])}_")
        st.markdown("**The agent replied:**")
        _reply_card(a["reply"])
        st.caption("1 = strongly disagree · 5 = strongly agree")
        vals = {}
        for key, text in TRIAL_ITEMS:
            vals[key] = st.radio(text, options=[1, 2, 3, 4, 5], index=None, horizontal=True, key=f"{key}_{t}")
        st.write(MANIP_CHECK_TEXT)
        manip = st.radio("manip", options=list(MANIP_OPTIONS), index=None, horizontal=True,
                         key=f"manip_{t}", label_visibility="collapsed")
        done = all(v is not None for v in vals.values()) and manip is not None
        if st.button("Save and continue", type="primary", disabled=not done, key=f"save_{t}"):
            try:
                append_study_row(
                    participant_id=s["pid"], trial_index=t, block=spec["block"], condition=spec["condition"],
                    correctness_effective=a["correctness_effective"], scripted=spec["scripted"],
                    scripted_prompt=spec["scripted_prompt"], felt_label=a["felt_label"], shown_label=a["shown_label"],
                    manip_check_label=manip, ratings={k: int(v) for k, v in vals.items()}, alpha=a["alpha"],
                    vision_branch=a["vision_branch"], had_face_image=a["had_face_image"], face_detected=a["face_detected"],
                    argmax_dissonant=a["argmax_dissonant"], valence_dissonant=a["valence_dissonant"],
                    text_argmax=a["text_label"], face_argmax=a["face_label"] or "none", fused_label=a["fused_label"],
                    generator=a["generator"], attempts=s.get("attempts", 1), e2e_ms=a["e2e_ms"],
                    utterance=a["utterance"], response=a["reply"],
                )
            except Exception as exc:
                st.error(f"Could not save this turn ({type(exc).__name__}). Please tell the researcher.")
                return
            s["trial"] = t + 1
            s["trial_phase"] = "input"
            s["analysis"] = None
            s["attempts"] = 0
            if s["trial"] >= TRIALS_PER_PARTICIPANT:
                s["phase"] = "pets"
            st.rerun()


def pets_screen() -> None:
    s = _state()
    st.markdown("### Last step: the agent overall")
    st.caption("Thinking about all the replies you received. 1 = strongly disagree · 5 = strongly agree")
    vals = {}
    for key, text in PETS_ITEMS:
        vals[key] = st.radio(text, options=[1, 2, 3, 4, 5], index=None, horizontal=True, key=f"pets_{key}")
    st.write("Did you notice any pattern in how the agent replied?")
    funnel = st.radio("funnel", options=["No", "Unsure", "Yes"], index=None, horizontal=True,
                      key="funnel_radio", label_visibility="collapsed")
    free = st.text_input("If yes, what did you notice? (optional)", key="funnel_text")
    done = all(v is not None for v in vals.values()) and funnel is not None
    if st.button("Finish", type="primary", disabled=not done):
        try:
            append_pets_global(
                participant_id=s["pid"], pets={k: int(v) for k, v in vals.items()},
                funnel_noticed_pattern=funnel.lower(), funnel_free_text=free,
                consent_version=CONSENT_VERSION, consent_ts_iso_utc=s.get("consent_ts", ""),
            )
        except Exception as exc:
            st.error(f"Could not save ({type(exc).__name__}). Please tell the researcher.")
            return
        s["phase"] = "debrief"
        st.rerun()


def debrief_screen() -> None:
    s = _state()
    st.markdown(DEBRIEF_MD)
    if s.get("withdrawn"):
        st.success("Your data have been deleted. Thank you for your time.")
        return
    col1, col2 = st.columns(2)
    if col1.button("Keep my data (finish)", type="primary"):
        s["phase"] = "done"
        st.rerun()
    if col2.button("Withdraw my data"):
        n = withdraw_participant(s["pid"])
        s["withdrawn"] = True
        s["phase"] = "done"
        st.session_state["_withdraw_n"] = n
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Conversation study", page_icon="💬", layout="centered",
                       initial_sidebar_state="collapsed")
    _css()
    if not llm_configured(REQUIRED_PROVIDER):
        st.error("Study configuration error: the reply generator is not available. Please call the researcher.")
        st.caption(f"Needs LLM_PROVIDER={REQUIRED_PROVIDER} and GROQ_API_KEY in .env (run from the repo root).")
        return
    s = _state()
    phase = s["phase"]
    if phase == "consent":
        consent_screen()
    elif phase == "start":
        start_screen()
    elif phase == "trials":
        trial_screen()
    elif phase == "pets":
        pets_screen()
    elif phase == "debrief":
        debrief_screen()
    else:
        if s.get("withdrawn"):
            st.success(f"Your data have been deleted ({st.session_state.get('_withdraw_n', 0)} records). Thank you.")
        else:
            st.success("All done — thank you for taking part. You can close this window.")
        st.caption("Researcher: restart the app for the next participant.")


if __name__ == "__main__":
    main()
