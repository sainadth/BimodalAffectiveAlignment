import pytest

from bimodal_empathy.llm_generator import AffectContext, build_user_prompt, resolve_provider

CTX_DISS = AffectContext(text_label="Happy", face_label="Sad", fused_label="Sad", dissonant=True)
CTX_CONG = AffectContext(text_label="Happy", face_label="Happy", fused_label="Happy", dissonant=False)
CTX_NOFACE = AffectContext(text_label="Happy", face_label=None, fused_label="Happy", dissonant=False)


def test_text_only_never_leaks_face():
    p = build_user_prompt("I'm fine, really.", "text_only", CTX_DISS)
    assert "happy" in p and "face" not in p.lower() and "sad" not in p.lower()


def test_bimodal_names_dissonance_when_present():
    p = build_user_prompt("I'm fine, really.", "bimodal", CTX_DISS)
    assert "looks sad" in p and "disagree" in p


def test_bimodal_congruent_gives_overall_only():
    p = build_user_prompt("Great day!", "bimodal", CTX_CONG)
    assert "disagree" not in p and "Overall the user seems happy" in p


def test_bimodal_without_face_says_so():
    p = build_user_prompt("Great day!", "bimodal", CTX_NOFACE)
    assert "No facial expression" in p and "disagree" not in p


def test_random_uses_given_label_only():
    p = build_user_prompt("Great day!", "random", CTX_DISS, random_label="Fear")
    assert "fear" in p and "happy" not in p.lower() and "sad" not in p.lower()
    with pytest.raises(ValueError):
        build_user_prompt("x", "random", CTX_DISS)


def test_llm_only_carries_no_sensor_output():
    p = build_user_prompt("I'm fine, really.", "llm_only", CTX_DISS)
    assert "sensor" not in p.lower() and "seems" not in p and "looks" not in p
    assert "I'm fine, really." in p


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        build_user_prompt("x", "nope", CTX_DISS)


def test_resolve_provider_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert resolve_provider() == ("groq", "openai/gpt-oss-120b")
    assert resolve_provider("anthropic") == ("anthropic", "claude-opus-5")
    monkeypatch.setenv("LLM_MODEL", "x-model")
    assert resolve_provider("groq")[1] == "x-model"
    with pytest.raises(ValueError):
        resolve_provider("openai")


def test_v3_prompts_respect_move_and_label():
    from bimodal_empathy.llm_generator import build_user_prompt_v3

    a = build_user_prompt_v3("I'm fine.", "assert", "Sad")
    k = build_user_prompt_v3("I'm fine.", "ask", "Sad")
    n_a = build_user_prompt_v3("I'm fine.", "assert", None)
    n_k = build_user_prompt_v3("I'm fine.", "ask", "")
    # both moves must carry the exact label word, so they differ only in speech act
    assert "state how they seem" in a.lower() and '"sad" itself' in a and "you seem sad" in a
    assert "Do not state their emotion as a fact" in k and "are you feeling sad right now?" in k
    assert "Do not substitute a vaguer word" in a and "Do not substitute a vaguer word" in k
    assert "No affect sensor output" in n_a and "own reading" in n_a
    assert "No affect sensor output" in n_k and "open question about how they are feeling" in n_k
    with pytest.raises(ValueError):
        build_user_prompt_v3("x", "tell", "Sad")
