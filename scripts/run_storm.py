#!/usr/bin/env python3
"""Run Stanford STORM on the paper's research question for positioning / related-work scaffolding.

  python scripts/run_storm.py --out docs/storm
Uses Groq (openai/gpt-oss-120b via LiteLLM) and keyless DuckDuckGo retrieval. Output is a
drafting aid: every source must be verified before it is cited anywhere.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import DuckDuckGoSearchRM


class DDGSRM(DuckDuckGoSearchRM):
    """STORM's DuckDuckGo retriever on the maintained `ddgs` client (old package returns no results)."""

    def __init__(self, *a, **kw):
        import duckduckgo_search  # noqa: F401  (parent import check)
        super().__init__(*a, **kw)
        from ddgs import DDGS

        self.ddgs = DDGS()

    def request(self, query: str):
        import time

        for attempt in range(4):
            try:
                res = self.ddgs.text(query, max_results=self.k, safesearch="moderate", region=self.duck_duck_go_region)
                if res:
                    return res
            except Exception as exc:  # DDGSException("No results found.") or transient rate limit
                if "No results" in str(exc):
                    return []
                time.sleep(2 * (attempt + 1))
            time.sleep(1.0)
        return []

TOPIC = "Misattributed emotion in LLM chatbot replies: label correctness, asserting versus asking, and perceived empathy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/storm")
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--model", default="groq/openai/gpt-oss-120b")
    ap.add_argument("--perspectives", type=int, default=4)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        sys.exit("GROQ_API_KEY missing")
    common = dict(api_key=key, temperature=0.7, top_p=0.9, reasoning_effort="low")
    fast = LitellmModel(model=args.model, max_tokens=600, **common)
    long = LitellmModel(model=args.model, max_tokens=3000, **common)
    lm = STORMWikiLMConfigs()
    lm.set_conv_simulator_lm(fast)
    lm.set_question_asker_lm(fast)
    lm.set_outline_gen_lm(fast)
    lm.set_article_gen_lm(long)
    lm.set_article_polish_lm(long)

    rm = DDGSRM(k=args.top_k, safe_search="On", region="us-en")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    runner = STORMWikiRunner(
        STORMWikiRunnerArguments(output_dir=str(out), max_conv_turn=args.turns, max_perspective=args.perspectives,
                                 search_top_k=args.top_k, retrieve_top_k=args.top_k, max_thread_num=args.threads),
        lm, rm,
    )
    runner.run(topic=args.topic, do_research=True, do_generate_outline=True, do_generate_article=True, do_polish_article=True)
    runner.post_run()
    runner.summary()
    print("output dir:", out)


if __name__ == "__main__":
    main()
