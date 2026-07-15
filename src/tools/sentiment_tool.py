"""Sentiment/market-direction tool, routed through the fine-tuned domain
model (DOMAIN_FT_MODEL alias, served behind LiteLLM). Takes a retrieved AFR
article and the applicable RBA cash-rate target, and returns a sentiment
classification (positive/negative/mixed) plus a likely market-direction
call. Never asked for, and never returns, a fabricated numeric price/return.

The prompt shape here MUST match training/prepare_dataset.py's
STUDENT_PROMPT_TEMPLATE exactly -- that's what the LoRA adapter was trained
on, so train/inference stay consistent.
"""
from __future__ import annotations

import os
import re
from typing import Any

import httpx

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "EMPTY")
DOMAIN_FT_MODEL = os.environ.get("DOMAIN_FT_MODEL", "domain-ft")

ARTICLE_CHARS = 1200

# Must match training/prepare_dataset.py::STUDENT_PROMPT_TEMPLATE exactly.
STUDENT_PROMPT_TEMPLATE = """Article headline: {headline}
Applicable RBA cash rate target on publication date: {rate}%

Article text:
{text}

Classify this article's financial-market sentiment as positive, negative, or mixed \
(optionally qualified, e.g. "mixed with a negative bias"), and state the likely \
direction for the relevant shares or sector. Answer in exactly two short lines:
Sentiment: <label>
Likely direction: <short phrase>"""

LINE_RE = re.compile(
    r"Sentiment:\s*(?P<sentiment>[^\n]+)\s*\n\s*Likely direction:\s*(?P<direction>[^\n]+)",
    re.IGNORECASE,
)


def assess(headline: str, article_text: str, rba_target_pct: float) -> dict[str, Any]:
    prompt = STUDENT_PROMPT_TEMPLATE.format(
        headline=headline, rate=rba_target_pct, text=article_text[:ARTICLE_CHARS]
    )
    with httpx.Client() as client:
        resp = client.post(
            f"{LITELLM_BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": DOMAIN_FT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

    m = LINE_RE.search(raw)
    if not m:
        return {
            "headline": headline,
            "rba_target_pct": rba_target_pct,
            "sentiment": None,
            "direction": None,
            "raw_response": raw,
            "parse_error": True,
        }
    return {
        "headline": headline,
        "rba_target_pct": rba_target_pct,
        "sentiment": m.group("sentiment").strip().rstrip("."),
        "direction": m.group("direction").strip().rstrip("."),
        "parse_error": False,
    }


TOOL_SCHEMA = {
    "name": "assess_sentiment",
    "description": (
        "Classify an AFR article's financial-market sentiment (positive/negative/mixed) and likely "
        "market direction, given the article's headline/text and the RBA cash rate target that was "
        "in force on its publication date. Routed through the fine-tuned domain model. Does not "
        "produce a numeric price/return forecast -- pair with query_asx for actual price data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "article_text": {"type": "string"},
            "rba_target_pct": {"type": "number"},
        },
        "required": ["headline", "article_text", "rba_target_pct"],
    },
}


def run(args: dict) -> Any:
    return assess(args["headline"], args["article_text"], float(args["rba_target_pct"]))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tools import afr_search, rba_tool

    art = afr_search.get_article(184114)
    rate = rba_tool.get_dataset().rate_on_date("2021-02-23")
    result = assess(art["headline"], art["text"], rate["target_pct"])
    print(result)
