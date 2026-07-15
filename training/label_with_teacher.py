"""Stage 2 of the fine-tuning pipeline: label each candidate with a
sentiment + market-direction target using the larger `agent-brain` model
(Qwen3.6-35B) as a teacher, then re-express the target in the SHORT prompt
shape the student (Nemotron `domain-ft`) will actually see at inference time.

Labels are generated purely from the article's text + the applicable RBA rate
-- never from forward ASX returns -- so the trained model learns to read the
article, not to peek at hindsight price action.

Requires the LiteLLM proxy (agent-brain alias) to be reachable.
Run: python training/label_with_teacher.py [--limit N] [--workers 12]
Output: training/data/labeled.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.jsonl")
LABELED_PATH = os.path.join(DATA_DIR, "labeled.jsonl")
FAILED_PATH = os.path.join(DATA_DIR, "labeling_failures.jsonl")

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "EMPTY")
BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "agent-brain")

TEACHER_ARTICLE_CHARS = 1500

TEACHER_PROMPT_TEMPLATE = """You are a financial-markets analyst. Read the AFR article below together with \
the RBA cash rate that was in force on its publication date, then classify its financial-market sentiment \
and the likely near-term market direction it implies.

Guidance:
- Sentiment is about the tone/implications of THIS article's content, not any known future outcome.
- positive = clearly bullish/favorable for the relevant shares/sector; negative = clearly bearish/unfavorable; \
mixed = contains both good and bad signals or is ambiguous, optionally qualified with a bias \
(e.g. "mixed with a negative bias").
- Rate-sensitive sectors (banks, REITs/property trusts, utilities, highly-leveraged companies) generally react \
unfavorably to rising-rate expectations and favorably to falling-rate/rate-cut expectations; exporters/miners \
can benefit from a weaker currency that often accompanies rate cuts.
- Be specific about which shares/sector the direction applies to when the article names one.
- Do not invent a specific numeric price target or return.

Article headline: {headline}
Published: {pubdate}
RBA cash rate target in force on publication date: {rate}%

Article text:
{text}

Respond in exactly two lines with no other text:
Sentiment: <label>
Likely direction: <short phrase>"""

LINE_RE = re.compile(
    r"Sentiment:\s*(?P<sentiment>[^\n]+)\s*\n\s*Likely direction:\s*(?P<direction>[^\n]+)",
    re.IGNORECASE,
)


def call_teacher(client: httpx.Client, prompt: str) -> str | None:
    resp = client.post(
        f"{LITELLM_BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_KEY}"},
        json={
            "model": BRAIN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def label_one(client: httpx.Client, cand: dict) -> dict | None:
    prompt = TEACHER_PROMPT_TEMPLATE.format(
        headline=cand["headline"],
        pubdate=cand["pubdate"],
        rate=cand["rba_target_pct"],
        text=cand["text"][:TEACHER_ARTICLE_CHARS],
    )
    raw = call_teacher(client, prompt)
    if raw is None:
        return None
    m = LINE_RE.search(raw)
    if not m:
        # one retry with a blunter instruction, in case the model added preamble
        raw2 = call_teacher(
            client, prompt + "\n\n(Output ONLY the two lines above. No other text.)"
        )
        m = LINE_RE.search(raw2 or "")
        if not m:
            return None
    sentiment = m.group("sentiment").strip().rstrip(".")
    direction = m.group("direction").strip().rstrip(".")
    return {
        **cand,
        "sentiment": sentiment,
        "direction": direction,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    candidates = []
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    if args.limit:
        candidates = candidates[: args.limit]

    labeled: list[dict] = []
    failed: list[dict] = []

    with httpx.Client() as client, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(label_one, client, c): c for c in candidates}
        done = 0
        for fut in as_completed(futures):
            cand = futures[fut]
            done += 1
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                result = None
                print(f"[{done}/{len(candidates)}] ERROR id={cand['id']}: {exc}", file=sys.stderr)
            if result is None:
                failed.append(cand)
            else:
                labeled.append(result)
            if done % 25 == 0 or done == len(candidates):
                print(f"[{done}/{len(candidates)}] labeled={len(labeled)} failed={len(failed)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LABELED_PATH, "w", encoding="utf-8") as f:
        for row in labeled:
            f.write(json.dumps(row) + "\n")
    with open(FAILED_PATH, "w", encoding="utf-8") as f:
        for row in failed:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(labeled)} labeled examples to {LABELED_PATH}")
    print(f"Wrote {len(failed)} failures to {FAILED_PATH}")


if __name__ == "__main__":
    main()
