"""Stage 1 of the fine-tuning pipeline: join AFR articles with the RBA rate in
force on their publication date, filter to market-relevant articles, and draw
a stratified (by year) sample as raw candidates for teacher labeling.

No sentiment/direction label is produced here — that is stage 2
(label_with_teacher.py). This stage only performs the date-join and topic
filter so the two concerns stay separable and inspectable.

Run: python training/build_training_corpus.py
Output: training/data/candidates.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tools import afr_search, rba_tool  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_PATH = os.path.join(OUT_DIR, "candidates.jsonl")

# Held out for evaluation — these are the exact articles referenced by the
# public calibration questions (MHQ058, MHQ067, MHQ080). Never train on them.
HELD_OUT_ARTICLE_IDS = {184114, 213882, 175725}

# Broad net for "financial-market news" so training examples match the shape
# of what the agent will actually be asked to classify: company/ticker names,
# market/economy vocabulary, and RBA/rate terms.
MARKET_KEYWORDS = [
    "RBA", "cash rate", "interest rate", "interest rates", "rate rise", "rate cut",
    "ASX", "shares", "share price", "stocks", "stock market", "investors", "earnings",
    "profit", "profits", "dividend", "economy", "economic", "inflation", "the dollar",
    "Australian dollar", "bond", "bonds", "trading", "market", "markets",
    "AGL", "AMP", "ANZ", "Aurizon", "BHP", "CBA", "Cromwell", "GPT", "IAG", "NAB",
    "Qantas", "QBE", "Rio Tinto", "Stockland", "Suncorp", "Tabcorp", "TPG",
    "Transurban", "Commonwealth Bank", "National Australia Bank",
]

PER_YEAR_TARGET = 90
YEARS = list(range(2015, 2022))  # AFR + ASX coverage window


def main(seed: int = 0) -> None:
    random.seed(seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    rba = rba_tool.get_dataset()
    candidates_by_year: dict[int, list[dict]] = defaultdict(list)

    for year in YEARS:
        start, end = f"{year}-01-01", f"{year}-12-31"
        result = afr_search.search_any(
            MARKET_KEYWORDS, start_date=start, end_date=end, whole_word=True, max_examples=100_000
        )
        # search_any only returns up to max_examples in "examples"; request a
        # generous cap so we effectively get the full matched set for the year.
        pool = [e for e in result["examples"] if e["id"] not in HELD_OUT_ARTICLE_IDS]
        random.shuffle(pool)
        take = pool[:PER_YEAR_TARGET]
        for e in take:
            article = afr_search.get_article(e["id"])
            pubdate_iso = f"{e['pubdate'][0:4]}-{e['pubdate'][4:6]}-{e['pubdate'][6:8]}"
            rate_info = rba.rate_on_date(pubdate_iso)
            candidates_by_year[year].append(
                {
                    "id": e["id"],
                    "headline": article["headline"],
                    "subhead": article["subhead"],
                    "intro": article["intro"],
                    "text": article["text"],
                    "newspaper": article["newspaper"],
                    "pubdate": e["pubdate"],
                    "rba_target_pct": rate_info["target_pct"],
                }
            )
        print(f"{year}: matched={result['matched_record_count']} sampled={len(take)}")

    total = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for year in YEARS:
            for row in candidates_by_year[year]:
                f.write(json.dumps(row) + "\n")
                total += 1

    print(f"\nWrote {total} candidates to {OUT_PATH}")
    print(f"Held out {len(HELD_OUT_ARTICLE_IDS)} calibration articles from sampling.")


if __name__ == "__main__":
    main()
