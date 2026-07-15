"""Full-text search tool over the AFR news corpus, honoring the challenge's
non-negotiable search scope: HEADLINE + SUBHEAD + INTRO + TEXT combined,
case-insensitive, matched once per record (a record counts once even if the
pattern appears in multiple fields), with word-boundary anchors for
whole-word / acronym searches.
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Optional

from tools.build_afr_index import DB_PATH, build as build_index


def _ensure_index() -> str:
    if not os.path.exists(DB_PATH):
        build_index()
    return DB_PATH


def _to_pubdate(s: str) -> str:
    """Accepts 'YYYY-MM-DD', 'YYYYMMDD', or 'D Mon YYYY' and returns 'YYYYMMDD'."""
    s = s.strip()
    if re.fullmatch(r"\d{8}", s):
        return s
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    if re.fullmatch(r"\d{4}", s):
        return s + "0101"
    raise ValueError(f"Unrecognized date format: {s!r}")


def _compile_pattern(pattern: str, whole_word: bool) -> re.Pattern:
    escaped = re.escape(pattern.strip())
    if whole_word:
        escaped = rf"\b{escaped}\b"
    return re.compile(escaped, re.IGNORECASE)


def search(
    pattern: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    whole_word: bool = True,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Count + sample of AFR records whose combined HEADLINE/SUBHEAD/INTRO/TEXT
    matches `pattern` at least once, matched once per record."""
    db_path = _ensure_index()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where = []
    params: list[Any] = []
    if start_date:
        where.append("pubdate >= ?")
        params.append(_to_pubdate(start_date))
    if end_date:
        where.append("pubdate <= ?")
        params.append(_to_pubdate(end_date))
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    regex = _compile_pattern(pattern, whole_word)
    cur = conn.execute(
        f"SELECT id, headline, subhead, newspaper, pubdate, combined FROM articles {where_clause}",
        params,
    )

    matched = 0
    examples = []
    scanned = 0
    for row in cur:
        scanned += 1
        if regex.search(row["combined"]):
            matched += 1
            if len(examples) < max_examples:
                m = regex.search(row["combined"])
                start = max(0, m.start() - 60)
                end = min(len(row["combined"]), m.end() + 60)
                snippet = row["combined"][start:end].replace("\n", " ").strip()
                examples.append(
                    {
                        "id": row["id"],
                        "headline": row["headline"],
                        "newspaper": row["newspaper"],
                        "pubdate": row["pubdate"],
                        "snippet": snippet,
                    }
                )
    conn.close()
    return {
        "pattern": pattern,
        "whole_word": whole_word,
        "records_scanned": scanned,
        "matched_record_count": matched,
        "examples": examples,
    }


def counts_by_period(
    pattern: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    whole_word: bool = True,
) -> dict[str, Any]:
    """Matched-record counts broken down by year AND by year-month, in one
    pass -- use this instead of calling `search` once per year/month when a
    question asks which period has the highest count."""
    db_path = _ensure_index()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where = []
    params: list[Any] = []
    if start_date:
        where.append("pubdate >= ?")
        params.append(_to_pubdate(start_date))
    if end_date:
        where.append("pubdate <= ?")
        params.append(_to_pubdate(end_date))
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    regex = _compile_pattern(pattern, whole_word)
    cur = conn.execute(f"SELECT pubdate, combined FROM articles {where_clause}", params)

    by_year: dict[str, int] = {}
    by_year_month: dict[str, int] = {}
    total = 0
    for row in cur:
        if not row["pubdate"] or len(row["pubdate"]) < 6:
            continue
        if regex.search(row["combined"]):
            total += 1
            year, month = row["pubdate"][:4], row["pubdate"][:6]
            by_year[year] = by_year.get(year, 0) + 1
            by_year_month[month] = by_year_month.get(month, 0) + 1
    conn.close()

    peak_year = max(by_year, key=by_year.get) if by_year else None
    peak_month = max(by_year_month, key=by_year_month.get) if by_year_month else None
    return {
        "pattern": pattern,
        "whole_word": whole_word,
        "total_matched_record_count": total,
        "counts_by_year": dict(sorted(by_year.items())),
        "counts_by_year_month": dict(sorted(by_year_month.items())),
        "peak_year": peak_year,
        "peak_year_count": by_year.get(peak_year) if peak_year else None,
        "peak_month": peak_month,
        "peak_month_count": by_year_month.get(peak_month) if peak_month else None,
    }


def search_any(
    patterns: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    whole_word: bool = True,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Records matching ANY of the given patterns (still once per record)."""
    db_path = _ensure_index()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where = []
    params: list[Any] = []
    if start_date:
        where.append("pubdate >= ?")
        params.append(_to_pubdate(start_date))
    if end_date:
        where.append("pubdate <= ?")
        params.append(_to_pubdate(end_date))
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    regexes = [_compile_pattern(p, whole_word) for p in patterns]
    cur = conn.execute(
        f"SELECT id, headline, subhead, newspaper, pubdate, combined FROM articles {where_clause}",
        params,
    )
    matched = 0
    examples = []
    scanned = 0
    for row in cur:
        scanned += 1
        if any(r.search(row["combined"]) for r in regexes):
            matched += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "id": row["id"],
                        "headline": row["headline"],
                        "newspaper": row["newspaper"],
                        "pubdate": row["pubdate"],
                    }
                )
    conn.close()
    return {
        "patterns": patterns,
        "whole_word": whole_word,
        "records_scanned": scanned,
        "matched_record_count": matched,
        "examples": examples,
    }


def get_article(article_id: int) -> dict[str, Any]:
    db_path = _ensure_index()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, headline, subhead, intro, text, newspaper, pubdate FROM articles WHERE id = ?",
        (article_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return {"found": False, "id": article_id}
    return {"found": True, **dict(row)}


def dataset_summary() -> dict[str, Any]:
    db_path = _ensure_index()
    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    min_d, max_d = conn.execute(
        "SELECT MIN(pubdate), MAX(pubdate) FROM articles WHERE pubdate != ''"
    ).fetchone()
    undated = conn.execute("SELECT COUNT(*) FROM articles WHERE pubdate = ''").fetchone()[0]
    conn.close()
    return {
        "total_articles": total,
        "earliest_pubdate": min_d,
        "latest_pubdate": max_d,
        "records_missing_pubdate": undated,
    }


TOOL_SCHEMA = {
    "name": "search_afr",
    "description": (
        "Query the AFR news corpus (219,538 articles). operation='dataset_summary': total count + "
        "earliest/latest date (use this for date coverage, NOT a keyword search). "
        "operation='get_article' + article_id: fetch one article's full text. operation omitted/'search': "
        "full-text search over HEADLINE+SUBHEAD+INTRO+TEXT combined, case-insensitive, once per record; "
        "whole_word=true (default) for names/tickers/acronyms like 'NAB'. Returns matched_record_count + "
        "up to max_examples sample articles. operation='counts_by_period': counts by year AND year-month "
        "in ONE call with peak_year/peak_month precomputed -- use instead of looping search per period."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["search", "dataset_summary", "get_article", "counts_by_period"],
                "description": "Omit or 'search' for a pattern/patterns search; 'dataset_summary' for coverage; 'get_article' with article_id to fetch one article; 'counts_by_period' for a year/month breakdown in one call.",
            },
            "article_id": {"type": "integer", "description": "Required when operation='get_article'."},
            "pattern": {"type": "string", "description": "Word or phrase to search for."},
            "patterns": {"type": "array", "items": {"type": "string"}, "description": "Alternative: match ANY of these patterns."},
            "start_date": {"type": "string", "description": "Optional inclusive start date (YYYY-MM-DD)."},
            "end_date": {"type": "string", "description": "Optional inclusive end date (YYYY-MM-DD)."},
            "whole_word": {"type": "boolean", "description": "Word-boundary match. Default true."},
            "max_examples": {"type": "integer"},
        },
        "required": [],
    },
}


def run(args: dict) -> Any:
    if args.get("operation") == "get_article" or "article_id" in args:
        return get_article(int(args["article_id"]))
    if args.get("operation") == "dataset_summary":
        return dataset_summary()
    if args.get("operation") == "counts_by_period":
        return counts_by_period(
            args["pattern"], args.get("start_date"), args.get("end_date"), bool(args.get("whole_word", True))
        )
    whole_word = bool(args.get("whole_word", True))
    max_examples = int(args.get("max_examples", 10))
    if args.get("patterns"):
        return search_any(
            args["patterns"], args.get("start_date"), args.get("end_date"), whole_word, max_examples
        )
    return search(
        args["pattern"], args.get("start_date"), args.get("end_date"), whole_word, max_examples
    )


if __name__ == "__main__":
    import time

    t0 = time.time()
    print(dataset_summary())
    t1 = time.time()
    print(f"summary query: {t1 - t0:.2f}s")
    result = search("NAB", whole_word=True)
    t2 = time.time()
    print(f"NAB search: matched={result['matched_record_count']} scanned={result['records_scanned']} in {t2 - t1:.2f}s")
    print(result["examples"][:2])
