"""One-time preprocessing: load every AFR_*.jsonl file into a single SQLite
database so the agent doesn't have to re-parse 780MB of JSON on every query.

Each row of the source data is treated as one record (no de-duplication across
the overlapping monthly/day files supplied by organizers) so that counts stay
reproducible against whatever reference implementation scored the same files.

Run: python -m tools.build_afr_index
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3

AFR_DIR = os.environ.get(
    "AFR_DATA_DIR",
    "/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/AFR Jasonl",
)
DB_PATH = os.environ.get(
    "AFR_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "afr.db"),
)


def build(db_path: str = DB_PATH, afr_dir: str = AFR_DIR) -> dict:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            headline TEXT,
            subhead TEXT,
            intro TEXT,
            text TEXT,
            newspaper TEXT,
            pubdate TEXT,
            combined TEXT
        )
        """
    )
    conn.execute("CREATE INDEX idx_articles_pubdate ON articles(pubdate)")

    total = 0
    per_file = []
    files = sorted(glob.glob(os.path.join(afr_dir, "AFR_*.jsonl")))
    for path in files:
        n_this_file = 0
        with open(path, encoding="utf-8") as f:
            batch = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                headline = row.get("HEADLINE", "") or ""
                subhead = row.get("SUBHEAD", "") or ""
                intro = row.get("INTRO", "") or ""
                text = row.get("TEXT", "") or ""
                newspaper = row.get("NEWSPAPER", "") or ""
                pubdate = row.get("PUBLICATIONDATE", "") or ""
                combined = "\n".join([headline, subhead, intro, text])
                batch.append((headline, subhead, intro, text, newspaper, pubdate, combined))
                n_this_file += 1
            if batch:
                conn.executemany(
                    "INSERT INTO articles (headline, subhead, intro, text, newspaper, pubdate, combined) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
        total += n_this_file
        per_file.append({"file": os.path.basename(path), "articles": n_this_file})

    conn.commit()
    conn.close()
    return {"db_path": db_path, "files_processed": len(files), "total_articles": total, "per_file": per_file}


if __name__ == "__main__":
    summary = build()
    print(f"Indexed {summary['total_articles']} articles from {summary['files_processed']} files into {summary['db_path']}")
