import json, glob, os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from config import emb

RAW, VDB = {}, None


def _as_records(data):
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        recs = [r for v in data.values() if isinstance(v, list)
                for r in v if isinstance(r, dict)]
        return recs or [data]
    return []


def _exact(rec, candidates):
    up = {k.upper(): k for k in rec}
    for c in candidates:
        if c in up:
            return up[c]
    return None


def _contains(rec, needle, exclude=""):
    for k in rec:
        ku = k.upper()
        if needle in ku and (not exclude or exclude not in ku):
            return k
    return None


def verbalize(name, rec):
    # --- ASX daily stock bar ---
    if "ticker" in rec and "close" in rec:
        try:
            return (f"{rec['ticker']} stock on {rec.get('date', '?')}: "
                    f"open {round(float(rec['open']), 2)}, high {round(float(rec['high']), 2)}, "
                    f"low {round(float(rec['low']), 2)}, close {round(float(rec['close']), 2)}, "
                    f"volume {rec.get('volume', '?')}.")
        except (TypeError, ValueError):
            pass

    # --- RBA cash rate decision ---
    if _contains(rec, "CASH RATE"):
        d = rec.get(_contains(rec, "DATE"), "?")
        rate = rec.get(_contains(rec, "CASH RATE"), "?")
        chg = rec.get(_contains(rec, "CHANGE"), "?")
        return (f"RBA cash rate decision effective {d}: "
                f"target {rate}%, change {chg} percentage points.")

    # --- AFR news article ---
    if _contains(rec, "TEXT"):
        d = rec.get(_contains(rec, "DATE"), "") or ""
        head_key = _exact(rec, ["HEADLINE", "HEAD", "TITLE", "HL"]) or \
                   _contains(rec, "HEAD", exclude="SUB")
        head = (rec.get(head_key) or "") if head_key else ""
        intro = (rec.get(_contains(rec, "INTRO")) or "")[:200]
        text = (rec.get(_contains(rec, "TEXT")) or "")[:400].replace("\n", " ")
        return f"AFR news {d}: {head}. {intro} {text}".strip()

    # --- generic fallback ---
    return name + " record: " + ", ".join(f"{k}={str(v)[:200]}" for k, v in rec.items())


def _load_any(path):
    """.jsonl = newline-delimited JSON (the real AFR/ASX/RBA source format);
    .json = a single JSON document (the sampled subset uses this)."""
    if path.endswith(".jsonl"):
        rows = []
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.load(open(path, encoding="utf-8-sig"))


AFR_FULL_FOLDER = os.path.join("data_full", "Jsonl format DataSets", "AFR Jasonl")


def build(folder="data", index_dir="afr_index_full"):
    global VDB
    if index_dir and os.path.exists(index_dir):
        VDB = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
        print(f"Loaded existing index from {index_dir}/")
        return
    if os.path.isdir(AFR_FULL_FOLDER):
        folder = AFR_FULL_FOLDER  # prefer the complete AFR corpus over the small dev sample
    docs = []
    for path in sorted(glob.glob(f"{folder}/*.json*")):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        for i, rec in enumerate(_as_records(_load_any(path))):
            tag = f"{name}#{i}"
            RAW[tag] = rec
            docs.append(Document(page_content=verbalize(name, rec),
                                 metadata={"source": tag}))
    if not docs:
        raise SystemExit(f"No records found — put the .json files in '{folder}/'.")
    VDB = FAISS.from_documents(docs, emb)
    print(f"Indexed {len(docs)} records from {folder}/")
    if index_dir:
        VDB.save_local(index_dir)
        print(f"Saved index to {index_dir}/")


def search(query, k=6, kind=None):
    """kind: 'news' = AFR articles only, 'numeric' = stocks/RBA only, None = all."""
    hits = VDB.similarity_search(query, k=k * 3 if kind else k)
    out = []
    for d in hits:
        src = d.metadata["source"]
        is_news = src.startswith("AFR")
        if kind == "news" and not is_news:
            continue
        if kind == "numeric" and is_news:
            continue
        out.append({"source": src, "record": d.page_content})
        if len(out) >= k:
            break
    return out