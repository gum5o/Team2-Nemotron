import json, glob, os, re
import pandas as pd
from collections import Counter


def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if line:
                rows.append(json.loads(line))
    return rows


def _load_any(path):
    if path.endswith(".jsonl"):
        return _load_jsonl(path)
    d = json.load(open(path, encoding="utf-8-sig"))
    return d if isinstance(d, list) else []


def _find_files(pattern):
    for folder in ("data_full", "data"):
        hits = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
        if hits:
            return hits
    return []


def _dt(x):
    s = str(x)
    if re.match(r"^\d{4}-", s):
        return pd.to_datetime(s)
    return pd.to_datetime(s, dayfirst=True)


def _nice_date(d):
    return f"{d.day} {d.strftime('%b %Y')}"


# ---------- load ASX ----------
_frames = []
NAMES = {}
for _p in _find_files("*-ASX-*.json*"):
    _rows = _load_any(_p)
    if _rows:
        _df = pd.DataFrame(_rows)
        _df["date"] = pd.to_datetime(_df["date"])
        _company = os.path.basename(_p).split("-ASX")[0]
        for _t in _df["ticker"].unique():
            NAMES[_t] = _company
        _frames.append(_df)
ASX = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()

# ---------- load RBA ----------
RBA = pd.DataFrame()
_rba_files = _find_files("RBA*.json*") or _find_files("*rates*.json*")
if _rba_files:
    RBA = pd.DataFrame(_load_any(_rba_files[0]))
    _cols = {}
    for c in RBA.columns:
        cu = c.upper()
        if "DATE" in cu:
            _cols[c] = "date"
        elif "CHANGE" in cu:
            _cols[c] = "change"
        elif "CASH" in cu or "TARGET" in cu:
            _cols[c] = "target"
    RBA = RBA.rename(columns=_cols)
    RBA["date"] = RBA["date"].apply(_dt)
    RBA["change"] = pd.to_numeric(RBA["change"], errors="coerce").fillna(0.0)
    RBA["target"] = pd.to_numeric(RBA["target"], errors="coerce")
    RBA = RBA.sort_values("date").reset_index(drop=True)

# ---------- lazy AFR loader ----------
_AFR = None

def _afr_records():
    global _AFR
    if _AFR is None:
        _AFR = []
        for p in _find_files("AFR_*.json*"):
            for rec in _load_any(p):
                d = str(rec.get("PUBLICATIONDATE") or "")
                blob = " ".join(str(rec.get(k) or "")
                                for k in ("HEADLINE", "SUBHEAD", "INTRO", "TEXT"))
                _AFR.append({"date": d, "headline": rec.get("HEADLINE") or "", "blob": blob})
    return _AFR


def _excluded(ticker, exclude):
    for x in (exclude or []):
        xu = str(x).upper()
        if xu in ticker.upper() or xu in NAMES.get(ticker, "").upper():
            return True
    return False


# ================= tools =================
def rba_summary(start=None, end=None):
    df = RBA
    if df.empty:
        return {"error": "RBA data not loaded"}
    if start:
        df = df[df["date"] >= _dt(start)]
    if end:
        df = df[df["date"] <= _dt(end)]
    if df.empty:
        return {"error": "no RBA rows in that range"}
    changed = df[df["change"] != 0]
    per_year = {int(y): {"cuts": int((g["change"] < 0).sum()),
                         "hikes": int((g["change"] > 0).sum())}
                for y, g in changed.groupby(changed["date"].dt.year)}
    return {"decisions": int(len(df)),
            "changed_rate": int(len(changed)),
            "increases": int((df["change"] > 0).sum()),
            "decreases": int((df["change"] < 0).sum()),
            "start_date": str(df.iloc[0]["date"].date()),
            "start_target_pct": float(df.iloc[0]["target"]),
            "end_date": str(df.iloc[-1]["date"].date()),
            "end_target_pct": float(df.iloc[-1]["target"]),
            "total_change_pp": round(float(df["change"].sum()), 2),
            "per_year_changes": per_year}


def rba_decisions(start=None, end=None, changed_only=True):
    """Exact dates of RBA decisions — use these real dates, never guess dates."""
    if RBA.empty:
        return {"error": "RBA data not loaded"}
    df = RBA
    if start:
        df = df[df["date"] >= _dt(start)]
    if end:
        df = df[df["date"] <= _dt(end)]
    if changed_only:
        df = df[df["change"] != 0]
    if df.empty:
        return {"error": "no decisions in that range"}
    decs = [{"date": str(r["date"].date()), "change_pp": float(r["change"]),
             "new_target_pct": float(r["target"])} for _, r in df.iterrows()]
    truncated = len(decs) > 24
    return {"count": len(decs), "decisions": decs[:24], "truncated": truncated}


def rba_rate_on(date):
    if RBA.empty:
        return {"error": "RBA data not loaded"}
    d = _dt(date)
    df = RBA[RBA["date"] <= d]
    if df.empty:
        return {"error": f"no RBA decision on or before {date}"}
    row = df.iloc[-1]
    return {"as_of": str(d.date()),
            "target_pct": float(row["target"]),
            "target_formatted": f"{float(row['target']):.2f}%",
            "in_force_since": str(row["date"].date())}


def asx_overview():
    if ASX.empty:
        return {"error": "ASX data not loaded"}
    per = ASX.groupby("ticker").agg(rows=("date", "count"),
                                    first=("date", "min"), last=("date", "max"))
    rows = per["rows"].unique().tolist()
    return {"tickers": int(len(per)),
            "ticker_list": sorted(per.index.tolist()),
            "company_names": {t: NAMES.get(t, "?") for t in sorted(per.index.tolist())},
            "rows_per_ticker": int(rows[0]) if len(rows) == 1 else rows,
            "common_start": str(per["first"].max().date()),
            "common_end": str(per["last"].min().date())}


def asx_yearly_returns(year, exclude=None):
    if ASX.empty:
        return {"error": "ASX data not loaded"}
    out = {}
    for t, g in ASX.groupby("ticker"):
        if _excluded(t, exclude):
            continue
        g = g[g["date"].dt.year == int(year)].sort_values("date")
        if len(g) < 2:
            continue
        out[t] = round(float((g.iloc[-1]["close"] / g.iloc[0]["close"] - 1) * 100), 2)
    if not out:
        return {"error": f"no data for {year}"}
    ranked = dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
    items = list(ranked.items())
    avg = round(sum(out.values()) / len(out), 2)
    return {"year": int(year), "returns_pct": ranked,
            "simple_average_return_pct": avg,
            "best": {"ticker": items[0][0], "return_pct": items[0][1]},
            "worst": {"ticker": items[-1][0], "return_pct": items[-1][1]}}


def asx_avg_volume(exclude=None, start=None, end=None):
    if ASX.empty:
        return {"error": "ASX data not loaded"}
    df = ASX
    if start:
        df = df[df["date"] >= _dt(start)]
    if end:
        df = df[df["date"] <= _dt(end)]
    out = {}
    for t, g in df.groupby("ticker"):
        if _excluded(t, exclude):
            continue
        out[t] = int(g["volume"].mean())
    ranked = dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
    items = list(ranked.items())
    return {"avg_daily_volume": {k: f"{v:,}" for k, v in ranked.items()},
            "highest": {"ticker": items[0][0], "avg_daily_volume": f"{items[0][1]:,}"},
            "lowest": {"ticker": items[-1][0], "avg_daily_volume": f"{items[-1][1]:,}"}}


def asx_price(ticker, date):
    if ASX.empty:
        return {"error": "ASX data not loaded"}
    tu = str(ticker).upper().replace(".AX", "")
    g = ASX[[tu in t.upper() or tu in NAMES.get(t, "").upper() for t in ASX["ticker"]]]
    if g.empty:
        return {"error": f"unknown ticker {ticker}"}
    d = _dt(date)
    row = g.iloc[(g["date"] - d).abs().argsort().iloc[0]]
    return {"ticker": row["ticker"], "date": str(row["date"].date()),
            "open": round(float(row["open"]), 2), "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2), "close": round(float(row["close"]), 2),
            "volume": int(row["volume"])}


def asx_period_change(ticker, start, end):
    if ASX.empty:
        return {"error": "ASX data not loaded"}
    tu = str(ticker).upper().replace(".AX", "")
    g = ASX[[tu in t.upper() or tu in NAMES.get(t, "").upper() for t in ASX["ticker"]]]
    g = g[(g["date"] >= _dt(start)) & (g["date"] <= _dt(end))].sort_values("date")
    if len(g) < 2:
        return {"error": "not enough rows in that range"}
    a, b = g.iloc[0], g.iloc[-1]
    return {"ticker": a["ticker"],
            "from": str(a["date"].date()), "from_close": round(float(a["close"]), 2),
            "to": str(b["date"].date()), "to_close": round(float(b["close"]), 2),
            "change_pct": round(float((b["close"] / a["close"] - 1) * 100), 2)}


def asx_basket_return(start, end, exclude=None):
    """Equal-weight basket: per-ticker close-to-close % return over the window, averaged."""
    if ASX.empty:
        return {"error": "ASX data not loaded"}
    raw, used = {}, None
    for t, g in ASX.groupby("ticker"):
        if _excluded(t, exclude):
            continue
        g = g[(g["date"] >= _dt(start)) & (g["date"] <= _dt(end))].sort_values("date")
        if len(g) < 2:
            continue
        raw[t] = float((g.iloc[-1]["close"] / g.iloc[0]["close"] - 1) * 100)
        used = (str(g.iloc[0]["date"].date()), str(g.iloc[-1]["date"].date()))
    if not raw:
        return {"error": "no data in that window"}
    ranked = dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True))
    return {"window": {"from": used[0], "to": used[1]},
            # basket_equal_weight_return_pct listed before the verbose per-ticker breakdown
            # so it survives if this result ever gets string-truncated downstream.
            "basket_equal_weight_return_pct": round(sum(raw.values()) / len(raw), 2),
            "per_ticker_return_pct": {k: round(v, 2) for k, v in ranked.items()}}


def asx_drawdowns(exclude=None, top=3):
    """Worst full-sample maximum drawdowns (close-based) with peak and trough dates."""
    if ASX.empty:
        return {"error": "ASX data not loaded"}
    results = []
    for t, g in ASX.groupby("ticker"):
        if _excluded(t, exclude):
            continue
        g = g.sort_values("date").reset_index(drop=True)
        runmax = g["close"].cummax()
        dd = g["close"] / runmax - 1
        ti = int(dd.values.argmin())
        pi = int(g.loc[:ti, "close"].values.argmax())
        results.append({"ticker": t,
                        "max_drawdown_pct": round(float(dd.iloc[ti] * 100), 2),
                        "peak_date": _nice_date(g.iloc[pi]["date"]),
                        "peak_close": round(float(g.iloc[pi]["close"]), 2),
                        "trough_date": _nice_date(g.iloc[ti]["date"]),
                        "trough_close": round(float(g.iloc[ti]["close"]), 2)})
    results.sort(key=lambda r: r["max_drawdown_pct"])
    return {"worst_drawdowns": results[:int(top)]}


def afr_count(terms, year=None):
    """Case-insensitive whole-word search, counted once per record. terms: word or list (OR)."""
    recs = _afr_records()
    if not recs:
        return {"error": "AFR data not loaded"}
    if isinstance(terms, str):
        terms = [terms]
    pats = [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in terms]
    per_year, per_month = Counter(), Counter()
    total = 0
    for r in recs:
        if year and not r["date"].startswith(str(year)):
            continue
        if any(p.search(r["blob"]) for p in pats):
            total += 1
            per_year[r["date"][:4]] += 1
            per_month[r["date"][:6]] += 1
    def _fmt_month(m):
        return pd.to_datetime(m, format="%Y%m").strftime("%B %Y")
    top_months = [{"month": _fmt_month(m), "count": c} for m, c in per_month.most_common(3)]
    return {"terms": terms, "year_filter": year, "total_matching_records": total,
            "per_year": dict(sorted(per_year.items())),
            "peak_year": max(per_year, key=per_year.get) if per_year else None,
            "top_months": top_months}


def afr_find(query, date=None):
    """Find AFR article(s) by title text. If a date is given, only an article on that exact
    date counts — near-misses are reported but not returned as the article."""
    recs = _afr_records()
    if not recs:
        return {"error": "AFR data not loaded"}
    ql = str(query).lower()
    matches = [r for r in recs if ql in r["headline"].lower()]
    if not matches:
        words = [w for w in re.findall(r"\w+", ql) if len(w) > 3]
        matches = [r for r in recs
                   if sum(w in r["headline"].lower() for w in words) >= max(2, len(words) // 2)]
    if date:
        d8 = _dt(date).strftime("%Y%m%d")
        exact = [r for r in matches if r["date"] == d8]
        if not exact:
            return {"error": f"no AFR article matching '{query}' on {date} in the loaded AFR data",
                    "near_matches": [{"date": r["date"], "headline": r["headline"]}
                                     for r in matches[:3]]}
        matches = exact
    if not matches:
        return {"error": f"no AFR article matching '{query}' in the loaded AFR data"}
    return {"matches": [{"date": r["date"], "headline": r["headline"],
                         "excerpt": r["blob"][:700]} for r in matches[:2]]}


def afr_sentiment(query, date=None):
    """Retrieve an AFR article and classify its financial-market sentiment with the
    fine-tuned finance model (nemotron-8b-finance)."""
    found = afr_find(query, date)
    if "error" in found:
        return found
    from config import llm_finance
    art = found["matches"][0]
    verdict = llm_finance.invoke(
        "Classify the financial-market sentiment of this news article as exactly one of: "
        "positive, negative, or mixed. Then state the likely short-term direction for the "
        "relevant ASX shares (upward, downward, or mixed) in one sentence.\n\n"
        f"HEADLINE: {art['headline']}\nARTICLE: {art['excerpt']}").content
    return {"headline": art["headline"], "date": art["date"],
            "model": "nemotron-8b-finance (team fine-tune)",
            "sentiment_analysis": verdict[:500]}


def dataset_coverage():
    """Date coverage of all three datasets — for questions about what analysis the data supports."""
    out = {}
    if not RBA.empty:
        out["RBA"] = {"from": str(RBA["date"].min().date()), "to": str(RBA["date"].max().date()),
                      "decisions": int(len(RBA))}
    if not ASX.empty:
        out["ASX"] = {"from": str(ASX["date"].min().date()), "to": str(ASX["date"].max().date()),
                      "tickers": int(ASX["ticker"].nunique())}
    afr_files = _find_files("AFR_*.json*")
    dates = re.findall(r"AFR_(\d{8})-(\d{8})", " ".join(os.path.basename(p) for p in afr_files))
    if dates:
        starts = sorted(d[0] for d in dates)
        ends = sorted(d[1] for d in dates)
        out["AFR"] = {"from": pd.to_datetime(starts[0]).strftime("%Y-%m-%d"),
                      "to": pd.to_datetime(ends[-1]).strftime("%Y-%m-%d"),
                      "files": len(afr_files)}
    return out or {"error": "no datasets loaded"}


TOOLS = {"rba_summary": rba_summary, "rba_decisions": rba_decisions,
         "rba_rate_on": rba_rate_on,
         "asx_overview": asx_overview, "asx_yearly_returns": asx_yearly_returns,
         "asx_avg_volume": asx_avg_volume, "asx_price": asx_price,
         "asx_period_change": asx_period_change, "asx_basket_return": asx_basket_return,
         "asx_drawdowns": asx_drawdowns, "afr_count": afr_count,
         "afr_find": afr_find, "afr_sentiment": afr_sentiment,
         "dataset_coverage": dataset_coverage}

CATALOG = """rba_summary(start?, end?) -> RBA decision counts, start/end target %, total change in pp, per-year cuts/hikes. NOTE: gives counts only, not dates.
rba_decisions(start?, end?) -> the EXACT dates of rate cuts/hikes with change and new target. ALWAYS use this to get real decision dates before computing windows around cuts — never guess dates.
rba_rate_on(date) -> the cash-rate target in force on a given date.
asx_overview() -> tickers, company names, rows per ticker, common date range.
asx_yearly_returns(year, exclude?) -> per-ticker % return for a year, best/worst, simple average. exclude accepts tickers or company names, e.g. ["Tabcorp"].
asx_avg_volume(exclude?, start?, end?) -> average daily volume per ticker, highest/lowest.
asx_price(ticker, date) -> OHLCV for the nearest trading day.
asx_period_change(ticker, start, end) -> one ticker's close-to-close % change over a window.
asx_basket_return(start, end, exclude?) -> per-ticker AND equal-weight basket % return over a date window. Use for "basket" questions.
asx_drawdowns(exclude?, top?) -> worst maximum drawdowns with peak/trough dates.
afr_count(terms, year?) -> whole-word once-per-record count of a word (or list of words, OR-matched) across AFR articles; per-year and peak months.
afr_find(query, date?) -> retrieve AFR article(s) by headline text, with excerpt.
afr_sentiment(query, date?) -> retrieve an AFR article AND classify its sentiment (positive/negative/mixed) plus likely ASX direction, using the team's fine-tuned finance model. Use for sentiment questions.
dataset_coverage() -> date ranges of RBA, ASX, AFR datasets. Use for questions about dataset dimensions, coverage, or whether the data can support an analysis."""