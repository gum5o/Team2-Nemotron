"""Deterministic query tool over the ASX 18-company price dataset.

Dataset schema (per Technical Reference): ticker, date, open, high, low, close, volume.
One JSONL file per company, 2015-01-02 .. 2021-12-30 (1,774 rows each).
"""
from __future__ import annotations

import glob
import json
import os
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

ASX_DIR = os.environ.get(
    "ASX_DATA_DIR",
    "/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/ASX-18-companies-2015-2021-Jasonl",
)

# Friendly company-name aliases -> ticker symbol used in the data (without .AX)
_ALIAS = {
    "agl": "AGL",
    "amp": "AMP",
    "anz": "ANZ",
    "aurizon": "Aurizon",
    "bhp": "BHP",
    "cba": "CBA",
    "commonwealth bank": "CBA",
    "cromwell": "Cromwell",
    "gpt": "GPT",
    "iag": "IAG",
    "nab": "NAB",
    "national australia bank": "NAB",
    "qantas": "Qantas",
    "qbe": "QBE",
    "rio": "Rio",
    "rio tinto": "Rio",
    "stockland": "Stockland",
    "suncorp": "Suncorp",
    "tabcorp": "Tabcorp",
    "tpg": "TPG",
    "transurban": "Transurban",
}


def _file_for(ticker_stub: str) -> str:
    return os.path.join(ASX_DIR, f"{ticker_stub}-ASX-2015-2021.jsonl")


def normalize_ticker(name: str) -> Optional[str]:
    key = name.strip().lower().replace(".ax", "")
    if key in _ALIAS:
        return _ALIAS[key]
    for stub in _ALIAS.values():
        if stub.lower() == key:
            return stub
    return None


class AsxDataset:
    def __init__(self, directory: str = ASX_DIR):
        self.directory = directory
        self.frames: dict[str, pd.DataFrame] = {}
        self._load()

    def _load(self) -> None:
        for path in sorted(glob.glob(os.path.join(self.directory, "*-ASX-2015-2021.jsonl"))):
            stub = os.path.basename(path).split("-ASX-2015-2021.jsonl")[0]
            rows = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self.frames[stub] = df

    def tickers(self) -> list[str]:
        return sorted(self.frames.keys())

    def _df(self, ticker: str) -> pd.DataFrame:
        stub = normalize_ticker(ticker) or ticker
        if stub not in self.frames:
            raise ValueError(f"Unknown ticker '{ticker}'. Known: {self.tickers()}")
        return self.frames[stub]

    def _sliced(self, ticker: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        df = self._df(ticker)
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df

    # ---- aggregates -------------------------------------------------

    def dataset_summary(self) -> dict[str, Any]:
        counts = {t: len(df) for t, df in self.frames.items()}
        all_dates = pd.concat([df["date"] for df in self.frames.values()])
        return {
            "ticker_count": len(self.frames),
            "tickers": self.tickers(),
            "rows_per_ticker": counts,
            "rows_per_ticker_common_value": (
                next(iter(counts.values())) if len(set(counts.values())) == 1 else None
            ),
            "common_start_date": all_dates.min().date().isoformat(),
            "common_end_date": all_dates.max().date().isoformat(),
        }

    def price_on_date(self, ticker: str, on: str) -> dict[str, Any]:
        df = self._df(ticker)
        target = pd.Timestamp(on)
        row = df[df["date"] == target]
        if row.empty:
            # nearest prior trading day
            prior = df[df["date"] <= target]
            if prior.empty:
                return {"ticker": ticker, "date": on, "found": False}
            row = prior.tail(1)
        r = row.iloc[0]
        return {
            "ticker": ticker,
            "date": r["date"].date().isoformat(),
            "open": round(float(r["open"]), 4),
            "high": round(float(r["high"]), 4),
            "low": round(float(r["low"]), 4),
            "close": round(float(r["close"]), 4),
            "volume": int(r["volume"]),
        }

    def return_over_range(self, ticker: str, start: str, end: str) -> dict[str, Any]:
        df = self._sliced(ticker, start, end)
        if df.empty:
            return {"ticker": ticker, "start": start, "end": end, "found": False}
        first, last = df.iloc[0], df.iloc[-1]
        pct = (float(last["close"]) - float(first["close"])) / float(first["close"]) * 100
        return {
            "ticker": ticker,
            "start_date": first["date"].date().isoformat(),
            "end_date": last["date"].date().isoformat(),
            "start_close": round(float(first["close"]), 4),
            "end_close": round(float(last["close"]), 4),
            "pct_return": round(pct, 2),
        }

    def _basket(self, exclude: Optional[list[str]] = None) -> list[str]:
        exclude_norm = {normalize_ticker(x) or x for x in (exclude or [])}
        return [t for t in self.frames if t not in exclude_norm]

    def rank_by_return(
        self, start: str, end: str, top: int = 5, ascending: bool = False, exclude: Optional[list[str]] = None
    ) -> list[dict]:
        results = []
        for stub in self._basket(exclude):
            r = self.return_over_range(stub, start, end)
            if r.get("found", True):
                results.append(r)
        results.sort(key=lambda r: r["pct_return"], reverse=not ascending)
        return results[:top]

    def basket_return(self, start: str, end: str, exclude: Optional[list[str]] = None) -> dict[str, Any]:
        """Equal-weighted average simple return of all tickers (minus any
        excluded) over [start, end]. Matches the 'non-Tabcorp basket' style
        of question: exclude=['Tabcorp']."""
        basket = self._basket(exclude)
        rets = []
        for stub in basket:
            r = self.return_over_range(stub, start, end)
            if r.get("found", True):
                rets.append(r["pct_return"])
        return {
            "basket": basket,
            "basket_size": len(basket),
            "excluded": exclude or [],
            "start": start,
            "end": end,
            "simple_average_pct_return": round(sum(rets) / len(rets), 2) if rets else None,
            "per_ticker_returns": {
                stub: self.return_over_range(stub, start, end)["pct_return"] for stub in basket
            },
        }

    def ticker_rank_in_basket(
        self, ticker: str, start: str, end: str, exclude: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Rank (1 = best return) of `ticker` among the basket (excluding any
        listed tickers, e.g. exclude=['Tabcorp']) over [start, end]."""
        ranked = self.rank_by_return(start, end, top=len(self.frames), exclude=exclude)
        stub = normalize_ticker(ticker) or ticker
        for i, r in enumerate(ranked, start=1):
            if r["ticker"] == stub:
                return {"ticker": stub, "rank": i, "basket_size": len(ranked), "pct_return": r["pct_return"]}
        return {"ticker": stub, "rank": None, "basket_size": len(ranked), "note": "not found in basket"}

    def volatility(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
        df = self._sliced(ticker, start, end)
        rets = df["close"].pct_change().dropna()
        return {
            "ticker": ticker,
            "start_date": df.iloc[0]["date"].date().isoformat() if not df.empty else None,
            "end_date": df.iloc[-1]["date"].date().isoformat() if not df.empty else None,
            "daily_return_std_pct": round(float(rets.std()) * 100, 2),
            "annualized_vol_pct": round(float(rets.std()) * (252 ** 0.5) * 100, 2),
        }

    def max_drawdown(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
        df = self._sliced(ticker, start, end).reset_index(drop=True)
        closes = df["close"]
        running_max = closes.cummax()
        idx_peak_running = running_max.idxmax()  # unused directly; peak is the running-max date AT the trough
        drawdown = (closes - running_max) / running_max
        idx_trough = drawdown.idxmin()
        peak_value = running_max.iloc[idx_trough]
        idx_peak = closes[: idx_trough + 1][closes[: idx_trough + 1] == peak_value].index[-1]
        return {
            "ticker": ticker,
            "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
            "peak_date": df.iloc[idx_peak]["date"].date().isoformat(),
            "trough_date": df.iloc[idx_trough]["date"].date().isoformat(),
        }

    def rank_by_drawdown(
        self, start: Optional[str] = None, end: Optional[str] = None, top: int = 5, exclude: Optional[list[str]] = None
    ) -> list[dict]:
        """Worst (most negative) max-drawdowns first, across the basket."""
        results = [self.max_drawdown(t, start, end) for t in self._basket(exclude)]
        results.sort(key=lambda r: r["max_drawdown_pct"])
        return results[:top]

    def average_volume(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, Any]:
        df = self._sliced(ticker, start, end)
        return {
            "ticker": ticker,
            "average_volume": round(float(df["volume"].mean()), 2) if not df.empty else None,
            "total_volume": int(df["volume"].sum()) if not df.empty else None,
            "n_days": len(df),
        }

    def rank_by_volume(self, start: Optional[str] = None, end: Optional[str] = None, top: int = 5) -> list[dict]:
        results = [self.average_volume(t, start, end) for t in self.frames]
        results = [r for r in results if r["average_volume"] is not None]
        results.sort(key=lambda r: r["average_volume"], reverse=True)
        return results[:top]


_dataset: Optional[AsxDataset] = None


def get_dataset() -> AsxDataset:
    global _dataset
    if _dataset is None:
        _dataset = AsxDataset()
    return _dataset


TOOL_SCHEMA = {
    "name": "query_asx",
    "description": (
        "Query the ASX 18-company daily price dataset (2015-01-02 to 2021-12-30). operations: "
        "dataset_summary (dims/tickers/dates); price_on_date (OHLCV); return_over_range (% return, "
        "one ticker); rank_by_return (top/bottom N by return, can exclude tickers); basket_return "
        "(equal-weighted avg return across all tickers minus excluded, e.g. exclude=['Tabcorp']); "
        "ticker_rank_in_basket; volatility (std dev of daily returns); max_drawdown (ONE ticker, with "
        "peak/trough dates); rank_by_drawdown (worst N drawdowns across ALL tickers in one call -- use "
        "instead of looping max_drawdown); average_volume; rank_by_volume. "
        f"Tickers: {', '.join(sorted(_ALIAS.values()))} (aliases like 'nab' accepted)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "dataset_summary",
                    "price_on_date",
                    "return_over_range",
                    "rank_by_return",
                    "basket_return",
                    "ticker_rank_in_basket",
                    "volatility",
                    "max_drawdown",
                    "rank_by_drawdown",
                    "average_volume",
                    "rank_by_volume",
                ],
            },
            "ticker": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "on": {"type": "string"},
            "top": {"type": "integer"},
            "ascending": {"type": "boolean"},
            "exclude": {"type": "array", "items": {"type": "string"}, "description": "Tickers to exclude, e.g. ['Tabcorp']"},
        },
        "required": ["operation"],
    },
}


def run(args: dict) -> Any:
    ds = get_dataset()
    op = args.get("operation")
    if op == "dataset_summary":
        return ds.dataset_summary()
    if op == "price_on_date":
        return ds.price_on_date(args["ticker"], args["on"])
    if op == "return_over_range":
        return ds.return_over_range(args["ticker"], args["start"], args["end"])
    if op == "rank_by_return":
        return ds.rank_by_return(
            args["start"], args["end"], int(args.get("top", 5)), bool(args.get("ascending", False)), args.get("exclude")
        )
    if op == "basket_return":
        return ds.basket_return(args["start"], args["end"], args.get("exclude"))
    if op == "ticker_rank_in_basket":
        return ds.ticker_rank_in_basket(args["ticker"], args["start"], args["end"], args.get("exclude"))
    if op == "volatility":
        return ds.volatility(args["ticker"], args.get("start"), args.get("end"))
    if op == "max_drawdown":
        return ds.max_drawdown(args["ticker"], args.get("start"), args.get("end"))
    if op == "rank_by_drawdown":
        return ds.rank_by_drawdown(args.get("start"), args.get("end"), int(args.get("top", 5)), args.get("exclude"))
    if op == "average_volume":
        return ds.average_volume(args["ticker"], args.get("start"), args.get("end"))
    if op == "rank_by_volume":
        return ds.rank_by_volume(args.get("start"), args.get("end"), int(args.get("top", 5)))
    raise ValueError(f"Unknown operation: {op}")


if __name__ == "__main__":
    ds = get_dataset()
    print("summary:", ds.dataset_summary())
