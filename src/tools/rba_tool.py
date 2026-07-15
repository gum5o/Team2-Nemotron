"""Deterministic query tool over the RBA cash-rate decision dataset.

Dataset schema (per Technical Reference): Effective Date, Change % points,
Cash rate target% (source file is UTF-8 with a BOM).
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, Optional

RBA_PATH = os.environ.get(
    "RBA_DATA_PATH",
    "/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/RBA-Rates-2010-2026/RBA-rates.jsonl",
)


def _parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%d %b %Y").date()


class RbaRecord:
    __slots__ = ("effective_date", "change", "target")

    def __init__(self, effective_date: date, change: float, target: float):
        self.effective_date = effective_date
        self.change = change
        self.target = target

    def to_dict(self) -> dict:
        return {
            "effective_date": self.effective_date.isoformat(),
            "change_pct_points": self.change,
            "cash_rate_target_pct": self.target,
        }


class RbaDataset:
    """Loads and serves the full RBA decision history, sorted chronologically."""

    def __init__(self, path: str = RBA_PATH):
        self.path = path
        self.records: list[RbaRecord] = []
        self._load()

    def _load(self) -> None:
        records = []
        with open(self.path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                records.append(
                    RbaRecord(
                        effective_date=_parse_date(row["Effective Date"]),
                        change=round(float(row["Change % points"]), 4),
                        target=round(float(row["Cash rate target%"]), 4),
                    )
                )
        records.sort(key=lambda r: r.effective_date)
        self.records = records

    def _filtered(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> list[RbaRecord]:
        recs = self.records
        if start:
            sd = _parse_date_flex(start)
            recs = [r for r in recs if r.effective_date >= sd]
        if end:
            ed = _parse_date_flex(end)
            recs = [r for r in recs if r.effective_date <= ed]
        return recs

    # ---- high-level, pre-built aggregates -------------------------------

    def summary(self) -> dict[str, Any]:
        recs = self.records
        changed = [r for r in recs if r.change != 0]
        increases = [r for r in changed if r.change > 0]
        decreases = [r for r in changed if r.change < 0]
        return {
            "total_records": len(recs),
            "first_effective_date": recs[0].effective_date.isoformat(),
            "last_effective_date": recs[-1].effective_date.isoformat(),
            "first_target_pct": recs[0].target,
            "last_target_pct": recs[-1].target,
            "changed_count": len(changed),
            "increase_count": len(increases),
            "decrease_count": len(decreases),
            "held_count": len(recs) - len(changed),
        }

    def changes_in_range(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> dict[str, Any]:
        """Counts and net movement of rate changes within [start, end] (inclusive)."""
        recs = self._filtered(start, end)
        changed = [r for r in recs if r.change != 0]
        increases = [r for r in changed if r.change > 0]
        decreases = [r for r in changed if r.change < 0]
        by_year: dict[int, dict[str, int]] = {}
        for r in changed:
            y = r.effective_date.year
            bucket = by_year.setdefault(y, {"cuts": 0, "hikes": 0})
            if r.change > 0:
                bucket["hikes"] += 1
            else:
                bucket["cuts"] += 1
        return {
            "range_start": recs[0].effective_date.isoformat() if recs else None,
            "range_end": recs[-1].effective_date.isoformat() if recs else None,
            "total_records_in_range": len(recs),
            "changed_count": len(changed),
            "increase_count": len(increases),
            "decrease_count": len(decreases),
            "by_year": by_year,
            "starting_target_pct": recs[0].target if recs else None,
            "ending_target_pct": recs[-1].target if recs else None,
            "net_change_pct_points": (
                round(recs[-1].target - recs[0].target, 4) if recs else None
            ),
            "sum_of_changes_pct_points": round(sum(r.change for r in recs), 4),
        }

    def rate_on_date(self, on: str) -> dict[str, Any]:
        """The cash rate target in effect on a given calendar date (most recent
        decision with effective_date <= on)."""
        d = _parse_date_flex(on)
        applicable = [r for r in self.records if r.effective_date <= d]
        if not applicable:
            return {"date": d.isoformat(), "target_pct": None, "note": "before first record"}
        r = applicable[-1]
        return {
            "date": d.isoformat(),
            "as_of_decision_date": r.effective_date.isoformat(),
            "target_pct": r.target,
        }

    def list_records(
        self, start: Optional[str] = None, end: Optional[str] = None, only_changes: bool = False
    ) -> list[dict]:
        recs = self._filtered(start, end)
        if only_changes:
            recs = [r for r in recs if r.change != 0]
        return [r.to_dict() for r in recs]

    def longest_streak(self, kind: str = "hold") -> dict[str, Any]:
        """Longest run of consecutive decisions of a given kind: 'hold', 'hike', or 'cut'."""
        best_len, best_start, best_end = 0, None, None
        cur_len, cur_start = 0, None
        for r in self.records:
            is_kind = (
                (kind == "hold" and r.change == 0)
                or (kind == "hike" and r.change > 0)
                or (kind == "cut" and r.change < 0)
            )
            if is_kind:
                if cur_len == 0:
                    cur_start = r.effective_date
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start, best_end = cur_len, cur_start, r.effective_date
            else:
                cur_len = 0
        return {
            "kind": kind,
            "longest_streak_length": best_len,
            "start": best_start.isoformat() if best_start else None,
            "end": best_end.isoformat() if best_end else None,
        }


def _parse_date_flex(s: str) -> date:
    """Accepts 'YYYY-MM-DD', 'D Mon YYYY', or 'YYYY' (-> Jan 1 / Dec 31 handled by caller)."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%B %Y", "%b %Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt in ("%B %Y", "%b %Y"):
                return dt.date()
            return dt.date()
        except ValueError:
            continue
    if s.isdigit() and len(s) == 4:
        return date(int(s), 1, 1)
    raise ValueError(f"Unrecognized date format: {s!r}")


_dataset: Optional[RbaDataset] = None


def get_dataset() -> RbaDataset:
    global _dataset
    if _dataset is None:
        _dataset = RbaDataset()
    return _dataset


TOOL_SCHEMA = {
    "name": "query_rba",
    "description": (
        "Query the RBA cash-rate decision dataset (175 records, 2010-2026). "
        "operation is one of: 'summary' (whole-dataset stats), "
        "'changes_in_range' (counts/net movement of changes within a date window), "
        "'rate_on_date' (target rate in effect on a given date), "
        "'list_records' (raw records, optionally only changes, in a date window), "
        "'longest_streak' (longest run of hold/hike/cut decisions)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "summary",
                    "changes_in_range",
                    "rate_on_date",
                    "list_records",
                    "longest_streak",
                ],
            },
            "start": {"type": "string", "description": "Range start date, e.g. '2011-01-01' or '1 Jan 2011'"},
            "end": {"type": "string", "description": "Range end date"},
            "on": {"type": "string", "description": "Single date for rate_on_date"},
            "only_changes": {"type": "boolean"},
            "kind": {"type": "string", "enum": ["hold", "hike", "cut"]},
        },
        "required": ["operation"],
    },
}


def run(args: dict) -> Any:
    ds = get_dataset()
    op = args.get("operation")
    if op == "summary":
        return ds.summary()
    if op == "changes_in_range":
        return ds.changes_in_range(args.get("start"), args.get("end"))
    if op == "rate_on_date":
        return ds.rate_on_date(args["on"])
    if op == "list_records":
        return ds.list_records(args.get("start"), args.get("end"), bool(args.get("only_changes", False)))
    if op == "longest_streak":
        return ds.longest_streak(args.get("kind", "hold"))
    raise ValueError(f"Unknown operation: {op}")


if __name__ == "__main__":
    import sys

    ds = get_dataset()
    print("summary:", ds.summary())
    print("2011-2013 easing:", ds.changes_in_range("2011-01-01", "2013-12-31"))
