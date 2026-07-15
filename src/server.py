"""FastAPI server exposing the agent per submission-guide.md's contract.

POST /query  {"question": "..."}  -> {"answer": "...", "steps": N, "tool_trace": [...]}
GET  /health -> 200 when the tool indexes are loaded and the agent is ready.

Run: uvicorn server:app --host 0.0.0.0 --port 8080 --app-dir src
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from pydantic import BaseModel

from agent import orchestrator
from tools import afr_search, asx_tool, rba_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-server")

app = FastAPI(title="Cognitivo Hackathon Agent")

_ready = False


class Question(BaseModel):
    question: str


class ToolTraceEntry(BaseModel):
    tool: str
    args: dict
    result: str


class Answer(BaseModel):
    answer: str
    steps: int = 0
    tool_trace: list[ToolTraceEntry] = []


@app.on_event("startup")
def _load_indexes() -> None:
    global _ready
    t0 = time.time()
    rba_tool.get_dataset()
    asx_tool.get_dataset()
    afr_search.dataset_summary()  # ensures the SQLite index exists/opens
    _ready = True
    logger.info(f"Tool indexes loaded in {time.time() - t0:.1f}s")


@app.get("/health")
def health():
    return {"status": "ok" if _ready else "loading"}


@app.post("/query", response_model=Answer)
def query(q: Question):
    t0 = time.time()
    try:
        result = orchestrator.answer_question(q.question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent error")
        return Answer(answer=f"Agent error: {exc}", steps=0, tool_trace=[])
    logger.info(f"Answered in {time.time() - t0:.1f}s ({result['steps']} steps): {q.question[:80]!r}")
    return Answer(**result)
