"""Manual JSON-based ReAct loop over agent-brain.

Native OpenAI-style `tool_calls` were tested against this vLLM deployment
and never populated (the model just narrates a "Thinking Process" and never
emits a structured tool_calls field), so the agent instead asks the model to
respond with exactly one JSON object per turn -- either a tool call or a
final answer -- and parses it directly out of the completion text.

agent-brain is served with a 4096-token context window, so the running
conversation can't just grow unboundedly. Two safeguards:
  1. The reasoning loop only shows the model a sliding window of its most
     recent tool exchanges (older ones are dropped, not garbled into a
     confusing "[omitted]" placeholder that previously caused the model to
     doubt data it actually still had).
  2. The final answer is always synthesized FRESH from the complete,
     never-trimmed `tool_trace` -- so even if the reasoning loop's own
     working context only saw a recent window, nothing gathered earlier in
     the run is lost when it's time to answer.
"""
from __future__ import annotations

import json
import re

from agent import config, llm
from tools import registry

SYSTEM_PROMPT = """You are a financial-market analysis agent for a hackathon that grades answers \
against three approved local datasets: RBA cash-rate decisions, ASX daily prices for 18 companies \
(2015-01-02 to 2021-12-30), and an AFR news corpus. You must ground every factual claim in tool \
output -- never compute or recall a dataset fact from memory.

Available tools:
{tool_schemas}

On every turn, respond with EXACTLY ONE JSON object and nothing else (no prose before or after it):
- To call a tool: {{"tool": "<tool_name>", "args": {{...}}}}
- To give your final answer: {{"final_answer": "<answer text>"}}

Rules:
- Use tools for every number, date, count, ranking, or classification asked for.
- AFR counts: whole_word=true (default) unless substring matching is clearly wanted; pass date filters \
when a period is specified. If the target is described as a "pattern" or a set of related terms (e.g. \
"X/Y") rather than one exact word, use `patterns` with that set -- a single narrow term will undercount.
- Excluding one named company/ticker from an otherwise full-dataset question means query_asx with \
exclude=[<that company>].
- Never repeat an identical tool call -- reuse the result you already have ("Already called" below \
lists every call made so far).
- Prefer the most aggregated operation available (summary, changes_in_range, rank_by_return, \
rank_by_drawdown, basket_return, counts_by_period) over a raw dump (list_records) or looping one \
ticker/period at a time -- raw dumps are large and may be truncated, and your tool-call budget is \
limited. Only request raw/individual records for row-level detail an aggregate can't give.
- For AFR article sentiment/direction: retrieve the article (search_afr / article_id), get the \
applicable RBA rate (query_rba rate_on_date), then call assess_sentiment with headline+text+rate. \
Never invent a numeric price target -- assess_sentiment reports direction/tone, not a forecast.
- For "can these datasets support X" questions: check each dataset's actual coverage (query_rba \
summary, query_asx dataset_summary, search_afr dataset_summary) and state any date-range gap.
- final_answer must address every part of the question with specific numbers/dates/names, not a vague \
summary. State any genuine evidence gap explicitly rather than guessing.
- If asked for both a group aggregate AND individual named constituents, state the aggregate \
explicitly and separately -- don't fold it into the per-item list or drop it.
- If a question refers to a set of specific named events (e.g. a sequence of rate decisions, or "each" \
occurrence of something), identify every one of them concretely: its date/identifier AND its resulting \
value -- not just a downstream calculation derived from them.
- You have at most {max_steps} tool calls before you must answer with whatever you've gathered.
"""

FINAL_SYSTEM_PROMPT = """You already gathered evidence for a financial-market question using tools \
over RBA/ASX/AFR datasets. Below is the ORIGINAL QUESTION and the COMPLETE, exact set of tool results \
gathered (nothing has been omitted or lost -- do not express doubt about missing data).

Write the final answer as plain natural-language sentences that directly and completely address every \
part of the question, using only the exact numbers/dates/names from the tool results below. If a part \
of the question genuinely wasn't covered by any tool result, say so explicitly instead of guessing.

Checklist before you answer:
- If the question asks for both a group-level aggregate AND individual named constituents, state the \
aggregate number explicitly and separately -- do not fold it into the per-item list or drop it.
- If the question refers to a set of specific named events, identify every one of them concretely with \
its date/identifier AND its resulting value, not just a downstream calculation derived from them.

Respond with EXACTLY ONE JSON object and nothing else: {{"final_answer": "<answer text>"}}
"""

WINDOW_SIZE = 3  # number of most-recent tool exchanges kept in the reasoning loop's own context


def _extract_json(text: str) -> dict | None:
    """Find the first balanced {...} block in `text` and parse it."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


MAX_RESULT_CHARS = 900  # agent-brain's context window is only 4096 tokens -- keep tool results tight


def _summarize_result(result) -> str:
    text = json.dumps(result, default=str, separators=(",", ":"))
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "...(truncated)"
    return text


LOOP_WARNING_THRESHOLD = 3  # same tool name called this many times in one run -> inject a corrective nudge


def _detect_looping_tool(tool_trace: list[dict]) -> str | None:
    """Return a tool name that's been called LOOP_WARNING_THRESHOLD+ times in
    this run (regardless of args), or None. This is a generic, question-
    agnostic pattern check -- it fires for ANY question where the model falls
    into calling one tool repeatedly instead of reaching for an aggregate."""
    counts: dict[str, int] = {}
    for t in tool_trace:
        counts[t["tool"]] = counts.get(t["tool"], 0) + 1
    for name, count in counts.items():
        if count >= LOOP_WARNING_THRESHOLD:
            return name
    return None


def _build_reasoning_messages(system: str, question: str, tool_trace: list[dict]) -> list[dict]:
    """System + question + a compact 'already called' list + only the last
    WINDOW_SIZE tool exchanges in full. Older exchanges are dropped (not
    garbled) -- full history is never needed here since the final synthesis
    step re-reads the complete tool_trace directly."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]

    if len(tool_trace) > WINDOW_SIZE:
        already = [f"{t['tool']}({json.dumps(t['args'], separators=(',', ':'))})" for t in tool_trace]
        messages.append(
            {"role": "user", "content": "Already called (don't repeat): " + "; ".join(already)}
        )

    looping_tool = _detect_looping_tool(tool_trace)
    if looping_tool:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"You've called {looping_tool} repeatedly. Use an aggregate/batch operation instead "
                    "of looping per item, or answer now with what you have."
                ),
            }
        )

    for t in tool_trace[-WINDOW_SIZE:]:
        messages.append({"role": "assistant", "content": json.dumps({"tool": t["tool"], "args": t["args"]})})
        messages.append({"role": "user", "content": f"Tool result for {t['tool']}: {t['result']}"})

    return messages


def _synthesize_final_answer(question: str, tool_trace: list[dict]) -> str:
    evidence_lines = [f"- {t['tool']}({json.dumps(t['args'], separators=(',', ':'))}) -> {t['result']}" for t in tool_trace]
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "(no tool calls were made)"
    messages = [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"ORIGINAL QUESTION:\n{question}\n\nTOOL RESULTS GATHERED:\n{evidence_text}",
        },
    ]
    raw = llm.chat(messages, max_tokens=450)
    action = _extract_json(raw) or {}
    return str(action.get("final_answer") or raw.strip())


def answer_question(question: str, max_steps: int | None = None) -> dict:
    max_steps = max_steps or config.MAX_AGENT_STEPS
    tool_schema_text = json.dumps(registry.schemas(), separators=(",", ":"))
    system = SYSTEM_PROMPT.format(tool_schemas=tool_schema_text, max_steps=max_steps)

    tool_trace: list[dict] = []
    steps = 0
    seen_calls: set[str] = set()
    model_signaled_done = False

    for step in range(max_steps):
        steps += 1
        messages = _build_reasoning_messages(system, question, tool_trace)
        raw = llm.chat(messages, max_tokens=280)
        action = _extract_json(raw)

        if action is None or ("tool" not in action and "final_answer" not in action):
            continue  # malformed turn; next loop iteration just re-asks with the same context

        if "final_answer" in action:
            # Treat this as a stop signal, not the final text: synthesize the
            # actual answer from the complete tool_trace below, so it isn't
            # limited to whatever fit in this step's sliding-window context.
            model_signaled_done = True
            break

        tool_name = action["tool"]
        tool_args = action.get("args", {}) or {}
        call_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

        if call_key in seen_calls:
            tool_trace.append(
                {"tool": tool_name, "args": tool_args, "result": "SKIPPED: identical call already made"}
            )
            continue

        seen_calls.add(call_key)
        try:
            result = registry.call(tool_name, tool_args)
            result_summary = _summarize_result(result)
        except Exception as exc:  # noqa: BLE001
            result_summary = f"ERROR: {exc}"
        tool_trace.append({"tool": tool_name, "args": tool_args, "result": result_summary})

    # Whether the model signaled it was done or the step budget ran out,
    # always synthesize the final answer fresh from the complete tool_trace.
    if not model_signaled_done:
        steps += 1
    final = _synthesize_final_answer(question, tool_trace)
    return {"answer": final, "steps": steps, "tool_trace": tool_trace}
