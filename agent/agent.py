import json, re, operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from config import llm
import datastore
import tools

class State(TypedDict):
    question: str
    route: str
    evidence: Annotated[list, operator.add]
    calculations: list
    draft: str
    grounded: bool
    retries: int
    final: dict

def _tool_errors(calcs):
    n = 0
    for c in calcs or []:
        if "error" in c or (isinstance(c.get("result"), dict) and "error" in c["result"]):
            n += 1
    return n

def route(state):
    label = llm.invoke(
        "Classify which data this question needs. Reply exactly one word: "
        "CALC, SEMANTIC, or HYBRID.\n"
        "CALC = computed from the datasets: prices, volumes, returns, drawdowns, rates, "
        "returns around rate decisions, counting articles, dataset dimensions/coverage, "
        "or whether the data can support an analysis.\n"
        "SEMANTIC = only about what news articles say: events, stories, opinions.\n"
        "HYBRID = ONLY when the question explicitly involves BOTH a news article's content "
        "AND numeric data (e.g. retrieve an article and use rates or returns).\n"
        "Question: " + state["question"]).content.strip().upper()
    return {"route": label if label in ("CALC", "SEMANTIC", "HYBRID") else "HYBRID"}

def analyze(state):
    calcs, ev = [], []
    for round_no in range(3):
        if round_no == 0:
            guidance = (
                "If the question involves windows around rate decisions, FIRST call "
                "rba_decisions to get the real dates — never guess dates. You will get "
                "follow-up rounds to request more calculations from the results.\n")
        else:
            guidance = (
                "Results so far: " + json.dumps(calcs)[:1200] + "\n"
                "Go through EVERY part of the question and check whether the results above "
                "already contain it. Output the tool calls for every part still missing.\n"
                "Worked example: if the question asks for one-week basket returns after "
                "each cut, and the results show cut dates 2019-06-05, 2019-07-03 and "
                "2019-10-02, you MUST reply:\n"
                '[{"tool":"asx_basket_return","args":{"start":"2019-06-05","end":"2019-06-12","exclude":["Tabcorp"]}},'
                '{"tool":"asx_basket_return","args":{"start":"2019-07-03","end":"2019-07-10","exclude":["Tabcorp"]}},'
                '{"tool":"asx_basket_return","args":{"start":"2019-10-02","end":"2019-10-09","exclude":["Tabcorp"]}}]\n'
                "Reply [] ONLY if every part of the question is already computed.\n")
        resp = llm.invoke(
            "Choose the tool call(s) that help answer the question. Reply ONLY a JSON "
            'array, e.g. [{"tool":"asx_yearly_returns","args":{"year":2018,'
            '"exclude":["Tabcorp"]}}].\n'
            + guidance +
            "Available tools:\n" + tools.CATALOG +
            "\nQuestion: " + state["question"]).content
        try:
            calls = json.loads(re.search(r"\[.*\]", resp, re.S).group())
        except Exception:
            calls = []
        if not calls:
            break
        for c in calls[:4]:
            if len(calcs) >= 10:
                break
            try:
                result = tools.TOOLS[c["tool"]](**c.get("args", {}))
                calcs.append({"tool": c["tool"], "args": c.get("args", {}), "result": result})
                if isinstance(result, dict) and "error" in result:
                    continue  # errors are logged but are NOT citable evidence
                ev.append({"source": f"tool:{c['tool']}", "record": json.dumps(result)[:900]})
            except Exception as e:
                calcs.append({"tool": c.get("tool"), "args": c.get("args", {}), "error": str(e)})
    return {"calculations": calcs, "evidence": ev}

def semantic_search(state):
    ev = datastore.search(state["question"], k=6, kind="news")
    if len(ev) < 3:
        ev = ev + datastore.search(state["question"], k=6)
    seen, out = set(), []
    for e in ev:
        if e["source"] not in seen:
            seen.add(e["source"])
            out.append(e)
    return {"evidence": out[:6]}

def combine(state):
    retry_note = ""
    if state["retries"] > 0:
        retry_note = ("NOTE: your previous draft was rejected. Answer from the evidence and "
                      "cite [source] tags copied exactly.\n")
    draft = llm.invoke(
        "You are a market data analyst. Answer the question using ONLY the evidence and "
        "calculations below — never your own outside knowledge.\n"
        "Rules:\n"
        "1. Cite the [source] tag in square brackets, copied verbatim, for every claim, "
        "e.g. [tool:rba_summary] or [AFR_20190701-20190731.json#0].\n"
        "2. Numbers AND dates must come only from the calculations or evidence — never "
        "invent, estimate, or recall a date or figure that is not in the evidence.\n"
        "3. State key numbers explicitly: counts, percentages to 2 decimals, large numbers "
        "with thousands separators (e.g. 11,635,671), dates and tickers by name.\n"
        "4. You MAY draw simple analytical inferences (sentiment, likely market direction) "
        "but ONLY from retrieved article text in the evidence — NEVER from the question's "
        "own wording. If an article was not found, say that part cannot be determined from "
        "the loaded data and do not guess.\n"
        "5. If a tool reported an error for part of the question, state plainly that this "
        "part is unavailable; answer the parts that have evidence.\n"
        "6. Approximate references like 'mid January' are fine — use the closest evidence "
        "and say which date(s) you used.\n"
        "7. Reply INSUFFICIENT only when NO evidence relates to the question at all.\n"
        "8. Be concise but complete: cover every part of the question.\n"
        + retry_note +
        # Both evidence and calculations can accumulate across up to 3 analyze() rounds --
        # cap the combined dump so heavy multi-tool questions don't overflow the 4096-token
        # context window (this was silently truncating some questions to "unable to answer").
        f"Evidence: {json.dumps(state['evidence'])[:3500]}\n"
        f"Calculations: {json.dumps(state.get('calculations', []))[:2200]}\n"
        f"Question: {state['question']}").content
    return {"draft": draft}

def ground_check(state):
    cited = re.findall(r"\[([^\]]+)\]", state["draft"])
    known = {e["source"] for e in state["evidence"]}
    valid = [c for c in cited if c in known]
    insufficient = state["draft"].strip().upper().startswith("INSUFFICIENT")
    grounded = bool(valid) and len(valid) >= max(1, len(cited) // 2) and not insufficient
    return {"grounded": grounded, "retries": state["retries"] + (0 if grounded else 1)}

def answer(state):
    cited = list(dict.fromkeys(re.findall(r"\[([^\]]+)\]", state["draft"])))
    known = {e["source"]: e for e in state["evidence"]}
    used = [known[c] for c in cited if c in known]
    citation_rate = len(used) / len(cited) if cited else 0.0
    coverage = min(1.0, len(used) / 2)
    math_ok = 1.0 if state.get("calculations") else 0.7
    conf = 0.45 * coverage + 0.4 * citation_rate + 0.15 * math_ok
    conf -= 0.25 * min(_tool_errors(state.get("calculations")), 2)
    conf = round(max(0.15, min(conf, 0.95)), 2)
    insufficient = state["draft"].strip().upper().startswith("INSUFFICIENT")
    return {"final": {"answer": state["draft"],
                      "evidence": used,
                      "calculations": state.get("calculations", []),
                      "confidence": 0.15 if insufficient else conf,
                      "abstained": insufficient or not state["grounded"]}}

g = StateGraph(State)
for name, fn in [("route", route), ("analyze", analyze),
                 ("semantic_search", semantic_search), ("combine", combine),
                 ("ground_check", ground_check), ("answer", answer)]:
    g.add_node(name, fn)

g.add_edge(START, "route")
g.add_conditional_edges("route",
    lambda s: "semantic_search" if s["route"] == "SEMANTIC" else "analyze",
    {"analyze": "analyze", "semantic_search": "semantic_search"})
g.add_conditional_edges("analyze",
    lambda s: "semantic_search" if s["route"] == "HYBRID" else "combine",
    {"semantic_search": "semantic_search", "combine": "combine"})
g.add_edge("semantic_search", "combine")
g.add_edge("combine", "ground_check")
g.add_conditional_edges("ground_check",
    lambda s: "answer" if s["grounded"] or s["retries"] >= 2 else "combine",
    {"answer": "answer", "combine": "combine"})
g.add_edge("answer", END)
graph = g.compile()