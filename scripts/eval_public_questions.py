"""Send all 15 public calibration questions through the running agent server,
and do a lightweight self-check of whether each expected_fact's key tokens
show up in the answer. This is NOT the official grader (that logic is
private to the organizers) -- it's a fast local signal for "did we clearly
state this fact" so we can catch obvious misses before submission.

Usage: python scripts/eval_public_questions.py [--server http://localhost:8080]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

QUESTIONS_PATH = (
    "/home/cognitivo/Desktop/Cognitivo_Training/Mock_Hackathon_Participant_Package/public_questions.jsonl"
)


def query(server: str, question: str, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"{server}/query",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9.%\-]+", " ", s.lower())


def fact_present(answer: str, expected_fact: str) -> bool:
    """Heuristic: pull out the numeric/percent tokens and key words from the
    expected fact and check they all show up (in some form) in the answer."""
    norm_answer = normalize(answer)
    # numbers / percentages, tolerant of comma separators
    nums = re.findall(r"-?\d[\d,]*\.?\d*%?", expected_fact)
    nums = [n.replace(",", "") for n in nums]
    for n in nums:
        bare = n.rstrip("%")
        if bare and bare not in norm_answer.replace(",", ""):
            return False
    # a couple of the most distinctive non-numeric words
    words = [w for w in re.findall(r"[A-Za-z]{3,}", expected_fact) if w.lower() not in
             {"the", "and", "for", "with", "records", "total", "return", "target"}]
    if words:
        hits = sum(1 for w in words if w.lower() in norm_answer)
        if hits == 0:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--out", default="logs/eval_report.json")
    args = ap.parse_args()

    questions = []
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    report = []
    total_possible = 0
    total_hit_estimate = 0

    for q in questions:
        t0 = time.time()
        try:
            result = query(args.server, q["prompt"])
            error = None
        except Exception as exc:  # noqa: BLE001
            result = {"answer": "", "steps": 0, "tool_trace": []}
            error = str(exc)
        elapsed = time.time() - t0

        components = []
        for c in q["grading"]["components"]:
            hit = fact_present(result["answer"], c["expected_fact"]) if not error else False
            components.append(
                {
                    "expected_fact": c["expected_fact"],
                    "points": c["points"],
                    "heuristic_hit": hit,
                }
            )
            total_possible += c["points"]
            if hit:
                total_hit_estimate += c["points"]

        row = {
            "id": q["id"],
            "difficulty": q["difficulty"],
            "datasets": q["datasets"],
            "prompt": q["prompt"],
            "answer": result["answer"],
            "steps": result["steps"],
            "elapsed_s": round(elapsed, 1),
            "error": error,
            "components": components,
        }
        report.append(row)
        est = sum(c["points"] for c in components if c["heuristic_hit"])
        max_pts = sum(c["points"] for c in components)
        print(f"[{q['id']}] ~{est:.0f}/{max_pts:.0f} pts ({elapsed:.1f}s, {result['steps']} steps)"
              f"{' ERROR: ' + error if error else ''}")

    print(f"\nEstimated total: {total_hit_estimate:.0f}/{total_possible:.0f} "
          f"({100 * total_hit_estimate / total_possible:.1f}%)")
    print("(This is a rough local heuristic, not the official grader.)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "estimated_total": total_hit_estimate,
                "max_possible": total_possible,
                "results": report,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
