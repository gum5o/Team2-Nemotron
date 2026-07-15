import json, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor
import datastore
from agent import graph

def answer_one(q):
    try:
        r = graph.invoke({"question": q["text"], "evidence": [],
                          "calculations": [], "retries": 0},
                         config={"recursion_limit": 15})
        return {"id": q["id"], **r["final"]}
    except Exception as e:
        return {"id": q["id"], "answer": "Unable to answer from the supplied data.",
                "evidence": [], "calculations": [], "confidence": 0.0,
                "abstained": True, "error": str(e)}

def load_questions(path):
    text = open(path, encoding="utf-8-sig").read().strip()
    if text.startswith("["):
        qs = json.loads(text)
    else:
        qs = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for i, q in enumerate(qs):
        t = q.get("prompt") or q.get("text") or q.get("question") or q.get("q")
        if not t:
            print(f"WARNING: question {q.get('id', i)} has no recognizable text field: {list(q)}")
            continue
        out.append({"id": q.get("id", i), "text": t})
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("infile")
    p.add_argument("-o", default="answers.jsonl")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    datastore.build()
    qs = load_questions(a.infile)
    print(f"Loaded {len(qs)} questions")
    t0 = time.time()
    with open(a.o, "w") as out, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(answer_one, qs), 1):
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(f"[{i}/{len(qs)}] id={res['id']} conf={res.get('confidence')} "
                  f"abstained={res.get('abstained')} ({time.time()-t0:.0f}s)")
    print(f"Done: {len(qs)} -> {a.o}")