import json

qs = {}
for line in open("public_questions.jsonl", encoding="utf-8-sig"):
    if line.strip():
        q = json.loads(line)
        qs[q["id"]] = q

for line in open("answers.jsonl", encoding="utf-8"):
    a = json.loads(line)
    q = qs.get(a["id"], {})
    print("=" * 70)
    print(a["id"], "| difficulty:", q.get("difficulty"), "| datasets:", q.get("datasets"))
    print("Q:", q.get("prompt"))
    print("ANSWER:", str(a.get("answer"))[:400])
    print("conf:", a.get("confidence"), "| abstained:", a.get("abstained"))
    print("tools used:", [c.get("tool") for c in a.get("calculations", [])])
    exp = [c["expected_fact"] for c in q.get("grading", {}).get("components", [])]
    print("EXPECTED:", exp)