import json
import datastore
from agent import graph

datastore.build()
print("Agent ready. Type a question, or 'exit' to quit.")
while True:
    q = input("\nQuestion> ").strip()
    if q.lower() in ("exit", "quit", ""):
        break
    result = graph.invoke({"question": q, "evidence": [],
                           "calculations": [], "retries": 0},
                          config={"recursion_limit": 15})
    print(json.dumps(result["final"], indent=2))