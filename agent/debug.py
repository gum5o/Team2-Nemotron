import datastore
from agent import graph

datastore.build()
q = "Across the three 2019 RBA cuts, what was the non-Tabcorp basket's one-week return after each effective date?"
for step in graph.stream({"question": q, "evidence": [],
                          "calculations": [], "retries": 0},
                         config={"recursion_limit": 15}):
    for node, out in step.items():
        print("=== node:", node)
        print(str(out)[:800])
        print()