import json, os, glob, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
outdir = os.path.expanduser("~/sample")
os.makedirs(outdir, exist_ok=True)

paths = glob.glob(os.path.join(SRC, "**", "*.json*"), recursive=True)
print("found", len(paths), "files")
for p in paths:
    base = os.path.basename(p)
    if base.startswith("_"):
        continue
    recs = []
    try:
        if p.endswith(".jsonl"):
            with open(p, encoding="utf-8-sig") as f:
                for i, line in enumerate(f):
                    if i >= N:
                        break
                    if line.strip():
                        recs.append(json.loads(line))
        else:
            d = json.load(open(p, encoding="utf-8-sig"))
            recs = d[:N] if isinstance(d, list) else [d]
    except Exception as e:
        print("skip", base, e)
        continue
    if recs:
        out = os.path.join(outdir, base.replace(".jsonl", ".json"))
        json.dump(recs, open(out, "w"))
        print("wrote", base)
print("done ->", outdir)