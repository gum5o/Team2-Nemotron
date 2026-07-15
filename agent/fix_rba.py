import json, os, glob

src = glob.glob(os.path.expanduser(
    "~/Desktop/HackathonDataset/**/RBA-rates.jsonl"), recursive=True)[0]
recs = []
with open(src, encoding="utf-8-sig") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        line = line.strip().lstrip("\ufeff")
        if line:
            recs.append(json.loads(line))
json.dump(recs, open(os.path.expanduser("~/sample/RBA-rates.json"), "w"))
print("wrote RBA-rates.json,", len(recs), "records. First record:")
print(json.dumps(recs[0], indent=2)[:400])