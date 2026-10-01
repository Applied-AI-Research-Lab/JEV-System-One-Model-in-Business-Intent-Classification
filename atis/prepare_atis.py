import csv
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "Datasets" / "ATIS Airline Travel Information System"
rows, seen = [], set()
for split in ("train", "test"):
    with open(D / f"atis_intents_{split}.csv", newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.reader(f)):
            label, text = r[0].strip(), r[1].strip()
            key = (text, label)
            rows.append({"id": f"{split}-{i}", "split": split, "text": text, "label": label, "dup": int(key in seen)})
            seen.add(key)
with open(D / "atis_all.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "split", "text", "label", "dup"])
    w.writeheader(); w.writerows(rows)
lab = Counter(r["label"] for r in rows)
print(f"rows={len(rows)} unique={len(rows)-sum(r['dup'] for r in rows)} labels={len(lab)}")
for l, n in lab.most_common():
    print(f"  {l:22s} {n:5d}  {n/len(rows):.1%}")
