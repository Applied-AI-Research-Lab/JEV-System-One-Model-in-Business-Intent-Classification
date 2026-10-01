import csv
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent / "Datasets" / "banking77"
rows = []
for split in ("train", "test"):
    with open(D / f"{split}.csv", newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append({"id": f"{split}-{i}", "split": split, "text": r["text"].strip(),
                         "label": r["category"].strip().rstrip("?")})
with open(D / "banking77_all.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "split", "text", "label"])
    w.writeheader(); w.writerows(rows)
labels = Counter(r["label"] for r in rows)
dups = len(rows) - len({r["text"] for r in rows})
print(f"rows={len(rows)} labels={len(labels)} min/max per label={min(labels.values())}/{max(labels.values())} duplicate_texts={dups}")
