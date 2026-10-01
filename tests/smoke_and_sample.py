import csv, random, statistics, sys, time
from pathlib import Path
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

ROOT = Path(__file__).resolve().parent.parent
KEY = next(l.split("=", 1)[1].strip() for l in (ROOT / "api-keys.txt").read_text().splitlines()
           if l.startswith(("jev=", "TYPESAFE_API_KEY=")))
MODEL = "jev-1.13.0"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

client = TypeSafeClient(api_key=KEY, model=MODEL)

r = client.system_one(
    state="I was charged twice for my card payment, please fix ASAP.",
    questions={
        "topic": Choice(instructions="Which team should handle this?",
                        criteria={"billing": "Charges and payments", "technical": "Bugs", "sales": None}),
        "urgency": Score(instructions="How urgent is this?", criteria=["can wait", "this week", "today"]),
        "billing": Noul(instructions="Is this about billing?"),
    },
)
print("SMOKE", r.model, r.usage,
      r.answers["topic"].choice, round(r.answers["topic"].confidence, 2),
      round(r.answers["urgency"].score, 2), round(r.answers["billing"].noul, 2))

def load(name):
    with open(ROOT / "Datasets/banking77" / name, newline="") as f:
        return list(csv.DictReader(f))

train, test = load("train.csv"), load("test.csv")
labels = sorted({row["category"] for row in train})
print("labels:", len(labels))
QUESTION = {"intent": Choice(instructions="Which banking customer-service intent does this message express?",
                             criteria={l: None for l in labels})}
random.seed(0)
sample = random.sample(test, N)

rows, lat, toks = [], [], []
for row in sample:
    t = time.time()
    resp = client.system_one(state=row["text"], questions=QUESTION)
    lat.append(time.time() - t)
    a = resp.answers["intent"]
    toks.append(resp.usage.input_tokens)
    top3 = sorted(a.probabilities.items(), key=lambda kv: -kv[1])[:3]
    rows.append((row["category"], a.choice, a.confidence, [k for k, _ in top3]))

acc = statistics.mean(g == p for g, p, _, _ in rows)
top3 = statistics.mean(g in t3 for g, _, _, t3 in rows)
print(f"SAMPLE n={N} acc={acc:.3f} top3={top3:.3f} "
      f"lat_mean={statistics.mean(lat):.2f}s in_tokens={statistics.mean(toks):.0f}")
hi = [g == p for g, p, c, _ in rows if c >= 0.9]
lo = [g == p for g, p, c, _ in rows if c < 0.9]
print(f"conf>=0.9: n={len(hi)} acc={statistics.mean(hi) if hi else float('nan'):.3f} | "
      f"conf<0.9: n={len(lo)} acc={statistics.mean(lo) if lo else float('nan'):.3f}")
print("errors:")
for g, p, c, _ in rows:
    if g != p:
        print(f"  gold={g} pred={p} conf={c:.2f}")

outs = [client.system_one(state=sample[0]["text"], questions=QUESTION).answers["intent"] for _ in range(3)]
print("REPEAT", [(o.choice, round(o.confidence, 3)) for o in outs])
