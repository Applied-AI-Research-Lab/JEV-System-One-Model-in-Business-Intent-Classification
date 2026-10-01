import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent; J = ROOT / "JEVvsLLMs"; E = ROOT / "extra/predictions"
ld = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
CFG = {
    "Banking77": dict(s0=J / "jev_predictions/jev-1.13.0__raw__s0.jsonl", pat="banking__jev__raw__s{}.jsonl", n=5,
                      others={"Qwen3.5-4B": J / "predictions/qwen3.5-4b__raw__s0.jsonl", "Gemma-3-4B": J / "predictions/gemma-3-4b-it__raw__s0.jsonl",
                              "GLM-5.3": ROOT / "cloud/predictions/banking__glm-5.3-cloud__raw.jsonl"}),
    "ATIS": dict(s0=J / "jev_predictions_atis/jev-1.13.0__raw__s0.jsonl", pat="atis__jev__raw__s{}.jsonl", n=7,
                 others={"Qwen3.5-4B": J / "predictions_atis/qwen3.5-4b__raw__s0.jsonl", "Gemma-3-4B": J / "predictions_atis/gemma-3-4b-it__raw__s0.jsonl",
                         "GLM-5.3": ROOT / "cloud/predictions/atis__glm-5.3-cloud__raw.jsonl"}),
}


def macro_f1(g, p):
    f = []
    for l in set(g):
        tp = sum(a == l and b == l for a, b in zip(g, p)); fp = sum(a != l and b == l for a, b in zip(g, p)); fn = sum(a == l and b != l for a, b in zip(g, p))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def ece(conf, corr, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo else (conf <= hi)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e


def acc_at(conf, corr, cov):
    c = corr[np.argsort(-conf, kind="stable")]; return c[: int(cov * len(c))].mean()


out = ["# Jev across option orders\n"]
for ds, c in CFG.items():
    runs = [ld(c["s0"])] + [ld(E / c["pat"].format(s)) for s in range(1, c["n"])]
    ids = sorted(set.intersection(*[set(r) for r in runs]))
    gold = [runs[0][i]["gold"] for i in ids]
    rows, corrs, preds = [], [], []
    for k, r in enumerate(runs):
        p = [r[i]["pred"] for i in ids]; cr = np.array([a == b for a, b in zip(p, gold)]); cf = np.array([r[i]["p_top"] for i in ids])
        corrs.append(cr); preds.append(p)
        rows.append((k, cr.mean(), macro_f1(gold, p), ece(cf, cr), acc_at(cf, cr, .5)))
    out += [f"\n## {ds}: {len(ids)} rows, {len(runs)} orders (s0 = alphabetical)\n", "| order | accuracy | macro-F1 | ECE | acc@50% coverage |", "|---|---|---|---|---|"]
    out += [f"| s{k} | {a:.4f} | {f:.4f} | {e:.3f} | {c5:.3f} |" for k, a, f, e, c5 in rows]
    A = np.array([r[1] for r in rows]); F = np.array([r[2] for r in rows]); Ec = np.array([r[3] for r in rows])
    out.append(f"| **mean ± sd** | **{A.mean():.4f} ± {A.std(ddof=1):.4f}** | **{F.mean():.4f} ± {F.std(ddof=1):.4f}** | **{Ec.mean():.3f} ± {Ec.std(ddof=1):.3f}** | |")
    out.append(f"| min / max | {A.min():.4f} / {A.max():.4f} | {F.min():.4f} / {F.max():.4f} | | |")
    vote = []
    for j in range(len(ids)):
        cnt = Counter(p[j] for p in preds).most_common(); vote.append(cnt[0][0] if len(cnt) == 1 or cnt[0][1] > cnt[1][1] else preds[0][j])
    agree = np.array([Counter(p[j] for p in preds).most_common(1)[0][1] / len(preds) for j in range(len(ids))])
    out.append(f"\nMajority vote over {len(runs)} orders: accuracy {np.mean([v == g for v, g in zip(vote, gold)]):.4f}, macro-F1 {macro_f1(gold, vote):.4f}.")
    vc = np.array([v == g for v, g in zip(vote, gold)])
    out.append(f"Rows where all orders agree: {np.mean(agree == 1):.1%} (vote accuracy there {vc[agree == 1].mean():.3f}); the rest: vote accuracy {vc[agree < 1].mean():.3f}.")
    out.append("\nSingle-order comparison systems (alphabetical order, same rows):\n\n| system | accuracy | macro-F1 | Jev orders above this accuracy |\n|---|---|---|---|")
    for n, p in c["others"].items():
        d = ld(p); pr = [d[i]["pred"] for i in ids]; ac = np.mean([a == b for a, b in zip(pr, gold)])
        out.append(f"| {n} | {ac:.4f} | {macro_f1(gold, pr):.4f} | {int((A > ac).sum())} of {len(A)} |")
text = "\n".join(out) + "\n"
(Path(__file__).resolve().parent / "analysis_orders.md").write_text(text)
print(text)
