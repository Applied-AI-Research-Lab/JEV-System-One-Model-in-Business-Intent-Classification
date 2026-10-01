import json, math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "extra/predictions"
ld = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
out = []
P = lambda s="": out.append(s)


def mcnemar_p(a, b):
    n01, n10 = int(((~a) & b).sum()), int((a & (~b)).sum()); n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)


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


base = {"Banking77": ROOT / "JEVvsLLMs/jev_predictions/jev-1.13.0__raw__s0.jsonl", "ATIS": ROOT / "JEVvsLLMs/jev_predictions_atis/jev-1.13.0__raw__s0.jsonl"}
shuf = {"Banking77": [E / "banking__jev__raw__s1.jsonl", E / "banking__jev__raw__s2.jsonl"], "ATIS": [E / "atis__jev__raw__s1.jsonl", E / "atis__jev__raw__s2.jsonl"]}
det = {"Banking77": [E / f"banking__jev__det{i}.jsonl" for i in (1, 2, 3)], "ATIS": [E / f"atis__jev__det{i}.jsonl" for i in (1, 2, 3)]}

P("# Extra Jev-API analyses\n")
P("## 1. Sensitivity to the order of the options (alphabetical s0 vs two random orders)\n")
for ds in base:
    runs = {"s0 (alphabetical)": ld(base[ds]), "s1": ld(shuf[ds][0]), "s2": ld(shuf[ds][1])}
    ids = sorted(set.intersection(*[set(r) for r in runs.values()]))
    corr = {k: np.array([r[i]["pred"] == r[i]["gold"] for i in ids]) for k, r in runs.items()}
    preds = {k: [r[i]["pred"] for i in ids] for k, r in runs.items()}
    gold = [runs["s0 (alphabetical)"][i]["gold"] for i in ids]
    P(f"### {ds} ({len(ids)} rows)\n")
    P("| order | accuracy | macro-F1 | ECE |\n|---|---|---|---|")
    for k, r in runs.items():
        conf = np.array([r[i]["p_top"] for i in ids])
        P(f"| {k} | {corr[k].mean():.4f} | {macro_f1(gold, preds[k]):.4f} | {ece(conf, corr[k]):.3f} |")
    accs = np.array([c.mean() for c in corr.values()])
    P(f"\nAccuracy mean {accs.mean():.4f}, range {accs.max() - accs.min():.4f} (max-min).")
    ks = list(runs)
    for a_, b_ in ((0, 1), (0, 2), (1, 2)):
        same = np.mean([x == y for x, y in zip(preds[ks[a_]], preds[ks[b_]])])
        P(f"- {ks[a_]} vs {ks[b_]}: identical prediction on {same:.1%} of rows; McNemar p = {mcnemar_p(corr[ks[a_]], corr[ks[b_]]):.2g}")
    allsame = np.mean([len({preds[k][j] for k in ks}) == 1 for j in range(len(ids))])
    vote = []
    for j in range(len(ids)):
        c = Counter(preds[k][j] for k in ks).most_common(); vote.append(c[0][0] if c[0][1] >= 2 else preds[ks[0]][j])
    P(f"- All three orders agree on {allsame:.1%} of rows. Majority vote over the three orders: accuracy {np.mean([v == g for v, g in zip(vote, gold)]):.4f} (single run {corr[ks[0]].mean():.4f}).")
    agree = np.array([len({preds[k][j] for k in ks}) == 1 for j in range(len(ids))])
    P(f"- Accuracy when the three orders agree: {corr[ks[0]][agree].mean():.3f} ({int(agree.sum())} rows); when they disagree: {corr[ks[0]][~agree].mean():.3f} ({int((~agree).sum())} rows) -> order-disagreement is itself an uncertainty signal.\n")

P("## 2. Repeatability of the API (same 500 rows, 3 repetitions, temperature-free)\n")
for ds in det:
    runs = [ld(p) for p in det[ds]]
    ids = sorted(set.intersection(*[set(r) for r in runs]))
    same = np.mean([len({r[i]["pred"] for r in runs}) == 1 for i in ids])
    dp = [max(r[i]["p_top"] for r in runs) - min(r[i]["p_top"] for r in runs) for i in ids]
    P(f"- {ds}: {len(ids)} rows x 3 runs -> identical prediction in {same:.1%} of rows; top-probability spread (max-min across runs): mean {np.mean(dp):.4f}, max {np.max(dp):.4f}.")

P("\n## 3. ATIS with option descriptions (post-hoc diagnostic, NOT pure zero-shot)\n")
raw = ld(base["ATIS"]); desc = ld(E / "atis__jev__desc.jsonl")
ids = sorted(set(raw) & set(desc)); gold = [raw[i]["gold"] for i in ids]
cr = np.array([raw[i]["pred"] == raw[i]["gold"] for i in ids]); cd = np.array([desc[i]["pred"] == desc[i]["gold"] for i in ids])
conf_d = np.array([desc[i]["p_top"] for i in ids]); conf_r = np.array([raw[i]["p_top"] for i in ids])
P("| condition | accuracy | macro-F1 | ECE | McNemar p vs names-only |\n|---|---|---|---|---|")
P(f"| names only (zero-shot) | {cr.mean():.4f} | {macro_f1(gold, [raw[i]['pred'] for i in ids]):.4f} | {ece(conf_r, cr):.3f} | - |")
P(f"| + descriptions | {cd.mean():.4f} | {macro_f1(gold, [desc[i]['pred'] for i in ids]):.4f} | {ece(conf_d, cd):.3f} | {mcnemar_p(cd, cr):.2g} |")
P("\nPer-label F1 (precision / recall):\n\n| label | names only | + descriptions |\n|---|---|---|")
for l in sorted(set(gold), key=lambda l: -gold.count(l)):
    cells = []
    for d in (raw, desc):
        p_ = [d[i]["pred"] for i in ids]
        tp = sum(g == l and q == l for g, q in zip(gold, p_)); fp = sum(g != l and q == l for g, q in zip(gold, p_)); fn = sum(g == l and q != l for g, q in zip(gold, p_))
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        cells.append(f"{2*pr*rc/(pr+rc) if pr+rc else 0:.2f} ({pr:.2f} / {rc:.2f})")
    P(f"| {l} | {cells[0]} | {cells[1]} |")

P("\n## 4. ATIS with eight Noul questions (one per intent, absolute probabilities)\n")
nl = ld(E / "atis__jev__noul.jsonl"); ids = sorted(set(raw) & set(nl)); gold = [raw[i]["gold"] for i in ids]
cn = np.array([nl[i]["pred"] == nl[i]["gold"] for i in ids]); cr = np.array([raw[i]["pred"] == raw[i]["gold"] for i in ids])
conf_n = np.array([nl[i]["p_top"] for i in ids])
P(f"- argmax of the eight Noul values: accuracy {cn.mean():.4f}, macro-F1 {macro_f1(gold, [nl[i]['pred'] for i in ids]):.4f} (Choice names-only: {cr.mean():.4f} / {macro_f1(gold, [raw[i]['pred'] for i in ids]):.4f}); McNemar p = {mcnemar_p(cn, cr):.2g}.")
P(f"- ECE of the winning Noul value: {ece(conf_n, cn):.3f}. Sum of the eight probabilities: mean {np.mean([sum(nl[i]['probs'].values()) for i in ids]):.2f} (a Choice sums to 1).")
multi = np.mean([sum(v > 0.5 for v in nl[i]["probs"].values()) > 1 for i in ids]); none = np.mean([max(nl[i]["probs"].values()) < 0.5 for i in ids])
P(f"- Rows with more than one intent above 0.5: {multi:.1%}; rows with no intent above 0.5: {none:.1%}.")
P("\n| label | Noul F1 (precision / recall) | Choice F1 (precision / recall) |\n|---|---|---|")
for l in sorted(set(gold), key=lambda l: -gold.count(l)):
    cells = []
    for d in (nl, raw):
        p_ = [d[i]["pred"] for i in ids]
        tp = sum(g == l and q == l for g, q in zip(gold, p_)); fp = sum(g != l and q == l for g, q in zip(gold, p_)); fn = sum(g == l and q != l for g, q in zip(gold, p_))
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        cells.append(f"{2*pr*rc/(pr+rc) if pr+rc else 0:.2f} ({pr:.2f} / {rc:.2f})")
    P(f"| {l} | {cells[0]} | {cells[1]} |")
text = "\n".join(out) + "\n"
(Path(__file__).resolve().parent / "analysis_extra.md").write_text(text)
print(text)
