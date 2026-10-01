import glob, json, os
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
files = sorted(glob.glob(str(HERE / "predictions/*.jsonl")))


def ece(conf, corr, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo else (conf <= hi)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e


def risk_coverage(conf, corr):
    c = corr[np.argsort(-conf, kind="stable")]
    acc_at = np.cumsum(c) / np.arange(1, len(c) + 1)
    return (1 - acc_at).mean(), acc_at[int(.5 * len(c)) - 1], acc_at[int(.8 * len(c)) - 1]


def prf(gold, pred, labels):
    out = {}
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred)); fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        pr, rc = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        out[l] = (pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0, tp + fn)
    return out


rng = np.random.default_rng(0)
summary = ["| system | variant | order | rows | n | acc [95% CI] | majority baseline | macro-F1 | weighted-F1 | top2 | top3 | ECE | AURC | acc@50% | acc@80% | invalid | p50 lat (s) |",
           "|" + "---|" * 17]
detail = []
for path in files:
    recs_all = [json.loads(l) for l in open(path)]
    name = Path(path).stem
    system, variant, order = name.split("__")
    for subset, recs in (("all", recs_all), ("unique", [r for r in recs_all if not r.get("dup", 0)])):
        gold = [r["gold"] for r in recs]; pred = [r["pred"] for r in recs]
        labels = sorted(set(gold))
        corr = np.array([g == p for g, p in zip(gold, pred)])
        ci = np.percentile([corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(1000)], [2.5, 97.5])
        stats = prf(gold, pred, labels)
        macro = np.mean([v[2] for v in stats.values()]); wf1 = sum(v[2] * v[3] for v in stats.values()) / len(gold)
        maj = Counter(gold).most_common(1)[0][1] / len(gold)
        topk = lambda k: np.mean([r["gold"] in [l for l, _ in r["top5"][:k]] for r in recs]) if "top5" in recs[0] else float("nan")
        conf = np.array([r["p_top"] for r in recs])
        au, a5, a8 = risk_coverage(conf, corr)
        inv = np.mean([r.get("valid", True) is False for r in recs])
        summary.append(f"| {system} | {variant} | {order} | {subset} | {len(recs)} | {corr.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {maj:.3f} | "
                       f"{macro:.3f} | {wf1:.3f} | {topk(2):.3f} | {topk(3):.3f} | {ece(conf, corr):.3f} | {au:.3f} | {a5:.3f} | {a8:.3f} | "
                       f"{inv:.3f} | {np.median([r['latency_s'] for r in recs]):.2f} |")
        if subset == "all":
            detail.append(f"\n### {system} / {variant} / {order} (all rows)\n\n| label | precision | recall | F1 | support |\n|---|---|---|---|---|")
            for l, (pr, rc, f1, sup) in sorted(stats.items(), key=lambda kv: -kv[1][3]):
                detail.append(f"| {l} | {pr:.3f} | {rc:.3f} | {f1:.3f} | {sup} |")
            conf_pairs = Counter((g, p) for g, p in zip(gold, pred) if g != p).most_common(6)
            detail.append("\nTop confusions (gold → predicted): " + "; ".join(f"{g} → {p} ({n})" for (g, p), n in conf_pairs))
text = "# ATIS results\n\n" + "\n".join(summary) + "\n" + "\n".join(detail) + "\n"
(HERE / "results_atis.md").write_text(text)
print(text)
