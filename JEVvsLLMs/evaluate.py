import glob, json, math, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_jd = Path(os.environ.get("JEV_PRED_DIR") or HERE / "jev_predictions")
if not _jd.exists():
    _jd = HERE / "jev_predictions" if (HERE / "jev_predictions").exists() else HERE.parent / "jev/predictions"
files = sorted(glob.glob(str(_jd / "*.jsonl"))) + sorted(glob.glob(str(HERE / "predictions/*.jsonl")))


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def ece(conf, correct, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo else (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def risk_coverage(conf, correct):
    o = np.argsort(-conf, kind="stable")
    c = correct[o]
    k = np.arange(1, len(c) + 1)
    acc_at = np.cumsum(c) / k
    return (1 - acc_at).mean(), acc_at[int(.5 * len(c)) - 1], acc_at[int(.8 * len(c)) - 1]


def macro_f1(gold, pred):
    f = []
    for l in set(gold):
        tp = sum(g == l and p == l for g, p in zip(gold, pred)); fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def mcnemar_p(a, b):
    n01 = int(((~a) & b).sum()); n10 = int((a & (~b)).sum()); n = n01 + n10
    if n == 0: return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


preds = {Path(p).parent.parent.name + "/" + Path(p).stem: load(p) for p in files}
ref_name = next((k for k in preds if k.split("/", 1)[1].startswith("jev-")), None)
lines = ["| system | variant | order | n | acc [95% CI] | macro-F1 | top5 | ECE | AURC | acc@50% | acc@80% | invalid | p50 lat (s) | in tok | McNemar p vs ref |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
rng = np.random.default_rng(0)
for name, d in preds.items():
    ids = sorted(d)
    recs = [d[i] for i in ids]
    system, rest = name.split("/", 1)[1].split("__", 1)[0], name.split("__")
    variant, order = rest[1], rest[2]
    gold = [r["gold"] for r in recs]; pred = [r["pred"] for r in recs]
    corr = np.array([g == p for g, p in zip(gold, pred)])
    boots = [corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(1000)]
    ci = np.percentile(boots, [2.5, 97.5])
    top5 = "-" if "top5" not in recs[0] else f"{np.mean([r['gold'] in [l for l, _ in r['top5']] for r in recs]):.3f}"
    conf = np.array([r["p_top"] if r["p_top"] is not None else np.nan for r in recs])
    if np.isnan(conf).any():
        e = aurc = a50 = a80 = "n/a"
    else:
        e = f"{ece(conf, corr):.3f}"; au, a5, a8 = risk_coverage(conf, corr); aurc, a50, a80 = f"{au:.3f}", f"{a5:.3f}", f"{a8:.3f}"
    mc = "-"
    if ref_name and name != ref_name:
        common = sorted(set(d) & set(preds[ref_name]))
        a = np.array([d[i]["gold"] == d[i]["pred"] for i in common]); b = np.array([preds[ref_name][i]["gold"] == preds[ref_name][i]["pred"] for i in common])
        mc = f"{mcnemar_p(a, b):.2g}"
    inv = np.mean([r.get('valid', True) is False for r in recs])
    lat = np.median([r["latency_s"] for r in recs]); tok = np.mean([r["in_tokens"] or 0 for r in recs])
    lines.append(f"| {system} | {variant} | {order} | {len(recs)} | {corr.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {macro_f1(gold, pred):.3f} | {top5} | {e} | {aurc} | {a50} | {a80} | {inv:.3f} | {lat:.2f} | {tok:.0f} | {mc} |")
out = "\n".join(lines)
(HERE / "results.md").write_text(out + "\n")
print(out)
