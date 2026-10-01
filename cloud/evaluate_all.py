import glob, json, math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / "JEVvsLLMs"
SETS = {
    "Banking77 (77 intents)": [
        ("Jev", J / "jev_predictions/jev-1.13.0__{v}__s0.jsonl"),
        ("Qwen3.5-4B", J / "predictions/qwen3.5-4b__{v}__s0.jsonl"),
        ("Gemma-3-4B", J / "predictions/gemma-3-4b-it__{v}__s0.jsonl"),
        ("GLM-5.3 (cloud)", ROOT / "cloud/predictions/banking__glm-5.3-cloud__{v}.jsonl")],
    "ATIS (8 intents)": [
        ("Jev", J / "jev_predictions_atis/jev-1.13.0__{v}__s0.jsonl"),
        ("Qwen3.5-4B", [J / "predictions_atis/qwen3.5-4b__{v}__s0.jsonl", J / "jev_predictions_atis/qwen3.5-4b__{v}__s0.jsonl"]),
        ("Gemma-3-4B", [J / "predictions_atis/gemma-3-4b-it__{v}__s0.jsonl", J / "jev_predictions_atis/gemma-3-4b-it__{v}__s0.jsonl"]),
        ("GLM-5.3 (cloud)", ROOT / "cloud/predictions/atis__glm-5.3-cloud__{v}.jsonl")],
}


def load(p):
    for q in (p if isinstance(p, list) else [p]):
        if q.exists():
            return {r["id"]: r for r in map(json.loads, open(q))}
    return None


def ece(conf, corr, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo else (conf <= hi)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e


def rc(conf, corr):
    c = corr[np.argsort(-conf, kind="stable")]
    a = np.cumsum(c) / np.arange(1, len(c) + 1)
    return (1 - a).mean(), a[int(.5 * len(c)) - 1], a[int(.8 * len(c)) - 1]


def macro_f1(gold, pred):
    f = []
    for l in set(gold):
        tp = sum(g == l and p == l for g, p in zip(gold, pred)); fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def mcnemar_p(a, b):
    n01, n10 = int(((~a) & b).sum()), int((a & (~b)).sum()); n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)


rng = np.random.default_rng(0)
out = ["# Jev vs small LLMs vs GLM-5.3 — comparison\n"]
for dname, systems in SETS.items():
    for variant in ("raw", "human"):
        data = {n: load([Path(str(q).format(v=variant)) for q in (p if isinstance(p, list) else [p])]) for n, p in systems}
        data = {n: d for n, d in data.items() if d}
        if "Jev" not in data or len(data) < 2:
            continue
        ids = sorted(set.intersection(*[set(d) for d in data.values()]))
        gold = [data["Jev"][i]["gold"] for i in ids]
        maj = Counter(gold).most_common(1)[0][1] / len(gold)
        out += [f"\n## {dname} — {variant} labels — {len(ids)} rows in common (majority baseline {maj:.3f})\n",
                "| system | acc [95% CI] | macro-F1 | invalid | ECE | AURC | acc@50% | acc@80% | in tok | out tok | s/utt (see note) | McNemar p vs Jev |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        jc = np.array([data["Jev"][i]["gold"] == data["Jev"][i]["pred"] for i in ids])
        for n, d in data.items():
            recs = [d[i] for i in ids]
            corr = np.array([r["gold"] == r["pred"] for r in recs])
            ci = np.percentile([corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(1000)], [2.5, 97.5])
            conf = [r.get("p_top") for r in recs]
            if any(c is None for c in conf):
                e = au = a5 = a8 = "n/a"
            else:
                c = np.array(conf); e = f"{ece(c, corr):.3f}"; x = rc(c, corr); au, a5, a8 = f"{x[0]:.3f}", f"{x[1]:.3f}", f"{x[2]:.3f}"
            inv = np.mean([r.get("valid", True) is False for r in recs])
            mc = "-" if n == "Jev" else f"{mcnemar_p(corr, jc):.2g}"
            it = np.mean([r.get("in_tokens") or 0 for r in recs]); ot = np.mean([r.get("out_tokens") or 0 for r in recs])
            out.append(f"| {n} | {corr.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {macro_f1([r['gold'] for r in recs], [r['pred'] for r in recs]):.3f} | {inv:.3f} | "
                       f"{e} | {au} | {a5} | {a8} | {it:.0f} | {ot:.0f} | {np.mean([r['latency_s'] for r in recs]):.2f} | {mc} |")
        out.append("\nNote: s/utt = mean latency per utterance; Jev = hosted API call, SLMs = batched (8) on one H100, GLM = Ollama cloud with reasoning. Not like-for-like: compare tokens and cost, not seconds.")
        if "GLM-5.3 (cloud)" in data:
            g = data["GLM-5.3 (cloud)"]
            gc = np.array([g[i]["gold"] == g[i]["pred"] for i in ids])
            out.append(f"\nPaired Jev vs GLM: both right {int((jc & gc).sum())}, only GLM right {int(((~jc) & gc).sum())}, only Jev right {int((jc & (~gc)).sum())}, both wrong {int(((~jc) & (~gc)).sum())}.")
            conf = np.array([data["Jev"][i]["p_top"] for i in ids])
            order = np.argsort(conf, kind="stable")
            out += ["\nCascade Jev -> GLM (least confident Jev answers go to GLM):\n", "| share sent to GLM | accuracy | GLM calls |", "|---|---|---|"]
            for share in (0, .05, .1, .2, .3, .5, 1.0):
                k = int(round(share * len(ids))); m = jc.copy(); m[order[:k]] = gc[order[:k]]
                out.append(f"| {share:.0%} | {m.mean():.3f} | {k} |")
text = "\n".join(out) + "\n"
(Path(__file__).resolve().parent / "results_comparison.md").write_text(text)
print(text)
