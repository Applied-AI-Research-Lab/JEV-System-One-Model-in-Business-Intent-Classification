import json, os
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent; J = ROOT / "JEVvsLLMs"; X = ROOT / "extra/predictions"
ORD = Path(os.environ.get("ORDERS_DIR") or J / "predictions_orders")
CFG = {
    "Banking77": dict(key="banking", s0={"Jev": J / "jev_predictions/jev-1.13.0__raw__s0.jsonl", "Qwen3.5-4B": J / "predictions/qwen3.5-4b__raw__s0.jsonl", "Gemma-3-4B": J / "predictions/gemma-3-4b-it__raw__s0.jsonl"}),
    "ATIS": dict(key="atis", s0={"Jev": J / "jev_predictions_atis/jev-1.13.0__raw__s0.jsonl", "Qwen3.5-4B": J / "predictions_atis/qwen3.5-4b__raw__s0.jsonl", "Gemma-3-4B": J / "predictions_atis/gemma-3-4b-it__raw__s0.jsonl"}),
}
TAG = {"Qwen3.5-4B": "qwen3.5-4b", "Gemma-3-4B": "gemma-3-4b-it"}
ld = lambda p: {r["id"]: r for r in map(json.loads, open(p))}


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


out = ["# Option-order sensitivity: Jev vs Qwen3.5-4B vs Gemma-3-4B\n", "Seed 0 = alphabetical order; seeds >= 1 = the same random orders for every system.\n"]
for ds, c in CFG.items():
    runs = {}
    for sys_, p in c["s0"].items():
        if p.exists(): runs.setdefault(sys_, {})[0] = ld(p)
    for seed in range(1, 12):
        if (X / f"{c['key']}__jev__raw__s{seed}.jsonl").exists():
            runs.setdefault("Jev", {})[seed] = ld(X / f"{c['key']}__jev__raw__s{seed}.jsonl")
        for sys_, tag in TAG.items():
            p = ORD / c["key"] / f"{tag}__raw__s{seed}.jsonl"
            if p.exists(): runs.setdefault(sys_, {})[seed] = ld(p)
    seeds_common = sorted(set.intersection(*[set(r) for r in runs.values()])) if runs else []
    if len(runs) < 3 or len(seeds_common) < 2:
        out.append(f"\n## {ds}\n\nNot enough matched runs yet (systems found: {{ {', '.join(f'{k}: seeds {sorted(v)}' for k, v in runs.items())} }}).\n"); continue
    ids = sorted(set.intersection(*[set(r) for s in runs.values() for sd, r in s.items() if sd in seeds_common]))
    gold = [runs["Jev"][0][i]["gold"] for i in ids]
    out += [f"\n## {ds}: {len(ids)} rows, seeds used: {seeds_common}\n", "| system | order | accuracy | macro-F1 | ECE |", "|---|---|---|---|---|"]
    stats, preds = {}, {}
    for sys_ in ("Jev", "Qwen3.5-4B", "Gemma-3-4B"):
        A, F, Ec, preds[sys_] = [], [], [], []
        for sd in seeds_common:
            r = runs[sys_][sd]; p = [r[i]["pred"] for i in ids]; cr = np.array([a == b for a, b in zip(p, gold)]); cf = np.array([r[i]["p_top"] for i in ids])
            A.append(cr.mean()); F.append(macro_f1(gold, p)); Ec.append(ece(cf, cr)); preds[sys_].append(p)
            out.append(f"| {sys_} | s{sd} | {A[-1]:.4f} | {F[-1]:.4f} | {Ec[-1]:.3f} |")
        stats[sys_] = (np.array(A), np.array(F), np.array(Ec))
    out += ["\n### Summary over orders\n", "| system | accuracy mean ± sd | range (max-min) | macro-F1 mean ± sd | ECE mean | orders agreeing on a row (all) | accuracy: agree / disagree | vote accuracy |", "|---|---|---|---|---|---|---|---|"]
    for sys_, (A, F, Ec) in stats.items():
        P = preds[sys_]; agree = np.array([len({p[j] for p in P}) == 1 for j in range(len(ids))])
        corr0 = np.array([P[0][j] == gold[j] for j in range(len(ids))])
        vote = [Counter(p[j] for p in P).most_common(1)[0][0] for j in range(len(ids))]
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")
        out.append(f"| {sys_} | {A.mean():.4f} ± {sd(A):.4f} | {A.max() - A.min():.4f} | {F.mean():.4f} ± {sd(F):.4f} | {Ec.mean():.3f} | {agree.mean():.1%} | "
                   f"{corr0[agree].mean():.3f} / {corr0[~agree].mean() if (~agree).any() else float('nan'):.3f} | {np.mean([v == g for v, g in zip(vote, gold)]):.4f} |")
    out.append("\n### Does Jev beat each small model under the SAME option order?\n\n| comparison | seeds where Jev accuracy is higher | seeds where Jev macro-F1 is higher |\n|---|---|---|")
    for sys_ in ("Qwen3.5-4B", "Gemma-3-4B"):
        out.append(f"| Jev vs {sys_} | {int((stats['Jev'][0] > stats[sys_][0]).sum())} of {len(seeds_common)} | {int((stats['Jev'][1] > stats[sys_][1]).sum())} of {len(seeds_common)} |")
text = "\n".join(out) + "\n"
(Path(__file__).resolve().parent / "analysis_slm_orders.md").write_text(text)
print(text)
