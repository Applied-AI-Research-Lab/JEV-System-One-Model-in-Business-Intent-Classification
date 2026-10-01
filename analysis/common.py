import json, math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
J, X, C, ORD = ROOT / "JEVvsLLMs", ROOT / "extra/predictions", ROOT / "cloud/predictions", ROOT / "JEVvsLLMs/predictions_orders"
PAPER = ROOT / "paper"
SYS = ["Jev", "Qwen", "Gemma", "GLM"]
NAME = {"Jev": "Jev", "Qwen": "Qwen3.5-4B", "Gemma": "Gemma-3-4B", "GLM": "GLM-5.3"}
DS = {"bank": "Banking77", "atis": "ATIS"}
ld = lambda p: {r["id"]: r for r in map(json.loads, open(p))}


def load_all():
    D = {ds: {s: {"raw": {}, "human": {}} for s in SYS} for ds in DS}
    jev0 = {"bank": J / "jev_predictions/jev-1.13.0__{v}__s0.jsonl", "atis": J / "jev_predictions_atis/jev-1.13.0__{v}__s0.jsonl"}
    slm0 = {"bank": J / "predictions", "atis": J / "predictions_atis"}
    tag = {"Qwen": "qwen3.5-4b", "Gemma": "gemma-3-4b-it"}
    key = {"bank": "banking", "atis": "atis"}
    for ds in DS:
        for v in ("raw", "human"):
            D[ds]["Jev"][v][0] = ld(str(jev0[ds]).format(v=v))
            for s in ("Qwen", "Gemma"):
                D[ds][s][v][0] = ld(slm0[ds] / f"{tag[s]}__{v}__s0.jsonl")
        for seed in range(1, 12):
            p = X / f"{key[ds]}__jev__raw__s{seed}.jsonl"
            if p.exists(): D[ds]["Jev"]["raw"][seed] = ld(p)
            for s in ("Qwen", "Gemma"):
                p = ORD / key[ds] / f"{tag[s]}__raw__s{seed}.jsonl"
                if p.exists(): D[ds][s]["raw"][seed] = ld(p)
        D[ds]["GLM"]["raw"][0] = ld(C / f"{key[ds]}__glm-5.3-cloud__raw.jsonl")
    return D


def macro_f1(g, p):
    f = []
    for l in set(g):
        tp = sum(a == l and b == l for a, b in zip(g, p)); fp = sum(a != l and b == l for a, b in zip(g, p)); fn = sum(a == l and b != l for a, b in zip(g, p))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def per_label(g, p):
    out = {}
    for l in sorted(set(g)):
        tp = sum(a == l and b == l for a, b in zip(g, p)); fp = sum(a != l and b == l for a, b in zip(g, p)); fn = sum(a == l and b != l for a, b in zip(g, p))
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        out[l] = dict(p=pr, r=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, n=tp + fn)
    return out


def ece(conf, corr, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo else (conf <= hi)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def risk_coverage(conf, corr):
    c = corr[np.argsort(-conf, kind="stable")]
    acc_at = np.cumsum(c) / np.arange(1, len(c) + 1)
    return acc_at


def aurc(conf, corr): return float((1 - risk_coverage(conf, corr)).mean())
def acc_at(conf, corr, cov): return float(risk_coverage(conf, corr)[int(cov * len(corr)) - 1])


def auroc(score, positive):
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(score, kind="stable"); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):
        m = score == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mcnemar_p(a, b):
    n01, n10 = int(((~a) & b).sum()), int((a & (~b)).sum()); n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)


def boot_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def vec(run, ids, field="pred"): return [run[i][field] for i in ids]
def correct(run, ids): return np.array([run[i]["pred"] == run[i]["gold"] for i in ids])
def conf(run, ids):
    v = [run[i].get("p_top") for i in ids]
    return None if any(x is None for x in v) else np.array(v, float)
