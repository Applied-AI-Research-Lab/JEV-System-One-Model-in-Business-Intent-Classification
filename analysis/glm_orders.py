import json, numpy as np
from collections import Counter
from common import *

D = load_all()
glm = {0: D["atis"]["GLM"]["raw"][0]}
for k in (1, 2): glm[k] = ld(C / f"atis__glm-5.3-cloud__raw__s{k}.jsonl")
ids = sorted(set.intersection(*[set(glm[k]) for k in glm]))
gold = [glm[0][i]["gold"] for i in ids]
sd = lambda v: float(np.std(v, ddof=1))
out = {"n": len(ids), "seeds": [0, 1, 2], "sys": {}}
cors = {}
for s in SYS:
    runs = {k: (glm[k] if s == "GLM" else D["atis"][s]["raw"][k]) for k in (0, 1, 2)}
    A, F, P = [], [], []
    for k in (0, 1, 2):
        p = [runs[k][i]["pred"] for i in ids]; P.append(p)
        A.append(float(np.mean([a == b for a, b in zip(p, gold)]))); F.append(macro_f1(gold, p))
    agree = float(np.mean([len({p[j] for p in P}) == 1 for j in range(len(ids))]))
    out["sys"][s] = dict(acc=A, f1=F, acc_mean=float(np.mean(A)), acc_sd=sd(A), acc_range=max(A) - min(A), f1_mean=float(np.mean(F)), f1_sd=sd(F), agree=agree)
    cors[s] = [np.array([runs[k][i]["pred"] == runs[k][i]["gold"] for i in ids]) for k in (0, 1, 2)]
out["glm_invalid"] = {k: float(np.mean([not glm[k][i].get("valid", True) for i in ids])) for k in (0, 1, 2)}
casc = {}
for frac in (0.1, 0.2, 0.3):
    accs = []
    for k in (0, 1, 2):
        conf = np.array([D["atis"]["Jev"]["raw"][k][i]["p_top"] for i in ids]); n = int(round(frac * len(ids)))
        route = np.argsort(conf)[:n]; c = cors["Jev"][k].copy(); c[route] = cors["GLM"][k][route]; accs.append(float(c.mean()))
    casc[frac] = dict(accs=accs, mean=float(np.mean(accs)), sd=sd(accs))
out["cascade_same_order"] = casc
json.dump(out, open(Path(__file__).resolve().parent / "glm_orders.json", "w"), indent=1)
for s, v in out["sys"].items(): print(s, [round(a * 100, 1) for a in v["acc"]], "mean %.1f sd %.1f range %.1f f1 %.3f sd %.3f agree %.1f" % (v["acc_mean"] * 100, v["acc_sd"] * 100, v["acc_range"] * 100, v["f1_mean"], v["f1_sd"], v["agree"] * 100))
print("glm invalid", out["glm_invalid"]); print({k: (round(v["mean"] * 100, 1), round(v["sd"] * 100, 1)) for k, v in casc.items()})
