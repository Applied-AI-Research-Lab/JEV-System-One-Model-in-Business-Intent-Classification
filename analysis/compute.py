import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

D = load_all()
R = {}
SHARES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
PRICE_PER_MTOK = 0.042


def stats_of(run, ids, with_conf=True):
    g = vec(run, ids, "gold"); p = vec(run, ids); c = correct(run, ids); cf = conf(run, ids) if with_conf else None
    lo, hi = boot_ci(c)
    d = dict(n=len(ids), acc=float(c.mean()), ci=[lo, hi], f1=macro_f1(g, p), invalid=float(np.mean([run[i].get("valid", True) is False for i in ids])))
    if cf is not None:
        d.update(ece=ece(cf, c), aurc=aurc(cf, c), acc50=acc_at(cf, c, .5), acc80=acc_at(cf, c, .8), auroc=auroc(cf, c), meanconf=float(cf.mean()))
    return d


for ds in DS:
    r = {}
    runs0 = {s: D[ds][s]["raw"][0] for s in SYS}
    ids = sorted(set.intersection(*[set(v) for v in runs0.values()]))
    gold = vec(runs0["Jev"], ids, "gold")
    r["n"] = len(ids)
    cls = Counter(gold)
    texts = [runs0["Jev"][i]["text"] for i in ids]
    r["data"] = dict(rows=len(ids), classes=len(cls), majority=cls.most_common(1)[0][1] / len(ids), min_class=min(cls.values()), max_class=max(cls.values()),
                     median_words=float(np.median([len(t.split()) for t in texts])), mean_words=float(np.mean([len(t.split()) for t in texts])),
                     unique_rows=len({(runs0["Jev"][i]["text"], runs0["Jev"][i]["gold"]) for i in ids}))
    jc = correct(runs0["Jev"], ids)

    r["main"] = {}
    for s in SYS:
        d = stats_of(runs0[s], ids)
        d["mcnemar_vs_jev"] = None if s == "Jev" else mcnemar_p(correct(runs0[s], ids), jc)
        d["in_tok"] = float(np.mean([runs0[s][i].get("in_tokens") or 0 for i in ids]))
        d["out_tok"] = float(np.mean([runs0[s][i].get("out_tokens") or 0 for i in ids])) if s == "GLM" else None
        d["lat_mean"] = float(np.mean([runs0[s][i]["latency_s"] for i in ids])); d["lat_median"] = float(np.median([runs0[s][i]["latency_s"] for i in ids]))
        r["main"][s] = d
    rng = np.random.default_rng(1); r["diff_vs_jev"] = {}
    for s in ("Qwen", "Gemma", "GLM"):
        dlt = jc.astype(float) - correct(runs0[s], ids).astype(float)
        bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
        r["diff_vs_jev"][s] = dict(diff=float(dlt.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    gc = correct(runs0["GLM"], ids)
    r["jev_vs_glm"] = dict(both=int((jc & gc).sum()), only_glm=int(((~jc) & gc).sum()), only_jev=int((jc & ~gc).sum()), neither=int((~jc & ~gc).sum()))

    r["human"] = {}
    for s in ("Jev", "Qwen", "Gemma"):
        h = stats_of(D[ds][s]["human"][0], ids); rw = r["main"][s]
        r["human"][s] = dict(acc=h["acc"], f1=h["f1"], ece=h.get("ece"), d_acc=h["acc"] - rw["acc"], d_f1=h["f1"] - rw["f1"],
                             mcnemar=mcnemar_p(correct(D[ds][s]["human"][0], ids), correct(runs0[s], ids)))

    seeds = sorted(set(D[ds]["Jev"]["raw"]) & set(D[ds]["Qwen"]["raw"]) & set(D[ds]["Gemma"]["raw"]))
    oids = sorted(set.intersection(*[set(D[ds][s]["raw"][k]) for s in ("Jev", "Qwen", "Gemma") for k in seeds]))
    og = vec(D[ds]["Jev"]["raw"][0], oids, "gold")
    ordr = {"seeds": seeds, "n": len(oids), "sys": {}}
    for s in ("Jev", "Qwen", "Gemma"):
        A, F, E_, P = [], [], [], []
        for k in seeds:
            run = D[ds][s]["raw"][k]; c = correct(run, oids)
            A.append(float(c.mean())); F.append(macro_f1(og, vec(run, oids))); E_.append(ece(conf(run, oids), c)); P.append(vec(run, oids))
        agree = np.array([len({p[j] for p in P}) == 1 for j in range(len(oids))]); c0 = np.array([P[0][j] == og[j] for j in range(len(oids))])
        vote = [Counter(p[j] for p in P).most_common(1)[0][0] for j in range(len(oids))]
        sd = lambda v: float(np.std(v, ddof=1))
        ordr["sys"][s] = dict(acc=A, f1=F, ece=E_, acc_mean=float(np.mean(A)), acc_sd=sd(A), acc_range=float(max(A) - min(A)), f1_mean=float(np.mean(F)), f1_sd=sd(F),
                              ece_mean=float(np.mean(E_)), agree=float(agree.mean()), acc_agree=float(c0[agree].mean()), acc_disagree=float(c0[~agree].mean()),
                              vote=float(np.mean([v == g for v, g in zip(vote, og)])))
    ordr["jev_wins_acc"] = {s: int(sum(a > b for a, b in zip(ordr["sys"]["Jev"]["acc"], ordr["sys"][s]["acc"]))) for s in ("Qwen", "Gemma")}
    ordr["jev_wins_f1"] = {s: int(sum(a > b for a, b in zip(ordr["sys"]["Jev"]["f1"], ordr["sys"][s]["f1"]))) for s in ("Qwen", "Gemma")}
    r["orders"] = ordr
    jall = D[ds]["Jev"]["raw"]; ks = sorted(jall)
    A = [float(correct(jall[k], ids).mean()) for k in ks]; F = [macro_f1(gold, vec(jall[k], ids)) for k in ks]; Ec = [ece(conf(jall[k], ids), correct(jall[k], ids)) for k in ks]
    P = [vec(jall[k], ids) for k in ks]
    vote = [Counter(p[j] for p in P).most_common(1)[0][0] for j in range(len(ids))]
    agree_frac = np.array([Counter(p[j] for p in P).most_common(1)[0][1] / len(ks) for j in range(len(ids))])
    vc = np.array([v == g for v, g in zip(vote, gold)])
    r["jev_all_orders"] = dict(seeds=ks, acc=A, f1=F, ece=Ec, acc_mean=float(np.mean(A)), acc_sd=float(np.std(A, ddof=1)), acc_min=min(A), acc_max=max(A), f1_mean=float(np.mean(F)),
                               f1_sd=float(np.std(F, ddof=1)), ece_mean=float(np.mean(Ec)), vote_acc=float(vc.mean()), vote_f1=macro_f1(gold, vote), s0_rank_worst=int(np.argsort(A).tolist().index(0)) + 1,
                               all_agree=float((agree_frac == 1).mean()), vote_acc_agree=float(vc[agree_frac == 1].mean()), vote_acc_disagree=float(vc[agree_frac < 1].mean()))
    mp = np.mean([[jall[k][i]["p_top"] for i in ids] for k in ks], axis=0)
    r["jev_auroc"] = dict(p_top_s0=auroc(conf(jall[0], ids), correct(jall[0], ids)), agreement_vote=auroc(agree_frac + 1e-3 * mp, vc), mean_ptop_vote=auroc(mp, vc),
                          ptop_vote=auroc(mp, vc))
    bins = {}
    for v in sorted(set(np.round(agree_frac, 3))):
        m = np.round(agree_frac, 3) == v; bins[f"{v:.3f}"] = dict(n=int(m.sum()), acc=float(vc[m].mean()))
    r["agreement_bins"] = bins

    r["cascade"] = {"shares": SHARES, "ptop": [], "agree": {}}
    glm_c = correct(runs0["GLM"], ids)
    for k in ks:
        c = correct(jall[k], ids); cf = conf(jall[k], ids); order = np.argsort(cf, kind="stable"); row = []
        for sh in SHARES:
            n = int(round(sh * len(ids))); m = c.copy(); m[order[:n]] = glm_c[order[:n]]; row.append(float(m.mean()))
        r["cascade"]["ptop"].append(row)
    score = agree_frac + 1e-3 * mp; order = np.argsort(score, kind="stable"); row = []
    for sh in SHARES:
        n = int(round(sh * len(ids))); m = vc.copy(); m[order[:n]] = glm_c[order[:n]]; row.append(float(m.mean()))
    r["cascade"]["agree"] = dict(acc=row, vote_alone=float(vc.mean()))
    pt = np.array(r["cascade"]["ptop"]); r["cascade"]["ptop_mean"] = pt.mean(0).tolist(); r["cascade"]["ptop_sd"] = pt.std(0, ddof=1).tolist()
    r["cascade"]["glm_alone"] = float(glm_c.mean()); r["cascade"]["jev_alone_mean"] = float(pt[:, 0].mean())
    gain = (pt - pt[:, [0]]) / (glm_c.mean() - pt[:, [0]]); r["cascade"]["gain_recovered_mean"] = gain.mean(0).tolist()

    r["cascade_slm"] = {}
    for s_ in ("Jev", "Qwen", "Gemma"):
        c_ = correct(runs0[s_], ids); cf_ = conf(runs0[s_], ids); o_ = np.argsort(cf_, kind="stable"); row_ = []
        for sh in SHARES:
            n = int(round(sh * len(ids))); m = c_.copy(); m[o_[:n]] = glm_c[o_[:n]]; row_.append(float(m.mean()))
        r["cascade_slm"][s_] = row_
    names = SYS
    preds = {s: vec(runs0[s], ids) for s in names}
    cor = {s: correct(runs0[s], ids) for s in names}
    same_wrong = [j for j in range(len(ids)) if len({preds[s][j] for s in names}) == 1 and preds["Jev"][j] != gold[j]]
    keep = np.ones(len(ids), bool); keep[same_wrong] = False
    solved = np.sum([cor[s] for s in names], axis=0)
    pairs = Counter((gold[j], preds["Jev"][j]) for j in same_wrong).most_common(8)
    r["consensus"] = dict(n_same_wrong=len(same_wrong), frac_same_wrong=len(same_wrong) / len(ids), n_all_wrong=int((solved == 0).sum()), frac_all_wrong=float((solved == 0).mean()),
                          oracle=float((solved > 0).mean()), solved_by=[int((solved == k).sum()) for k in range(len(names) + 1)],
                          adj_acc={s: float(cor[s][keep].mean()) for s in names}, top_pairs=[[a, b, n] for (a, b), n in pairs],
                          examples=[[gold[j], preds["Jev"][j], runs0["Jev"][ids[j]]["text"]] for j in same_wrong[:10]])

    r["per_label"] = {s: per_label(gold, preds[s]) for s in names}
    if ds == "bank":
        acc_lab = {l: {s: float(np.mean([cor[s][j] for j in range(len(ids)) if gold[j] == l])) for s in names} for l in sorted(set(gold))}
        worst = sorted(acc_lab, key=lambda l: np.mean(list(acc_lab[l].values())))[:8]
        r["worst_labels"] = [[l, cls[l]] + [acc_lab[l][s] for s in names] for l in worst]
    words = np.array([len(runs0["Jev"][i]["text"].split()) for i in ids]); q = np.percentile(words, [25, 50, 75])
    r["length"] = dict(q=q.tolist(), acc={s: [float(cor[s][m].mean()) for m in ((words <= q[0]), (words > q[0]) & (words <= q[1]), (words > q[1]) & (words <= q[2]), (words > q[2]))] for s in names})

    r["eff"] = {}
    for s in names:
        recs = [runs0[s][i] for i in ids]
        r["eff"][s] = dict(in_tok=float(np.mean([x.get("in_tokens") or 0 for x in recs])), lat_median=float(np.median([x["latency_s"] for x in recs])), lat_mean=float(np.mean([x["latency_s"] for x in recs])),
                           out_tok=float(np.mean([x.get("out_tokens") or 0 for x in recs])) if s == "GLM" else None)
    r["eff"]["Jev"]["usd_per_1k"] = r["eff"]["Jev"]["in_tok"] * PRICE_PER_MTOK / 1e6 * 1000
    for s in ("Qwen", "Gemma"): r["eff"][s]["sec_per_1k"] = r["eff"][s]["lat_mean"] * 1000
    r["eff"]["GLM"]["sec_per_1k_parallel4"] = r["eff"]["GLM"]["lat_mean"] * 1000 / 4

    R[ds] = r

E_ = ROOT / "extra/predictions"
ids = sorted(D["atis"]["Jev"]["raw"][0]); gold = vec(D["atis"]["Jev"]["raw"][0], ids, "gold")
desc, noul = ld(E_ / "atis__jev__desc.jsonl"), ld(E_ / "atis__jev__noul.jsonl")
R["atis_extra"] = {}
for nm, run in (("names", D["atis"]["Jev"]["raw"][0]), ("desc", desc), ("noul", noul)):
    c = correct(run, ids); R["atis_extra"][nm] = dict(acc=float(c.mean()), f1=macro_f1(gold, vec(run, ids)), ece=ece(conf(run, ids), c),
                                                     pl=per_label(gold, vec(run, ids)), mcnemar_vs_names=None if nm == "names" else mcnemar_p(c, correct(D["atis"]["Jev"]["raw"][0], ids)))
nm = np.array([[noul[i]["probs"][l] for l in sorted(set(gold))] for i in ids])
R["atis_extra"]["noul_sum_mean"] = float(nm.sum(1).mean()); R["atis_extra"]["noul_multi"] = float(((nm > .5).sum(1) > 1).mean()); R["atis_extra"]["noul_none"] = float((nm.max(1) < .5).mean())
R["repeat"] = {}
for ds, key in (("bank", "banking"), ("atis", "atis")):
    runs = [ld(E_ / f"{key}__jev__det{i}.jsonl") for i in (1, 2, 3)]; rid = sorted(set.intersection(*[set(x) for x in runs]))
    R["repeat"][ds] = dict(n=len(rid), identical=float(np.mean([len({x[i]["pred"] for x in runs}) == 1 for i in rid])),
                           dp_mean=float(np.mean([max(x[i]["p_top"] for x in runs) - min(x[i]["p_top"] for x in runs) for i in rid])),
                           dp_max=float(np.max([max(x[i]["p_top"] for x in runs) - min(x[i]["p_top"] for x in runs) for i in rid])))
json.dump(R, open(Path(__file__).resolve().parent / "results.json", "w"), indent=1, default=float)
print("saved results.json")
for ds in DS:
    m = R[ds]["main"]
    print(ds, {s: (round(m[s]["acc"], 4), round(m[s]["f1"], 4), round(m[s].get("ece", float("nan")), 3)) for s in SYS})
    print("  orders", {s: (round(R[ds]["orders"]["sys"][s]["acc_mean"], 4), round(R[ds]["orders"]["sys"][s]["acc_sd"], 4)) for s in ("Jev", "Qwen", "Gemma")}, "jev all", round(R[ds]["jev_all_orders"]["acc_mean"], 4))
    print("  cascade ptop mean", [round(x, 4) for x in R[ds]["cascade"]["ptop_mean"]], "agree", [round(x, 4) for x in R[ds]["cascade"]["agree"]["acc"]], "glm", round(R[ds]["cascade"]["glm_alone"], 4))
    print("  auroc", {k: round(v, 3) for k, v in R[ds]["jev_auroc"].items()}, "s0 worst rank", R[ds]["jev_all_orders"]["s0_rank_worst"])
