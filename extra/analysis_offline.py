import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / "JEVvsLLMs"
PATHS = {
    "Banking77": {"Jev": J / "jev_predictions/jev-1.13.0__raw__s0.jsonl", "Qwen": J / "predictions/qwen3.5-4b__raw__s0.jsonl",
                  "Gemma": J / "predictions/gemma-3-4b-it__raw__s0.jsonl", "GLM": ROOT / "cloud/predictions/banking__glm-5.3-cloud__raw.jsonl"},
    "ATIS": {"Jev": J / "jev_predictions_atis/jev-1.13.0__raw__s0.jsonl", "Qwen": J / "predictions_atis/qwen3.5-4b__raw__s0.jsonl",
             "Gemma": J / "predictions_atis/gemma-3-4b-it__raw__s0.jsonl", "GLM": ROOT / "cloud/predictions/atis__glm-5.3-cloud__raw.jsonl"},
}
rng = np.random.default_rng(0)
out = ["# Offline analyses (raw labels)\n"]
P = lambda s="": out.append(s)


def auroc(score, positive):
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="stable"); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):
        m = score == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


for ds, paths in PATHS.items():
    data = {n: {r["id"]: r for r in map(json.loads, open(p))} for n, p in paths.items() if p.exists()}
    ids = sorted(set.intersection(*[set(d) for d in data.values()]))
    names = list(data)
    gold = {i: data["Jev"][i]["gold"] for i in ids}
    corr = {n: np.array([data[n][i]["pred"] == gold[i] for i in ids]) for n in names}
    P(f"\n## {ds}: {len(ids)} rows, systems: {', '.join(names)}\n")

    same_wrong = [i for i in ids if len({data[n][i]["pred"] for n in names}) == 1 and data["Jev"][i]["pred"] != gold[i]]
    all_wrong = sum(all(not corr[n][k] for n in names) for k in range(len(ids)))
    keep = np.array([i not in set(same_wrong) for i in ids])
    P(f"### A. Consensus disagreement with the gold label (possible label noise / ambiguity)\n")
    P(f"- All {len(names)} systems predict the SAME label and it differs from gold: **{len(same_wrong)}** rows ({len(same_wrong)/len(ids):.1%}).")
    P(f"- All systems wrong (labels may differ): {all_wrong} rows ({all_wrong/len(ids):.1%}).")
    P("- Accuracy if the consensus-disagreement rows are excluded: " + ", ".join(f"{n} {corr[n][keep].mean():.3f} (was {corr[n].mean():.3f})" for n in names))
    P("- Sample of consensus-disagreement rows for a manual audit (gold -> consensus prediction):\n")
    for i in same_wrong[:12]:
        P(f"  - `{gold[i]}` -> `{data['Jev'][i]['pred']}` : {data['Jev'][i]['text'][:110]}")

    P("\n### B. Calibration direction (bins of the top-label probability; + = overconfident)\n")
    P("| system | mean confidence | accuracy | gap | AUROC(conf -> correct) | " + " | ".join(f"[{a:.1f},{b:.1f})" for a, b in zip(np.linspace(0, .9, 10), np.linspace(.1, 1, 10))) + " |")
    P("|---|---|---|---|---|" + "---|" * 10)
    for n in names:
        conf = np.array([data[n][i]["p_top"] if data[n][i].get("p_top") is not None else np.nan for i in ids])
        if np.isnan(conf).any():
            P(f"| {n} | n/a | {corr[n].mean():.3f} | n/a | n/a |" + " |" * 10); continue
        cells = []
        for a, b in zip(np.linspace(0, .9, 10), np.linspace(.1, 1.0001, 10)):
            m = (conf >= a) & (conf < b)
            cells.append(f"{int(m.sum())}: {corr[n][m].mean():.2f}" if m.any() else "-")
        P(f"| {n} | {conf.mean():.3f} | {corr[n].mean():.3f} | {conf.mean() - corr[n].mean():+.3f} | {auroc(conf, corr[n]):.3f} | " + " | ".join(cells) + " |")
    P("\n(bin cells = rows in bin: accuracy in bin)")
    if "conf" in data["Jev"][ids[0]] and data["Jev"][ids[0]]["conf"] is not None:
        jc = np.array([data["Jev"][i]["conf"] for i in ids]); jp = np.array([data["Jev"][i]["p_top"] for i in ids])
        P(f"\nJev's own `confidence` field vs top probability as an error detector: AUROC {auroc(jc, corr['Jev']):.3f} vs {auroc(jp, corr['Jev']):.3f}.")

    P("\n### C. Paired bootstrap of the accuracy difference (Jev minus system), 95% CI\n")
    P("| system | difference | 95% CI |\n|---|---|---|")
    for n in names:
        if n == "Jev": continue
        d = corr["Jev"].astype(float) - corr[n].astype(float)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        P(f"| {n} | {d.mean():+.3f} | [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] |")

    solved = np.sum([corr[n] for n in names], axis=0)
    P("\n### D. Row difficulty: how many of the systems get the row right\n")
    P("| systems correct | " + " | ".join(str(k) for k in range(len(names) + 1)) + " |\n|---|" + "---|" * (len(names) + 1))
    P("| rows | " + " | ".join(f"{int((solved == k).sum())} ({(solved == k).mean():.1%})" for k in range(len(names) + 1)) + " |")
    P(f"\nOracle (at least one system right): {(solved > 0).mean():.3f}. Majority vote of systems right (>= half): {(solved >= len(names)/2).mean():.3f}.")

    words = np.array([len(data['Jev'][i]['text'].split()) for i in ids]); qs = np.percentile(words, [25, 50, 75])
    bins = [(0, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e9)]
    P("\n### E. Accuracy by utterance length (words, quartiles)\n")
    P("| length | rows | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names))
    for lo, hi in bins:
        m = (words > lo) & (words <= hi) if lo else (words <= hi)
        P(f"| {'<=' + str(int(hi)) if lo == 0 else '(' + str(int(lo)) + ',' + (str(int(hi)) if hi < 1e8 else 'max') + ']'} | {int(m.sum())} | " + " | ".join(f"{corr[n][m].mean():.3f}" for n in names) + " |")

    if "top5" in data["Jev"][ids[0]]:
        P("\n### F. Jev top-k accuracy\n")
        P("| k | accuracy |\n|---|---|")
        for k in (1, 2, 3, 5):
            P(f"| {k} | {np.mean([gold[i] in [l for l, _ in data['Jev'][i]['top5'][:k]] for i in ids]):.3f} |")

    by = defaultdict(lambda: defaultdict(list))
    for k, i in enumerate(ids):
        for n in names: by[gold[i]][n].append(corr[n][k])
    rows = sorted(by, key=lambda l: np.mean(by[l]["Jev"]))
    P("\n### G. Labels where Jev is weakest (per-label accuracy)\n")
    P("| label | rows | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names))
    for l in rows[:10]:
        P(f"| {l} | {len(by[l]['Jev'])} | " + " | ".join(f"{np.mean(by[l][n]):.2f}" for n in names) + " |")
    if ds == "Banking77":
        texts = defaultdict(set)
        for i in ids: texts[data["Jev"][i]["text"]].add(gold[i])
        P(f"\nData check: {sum(len(v) > 1 for v in texts.values())} identical texts carry more than one gold label; {len(ids) - len(texts)} rows repeat an earlier text.")

text = "\n".join(out) + "\n"
(Path(__file__).resolve().parent / "analysis_offline.md").write_text(text)
print(text)
