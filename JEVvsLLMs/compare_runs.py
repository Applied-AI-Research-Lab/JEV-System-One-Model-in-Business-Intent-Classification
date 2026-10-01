import json, sys
a = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
b = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
ids = sorted(set(a) & set(b))
same = sum(a[i]["pred"] == b[i]["pred"] for i in ids)
dp = [abs(a[i]["p_top"] - b[i]["p_top"]) for i in ids]
acc = lambda d: sum(d[i]["pred"] == d[i]["gold"] for i in ids) / len(ids)
print(f"common rows={len(ids)}  same prediction={same}/{len(ids)}  mean|dp|={sum(dp)/len(dp):.3f}  max|dp|={max(dp):.3f}  acc A={acc(a):.2f} B={acc(b):.2f}")
print("differences:", [(i, a[i]["pred"], b[i]["pred"]) for i in ids if a[i]["pred"] != b[i]["pred"]][:10])
