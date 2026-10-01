import argparse, asyncio, csv, json, random, re, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "predictions"
DATASETS = {
    "banking": dict(
        path=ROOT / "Datasets/banking77/banking77_all.csv",
        task="Which intent does this customer message to a banking app's support chat express?",
        who="a banking app's customer support",
        show=lambda l, v: l.replace("_", " ") if v == "human" else l),
    "atis": dict(
        path=ROOT / "Datasets/ATIS Airline Travel Information System/atis_all.csv",
        task="Which intent does this request to an airline travel information system express?",
        who="an airline travel information system",
        show=lambda l, v: l.removeprefix("atis_").replace("_", " ") if v == "human" else l),
}


def parse_label(content, options):
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip()).strip()
    lower = {o.lower(): o for o in options}
    try:
        v = json.loads(txt)
        v = v.get("intent") if isinstance(v, dict) else v
        if isinstance(v, str):
            v = v.strip()
            if v in options:
                return v, "json"
            if v.lower() in lower:
                return lower[v.lower()], "json_case"
    except Exception:
        pass
    hits = [o for o in options if o.lower() in txt.lower()]
    return (max(hits, key=len), "text") if hits else (None, "none")


def build_messages(cfg, options, text):
    system = (f"You are an intent classifier for {cfg['who']}. "
              "Answer with exactly one intent from the list, as JSON: {\"intent\": \"<option>\"}.")
    user = ("Intents:\n" + "\n".join(f"- {o}" for o in options) +
            f"\n\nQuestion: {cfg['task']}\n\nMessage: {text}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--model", default="glm-5.3:cloud")
    ap.add_argument("--variant", choices=["raw", "human"], default="raw")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", type=int, default=0, help="seed for the option order (0 = alphabetical); same seeds/orders as the Jev and SLM runs")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-consecutive-errors", type=int, default=30)
    ap.add_argument("--out", default=None, help="file inside cloud/predictions/ (default derived from dataset/model/variant)")
    a = ap.parse_args()
    cfg = DATASETS[a.dataset]

    rows = list(csv.DictReader(open(cfg["path"], encoding="utf-8")))
    labels = sorted({r["label"] for r in rows})
    if a.shuffle:
        random.Random(a.shuffle).shuffle(labels)
    if a.limit:
        random.Random(0).shuffle(rows)
        rows = rows[: a.limit]
    options = [cfg["show"](l, a.variant) for l in labels]
    back = dict(zip(options, labels))
    tag = re.sub(r"[^A-Za-z0-9.-]", "-", a.model)
    name = a.out or f"{a.dataset}__{tag}__{a.variant}" + (f"__s{a.shuffle}" if a.shuffle else "") + ".jsonl"
    out = (OUT_DIR / Path(name).name)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{a.model} on {a.dataset} [{a.variant}]: {len(todo)} to do ({len(done)} done) -> {out}", flush=True)

    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    st = {"ok": 0, "err": 0, "inv": 0, "consec": 0, "stop": False}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as http:
        with open(out, "a") as f:
            async def one(r):
                if st["stop"]:
                    return
                async with sem:
                    if st["stop"]:
                        return
                    t = time.perf_counter()
                    try:
                        resp = await http.post(f"{a.host}/api/chat", json={
                            "model": a.model, "stream": False, "messages": build_messages(cfg, options, r["text"]),
                            "options": {"temperature": 0, "seed": 0, "num_predict": 1024}})
                        if resp.status_code >= 400:
                            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")
                        j = resp.json()
                        if not (j.get("message") or {}).get("content", "").strip() and (j.get("eval_count") or 0) >= 1000:
                            resp = await http.post(f"{a.host}/api/chat", json={
                                "model": a.model, "stream": False, "messages": build_messages(cfg, options, r["text"]),
                                "options": {"temperature": 0, "seed": 0, "num_predict": 4096}})
                            if resp.status_code >= 400:
                                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")
                            j = resp.json()
                    except Exception as e:
                        st["err"] += 1; st["consec"] += 1
                        print("error", r["id"], type(e).__name__, str(e)[:150], flush=True)
                        if st["consec"] >= a.max_consecutive_errors:
                            st["stop"] = True
                            print(f"STOP: {st['consec']} consecutive errors (quota / rate limit / server down?). Rerun later to resume.", flush=True)
                        return
                    lat = time.perf_counter() - t
                st["consec"] = 0
                content = (j.get("message") or {}).get("content", "")
                lab, how = parse_label(content, back)
                pred = back[lab] if lab else ""
                rec = {"id": r["id"], "split": r["split"], "dup": int(r.get("dup", 0) or 0), "text": r["text"], "gold": r["label"],
                       "pred": pred, "valid": pred != "", "parse": how, "raw": content[:200], "p_top": None, "conf": None,
                       "latency_s": lat, "in_tokens": j.get("prompt_eval_count"), "out_tokens": j.get("eval_count"),
                       "model": a.model, "variant": a.variant, "dataset": a.dataset, "shuffle": a.shuffle}
                async with lock:
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    st["ok"] += 1; st["inv"] += pred == ""
                    if st["ok"] % 250 == 0:
                        print(f"{st['ok']}/{len(todo)}  invalid={st['inv']}  errors={st['err']}", flush=True)
            await asyncio.gather(*(one(r) for r in todo))
    print(f"done ok={st['ok']} invalid={st['inv']} errors={st['err']}" + ("  (STOPPED EARLY)" if st["stop"] else ""), flush=True)

asyncio.run(main())
