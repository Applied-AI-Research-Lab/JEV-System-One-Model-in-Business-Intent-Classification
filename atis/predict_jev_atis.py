import argparse, asyncio, csv, json, os, random, time
from pathlib import Path
from typesafe_sdk import AsyncTypeSafeClient, Choice

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Datasets/ATIS Airline Travel Information System/atis_all.csv"
TASK = "Which intent does this request to an airline travel information system express?"


def load_key():
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    for line in (ROOT / "api-keys.txt").read_text().splitlines():
        if line.startswith(("jev=", "TYPESAFE_API_KEY=")):
            os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip()
            return
    raise SystemExit("No TypeSafe key: set TYPESAFE_API_KEY or add it to api-keys.txt")


def show(label, variant):
    if variant == "human":
        label = label.removeprefix("atis_")
        return label.replace("_", " ")
    return label


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--variant", choices=["raw", "human"], default="raw")
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    load_key()

    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    labels = sorted({r["label"] for r in rows})
    if a.shuffle:
        random.Random(a.shuffle).shuffle(labels)
    if a.limit:
        random.Random(0).shuffle(rows)
        rows = rows[: a.limit]
    shown = {show(l, a.variant): l for l in labels}
    assert len(shown) == len(labels), "label names collide after humanising"
    question = {"intent": Choice(instructions=TASK, criteria={show(l, a.variant): None for l in labels})}

    out = Path(a.out or ROOT / f"atis/predictions/{a.model}__{a.variant}__s{a.shuffle}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} to do ({len(done)} already done) -> {out}", flush=True)

    sem, lock, n_ok, n_err = asyncio.Semaphore(a.concurrency), asyncio.Lock(), 0, 0
    with open(out, "a") as f:
        async with AsyncTypeSafeClient(model=a.model, timeout=120.0) as client:
            async def one(r):
                nonlocal n_ok, n_err
                async with sem:
                    t = time.perf_counter()
                    try:
                        resp = await client.system_one(state=r["text"], questions=question)
                    except Exception as e:
                        n_err += 1
                        print("error", r["id"], type(e).__name__, str(e)[:120], flush=True)
                        return
                    lat = time.perf_counter() - t
                ans = resp.answers["intent"]
                probs = {shown[k]: v for k, v in ans.probabilities.items()}
                rec = {"id": r["id"], "split": r["split"], "dup": int(r["dup"]), "text": r["text"], "gold": r["label"],
                       "pred": shown[ans.choice], "valid": True, "p_top": probs[shown[ans.choice]], "conf": ans.confidence,
                       "top5": sorted(probs.items(), key=lambda kv: -kv[1])[:5], "latency_s": lat,
                       "in_tokens": resp.usage.input_tokens, "out_tokens": resp.usage.output_tokens,
                       "model": resp.model, "variant": a.variant, "shuffle": a.shuffle}
                async with lock:
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    n_ok += 1
                    if n_ok % 500 == 0:
                        print(f"{n_ok}/{len(todo)}", flush=True)
            await asyncio.gather(*(one(r) for r in todo))
    print(f"done ok={n_ok} errors={n_err} (rerun to retry errors)", flush=True)

asyncio.run(main())
