import argparse, asyncio, csv, json, os, random, time
from pathlib import Path
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Datasets/ATIS Airline Travel Information System/atis_all.csv"
TASK = "Which intent does this request to an airline travel information system express?"
DESC = {
    "atis_flight": "The user wants a list of flights or flight options between places (by date, time, airline, stops or class). Not a question about the fare, the time of a specific flight, or other attributes.",
    "atis_airfare": "The user asks about the price or cost of a flight or ticket, fares, or the cheapest fare.",
    "atis_ground_service": "The user asks about ground transportation such as taxis, limousines, car rental, buses, trains, or how to get to or from the airport.",
    "atis_airline": "The user asks which airline(s) operate a route or serve a city, or asks about airline names.",
    "atis_abbreviation": "The user asks what an abbreviation, code or term means, for example a fare code, airline code, airport code or meal code.",
    "atis_aircraft": "The user asks what type of aircraft is used on a flight, or which aircraft types exist.",
    "atis_flight_time": "The user asks for the departure or arrival time of a specific flight, or how long a flight takes.",
    "atis_quantity": "The user asks how many of something there are (a count of flights, seats, airports, and so on).",
}


def load_key():
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    for line in (ROOT / "api-keys.txt").read_text().splitlines():
        if line.startswith(("jev=", "TYPESAFE_API_KEY=")):
            os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip(); return
    raise SystemExit("no key")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["desc", "noul"], required=True)
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); load_key()
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    labels = sorted({r["label"] for r in rows})
    if a.limit:
        random.Random(0).shuffle(rows); rows = rows[: a.limit]
    if a.mode == "desc":
        questions = {"intent": Choice(instructions=TASK, criteria={l: DESC[l] for l in labels})}
    else:
        questions = {l: Noul(instructions=f"Does this request to an airline travel information system express the intent '{l.removeprefix('atis_').replace('_', ' ')}'?") for l in labels}
    out = Path(ROOT / "extra/predictions" / Path(a.out or f"atis__jev__{a.mode}.jsonl").name)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{a.mode}: {len(todo)} to do -> {out}", flush=True)
    sem, lock, n_ok, n_err = asyncio.Semaphore(a.concurrency), asyncio.Lock(), 0, 0
    with open(out, "a") as f:
        async with AsyncTypeSafeClient(model=a.model, timeout=120.0) as client:
            async def one(r):
                nonlocal n_ok, n_err
                async with sem:
                    t = time.perf_counter()
                    try:
                        resp = await client.system_one(state=r["text"], questions=questions)
                    except Exception as e:
                        n_err += 1; print("error", r["id"], type(e).__name__, str(e)[:100], flush=True); return
                    lat = time.perf_counter() - t
                if a.mode == "desc":
                    ans = resp.answers["intent"]; probs = dict(ans.probabilities); pred = ans.choice; conf = ans.confidence
                else:
                    probs = {l: resp.answers[l].noul for l in labels}; pred = max(probs, key=probs.get); conf = None
                rec = {"id": r["id"], "split": r["split"], "dup": int(r["dup"]), "text": r["text"], "gold": r["label"], "pred": pred, "valid": True,
                       "p_top": probs[pred], "conf": conf, "probs": probs, "latency_s": lat, "in_tokens": resp.usage.input_tokens,
                       "model": resp.model, "mode": a.mode}
                async with lock:
                    f.write(json.dumps(rec) + "\n"); f.flush(); n_ok += 1
            await asyncio.gather(*(one(r) for r in todo))
    print(f"done ok={n_ok} errors={n_err}", flush=True)

asyncio.run(main())
