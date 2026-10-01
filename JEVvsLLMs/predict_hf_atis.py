import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import argparse, csv, json, math, random, time
from pathlib import Path
import torch
from predict_hf import MODELS, LOAD_4BIT, load, encode, collate, build_trie, TrieProcessor

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ATIS_DATA") or HERE / "data/atis_all.csv")
if not DATA.exists():
    DATA = HERE / "data/atis_all.csv"
TASK = "Which intent does this request to an airline travel information system express?"


def show(label, variant):
    if variant == "human":
        return label.removeprefix("atis_").replace("_", " ")
    return label


def build_prompt(options, text):
    system = ("You are an intent classifier for an airline travel information system. "
              "Answer with exactly one intent from the list and nothing else.")
    user = ("Intents:\n" + "\n".join(f"- {o}" for o in options) +
            f"\n\nQuestion: {TASK}\n\nMessage: {text}")
    return system, user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--variant", choices=["raw", "human"], default="raw")
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    labels = sorted({r["label"] for r in rows})
    if a.shuffle:
        random.Random(a.shuffle).shuffle(labels)
    if a.limit:
        random.Random(0).shuffle(rows)
        rows = rows[: a.limit]
    options = [show(l, a.variant) for l in labels]
    back = dict(zip(options, labels))
    assert len(back) == len(labels), "label names collide after humanising"

    hf_name, tag = MODELS[a.model]
    out = Path(a.out or HERE / f"predictions_atis/{tag}__{a.variant}__s{a.shuffle}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{hf_name} [{a.variant}] ATIS: {len(todo)} to do ({len(done)} done) -> {out}", flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load(a.model)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    eos_ids = {tok.eos_token_id}
    ge = getattr(model.generation_config, "eos_token_id", None)
    eos_ids |= set(ge if isinstance(ge, (list, tuple)) else [ge]) - {None}
    for special in ("<end_of_turn>", "<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(special)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            eos_ids.add(i)
    eos_ids = sorted(x for x in eos_ids if x is not None)
    option_tokens = {o: tok(o, add_special_tokens=False)["input_ids"] for o in options}
    root, max_len = build_trie(option_tokens), max(len(t) for t in option_tokens.values())
    pref = [(a_, b_) for a_ in options for b_ in options if a_ != b_ and option_tokens[b_][:len(option_tokens[a_])] == option_tokens[a_]]
    print(f"eos ids: {eos_ids}; longest option = {max_len} tokens; options that are a token-prefix of another: {pref}", flush=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]

    n_ok = n_inv = 0
    B = max(1, a.batch_size)
    with open(out, "a") as f:
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            encs = [encode(a.model, tokenizer, *build_prompt(options, r["text"]), device) for r in batch]
            inputs = collate(encs, pad_id, device)
            pl = inputs["input_ids"].shape[1]
            proc = TrieProcessor(root, eos_ids, pl)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                g = model.generate(**inputs, max_new_tokens=max_len + 1, do_sample=False, temperature=None, top_p=None,
                                   top_k=None, logits_processor=[proc], output_scores=True, return_dict_in_generate=True,
                                   eos_token_id=eos_ids, pad_token_id=pad_id)
            if device == "cuda":
                torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) / len(batch)
            for j, r in enumerate(batch):
                seq, node, lp, label = g.sequences[j, pl:].tolist(), root, 0.0, None
                for s, t in enumerate(seq):
                    lp += g.scores[s][j, t].item()
                    if t in eos_ids:
                        label = node.label if node is not None else None
                        break
                    node = node.kids.get(t) if node is not None else None
                pred = back[label] if label else ""
                n_inv += pred == ""
                pv = math.exp(lp)
                f.write(json.dumps({"id": r["id"], "split": r["split"], "dup": int(r["dup"]), "text": r["text"], "gold": r["label"],
                                    "pred": pred, "valid": pred != "", "p_top": pv, "conf": pv, "latency_s": lat, "batch_size": len(batch),
                                    "in_tokens": int(encs[j]["input_ids"].shape[1]), "model": hf_name, "variant": a.variant,
                                    "shuffle": a.shuffle, "quant": "4bit" if LOAD_4BIT else "bf16"}) + "\n")
                n_ok += 1
            f.flush()
            if (n_ok // B) % max(1, 200 // B) == 0 or n_ok == len(batch):
                print(f"{n_ok}/{len(todo)}  invalid={n_inv}  last={batch[-1]['label']}->{pred}  p={pv:.3f}  {lat:.2f}s/utt", flush=True)
    print(f"done ok={n_ok} invalid={n_inv}", flush=True)

if __name__ == "__main__":
    main()
