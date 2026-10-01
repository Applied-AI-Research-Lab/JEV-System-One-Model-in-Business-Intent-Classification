import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import argparse, csv, json, math, random, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("B77_DATA") or HERE / "data/banking77_all.csv")
if not DATA.exists():
    DATA = HERE / "data/banking77_all.csv"
TASK = "Which intent does this customer message to a banking app's support chat express?"
MODELS = {"qwen": ("unsloth/Qwen3.5-4B", "qwen3.5-4b"), "gemma": ("unsloth/gemma-3-4b-it", "gemma-3-4b-it")}
MAX_SEQ, LOAD_4BIT = 2048, True


def show(label, variant):
    return label.replace("_", " ") if variant == "human" else label


def build_prompt(options, text):
    system = ("You are an intent classifier for a banking app's customer support. "
              "Answer with exactly one intent from the list and nothing else.")
    user = ("Intents:\n" + "\n".join(f"- {o}" for o in options) +
            f"\n\nQuestion: {TASK}\n\nMessage: {text}")
    return system, user


def apply_chat_template_compat(tokenizer, messages, enable_thinking=False, **kwargs):
    call_kwargs = dict(enable_thinking=enable_thinking, **kwargs)
    try:
        return tokenizer.apply_chat_template(messages, **call_kwargs)
    except TypeError as exc:
        err = str(exc)
        if "string indices must be integers" not in err and "enable_thinking" not in err:
            raise
        if "enable_thinking" in err:
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError as exc2:
                if "string indices must be integers" not in str(exc2):
                    raise
        block = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]
                  if isinstance(m.get("content"), str) else m.get("content")} for m in messages]
        try:
            return tokenizer.apply_chat_template(block, **call_kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(block, **kwargs)


def load(which):
    name = MODELS[which][0]
    if which == "qwen":
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=MAX_SEQ, load_in_4bit=LOAD_4BIT)
        FastLanguageModel.for_inference(model)
    else:
        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained(model_name=name, max_seq_length=MAX_SEQ, load_in_4bit=LOAD_4BIT,
                                                     load_in_8bit=False, full_finetuning=False)
        FastModel.for_inference(model)
    return model, tokenizer


def encode(which, tokenizer, system, user, device):
    if which == "qwen":
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = apply_chat_template_compat(tokenizer, msgs, enable_thinking=False, add_generation_prompt=True,
                                            tokenize=True, return_tensors="pt", return_dict=True)
    else:
        msgs = [{"role": "user", "content": [{"type": "text", "text": system + "\n\n" + user}]}]
        inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                               return_tensors="pt", return_dict=True)
    return {k: v for k, v in inputs.items() if getattr(v, "dim", lambda: 0)() == 2}


def collate(items, pad_id, device):
    L = max(i["input_ids"].shape[1] for i in items)
    out = {}
    for k in items[0]:
        fill = pad_id if k == "input_ids" else 0
        rows = []
        for i in items:
            t = i[k]
            n = L - t.shape[1]
            rows.append(torch.cat([torch.full((1, n), fill, dtype=t.dtype), t], dim=1) if n else t)
        out[k] = torch.cat(rows, dim=0).to(device)
    return out


class Node:
    __slots__ = ("kids", "label")
    def __init__(self): self.kids, self.label = {}, None


def build_trie(option_tokens):
    root = Node()
    for label, toks in option_tokens.items():
        n = root
        for t in toks:
            n = n.kids.setdefault(t, Node())
        n.label = label
    return root


class TrieProcessor:
    def __init__(self, root, eos_ids, prompt_len):
        self.root, self.eos, self.pl = root, list(eos_ids), prompt_len
    def __call__(self, input_ids, scores):
        out = torch.full_like(scores, float("-inf"))
        for r in range(input_ids.shape[0]):
            node = self.root
            for t in input_ids[r, self.pl:].tolist():
                node = node.kids.get(t) if node is not None else None
            allowed = self.eos if node is None else list(node.kids) + (self.eos if node.label is not None else [])
            idx = torch.tensor(allowed, device=scores.device)
            out[r, idx] = scores[r, idx]
        return torch.log_softmax(out.float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--variant", choices=["raw", "human"], default="raw")
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", choices=["all", "train", "test"], default="all")
    ap.add_argument("--batch-size", type=int, default=8, help="utterances per generate() call (left-padded); 1 = one at a time")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    labels = sorted({r["label"] for r in rows})
    if a.shuffle:
        random.Random(a.shuffle).shuffle(labels)
    if a.split != "all":
        rows = [r for r in rows if r["split"] == a.split]
    if a.limit:
        random.Random(0).shuffle(rows)
        rows = rows[: a.limit]
    options = [show(l, a.variant) for l in labels]
    back = dict(zip(options, labels))

    hf_name, tag = MODELS[a.model]
    out = Path(a.out or HERE / f"predictions/{tag}__{a.variant}__s{a.shuffle}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{hf_name} [{a.variant}]: {len(todo)} to do ({len(done)} done) -> {out}", flush=True)
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
    print(f"eos ids: {eos_ids}; longest option = {max_len} tokens", flush=True)
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
                f.write(json.dumps({"id": r["id"], "split": r["split"], "text": r["text"], "gold": r["label"], "pred": pred,
                                    "valid": pred != "", "p_top": pv, "conf": pv, "latency_s": lat, "batch_size": len(batch),
                                    "in_tokens": int(encs[j]["input_ids"].shape[1]), "model": hf_name, "variant": a.variant,
                                    "shuffle": a.shuffle, "quant": "4bit" if LOAD_4BIT else "bf16"}) + "\n")
                n_ok += 1
            f.flush()
            if (n_ok // B) % max(1, 200 // B) == 0 or n_ok == len(batch):
                print(f"{n_ok}/{len(todo)}  invalid={n_inv}  last={batch[-1]['label']}->{pred}  p={pv:.3f}  {lat:.2f}s/utt", flush=True)
    print(f"done ok={n_ok} invalid={n_inv}", flush=True)

if __name__ == "__main__":
    main()
