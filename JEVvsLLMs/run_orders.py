import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = {"banking": "predict_hf.py", "atis": "predict_hf_atis.py"}
TAG = {"qwen": "qwen3.5-4b", "gemma": "gemma-3-4b-it"}
OUT_ROOT = HERE / "predictions_orders"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(TAG), required=True)
    ap.add_argument("--datasets", nargs="+", choices=list(SCRIPT), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--banking-limit", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if 0 in a.seeds:
        sys.exit("seed 0 is the existing alphabetical run; use seeds >= 1")
    for ds in a.datasets:
        for seed in a.seeds:
            out = OUT_ROOT / ds / f"{TAG[a.model]}__raw__s{seed}.jsonl"
            assert OUT_ROOT in out.parents, "output must stay inside predictions_orders/"
            cmd = [sys.executable, str(HERE / SCRIPT[ds]), "--model", a.model, "--variant", "raw", "--shuffle", str(seed),
                   "--batch-size", str(a.batch_size), "--out", str(out)]
            if ds == "banking":
                cmd += ["--limit", str(a.banking_limit)]
            print("\n>>> " + " ".join(cmd), flush=True)
            if a.dry_run:
                continue
            rc = subprocess.call(cmd, cwd=HERE)
            if rc != 0:
                sys.exit(f"FAILED (exit {rc}) for {ds} seed {seed}; fix and re-run the same command to resume")
    print("\nALL DONE", flush=True)

if __name__ == "__main__":
    main()
