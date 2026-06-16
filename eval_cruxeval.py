"""Stage 1 baseline eval: CRUXEval-O output prediction via full-trace generation.

Feed the training prompt (seeds frame 0), let the SFT model generate the trace,
take main()'s last return value as the predicted output, score by execution.
Greedy => pass@1 is the exact-match fraction. Reuses cwm_andre eval logic.
"""

import argparse
import json
import os
import re
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.dataset import _prompt_str, cruxeval_split
from tokens import add_trace_tokens, token_ids

ARG_SEP, FRAME_SEP, RETURN_SEP = "<|arg_sep|>", "<|frame_sep|>", "<|return_sep|>"


def extract_answer_trace_full(gen: str) -> str | None:
    """Value of main()'s last RETURN frame: ...<|arg_sep|>"value"<|frame_sep|>."""
    r = gen.rfind(RETURN_SEP)
    if r == -1:
        return None
    a = gen.find(ARG_SEP, r)
    if a == -1:
        return None
    rest = gen[a + len(ARG_SEP):]
    end = rest.find(FRAME_SEP)
    val = (rest[:end] if end != -1 else rest).strip()
    if not val:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return val


def check_correct(code: str, expected: str, predicted: str, timeout: float = 3.0) -> bool:
    """Execute `code; assert expected == predicted` (CRUXEval semantics)."""
    test = f"{code}\nassert {expected} == {predicted}"
    try:
        return subprocess.run(
            [sys.executable, "-c", test], timeout=timeout, capture_output=True
        ).returncode == 0
    except Exception:
        return False


def load_rows(split: str):
    local_dir = os.environ.get("CRUXEVAL_DIR")
    if local_dir and os.path.isdir(local_dir):
        from datasets import load_from_disk
        rows = list(load_from_disk(local_dir))
    else:
        from datasets import load_dataset
        rows = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    return cruxeval_split(rows, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    add_trace_tokens(tok)  # idempotent; ensures trace tokens present
    eot_id = token_ids(tok)["<|end_of_text|>"]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()

    rows = load_rows(args.split)
    if args.n_samples > 0:
        rows = rows[: args.n_samples]

    n_correct = n_fmt = 0
    results = []
    for i, r in enumerate(rows):
        prompt = _prompt_str(r["code"], r["input"])
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)]).cuda()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                eos_token_id=eot_id, pad_token_id=tok.pad_token_id or eot_id,
            )
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        pred = extract_answer_trace_full(gen)
        ok = pred is not None and check_correct(r["code"], r["output"], pred)
        n_fmt += pred is not None
        n_correct += ok
        results.append({"id": r["id"], "expected": r["output"], "predicted": pred, "correct": ok})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)}  pass@1={n_correct / (i + 1):.4f}", flush=True)

    n = len(rows)
    print(f"\nCRUXEval-O [{args.split}] pass@1={n_correct / n:.4f}  "
          f"valid_format={n_fmt / n:.4f}  (n={n}, greedy)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"pass_at_1": n_correct / n, "valid_format": n_fmt / n,
                       "n": n, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
