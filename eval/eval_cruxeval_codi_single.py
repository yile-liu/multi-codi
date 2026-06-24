"""CRUXEval-O single-block CODI eval: after the prompt, insert ONE latent block
(prompt -> latents -> answer), then greedily decode the answer (final RETURN frame).
"""

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data.dataset import _prompt_str
from data.sources import load_cruxeval
from eval.eval_cruxeval_sft import check_correct, extract_answer_trace_full
from tokens import add_trace_tokens, token_ids
from train.train_codi_single import CodiSingle


def load_codi_single(m, latent_steps, dev):
    tok = AutoTokenizer.from_pretrained(m, use_fast=True)
    add_trace_tokens(tok)
    ids = token_ids(tok)
    base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(m), torch_dtype=torch.bfloat16)
    model = CodiSingle(base, latent_start_id=ids["<|latent_start|>"],
                       latent_end_id=ids["<|latent_end|>"], latent_steps=latent_steps)
    if os.path.exists(f"{m}/pytorch_model.bin"):  # checkpoint: full CodiSingle
        model.load_state_dict(torch.load(f"{m}/pytorch_model.bin", map_location="cpu"))
    else:  # final export: backbone safetensors + separate projector
        model.model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16)
        model.prj.load_state_dict(torch.load(f"{m}/thought_projector.pt", map_location="cpu"))
    return tok, ids, model.to(dev).eval()


@torch.no_grad()
def gen_single(model, prompt_ids, eot, max_new):
    dev = prompt_ids.device
    cache = model.model.model(inputs_embeds=model._emb(prompt_ids[None]), use_cache=True).past_key_values
    cache, logits, _ = model._latent(cache)  # one latent block, then plain decode
    out = []
    for _ in range(max_new):
        t = int(logits.argmax(-1))
        if t == eot:
            break
        out.append(t)
        o = model.model(input_ids=torch.tensor([[t]], device=dev), past_key_values=cache, use_cache=True)
        cache, logits = o.past_key_values, o.logits[:, -1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--latent_steps", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl", timeout=timedelta(hours=1))
    torch.cuda.set_device(local_rank)

    tok, ids, model = load_codi_single(args.model, args.latent_steps, local_rank)
    eot = ids["<|end_of_text|>"]

    rows = load_cruxeval()
    if args.n_samples > 0:
        rows = rows[: args.n_samples]
    n = len(rows)
    shard = rows[rank::world]

    n_correct = n_fmt = 0
    results = []
    for i, r in enumerate(shard):
        enc = tok(_prompt_str(r["code"], r["input"]), return_tensors="pt",
                  add_special_tokens=False).to(local_rank)
        gen = tok.decode(gen_single(model, enc["input_ids"][0], eot, args.max_new_tokens),
                         skip_special_tokens=False)
        pred = extract_answer_trace_full(gen)
        ok = pred is not None and check_correct(r["code"], r["output"], pred)
        n_fmt += pred is not None
        n_correct += ok
        results.append({"id": r["id"], "expected": r["output"], "predicted": pred, "correct": ok, "generation": gen})
        if rank == 0 and (i + 1) % 20 == 0:
            print(f"  rank0 {i+1}/{len(shard)}  pass@1={n_correct/(i+1):.4f}", flush=True)

    if ddp:
        t = torch.tensor([n_correct, n_fmt], device=local_rank)
        dist.all_reduce(t)
        n_correct, n_fmt = int(t[0]), int(t[1])
        gathered = [None] * world
        dist.gather_object(results, gathered if rank == 0 else None, dst=0)
        if rank == 0:
            results = [x for part in gathered for x in part]

    if rank == 0:
        print(f"\nCRUXEval-O single-latent pass@1={n_correct / n:.4f}  "
              f"valid_format={n_fmt / n:.4f}  (n={n}, greedy)")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"pass_at_1": n_correct / n, "valid_format": n_fmt / n,
                           "n": n, "results": results}, f, indent=2)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
