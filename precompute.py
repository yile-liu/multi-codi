"""Parallel offline trace/token cache builder.

Raw rows are `{id, code, input, output}`. Output is a `datasets.save_to_disk`
cache consumed by train_sft.py/train_codi.py via `--cache_dir`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import time
from pathlib import Path

os.environ.setdefault("USE_TORCH", "0")
# Single-thread the Rust tokenizer; we parallelize at the process level, and
# one rayon pool per worker exhausts the node's thread limit (WouldBlock panic).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

from datasets import Dataset
from data.dataset import build_codi_example, build_example, rows_for_sources
from tokens import add_trace_tokens

TOK = MAX_LEN = MAX_FRAMES = MODE = TIMEOUT = None


def _alarm(*_):
    raise TimeoutError


def _no_net(*_a, **_k):
    raise OSError("network disabled")


def _init(model, max_len, max_frames, mode, timeout):
    import socket
    from transformers import AutoTokenizer

    global TOK, MAX_LEN, MAX_FRAMES, MODE, TIMEOUT
    TOK = AutoTokenizer.from_pretrained(model, use_fast=True)
    add_trace_tokens(TOK)
    MAX_LEN, MAX_FRAMES, MODE, TIMEOUT = max_len, max_frames, mode, timeout
    signal.signal(signal.SIGALRM, _alarm)
    # DNS (getaddrinfo) blocks in C and ignores SIGALRM, hanging the pool.
    socket.getaddrinfo = socket.create_connection = socket.socket = _no_net


def _work(row):
    signal.alarm(TIMEOUT)
    try:
        if MODE == "codi":
            ex = build_codi_example(row["code"], row["input"], TOK, max_seq_len=MAX_LEN, max_frames=MAX_FRAMES)
        else:
            pair = build_example(row["code"], row["input"], TOK, max_seq_len=MAX_LEN, max_frames=MAX_FRAMES)
            ex = None if pair is None else {"input_ids": pair[0], "labels": pair[1]}
        if ex is not None:
            ex["row_id"] = row["id"]
        return ex
    except Exception:
        return None
    finally:
        signal.alarm(0)


def _recon_stats(examples):
    lens = sorted(len(x) for ex in examples for x in ex.get("recon_targets", []))
    if not lens:
        return {}
    pct = lambda q: lens[min(len(lens) - 1, int(q * (len(lens) - 1)))]
    return {
        "recon_frames": len(lens),
        "recon_len_mean": sum(lens) / len(lens),
        "recon_len_p90": pct(0.90),
        "recon_len_p99": pct(0.99),
        "recon_len_max": lens[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["sft", "codi"], default="sft")
    ap.add_argument("--sources", nargs="+", default=["mbpp", "humaneval", "pyx"])
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_seq_len", type=int, default=6144)
    ap.add_argument("--max_frames", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--chunksize", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = rows_for_sources(args.sources)
    if args.n_samples > 0:
        rows = rows[: args.n_samples]

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    print(f"{n} rows -> {args.out} ({args.mode}, workers={args.workers})", flush=True)
    init_args = (args.model, args.max_seq_len, args.max_frames, args.mode, args.timeout)
    if args.workers == 1:
        _init(*init_args)
        results = map(_work, rows)
    else:
        pool = mp.Pool(args.workers, _init, init_args)
        results = pool.imap_unordered(_work, rows, chunksize=args.chunksize)

    built, t0 = [], time.time()
    for i, ex in enumerate(results, 1):
        built.append(ex)
        if i % 2000 == 0 or i == n:
            ok = sum(x is not None for x in built)
            print(f"  {i}/{n}  ok={ok}  {i / (time.time() - t0):.0f} rows/s", flush=True)
    if args.workers > 1:
        pool.close()
        pool.join()

    examples = [ex for ex in built if ex is not None]
    if not examples:
        raise RuntimeError("0 examples built")
    Dataset.from_list(examples).save_to_disk(str(out))

    stats = _recon_stats(examples)
    if stats:
        print("recon delta-locals lengths: "
              f"mean={stats['recon_len_mean']:.1f} p90={stats['recon_len_p90']} "
              f"p99={stats['recon_len_p99']} max={stats['recon_len_max']}", flush=True)
    cfg = {**vars(args), "n_rows": n, "n_saved": len(examples), **stats}
    (out / "precompute_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved {len(examples)}/{len(rows)} examples", flush=True)


if __name__ == "__main__":
    main()
