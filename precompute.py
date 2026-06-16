"""Parallel offline trace/token cache builder.

Raw rows are `{id, code, input, output}`. Output is a `datasets.save_to_disk`
cache consumed by train_sft.py/train_codi.py via `--cache_dir`.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import signal
from pathlib import Path

os.environ.setdefault("USE_TORCH", "0")

from datasets import Dataset
from data.dataset import build_codi_example, build_example, rows_for_sources
from tokens import add_trace_tokens

TOK = MAX_LEN = MAX_FRAMES = MODE = TIMEOUT = None


def _alarm(*_):
    raise TimeoutError


def _init(model, max_len, max_frames, mode, timeout):
    from transformers import AutoTokenizer

    global TOK, MAX_LEN, MAX_FRAMES, MODE, TIMEOUT
    TOK = AutoTokenizer.from_pretrained(model, use_fast=True)
    add_trace_tokens(TOK)
    MAX_LEN, MAX_FRAMES, MODE, TIMEOUT = max_len, max_frames, mode, timeout
    signal.signal(signal.SIGALRM, _alarm)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["sft", "codi"], default="sft")
    ap.add_argument("--sources", nargs="+", default=["cruxeval"])
    ap.add_argument("--split", choices=["train", "val", "all"], default="train")
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--chunksize", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = rows_for_sources(args.sources, args.split)
    if args.n_samples > 0:
        rows = rows[: args.n_samples]

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{len(rows)} rows -> {args.out} ({args.mode}, workers={args.workers})", flush=True)
    if args.workers == 1:
        _init(args.model, args.max_seq_len, args.max_frames, args.mode, args.timeout)
        built = [_work(r) for r in rows]
    else:
        with mp.Pool(args.workers, _init, (args.model, args.max_seq_len, args.max_frames, args.mode, args.timeout)) as pool:
            built = pool.map(_work, rows, chunksize=args.chunksize)

    examples = [ex for ex in built if ex is not None]
    if not examples:
        raise RuntimeError("0 examples built")
    Dataset.from_list(examples).save_to_disk(str(out))
    print(f"saved {len(examples)}/{len(rows)} examples", flush=True)


if __name__ == "__main__":
    main()
