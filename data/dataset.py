# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CRUXEval-O dataset: deterministic train/val split + ground-truth execution
traces -> (input_ids, labels) for teacher-forcing CODI.

Neutral data layer shared by training (``cwm.training.data``) and eval
(``evals.cruxeval.run_eval_codi``); depends on nothing in either, so the
split and trace format never drift. Thin HuggingFace-tokenizer wrapper over
the verbatim Table 9 trace generator (``.ground_truth`` / ``.trace_format``):
build the seeded prompt, tokenize ``prompt + render_frames_to_generation(frames)``,
and mask the prompt out of the labels (teacher-forced, so labels == input_ids
with the prompt prefix set to ``-100``).
"""

from __future__ import annotations

from .ground_truth import ground_truth_trace, make_trace_context
from .trace_format import (
    END_OF_TEXT,
    TraceEvent,
    render_frames_to_generation,
)

IGNORE_INDEX = -100
def _prompt_str(code: str, input_str: str) -> str:
    ctx = make_trace_context(code, input_str)
    return f"<|trace_context_start|>{ctx}<|frame_sep|><|call_sep|>{{}}<|action_sep|>def main():\n<|frame_sep|>"


def build_example(
    code: str, input_str: str, tokenizer, *, max_seq_len: int, max_frames: int = -1
) -> tuple[list[int], list[int]] | None:
    """Return ``(input_ids, labels)``, or None to skip (empty / too long).

    A raised program is kept: its EXCEPTION frame is part of the trace to predict.
    ``render_frames_to_generation`` already terminates the trace with ``<|end_of_text|>``.
    """
    frames, error = ground_truth_trace(code, input_str, align_to_prompt=True, max_frames=max_frames)
    if not frames or error == "frames_exceeded":
        return None
    # Qwen has no BOS (bos_token_id is None); CWM did. Prepend only if present.
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prompt_ids = bos + tokenizer.encode(
        _prompt_str(code, input_str), add_special_tokens=False
    )
    trace_ids = tokenizer.encode(render_frames_to_generation(frames), add_special_tokens=False)
    input_ids = prompt_ids + trace_ids
    if len(input_ids) > max_seq_len:
        return None
    return input_ids, [IGNORE_INDEX] * len(prompt_ids) + trace_ids


def build_codi_example(
    code: str, input_str: str, tokenizer, *, max_seq_len: int, max_frames: int = -1
) -> dict | None:
    """Single-span CODI example: split a trace into prompt / reasoning / answer.

    reasoning = all intermediate frames (the "thinking" the latents replace);
    answer = the final RETURN/EXCEPTION frame + end_of_text (what the student
    predicts). Teacher reads prompt+reasoning+answer; KD aligns the hidden that
    predicts the first answer token. Returns None to skip (degenerate/too long).
    """
    frames, error = ground_truth_trace(code, input_str, align_to_prompt=True, max_frames=max_frames)
    if error == "frames_exceeded":
        return None
    if len(frames) < 2 or frames[-1].event not in (TraceEvent.RETURN, TraceEvent.EXCEPTION):
        return None
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prompt_ids = bos + tokenizer.encode(_prompt_str(code, input_str), add_special_tokens=False)
    # render_frames_to_generation always ends with END_OF_TEXT; strip it for reasoning.
    reasoning_str = render_frames_to_generation(frames[:-1])[: -len(END_OF_TEXT)]
    reasoning_ids = tokenizer.encode(reasoning_str, add_special_tokens=False)
    answer_ids = tokenizer.encode(render_frames_to_generation(frames[-1:]), add_special_tokens=False)
    if len(prompt_ids) + len(reasoning_ids) + len(answer_ids) > max_seq_len:
        return None
    return {"prompt_ids": prompt_ids, "reasoning_ids": reasoning_ids, "answer_ids": answer_ids}


def _load_cache(cache_dir, n_samples):
    """Load precomputed tokenized examples (precompute.py); slice to n_samples."""
    from datasets import load_from_disk

    ex = list(load_from_disk(cache_dir))
    return ex[:n_samples] if n_samples > 0 else ex


def build_codi_dataset(
    tokenizer, *, sources=("cruxeval",), n_samples: int = -1,
    max_seq_len: int = 4096, max_frames: int = -1,
    split: str = "train", cache_dir: str | None = None
) -> list[dict]:
    """CODI examples (prompt/reasoning/answer) over ``sources``, or a precomputed cache."""
    if cache_dir:
        return _load_cache(cache_dir, n_samples)
    rows = rows_for_sources(sources, split)
    if n_samples > 0:
        rows = rows[:n_samples]
    out = [
        build_codi_example(
            r["code"], r["input"], tokenizer,
            max_seq_len=max_seq_len, max_frames=max_frames,
        )
        for r in rows
    ]
    return [ex for ex in out if ex is not None]


def cruxeval_split(rows, split: str = "all", val_stride: int = 5):
    """Deterministic interleaved train/val split of the CRUXEval rows.

    ``val`` = every ``val_stride``-th row (800/5 = 160 val, 640 train);
    interleaving keeps the length/difficulty distribution matched across splits.
    Single source of truth: both training (``build_dataset``) and eval
    (``run_eval_codi``) import this so the splits never drift.
    ``split`` in {"train", "val", "all"}.
    """
    if split == "all":
        return list(rows)
    is_val = lambda i: i % val_stride == 0
    if split == "val":
        return [r for i, r in enumerate(rows) if is_val(i)]
    if split == "train":
        return [r for i, r in enumerate(rows) if not is_val(i)]
    raise ValueError(f"split must be train/val/all, got {split!r}")



def rows_for_sources(sources, split: str):
    """Merge {id,code,input,output} rows across sources. cruxeval honors the
    train/val split; other sources are train-only augmentation (skipped for val)."""
    from . import sources as _src

    rows = []
    for name in sources:
        r = _src.load_one(name)
        if name.strip().lower() == "cruxeval":
            r = cruxeval_split(r, split)
        elif split == "val":
            continue
        for i, row in enumerate(r):
            missing = [k for k in ("id", "code", "input", "output") if k not in row]
            if missing:
                raise ValueError(f"{name} row {i} missing keys: {missing}")
            if not all(isinstance(row[k], str) for k in ("code", "input", "output")):
                raise TypeError(f"{name} row {i} must use string code/input/output")
            row = dict(row)
            row["id"] = str(row["id"])
            rows.append(row)
    return rows


def build_dataset(
    tokenizer, *, sources=("cruxeval",), n_samples: int = -1,
    max_seq_len: int = 8192, max_frames: int = -1,
    split: str = "all", cache_dir: str | None = None
) -> list[tuple[list[int], list[int]]]:
    """Tokenized trace examples over ``sources``, or a precomputed cache."""
    if cache_dir:
        return [(e["input_ids"], e["labels"]) for e in _load_cache(cache_dir, n_samples)]
    rows = rows_for_sources(sources, split)
    if n_samples > 0:
        rows = rows[:n_samples]
    examples = (
        build_example(
            r["code"], r["input"], tokenizer,
            max_seq_len=max_seq_len, max_frames=max_frames,
        )
        for r in rows
    )
    return [ex for ex in examples if ex is not None]
