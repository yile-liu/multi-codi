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
from .trace_format import render_frames_to_generation

IGNORE_INDEX = -100


def _prompt_str(code: str, input_str: str) -> str:
    ctx = make_trace_context(code, input_str)
    return f"<|trace_context_start|>{ctx}<|frame_sep|><|call_sep|>{{}}<|action_sep|>def main():\n<|frame_sep|>"


def build_example(
    code: str, input_str: str, tokenizer, *, max_seq_len: int
) -> tuple[list[int], list[int]] | None:
    """Return ``(input_ids, labels)``, or None to skip (empty / too long).

    A raised program is kept: its EXCEPTION frame is part of the trace to predict.
    ``render_frames_to_generation`` already terminates the trace with ``<|end_of_text|>``.
    """
    frames, _error = ground_truth_trace(code, input_str, align_to_prompt=True)
    if not frames:
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


def build_dataset(
    tokenizer, *, n_samples: int = -1, max_seq_len: int = 8192, split: str = "all"
) -> list[tuple[list[int], list[int]]]:
    """Tokenized CRUXEval-O traces. ``n_samples<=0`` uses all of ``split``."""
    import os

    # Prefer local save_to_disk copy; HF builder FileLock dies on NFS caches.
    local_dir = os.environ.get("CRUXEVAL_DIR")
    if local_dir and os.path.isdir(local_dir):
        from datasets import load_from_disk

        rows = list(load_from_disk(local_dir))
    else:
        from datasets import load_dataset

        rows = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    rows = cruxeval_split(rows, split)
    if n_samples > 0:
        rows = rows[:n_samples]
    examples = (build_example(r["code"], r["input"], tokenizer, max_seq_len=max_seq_len) for r in rows)
    return [ex for ex in examples if ex is not None]
