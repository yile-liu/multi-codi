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
    ACTION_SEP,
    LINE_SEP,
    TraceEvent,
    render_frames_to_generation,
)

IGNORE_INDEX = -100
def _prompt_str(code: str, input_str: str) -> str:
    ctx = make_trace_context(code, input_str)
    return f"<|trace_context_start|>{ctx}<|frame_sep|><|call_sep|>{{}}<|action_sep|>def main():\n<|frame_sep|>"


def _tokenize_trace(code, input_str, tokenizer, *, max_seq_len, max_frames):
    """``(prompt_ids, trace_ids, spans)``; None to skip. Trace must terminate in
    RETURN/EXCEPTION and have >=1 LINE span. Span ``(i, j)``: ``trace_ids[i]`` is
    ``<|line_sep|>``, ``j`` its ``<|action_sep|>``, ``trace_ids[i+1:j]`` the locals
    a CODI student swaps for a latent block. Single source of membership so the SFT
    baseline and CODI train on identical data."""
    frames, error = ground_truth_trace(code, input_str, align_to_prompt=True, max_frames=max_frames)
    if not frames or error == "frames_exceeded":
        return None
    if frames[-1].event not in (TraceEvent.RETURN, TraceEvent.EXCEPTION):
        return None
    # Qwen has no BOS (bos_token_id is None); CWM did. Prepend only if present.
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prompt_ids = bos + tokenizer.encode(_prompt_str(code, input_str), add_special_tokens=False)
    trace_ids = tokenizer.encode(render_frames_to_generation(frames), add_special_tokens=False)
    if len(prompt_ids) + len(trace_ids) > max_seq_len:
        return None
    ls = tokenizer.convert_tokens_to_ids(LINE_SEP)
    asep = tokenizer.convert_tokens_to_ids(ACTION_SEP)
    spans, i, n = [], 0, len(trace_ids)
    while i < n:
        if trace_ids[i] == ls:
            j = i + 1
            while j < n and trace_ids[j] != asep:
                j += 1
            if j == n:
                break
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    if not spans:
        return None
    return prompt_ids, trace_ids, spans


def build_example(code, input_str, tokenizer, *, max_seq_len, max_frames=-1):
    """SFT ``(input_ids, labels)`` with the prompt masked; None to skip."""
    r = _tokenize_trace(code, input_str, tokenizer, max_seq_len=max_seq_len, max_frames=max_frames)
    if r is None:
        return None
    prompt_ids, trace_ids, _ = r
    return prompt_ids + trace_ids, [IGNORE_INDEX] * len(prompt_ids) + trace_ids


def build_codi_example(code, input_str, tokenizer, *, max_seq_len, max_frames=-1):
    """Multi-span CODI example ``{prompt_ids, trace_ids, spans}``; None to skip."""
    r = _tokenize_trace(code, input_str, tokenizer, max_seq_len=max_seq_len, max_frames=max_frames)
    if r is None:
        return None
    prompt_ids, trace_ids, spans = r
    return {"prompt_ids": prompt_ids, "trace_ids": trace_ids, "spans": spans}


def _load_cache(cache_dir, n_samples):
    """Load precomputed tokenized examples (precompute.py); slice to n_samples."""
    from datasets import load_from_disk

    ex = list(load_from_disk(cache_dir))
    return ex[:n_samples] if n_samples > 0 else ex


def build_codi_dataset(
    tokenizer, *, sources=("mbpp", "humaneval", "pyx"), n_samples: int = -1,
    max_seq_len: int = 4096, max_frames: int = -1, cache_dir: str | None = None
) -> list[dict]:
    """CODI examples (prompt/reasoning/answer) over ``sources``, or a precomputed cache."""
    if cache_dir:
        ex = _load_cache(cache_dir, n_samples)
        return [e for e in ex if len(e["prompt_ids"]) + len(e["trace_ids"]) <= max_seq_len]
    rows = rows_for_sources(sources)
    if n_samples > 0:
        rows = rows[:n_samples]
    out = []
    for r in rows:
        try:
            out.append(build_codi_example(r["code"], r["input"], tokenizer,
                                          max_seq_len=max_seq_len, max_frames=max_frames))
        except Exception:
            pass
    return [ex for ex in out if ex is not None]


def build_codi_single_dataset(
    tokenizer, *, sources=("mbpp", "humaneval", "pyx"), n_samples: int = -1,
    max_seq_len: int = 4096, max_frames: int = -1, cache_dir: str | None = None
) -> list[dict]:
    """Faithful single-block CODI: split each trace at its last ``<|return_sep|>`` into
    ``{prompt_ids, reasoning_ids, answer_ids}`` (reasoning = whole trace, answer = final
    RETURN frame). Derived from the multi-span examples; no separate cache needed."""
    rsep = tokenizer.convert_tokens_to_ids("<|return_sep|>")
    out = []
    for e in build_codi_dataset(tokenizer, sources=sources, n_samples=n_samples,
                                max_seq_len=max_seq_len, max_frames=max_frames, cache_dir=cache_dir):
        t = e["trace_ids"]
        idx = [i for i, x in enumerate(t) if x == rsep]
        if not idx or idx[-1] == 0:
            continue
        out.append({"prompt_ids": e["prompt_ids"], "reasoning_ids": t[:idx[-1]], "answer_ids": t[idx[-1]:]})
    return out


def rows_for_sources(sources):
    """Merge {id,code,input,output} rows across sources (all rows; train vs test
    is split by dataset, e.g. cruxeval is held out for eval)."""
    from . import sources as _src

    rows = []
    for name in sources:
        for i, row in enumerate(_src.load_one(name)):
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
    tokenizer, *, sources=("mbpp", "humaneval", "pyx"), n_samples: int = -1,
    max_seq_len: int = 8192, max_frames: int = -1, cache_dir: str | None = None
) -> list[tuple[list[int], list[int]]]:
    """Tokenized trace examples over ``sources``, or a precomputed cache."""
    if cache_dir:
        ex = _load_cache(cache_dir, n_samples)
        return [(e["input_ids"], e["labels"]) for e in ex if len(e["input_ids"]) <= max_seq_len]
    rows = rows_for_sources(sources)
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
