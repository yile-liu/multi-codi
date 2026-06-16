# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Ground-truth execution-trace generation in CWM's trace format.

We execute the CRUXEval function under ``sys.settrace`` and reconstruct the
same frame sequence that CWM is trained to predict: a CALL frame on entering a
scope, a LINE frame before each executed line, and a RETURN/EXCEPTION frame on
leaving a scope. Locals use the diff-based representation (unchanged variables
render as ``".."``), and every value is rendered with Python ``repr``.

This mirrors the entry-point convention used by the trace prompt
(``evals.cruxeval.prompts.make_trace_full_prompt_tokens``): a synthetic
``def main(): return f(<input>)`` wraps the function under test, and the trace
starts when ``main`` is called. The prompt seeds the first ``call main()``
frame, so ``ground_truth_trace`` drops it by default (see ``drop_entry_call``).

Caveats (documented in README.md): the exact diff-reset rules used to build
CWM's original training traces are not published in this repo. Values are
rendered with ``repr`` (confirmed against real generations: single-quoted
strings, parenthesized tuples, bare ints), but exotic objects may differ from
CWM's internal renderer. Treat the resulting numbers as a faithful
re-implementation, not a bit-exact replica of Meta's internal tracer.
"""

from __future__ import annotations

import linecache
import sys
from types import FrameType
from typing import Any

from .trace_format import (
    DIFF_PLACEHOLDER,
    TraceEvent,
    TraceFrame,
    normalize_source,
)

_FILENAME = "<cwm_trace>"
_ENTRY = "main"


def make_trace_context(code: str, input_str: str) -> str:
    """Source context for trace prediction (matches cruxeval.prompts)."""
    return f"\n{code}\ndef main():  # << START_OF_TRACE\n    return f({input_str})\n"


def render_value(value: Any) -> str:
    """Render a Python value as CWM does: the Python source ``repr``.

    Confirmed against real CWM generations: tuples render as ``(4, 1)`` and
    strings as ``'x'`` (single-quoted), i.e. ``repr`` semantics, *not*
    ``json.dumps`` (which would emit ``[4, 1]`` / ``"x"``). The value string is
    then stored as a JSON string inside the frame's locals object.
    """
    try:
        return repr(value)
    except Exception:  # noqa: BLE001 - a broken __repr__ shouldn't crash eval
        return "<unrepr>"


class _GroundTruthTracer:
    def __init__(self, code: str, input_str: str) -> None:
        self.context = make_trace_context(code, input_str)
        # Register the context source so frame line numbers resolve to lines.
        src_lines = self.context.splitlines(keepends=True)
        linecache.cache[_FILENAME] = (
            len(self.context),
            None,
            src_lines,
            _FILENAME,
        )
        self._code_obj = compile(self.context, _FILENAME, "exec")
        self.frames: list[TraceFrame] = []
        # Per-scope (keyed by id(frame)) snapshot of last-rendered locals, to
        # compute the diff-based representation.
        self._scope_prev: dict[int, dict[str, str]] = {}
        self._entry_frame_id: int | None = None
        self.error: str | None = None

    # -- tracer callbacks ---------------------------------------------------

    def _source_line(self, frame: FrameType) -> str:
        line = linecache.getline(_FILENAME, frame.f_lineno)
        return normalize_source(line)

    def _diff_locals(self, frame: FrameType) -> dict[str, str]:
        scope = id(frame)
        prev = self._scope_prev.setdefault(scope, {})
        current: dict[str, str] = {}
        rendered: dict[str, str] = {}
        for name, val in frame.f_locals.items():
            r = render_value(val)
            rendered[name] = r
            if name in prev and prev[name] == r:
                current[name] = DIFF_PLACEHOLDER
            else:
                current[name] = r
        self._scope_prev[scope] = rendered
        return current

    def _trace(self, frame: FrameType, event: str, arg: Any):  # noqa: ANN001
        # Only follow execution at or below the entry point's scope.
        if self._entry_frame_id is None:
            if event == "call" and frame.f_code.co_name == _ENTRY:
                self._entry_frame_id = id(frame)
            else:
                return None

        if event == "call":
            self.frames.append(
                TraceFrame(
                    event=TraceEvent.CALL,
                    source=self._source_line(frame),
                    locals=self._diff_locals(frame),
                )
            )
            return self._trace
        if event == "line":
            self.frames.append(
                TraceFrame(
                    event=TraceEvent.LINE,
                    source=self._source_line(frame),
                    locals=self._diff_locals(frame),
                )
            )
            return self._trace
        if event == "return":
            self.frames.append(
                TraceFrame(
                    event=TraceEvent.RETURN,
                    source=self._source_line(frame),
                    arg=render_value(arg),
                )
            )
            return self._trace
        if event == "exception":
            exc_type = arg[0]
            self.frames.append(
                TraceFrame(
                    event=TraceEvent.EXCEPTION,
                    source=self._source_line(frame),
                    arg=render_value(getattr(exc_type, "__name__", str(exc_type))),
                )
            )
            return self._trace
        return self._trace

    # -- driver -------------------------------------------------------------

    def run(self, timeout_unused: float = 0.0) -> None:
        ns: dict[str, Any] = {}
        # Define f and main without tracing module-level execution.
        exec(self._code_obj, ns)
        main = ns[_ENTRY]
        old = sys.gettrace()
        sys.settrace(self._trace)
        try:
            main()
        except Exception as e:  # noqa: BLE001 - record but don't crash eval
            self.error = f"{type(e).__name__}: {e}"
        finally:
            sys.settrace(old)


def drop_entry_call(frames: list[TraceFrame]) -> list[TraceFrame]:
    """Drop the leading ``call main()`` frame.

    The full-trace prompt seeds ``<|call_sep|>{}<|action_sep|>def main():`` so
    the model only generates from the *next* frame onward. To align a generated
    trace with the ground truth we must drop this seeded entry frame.
    """
    if (
        frames
        and frames[0].event == TraceEvent.CALL
        and frames[0].source.startswith("def main()")
    ):
        return frames[1:]
    return frames


def ground_truth_trace(
    code: str, input_str: str, align_to_prompt: bool = True
) -> tuple[list[TraceFrame], str | None]:
    """Return (ground-truth frames, error) for executing ``f(input_str)``.

    When ``align_to_prompt`` is True (the default), the leading seeded
    ``call main()`` frame is dropped so the frames line up positionally with a
    model generation produced from ``make_trace_full_prompt_tokens``.

    ``error`` is non-None if the traced program raised; the frames captured up
    to that point are still returned.
    """
    tracer = _GroundTruthTracer(code, input_str)
    tracer.run()
    frames = drop_entry_call(tracer.frames) if align_to_prompt else tracer.frames
    return frames, tracer.error
