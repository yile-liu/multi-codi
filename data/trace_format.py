# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Shared CWM execution-trace representation and parsing.

CWM predicts an execution trace as a sequence of *frames*, each consisting of an
*observation* (the local-variable state) and an *action* (the executed source
line). The on-the-wire format (see PROMPTING_GUIDE.md and demos/cwmdbg.py) is:

    <|call_sep|>$LOCALS<|action_sep|>$SOURCE<|frame_sep|>
    <|line_sep|>$LOCALS<|action_sep|>$SOURCE<|frame_sep|>
    <|return_sep|><|action_sep|>$SOURCE<|arg_sep|>$VALUE<|frame_sep|>
    <|exception_sep|><|action_sep|>$SOURCE<|arg_sep|>$VALUE<|frame_sep|>

`$LOCALS` is a JSON object mapping variable names to *string* values; each value
is the JSON encoding of the underlying Python value (e.g. `"5"`, `"\"abc\""`,
`"[1, 2]"`). Locals use a diff-based representation: a variable whose value is
unchanged since the previous frame in the same scope is rendered as the
placeholder string `".."`. `$VALUE` (return/exception frames) is the JSON
encoding of the returned/raised value, stored as a JSON string.

This module is GPU-free and import-light so it can be unit-tested directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

# Literal piece strings as they appear when a generation is decoded with
# cut_at_stop_tokens=False (matches CWMInstructTokenizer.*_ID constants).
CALL_SEP = "<|call_sep|>"
LINE_SEP = "<|line_sep|>"
RETURN_SEP = "<|return_sep|>"
EXCEPTION_SEP = "<|exception_sep|>"
ACTION_SEP = "<|action_sep|>"
ARG_SEP = "<|arg_sep|>"
FRAME_SEP = "<|frame_sep|>"
END_OF_TEXT = "<|end_of_text|>"

DIFF_PLACEHOLDER = ".."
_START_MARKER = "  # << START_OF_TRACE"


class TraceEvent(Enum):
    CALL = "call"
    LINE = "line"
    RETURN = "return"
    EXCEPTION = "exception"


_EVENT_TOKENS: dict[str, TraceEvent] = {
    CALL_SEP: TraceEvent.CALL,
    LINE_SEP: TraceEvent.LINE,
    RETURN_SEP: TraceEvent.RETURN,
    EXCEPTION_SEP: TraceEvent.EXCEPTION,
}
_EVENT_TO_TOKEN: dict[TraceEvent, str] = {v: k for k, v in _EVENT_TOKENS.items()}


@dataclass
class TraceFrame:
    """A single execution-trace frame.

    `locals_str` is the raw `$LOCALS` text exactly as it appears between the
    event token and `<|action_sep|>` (empty string for return/exception
    frames). `locals` is its parsed form (a dict of name -> JSON-string-value),
    or None if it failed to parse as a JSON object. `source` is the action line
    with the START_OF_TRACE marker and trailing newline stripped.
    """

    event: TraceEvent
    source: str
    locals_str: str = ""
    locals: dict[str, str] | None = None
    arg: str | None = None
    malformed: bool = False
    # Token counts (filled when a tokenizer is available); used for the
    # "Avg State/Action Length (Token)" statistics rows of Table 9.
    state_tokens: int = 0
    action_tokens: int = 0

    @property
    def has_locals(self) -> bool:
        return self.event in (TraceEvent.CALL, TraceEvent.LINE)


def normalize_source(source: str) -> str:
    """Strip the trace start marker and trailing newline from a source line."""
    return source.rstrip("\n").rstrip(_START_MARKER).rstrip()


def parse_locals(locals_str: str) -> dict[str, str] | None:
    """Parse a `$LOCALS` payload into a dict, or None if it is not a JSON object."""
    locals_str = locals_str.strip()
    if locals_str == "":
        return {}
    try:
        obj = json.loads(locals_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    # Values are always JSON strings; coerce defensively.
    return {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in obj.items()}


def parse_generated_trace(generation: str) -> tuple[list[TraceFrame], bool]:
    """Parse a full-trace generation string into frames.

    Returns (frames, well_formed). `well_formed` is True when every frame had a
    leading event token and an `<|action_sep|>` (and an `<|arg_sep|>` for
    return/exception frames) and the generation contained no leftover garbage
    between the last frame and end-of-text. This drives the "Valid Trace Format"
    metric. Individual frames are still returned even when malformed so that the
    other metrics can be computed over whatever parsed cleanly.
    """
    # Everything after end-of-text is irrelevant.
    if END_OF_TEXT in generation:
        generation = generation.split(END_OF_TEXT, 1)[0]

    frames: list[TraceFrame] = []
    well_formed = True
    segments = generation.split(FRAME_SEP)
    # The final segment is the text after the last frame_sep; for a clean trace
    # it should be empty (the model emitted frame_sep then end_of_text).
    trailing = segments.pop() if segments else ""
    if trailing.strip() not in ("",):
        well_formed = False

    for seg in segments:
        if seg.strip() == "":
            # Stray empty segment (e.g. leading text before first token).
            continue
        frame, ok = _parse_segment(seg)
        if frame is None:
            well_formed = False
            continue
        well_formed = well_formed and ok
        frames.append(frame)

    if not frames:
        well_formed = False

    return frames, well_formed


def _parse_segment(seg: str) -> tuple[TraceFrame | None, bool]:
    # Identify the (first) event token.
    event: TraceEvent | None = None
    for tok, evt in _EVENT_TOKENS.items():
        idx = seg.find(tok)
        if idx != -1:
            event = evt
            seg = seg[idx + len(tok):]
            break
    if event is None:
        return None, False

    ok = True
    if event in (TraceEvent.CALL, TraceEvent.LINE):
        if ACTION_SEP not in seg:
            return (
                TraceFrame(event=event, source="", malformed=True),
                False,
            )
        locals_str, source = seg.split(ACTION_SEP, 1)
        parsed = parse_locals(locals_str)
        return (
            TraceFrame(
                event=event,
                source=normalize_source(source),
                locals_str=locals_str.strip(),
                locals=parsed,
                malformed=parsed is None,
            ),
            ok,
        )

    # RETURN / EXCEPTION
    if ACTION_SEP not in seg:
        return TraceFrame(event=event, source="", malformed=True), False
    seg = seg.split(ACTION_SEP, 1)[1]
    if ARG_SEP in seg:
        source, arg = seg.split(ARG_SEP, 1)
        arg = _parse_arg(arg)
    else:
        source, arg = seg, None
        ok = False
    return (
        TraceFrame(event=event, source=normalize_source(source), arg=arg),
        ok,
    )


def render_frames_to_generation(frames: list[TraceFrame]) -> str:
    """Render frames back to the on-the-wire generation string.

    Inverse of ``parse_generated_trace`` for well-formed frames. Used by tests
    (a ground-truth trace rendered this way must round-trip to a perfect score)
    and to materialize a reference trace string for inspection.
    """
    out: list[str] = []
    for f in frames:
        out.append(_EVENT_TO_TOKEN[f.event])
        if f.has_locals:
            out.append(json.dumps(f.locals if f.locals is not None else {}))
        out.append(ACTION_SEP)
        out.append(f.source)
        if f.event in (TraceEvent.RETURN, TraceEvent.EXCEPTION):
            out.append(ARG_SEP)
            out.append(json.dumps(f.arg))
        out.append(FRAME_SEP)
    out.append(END_OF_TEXT)
    return "".join(out)


def _parse_arg(arg_str: str) -> str | None:
    arg_str = arg_str.strip()
    if arg_str == "":
        return None
    try:
        # The frame stores json.dumps(value_string); unwrap one level so `arg`
        # is the source-literal value string (e.g. '"x9ja"' or '17').
        loaded = json.loads(arg_str)
        return loaded if isinstance(loaded, str) else arg_str
    except json.JSONDecodeError:
        return arg_str
