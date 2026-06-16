"""PyX (semcoder) -> {id, code, input, output}. Run `python -m data.PyX.convert` to save_to_disk ./data.

PyX is NL->code, but a large slice ships CRUXEval-style I/O-prediction examples:
a [PYTHON] block (function under test) + a response [ANSWER] holding a concrete
`assert f(args) == expected`. We keep only those (plain NL->code rows can't be
traced). output is reference-only; the trace pipeline recomputes it by execution.
"""

import ast
import re
from pathlib import Path

from datasets import Dataset, load_dataset

_PYTHON = re.compile(r"\[PYTHON\](.*?)\[/PYTHON\]", re.S)
_ANSWER = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.S)


def strip_trailing_asserts(block: str) -> str | None:
    """Block source with trailing assert(s) dropped (the probe assert may hold `??`)."""
    lines = block.splitlines()
    cut = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("assert ")), len(lines))
    code = "\n".join(lines[:cut]).strip()
    if not code:
        return None
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code


def call_args_source(src: str, call: ast.Call) -> str:
    """Reconstruct the call argument list source, e.g. `10, b=5, *xs, **kw`."""
    parts = [ast.get_source_segment(src, a) for a in call.args]
    for kw in call.keywords:
        v = ast.get_source_segment(src, kw.value)
        parts.append(f"**{v}" if kw.arg is None else f"{kw.arg}={v}")
    return ", ".join(p for p in parts if p)


def parse_answer(answer: str):
    """(entry, input, output) from an `assert f(args) == expected` line, or None."""
    src = answer.strip()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
            continue
        l, r = node.left, node.comparators[0]
        for call, exp in ((l, r), (r, l)):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                inp = call_args_source(src, call)
                out = ast.get_source_segment(src, exp)
                if inp and out:
                    return call.func.id, inp, out
    return None


def to_rows():
    rows = []
    for s in load_dataset("semcoder/PyX", split="train"):
        mnl = (s["fwd_mnl"] or "") or (s["bwd_mnl"] or "")
        pm = _PYTHON.search(mnl)
        am = _ANSWER.search(s["response"] or "")
        if not pm or not am:
            continue  # plain NL->code row: no concrete input to trace
        code = strip_trailing_asserts(pm.group(1))
        parsed = parse_answer(am.group(1))
        if not code or not parsed:
            continue
        entry, inp, out = parsed
        code = code + f"\nf = {entry}\n"
        rows.append({"id": f"pyx_{s['id']}", "code": code, "input": inp, "output": out})
    return rows


if __name__ == "__main__":
    rows = to_rows()
    out_dir = Path(__file__).parent / "data"
    Dataset.from_list(rows).save_to_disk(str(out_dir))
    print(f"PyX: {len(rows)} rows -> {out_dir}")
