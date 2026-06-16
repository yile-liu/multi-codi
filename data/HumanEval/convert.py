"""HumanEval -> {id, code, input, output}. Run `python -m data.HumanEval.convert` to save_to_disk ./data."""

import ast
from pathlib import Path

from datasets import Dataset, load_dataset


def extract_io(test_src, call_name):
    """Yield (input, output) for each clean `<call>(args) == <expected>` assert."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
            continue
        l, r = node.left, node.comparators[0]
        call = exp = None
        for a, b in ((l, r), (r, l)):
            if isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id == call_name:
                call, exp = a, b
        if call is None or call.keywords:
            continue
        inp = ", ".join(ast.get_source_segment(test_src, a) for a in call.args)
        out = ast.get_source_segment(test_src, exp)
        if inp and out:
            yield inp, out


def to_rows():
    rows = []
    for s in load_dataset("openai/openai_humaneval", split="test"):
        code = s["prompt"] + s["canonical_solution"] + f"\nf = {s['entry_point']}\n"
        tid = s["task_id"].replace("/", "_")
        for i, (inp, out) in enumerate(extract_io(s["test"], "candidate")):
            rows.append({"id": f"{tid}_{i}", "code": code, "input": inp, "output": out})
    return rows


if __name__ == "__main__":
    rows = to_rows()
    out_dir = Path(__file__).parent / "data"
    Dataset.from_list(rows).save_to_disk(str(out_dir))
    print(f"HumanEval: {len(rows)} rows -> {out_dir}")
