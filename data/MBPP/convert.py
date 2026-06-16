"""MBPP -> {id, code, input, output}. Run `python -m data.MBPP.convert` to save_to_disk ./data."""

import ast
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset


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


def entry_name(test_list):
    """Function name under test = first call in the first parseable assert."""
    for t in test_list:
        try:
            for n in ast.walk(ast.parse(t)):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    return n.func.id
        except SyntaxError:
            pass
    return None


def to_rows():
    ds = load_dataset("google-research-datasets/mbpp", "full")
    rows = []
    for s in concatenate_datasets(list(ds.values())):
        name = entry_name(s["test_list"])
        if not name:
            continue
        code = (s["test_setup_code"] or "") + "\n" + s["code"] + f"\nf = {name}\n"
        ios = [io for t in s["test_list"] for io in extract_io(t, name)]
        for i, (inp, out) in enumerate(ios):
            rows.append({"id": f"mbpp_{s['task_id']}_{i}", "code": code, "input": inp, "output": out})
    return rows


if __name__ == "__main__":
    rows = to_rows()
    out_dir = Path(__file__).parent / "data"
    Dataset.from_list(rows).save_to_disk(str(out_dir))
    print(f"MBPP: {len(rows)} rows -> {out_dir}")
