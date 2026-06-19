"""Offline post-hoc: stratify a CRUXEval-O eval by ground-truth trace length.

No change to eval/training. Reads eval_cruxeval_*.py's --out JSON (per-sample
id+correct), recomputes each id's LINE-frame count (= #latent spans = #KD
anchors, the verifiable-CODI difficulty axis) via the model-free tracer, bins,
and reports pass@1 per bin. Run from codi_trace/:
    python -m eval.stats_by_frames --results results.json
Compare two runs (e.g. per-frame vs single-anchor CODI) on the same id set to
see the gap widen with frame count.
"""

import argparse
import json

from data.ground_truth import ground_truth_trace
from data.sources import load_cruxeval
from data.trace_format import TraceEvent

DEFAULT_EDGES = [1, 2, 3, 4, 5, 7, 11, 21]  # right-open bins + [last, inf)


def line_frames(code: str, input_str: str, max_frames: int) -> int | None:
    """#LINE frames in the ground-truth trace, or None if it didn't trace."""
    frames, error = ground_truth_trace(code, input_str, align_to_prompt=True, max_frames=max_frames)
    if error or not frames:
        return None
    return sum(1 for f in frames if f.event == TraceEvent.LINE)


def bin_label(edges: list[int], k: int) -> str:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= k < hi:
            return f"{lo}-{hi - 1}" if hi - 1 > lo else f"{lo}"
    return f"{edges[-1]}+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="eval_cruxeval_*.py --out JSON")
    ap.add_argument("--edges", default=",".join(map(str, DEFAULT_EDGES)),
                    help="comma-sep bin lower edges; last bin is [edge,inf)")
    ap.add_argument("--max_frames", type=int, default=5000,
                    help="tracer cap (guards runaway loops); overflow -> untraceable")
    ap.add_argument("--out", default="", help="optional JSON dump of the table")
    args = ap.parse_args()

    edges = sorted(int(x) for x in args.edges.split(","))
    res = json.load(open(args.results))["results"]
    by_id = {str(r["id"]): r for r in load_cruxeval()}

    # bucket -> [n, n_correct, n_fmt]; plus a special "untraceable" bucket
    buckets: dict[str, list[int]] = {}
    untraceable = [0, 0, 0]
    missing = 0
    for r in res:
        row = by_id.get(str(r["id"]))
        if row is None:
            missing += 1
            continue
        k = line_frames(row["code"], row["input"], args.max_frames)
        tgt = untraceable if k is None else buckets.setdefault(bin_label(edges, k), [0, 0, 0])
        tgt[0] += 1
        tgt[1] += int(bool(r["correct"]))
        tgt[2] += int(r.get("predicted") is not None)

    def order(lbl: str) -> int:
        return int(lbl.split("-")[0].rstrip("+"))

    rows = sorted(buckets.items(), key=lambda kv: order(kv[0]))
    print(f"{'LINE frames':>12} {'n':>5} {'pass@1':>8} {'valid_fmt':>10}")
    tot = [0, 0, 0]
    table = []
    for lbl, (n, c, fmt) in rows:
        print(f"{lbl:>12} {n:>5} {c / n:>8.4f} {fmt / n:>10.4f}")
        table.append({"bin": lbl, "n": n, "pass_at_1": c / n, "valid_format": fmt / n})
        for i in range(3):
            tot[i] += [n, c, fmt][i]
    if untraceable[0]:
        n, c, fmt = untraceable
        print(f"{'untraceable':>12} {n:>5} {c / n:>8.4f} {fmt / n:>10.4f}")
        table.append({"bin": "untraceable", "n": n, "pass_at_1": c / n, "valid_format": fmt / n})
        for i in range(3):
            tot[i] += [n, c, fmt][i]
    n, c, fmt = tot
    print(f"{'ALL':>12} {n:>5} {c / n:>8.4f} {fmt / n:>10.4f}"
          + (f"   ({missing} ids not in CRUXEval-O, skipped)" if missing else ""))

    if args.out:
        json.dump({"edges": edges, "max_frames": args.max_frames,
                   "missing": missing, "bins": table,
                   "all": {"n": n, "pass_at_1": c / n, "valid_format": fmt / n}},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
