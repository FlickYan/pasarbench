"""
GRPO's learning curve, paired (RUNBOOK phase 4, GRPO, the learning curve's
second point): each fold's adapter on its own training tasks at temperature 1,
before an iteration and after it, compared task by task -- per fold and over
both folds together.

    python scripts/grpo_curve.py \\
        traces/P2-grpo-i1-collect-A/full traces/P2-grpo-i2-collect-A/full \\
        traces/P2-grpo-i1-collect-B/full traces/P2-grpo-i2-collect-B/full

The cells come in pairs, before then after: `traces/P2-grpo-iN-collect-<fold>`
is iteration N's rollouts, sampled by the adapter iteration N started from
(iteration 1: the RFT adapter's). Each task's passes out of its rollouts, and a
sign test over the tasks that differ. LEARNED when the adapters after pass more
at p < 0.05, WORSE when they pass less, NOT SHOWN otherwise -- the reading fixed
before grpo-measure ran. Beside it, never instead of it, the voucher trap read
as P6.3 and P10 read a declined voucher (WHAT_FAILED #46).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _rate(eps: list[dict]) -> float:
    return sum(e["passed"] for e in eps) / len(eps) if eps else 0.0


def _line(g: dict | None) -> str:
    if g is None:
        return "fewer than 5 shared tasks -- not testable"
    return (f"{g['diff']:+.3f}; {g['better']} tasks better, {g['worse']} worse, "
            f"sign test p={g['p']:.3f}")


def verdict(g: dict | None) -> str:
    """The words fixed before the run."""
    if g is None or not g["resolved"]:
        return "NOT SHOWN"
    return "LEARNED" if g["diff"] > 0 else "WORSE"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cells", nargs="+", help="pairs of cells: before, after")
    ap.add_argument("--checker", choices=["current", "recorded"], default="current",
                    help="today's checks on the replayed episodes (default) or the "
                         "verdicts the runs recorded")
    a = ap.parse_args(argv)
    if len(a.cells) % 2:
        raise SystemExit("the cells come in pairs: before, after")
    from pasarbench.diagnose import load_episodes, paired_episodes, voucher_choice

    cells = [Path(c) for c in a.cells]
    for c in cells:
        if not c.is_dir() or not any(c.glob("*.jsonl")):
            raise SystemExit(f"no traces in {c} (run from the repo root)")
    before, after, read_b, read_a = [], [], [], []
    declined = {"before": [0, 0], "after": [0, 0]}
    print("each fold's adapter on its own training tasks at temperature 1, before and "
          "after, paired by task")
    for b, c in zip(cells[::2], cells[1::2]):
        eb, ea = load_episodes(b, checker=a.checker), load_episodes(c, checker=a.checker)
        shared = {e["task_id"] for e in eb} & {e["task_id"] for e in ea}
        eb = [e for e in eb if e["task_id"] in shared]
        ea = [e for e in ea if e["task_id"] in shared]
        g = paired_episodes(eb, ea)
        print(f"\n{b.parent.name} -> {c.parent.name}: pass^1 {_rate(eb):.3f} -> "
              f"{_rate(ea):.3f} over {len(shared)} tasks ({len(eb)} and {len(ea)} rollouts)")
        print(f"  paired by task: {_line(g)}")
        before += eb
        after += ea
        for side, cell, eps, out in (("before", b, eb, read_b), ("after", c, ea, read_a)):
            read, n = voucher_choice(cell, eps, a.checker)
            out += read
            declined[side][0] += n["read_as_pass"]
            declined[side][1] += n["episodes"]

    g = paired_episodes(before, after)
    print(f"\nall the pairs together: pass^1 {_rate(before):.3f} -> {_rate(after):.3f}; "
          f"paired by task ({g['n'] if g else 0} tasks): {_line(g)}  ->  {verdict(g)}")

    by: dict[str, list[list[bool]]] = defaultdict(lambda: [[], []])
    for i, eps in enumerate((before, after)):
        for e in eps:
            by[e["trap"]][i].append(e["passed"])
    print("\nby trap, pass^1 before -> after (a lead to read, not a result):")
    for trap in sorted(by):
        x, y = by[trap]
        print(f"  {trap:40s} {sum(x) / len(x):.2f} -> {sum(y) / len(y):.2f}"
              f"   ({len(x)} and {len(y)} rollouts)")

    if declined["before"][0] or declined["after"][0]:
        gv = paired_episodes(read_b, read_a)
        print(f"\nthe voucher trap (WHAT_FAILED #46): she declined the offered voucher and the "
              f"agent escalated, failing on the voucher's lines alone, in "
              f"{declined['before'][0]} of {declined['before'][1]} rollouts before and "
              f"{declined['after'][0]} of {declined['after'][1]} after. Read as P6.3 and P10 "
              f"read them -- after the fact, beside the result: pass^1 {_rate(read_b):.3f} -> "
              f"{_rate(read_a):.3f}; {_line(gv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
