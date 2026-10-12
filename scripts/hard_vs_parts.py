"""
The hard edition against its parts (RUNBOOK 1s): each hard case paired with
the first edition's task of the trap whose rule it turns on, in the same
market and language, so the language is held fixed; a sign test over the
pairs. The hard edition is HARDER for an agent only if its cases fail more
often than their parts, task by task, at p < 0.05.

    python scripts/hard_vs_parts.py traces/V-dsk-parts/full traces/V-dsk/full

The first cell is a run of the parts --

    --suite generated --traps duplicate_refund_escalate,out_of_window_offer_voucher,cancel_while_processing

-- the second of the hard edition (`--suite edition3`): the same agent,
customer and settings, the same day. The pairs are generate.part_of's; a case
whose part did not run is listed and left out.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PARTS_TRAPS = ("duplicate_refund_escalate", "out_of_window_offer_voucher",
               "cancel_while_processing")


def _rates(eps: list[dict]) -> tuple[float, float, int]:
    """pass^1, pass^k and the number of tasks."""
    by = defaultdict(list)
    for e in eps:
        by[e["task_id"]].append(e["passed"])
    if not by:
        return 0.0, 0.0, 0
    return (sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values()),
            sum(all(v) for v in by.values()) / len(by), len(by))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parts", help="a run of the parts, e.g. traces/V-dsk-parts/full")
    ap.add_argument("hard", help="a run of the hard edition, e.g. traces/V-dsk/full")
    ap.add_argument("--checker", choices=["current", "recorded"], default="current",
                    help="today's checks on the replayed episodes (default) or the "
                         "verdicts the runs recorded")
    a = ap.parse_args(argv)
    from pasarbench.diagnose import load_episodes, paired_episodes
    from pasarbench.generate import EDITION3_PARTS, part_of
    from pasarbench.rescore import all_tasks

    cells = [Path(a.parts), Path(a.hard)]
    for c in cells:
        if not c.is_dir() or not any(c.glob("*.jsonl")):
            raise SystemExit(f"no traces in {c} (run from the repo root)")
    tasks = all_tasks()
    parts, every = (load_episodes(c, checker=a.checker) for c in cells)
    hard = [e for e in every if e["task_id"] in tasks
            and tasks[e["task_id"]].trap in EDITION3_PARTS]
    if not hard:
        raise SystemExit(f"no episode of the hard edition in {cells[1]}: run it with "
                         f"--suite edition3")
    twin = {e["task_id"]: part_of(tasks[e["task_id"]]) for e in hard}
    have = {e["task_id"] for e in parts}
    missing = sorted({t for t, p in twin.items() if p not in have})

    print(f"{'':44s} {'hard case':>14s} {'its part':>14s}")
    for trap, part in EDITION3_PARTS.items():
        mine = [e for e in hard if tasks[e["task_id"]].trap == trap]
        if not mine:
            continue
        theirs = [e for e in parts if e["task_id"] in {twin[x["task_id"]] for x in mine}]
        h1, hk, hn = _rates(mine)
        p1, pk, pn = _rates(theirs)
        print(f"{trap:44s} {f'{h1:.3f} ({hk:.2f})':>14s} {f'{p1:.3f} ({pk:.2f})':>14s}"
              f"   {hn} cases; parts: {part}, {pn} tasks")
    h1, hk, _ = _rates(hard)
    p1, pk, _ = _rates([e for e in parts if e["task_id"] in set(twin.values())])
    print(f"{'pass^1 (pass^k)':44s} {f'{h1:.3f} ({hk:.2f})':>14s} {f'{p1:.3f} ({pk:.2f})':>14s}")

    sims = sorted({e["simulator"] for e in parts + hard})
    if len(sims) > 1:
        print(f"!! the two cells had different simulated customers ({', '.join(sims)}): a "
              f"difference is partly the customer's. Run both with the same one.")
    if missing:
        print(f"!! {len(missing)} case(s) whose part is not in {cells[0]}, left out: "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    g = paired_episodes(parts, [{**e, "task_id": twin[e["task_id"]]} for e in hard])
    print()
    if g is None:
        print("paired: fewer than 5 cases with their part -- not testable")
        return 0
    verdict = (("HARDER" if g["diff"] < 0 else "EASIER") if g["resolved"]
               else "INCONCLUSIVE")
    print(f"each case against its part ({g['n']} pairs): {g['diff']:+.3f}; {g['worse']} "
          f"cases failed more than their part, {g['better']} less, sign test "
          f"p={g['p']:.3f}  ->  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
