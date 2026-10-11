"""
What the closing guard would have done to runs already recorded, before paying
for a guarded one (docs/RUNBOOK.md, 1o).

For every episode with a closing phase under the given trace directories, the
conversation is replayed to the note and the phase's calls walked in order:
each write that repeats one already made -- the review's own definition -- is
one the guard (`--closing guard`) would not have run. Beside it, the repeats
the closing review counts. Where the two differ, the episode is listed: the
guard would refuse what the review does not call a repeat, or the reverse.

    python scripts/closing_guard_dry_run.py traces              # every run
    python scripts/closing_guard_dry_run.py traces/R-check      # one run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def scan(roots: list[Path]) -> tuple[dict[str, list[int]], list[tuple]]:
    """({run/cell: [closing phases, repeats, would refuse, read]},
    [(run/cell, episode, repeats, would refuse)] where the two differ)."""
    from pasarbench.harness.closing import review, would_refuse
    from pasarbench.rescore import all_tasks
    tasks = all_tasks()
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    differ = []
    files = sorted({f for r in roots for f in
                    ([r] if r.is_file() else r.rglob("*.jsonl"))})
    for f in files:
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        head = next((r for r in recs if r.get("type") == "header"), None)
        if head is None or not any(r.get("type") == "step" and r.get("closing") for r in recs):
            continue
        cell = f"{f.parent.parent.name}/{f.parent.name}"
        c = counts[cell]
        c[0] += 1
        task = tasks.get(head.get("task_id"))
        guard = would_refuse(recs, task)
        r = review(recs, task)
        if guard is None or r.get("not_replayed") or task is None:
            continue            # not read: the calls do not replay
        c[3] += 1
        c[1] += len(r["repeated"])
        c[2] += len(guard)
        if len(guard) != len(r["repeated"]):
            differ.append((cell, f.stem, len(r["repeated"]), len(guard)))
    return dict(counts), differ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+", help="trace directories (or files)")
    a = ap.parse_args(argv)
    counts, differ = scan([Path(r) for r in a.roots])
    if not counts:
        raise SystemExit(f"no closing phases under {' '.join(a.roots)}")
    print(f"{'run/cell':36s} {'phases':>7s} {'read':>6s} {'repeats':>8s} {'would refuse':>13s}")
    for cell, (n, rep, ref, read) in sorted(counts.items()):
        print(f"{cell:36s} {n:7d} {read:6d} {rep:8d} {ref:13d}")
    print(f"\nepisodes where the guard and the review differ: {len(differ)}")
    for cell, ep, rep, ref in differ:
        print(f"  {cell:36s} {ep:24s} repeats {rep}, would refuse {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
