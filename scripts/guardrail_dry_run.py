"""
What the claim guardrail would have done to runs already recorded, before
paying for a guarded one (docs/RUNBOOK.md, 1i).

For every finished episode under the given trace directories: did a reply the
customer saw claim, in the first person and in English, a write action that no
successful call had done when it went out? That is exactly the reply
`--guardrail claims` holds back, so the count is how often it would have
fired, and the list is what to read for false alarms.

    python scripts/guardrail_dry_run.py traces            # every run
    python scripts/guardrail_dry_run.py traces/I-tools2   # one run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def scan(roots: list[Path]) -> tuple[dict[str, list[int]], list[tuple]]:
    """({run/cell: [finished, English, fired]}, [(run/cell, episode, passed, tool, quote)])."""
    from pasarbench.claims import ENGLISH
    from pasarbench.harness.guardrail import review
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    hits = []
    files = sorted({f for r in roots for f in
                    ([r] if r.is_file() else r.rglob("*.jsonl"))})
    for f in files:
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        head = next((r for r in recs if r.get("type") == "header"), None)
        foot = next((r for r in reversed(recs) if r.get("type") == "footer"), None)
        if head is None or foot is None:
            continue            # unfinished: no verdict to read it against
        cell = f"{f.parent.parent.name}/{f.parent.name}"
        c = counts[cell]
        c[0] += 1
        c[1] += head.get("language") in ENGLISH
        said = review([r for r in recs if r.get("type") == "step"])["delivered"]
        if said:
            c[2] += 1
            hits.append((cell, f.stem, foot.get("passed"), *said[0]))
    return dict(counts), hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+", help="trace directories (or files)")
    a = ap.parse_args(argv)
    counts, hits = scan([Path(r) for r in a.roots])
    if not counts:
        raise SystemExit(f"no finished episodes under {' '.join(a.roots)}")
    print(f"{'run/cell':40s} {'episodes':>9s} {'English':>8s} {'would fire':>11s}")
    for cell, (n, en, fired) in sorted(counts.items()):
        print(f"{cell:40s} {n:9d} {en:8d} {fired:11d}")
    tot = [sum(v[i] for v in counts.values()) for i in range(3)]
    print(f"{'total':40s} {tot[0]:9d} {tot[1]:8d} {tot[2]:11d}")
    print("\nthe replies it would have held back -- read each: one the policy would call "
          "true is a false alarm")
    for cell, ep, passed, tool, quote in hits:
        verdict = "passed" if passed else "failed"
        print(f"  {cell:30s} {ep:22s} {verdict:6s} {tool}: \"{quote[:100]}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
