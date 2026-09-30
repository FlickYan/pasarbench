"""
Re-score recorded runs against today's checks, and show what moved.

    python scripts/rescore.py                        # every run under traces/
    python scripts/rescore.py traces/P-base traces/P-rft --changed
    python scripts/rescore.py traces/I-tools2 --json rescored.json

Each episode's tool calls are replayed against the world it ran in, and
today's verifier scores the rebuilt state (pasarbench/rescore.py). No model is
called and nothing is written next to the traces. An episode whose replay does
not match its recording keeps its recorded verdict and is counted as not
re-scorable; RESULTS.md, which make_report.py builds on today's checks, counts
it the same way.

Read the per-trap lines first. A checker change should move the traps whose
checks changed and no others; a move anywhere else means the replay rebuilt a
different state than the run ended in, and the episode is worth reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pasarbench.rescore import cells, rescore_dir, summary   # noqa: E402


def _name(cell: Path) -> str:
    """run/cell, as the report names it."""
    parts = cell.resolve().parts
    if "traces" in parts:
        i = len(parts) - 1 - parts[::-1].index("traces")
        return "/".join(parts[i + 1:])
    return "/".join(parts[-2:])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", default=["traces"],
                    help="traces/, a run (traces/P-base) or a cell (traces/P-base/full)")
    ap.add_argument("--changed", action="store_true",
                    help="list every episode whose verdict moved, and every one that "
                         "could not be re-scored")
    ap.add_argument("--json", default="", help="write every episode's row here")
    a = ap.parse_args(argv)

    todo = []
    for p in a.paths:
        path = Path(p)
        if not path.is_dir():
            raise SystemExit(f"no directory {path} (run from the repo root, the folder "
                             f"that holds traces/)")
        todo += [(c, _name(c)) for c in cells(path)]
    if not todo:
        raise SystemExit(f"no traces under {', '.join(a.paths)}")

    print(f"{'cell':34s} {'episodes':>8s}  {'pass^1 recorded → now':>22s}  "
          f"{'pass^k recorded → now':>22s}  {'fail→pass':>9s} {'pass→fail':>9s} "
          f"{'not re-scored':>13s}  payloads verified")
    everything, by_trap = [], defaultdict(Counter)
    for cell, name in todo:
        rows = rescore_dir(cell)
        s = summary(rows)
        if not s["episodes"]:
            continue
        everything += [(name, r) for r in rows]
        for r in rows:
            if r.moved:
                by_trap[r.trap][r.moved] += 1
        v, t = s["payloads"]
        print(f"{name:34s} {s['episodes']:8d}  "
              f"{s['recorded']:10.3f} → {s['current']:.3f}  "
              f"{s['recorded_k']:10.3f} → {s['current_k']:.3f}  "
              f"{s['fail_to_pass']:9d} {s['pass_to_fail']:9d} {s['not_rescorable']:13d}  "
              f"{v}/{t}")

    print("\nmoved, by trap (all cells):")
    if not by_trap:
        print("  nothing moved: today's checks give every episode its recorded verdict")
    for trap, c in sorted(by_trap.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {trap:40s} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    worse = sum(c["pass→fail"] for c in by_trap.values())
    if worse:
        print(f"\n  {worse} episode(s) went from pass to fail. If no check got stricter, "
              f"the replay rebuilt a different state than the run ended in: read them "
              f"with --changed.")

    why = Counter(r.status for _, r in everything if r.current is None)
    if why:
        print("\nkept at their recorded verdict: " + ", ".join(
            f"{k} {v}" for k, v in why.most_common()))
        print("  diverged: a call succeeds on replay where it failed in the run, or the "
              "reverse\n  world_changed: a call's payload differs from the recording (an "
              "older world)\n  task_changed: the task's opening or facts changed since the "
              "run\n  unknown_task: the task is not in today's suite")

    if a.changed:
        print("\nepisodes that moved or could not be re-scored:")
        for name, r in everything:
            if r.moved or r.current is None:
                first = (r.failures_recorded or ["-"])[0]
                now = (r.failures_current or ["passes"])[0] if r.current is not None \
                    else r.detail
                print(f"  {name}/{r.transcript_id}: {r.moved or r.status}\n"
                      f"      recorded: {str(first)[:150]}\n      now:      {str(now)[:150]}")
    if a.json:
        Path(a.json).write_text(json.dumps([{"cell": n, **r.to_dict()} for n, r in everything],
                                           ensure_ascii=False, indent=1))
        print(f"\nwrote {len(everything)} rows to {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
