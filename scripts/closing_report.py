"""
One run's closing phase, read on its own (docs/RUNBOOK.md, 1p).

The closing phase acts only after the customer has left, so its whole effect
on a case is the case as she left it against the case at the end -- one
episode, the conversation held fixed (closing.review). Measuring that needs no
control arm. For one cell this prints its pass rate, what the closing phase did
and what it cost -- the same section compare_cells.py prints for each of two
cells -- then every call the guard refused beside the call it would have
repeated, every payment of another amount beside the one it came after (since
v41 no repeat: a correction, or a second payment), the failures the customer
ended, and the pass rate by trap.

    python scripts/closing_report.py traces/T-guard/full+search-300

A refusal is the guard working when the refused call is the one already made
again; it is the guard's cost when the call was a different write of the same
kind -- another amount, a message with something new in it, a correction the
policy asks for. The two are told apart by reading, which is why each refusal
is printed beside what it would have repeated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def refusals(cell: Path, eps: list[dict], tasks: dict) -> list[tuple[str, str, str]]:
    """(transcript, the refused call, the earlier call it would have repeated)
    for every refusal in the cell, as text. The earlier call is read off the
    action log the episode ended with: a write with the same target after the
    refusal would have been refused too, so the match can only be earlier. A
    run under the guard before v41 refused a payment of another amount as well
    (WHAT_FAILED #39): its refusal is shown beside the payment it was taken
    for, which today's target no longer calls the same."""
    from pasarbench.harness.closing import _shown, paid_before, repeat_of
    from pasarbench.harness.replay import replay

    def call(tool, args) -> str:
        return f"{tool}(" + ", ".join(f"{k}={_shown(v)}" for k, v in (args or {}).items()
                                      if v is not None) + ")"

    out = []
    for e in eps:
        recs = [json.loads(l) for l in (cell / f"{e['transcript_id']}.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        refused = [r for r in recs if r.get("type") == "event" and r.get("kind") == "closing_refused"]
        if not refused:
            continue
        task = tasks.get(e["task_id"])
        log = []
        if task is not None:
            head = [r for r in recs if r.get("type") == "header"][:1]
            steps = [r for r in recs if r.get("type") == "step"]
            log = replay(head + steps, task).db.action_log
        for r in refused:
            done = (repeat_of(log, r.get("tool"), r.get("args"))
                    or paid_before(log, r.get("tool"), r.get("args")))
            out.append((e["transcript_id"], call(r.get("tool"), r.get("args")),
                        call(done.tool, done.args) if done is not None else "(not found)"))
    return out


def topped_up(cell: Path, eps: list[dict], tasks: dict, cv: dict[str, dict]
              ) -> list[tuple[str, str, str]]:
    """(transcript, the payment of another amount, the payment it came after)
    for every payment closing.review lists as `another_amount` -- since v41 no
    repeat, and read: the rest of what is owed, or a second payment."""
    from pasarbench.harness.closing import _shown, paid_before
    from pasarbench.harness.replay import replay

    def call(tool, args) -> str:
        return f"{tool}(" + ", ".join(f"{k}={_shown(v)}" for k, v in (args or {}).items()
                                      if v is not None) + ")"

    out = []
    for e in eps:
        r = cv.get(e["transcript_id"], {})
        if not r.get("another_amount"):
            continue
        recs = [json.loads(l) for l in (cell / f"{e['transcript_id']}.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        head = [x for x in recs if x.get("type") == "header"][:1]
        steps = [x for x in recs if x.get("type") == "step"]
        log = replay(head + steps, tasks[e["task_id"]]).db.action_log
        for tool, args in r["another_amount"]:
            before = paid_before(log, tool, args)
            out.append((e["transcript_id"], call(tool, args),
                        call(before.tool, before.args) if before is not None else "(not found)"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cell", help="one run's cell, e.g. traces/T-guard/full+search-300")
    ap.add_argument("--checker", choices=["current", "recorded"], default="current",
                    help="today's checks on the replayed episodes (default) or the "
                         "verdicts the run recorded")
    ap.add_argument("--endings", action="store_true",
                    help="list every episode the customer ended, passes too")
    a = ap.parse_args(argv)
    import compare_cells as cc
    from pasarbench.diagnose import load_episodes
    from pasarbench.rescore import all_tasks

    cell = Path(a.cell)
    if not cell.is_dir() or not any(cell.glob("*.jsonl")):
        raise SystemExit(f"no traces in {cell} (run from the repo root)")
    eps = load_episodes(cell, checker=a.checker)
    tasks = all_tasks()
    head = cc._head(cell)
    name = f"{cell.parent.name}/{cell.name}"

    print(f"{'':16s} {name}")
    defaults = {"guardrail": "off", "closing": "off", "tool_checks": 1}
    for key in ("policy_mode", "exposure", "guardrail", "closing", "tool_checks",
                "requested_model", "simulator"):
        print(f"{key:16s} {cc._shown_head(head, key, defaults.get(key))}")
    s = cc.summary(eps)
    print(f"{'episodes':16s} {s['episodes']}\n{'tasks':16s} {s['tasks']}\n"
          f"{'pass^1':16s} {s['pass1']:.3f}\n{'pass^k':16s} {s['passk']:.3f}\n"
          f"{'tokens/ep':16s} {s['tokens']:,.0f}\n{'steps/ep':16s} {s['steps']:.1f}")
    broken = sum(e.get("stop_reason") == "backend_error" for e in eps)
    if broken:
        print(f"!! {broken} episode{'s' if broken != 1 else ''} ended with a server or API "
              f"error (backend_error) -- run the same sweep again with --resume, then read "
              f"this again.")

    cv = cc._closings(cell, eps, tasks)
    if not any(r["noted"] for r in cv.values()):
        print("\nno closing phase in this cell: the customer never left, or the run had "
              "--closing off")
    cc._print_closing(name, cv, mode=head.get("closing"))
    rows = refusals(cell, eps, tasks)
    if rows or head.get("closing") == "guard":
        print(f"\nevery call the guard refused, beside the call it would have repeated -- "
              f"{len(rows)}; read each: the same call again is the guard working, a different "
              f"write of the same kind (another amount, something new to say, a correction "
              f"the policy asks for) is its cost")
        for tid, refused, done in rows:
            print(f"    {tid:24s} refused {refused}\n    {'':24s} after   {done}")
    more = topped_up(cell, eps, tasks, cv)
    if more:
        print(f"\nevery payment of another amount after one already made, beside the one it "
              f"came after -- {len(more)}; read each: the rest of what the policy says is "
              f"owed (the shipping P4.6 refunds, say) is right, a second payment it does not "
              f"ask for is a cost")
        for tid, made, before in more:
            print(f"    {tid:24s} paid    {made}\n    {'':24s} after   {before}")
    cc._print_endings(name, cc._endings(cell, eps, every=a.endings), every=a.endings)
    cc._print_settlements(name, *cc._settlements(cell, eps))

    by: dict[str, list[bool]] = defaultdict(list)
    for e in eps:
        by[e["trap"]].append(e["passed"])
    print("\npass^1 by trap:")
    for trap, v in sorted(by.items(), key=lambda kv: (sum(kv[1]) / len(kv[1]), kv[0])):
        print(f"  {trap:40s} {sum(v):>4d}/{len(v):<4d} {sum(v) / len(v):.2f}")
    print(f"\nwhy each remaining failure failed: python scripts/audit_tool_arms.py "
          f"{cell.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
