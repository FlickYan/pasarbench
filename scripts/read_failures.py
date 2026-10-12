"""
Every episode a run failed on a task its control passed every time, set out to
be read as WHAT_FAILED #31 and #44 read them.

    python scripts/read_failures.py traces/P2-base/full traces/P2-rft/full
    python scripts/read_failures.py traces/P2-rft/full traces/P2-grpo-i1/full \\
        --trap customs_hold_escalate
    python scripts/read_failures.py <control> <treatment> --all   # every failure

A fine-tune's net change can be several moves in opposite directions (#31), and
the move worth finding is another trap's answer in the wrong place: the
livestream trap's refund on an out-of-window dispute, the peak sale's "wait" on
a customs hold, the photo rule on a cheap cash-on-delivery item (#44). No
pattern decides that; reading does. This prints what the reading needs, in the
order it is read:

  1. per trap, how many failures there are to read, and the writes made in
     them that the control never made on the same tasks. Another trap's answer
     usually shows first as a write that does not belong there -- an
     escalation on the voucher trap, a refund on a dispute, a seller message on
     a customs hold;
  2. each failure: its task and the control's passes on it, the writes with
     the arguments that say what they were for, the failure lines, a flag when
     the customer said she sent photos she was told she does not have (#43),
     and the conversation -- the customer's lines and the agent's, cut to
     --width.

Verdicts are today's checks (as RESULTS.md and compare_cells.py); with
`--checker recorded`, the ones the runs recorded, which is faster.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The arguments that say what a write was for, in the order they are shown.
SHOWN_ARGS = ("category", "amount_minor", "currency", "method", "order_id", "reason",
              "message")


def _short(text: str, width: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[:width - 1] + "…"


def _write(name: str, args: dict, width: int) -> str:
    shown = []
    for key in SHOWN_ARGS:
        if key in args:
            value = args[key]
            shown.append(f"{key}={_short(value, 60) if isinstance(value, str) else value}")
    return _short(f"{name}({', '.join(shown)})", width)


def episodes(cell: Path, checker: str, tasks: dict) -> list[dict]:
    """One record per finished episode: verdict, failures, writes, conversation."""
    from pasarbench.rescore import rescore_file
    from pasarbench.simqa import declined_voucher, photo_claim
    from pasarbench.tools import WRITE_TOOLS
    out = []
    for f in sorted(cell.glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        head = next((r for r in recs if r.get("type") == "header"), {})
        foot = next((r for r in reversed(recs) if r.get("type") == "footer"), None)
        if foot is None:
            continue                        # never finished: not a result
        if checker == "current":
            rs = rescore_file(f)
            passed, failures = rs.verdict, rs.failures
        else:
            passed, failures = bool(foot.get("passed")), foot.get("failures") or []
        task_id = head.get("task_id", f.stem.split("__")[0])
        task = tasks.get(task_id)
        writes, lines = [], []
        if task is not None:
            lines.append(("C", task.opening))
        for r in recs:
            if r.get("type") == "event" and r.get("kind") == "user_turn":
                lines.append(("C", r.get("text") or ""))
            elif r.get("type") == "step" and not r.get("closing"):
                made = []
                for tc, tr in zip(r.get("tool_calls") or [], r.get("tool_results") or []):
                    if tc["name"] in WRITE_TOOLS:
                        w = (tc["name"], tc.get("arguments") or {}, bool(tr.get("ok")))
                        writes.append(w)
                        made.append(w)
                text = (r.get("model_content") or "").strip()
                if text or made:
                    lines.append(("A", text, made))
        turns = [r.get("text") or "" for r in recs
                 if r.get("type") == "event" and r.get("kind") == "user_turn"]
        out.append({"id": f.stem, "task_id": task_id, "trap": head.get("trap", "?"),
                    "language": head.get("language", "?"), "passed": passed,
                    "failures": failures, "writes": writes, "lines": lines,
                    "stop": foot.get("stop_reason", "?"),
                    "photos": task is not None and photo_claim(task, turns) is not None,
                    "voucher": declined_voucher(head.get("trap", ""), recs)})
    return out


def select(control: list[dict], treatment: list[dict], every: bool = False,
           trap: str = "") -> list[tuple[dict, int, int]]:
    """(episode, the control's passes on its task, the control's episodes on it)
    for each treatment failure on a task the control passed every time -- or on
    any task the control ran, with `every`."""
    by_task: dict[str, list[bool]] = defaultdict(list)
    for e in control:
        by_task[e["task_id"]].append(e["passed"])
    out = []
    for e in treatment:
        if e["passed"] or (trap and e["trap"] != trap):
            continue
        ran = by_task.get(e["task_id"])
        if not ran:
            continue
        if every or all(ran):
            out.append((e, sum(ran), len(ran)))
    return sorted(out, key=lambda x: (x[0]["trap"], x[0]["id"]))


def new_writes(rows: list[tuple[dict, int, int]], control: list[dict]) -> dict[str, Counter]:
    """Per trap: the writes in the failures read that no control episode on the
    same tasks made -- counted by tool, and by category for an escalation."""
    def kinds(e: dict) -> set[str]:
        return {(f"{n}[{a.get('category')}]" if n == "escalate_to_human" else n)
                for n, a, ok in e["writes"] if ok}
    tasks = {e["task_id"] for e, _, _ in rows}
    made: dict[str, set[str]] = defaultdict(set)
    for e in control:
        if e["task_id"] in tasks:
            made[e["task_id"]] |= kinds(e)
    out: dict[str, Counter] = defaultdict(Counter)
    for e, _, _ in rows:
        for k in kinds(e) - made[e["task_id"]]:
            out[e["trap"]][k] += 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("control", help="the cell compared against, e.g. traces/P2-base/full")
    ap.add_argument("treatment", help="the cell whose failures are read")
    ap.add_argument("--all", action="store_true",
                    help="every failure on a task the control ran, not only the tasks it "
                         "always passed")
    ap.add_argument("--trap", default="", help="one trap only")
    ap.add_argument("--width", type=int, default=160, help="characters a line keeps")
    ap.add_argument("--no-conversation", action="store_true",
                    help="the writes and failure lines only")
    ap.add_argument("--checker", choices=["current", "recorded"], default="current")
    a = ap.parse_args(argv)
    from pasarbench.rescore import all_tasks

    cells = [Path(a.control), Path(a.treatment)]
    for c in cells:
        if not c.is_dir() or not any(c.glob("*.jsonl")):
            raise SystemExit(f"no traces in {c} (run from the repo root)")
    tasks = all_tasks()
    control, treatment = (episodes(c, a.checker, tasks) for c in cells)
    rows = select(control, treatment, a.all, a.trap)
    names = [f"{c.parent.name}/{c.name}" for c in cells]
    which = ("on a task the control ran" if a.all
             else "on a task the control passed every time")
    print(f"{names[1]}: {len(rows)} failure(s) {which} in {names[0]}"
          + (f", trap {a.trap}" if a.trap else ""))
    if not rows:
        return 0

    fresh = new_writes(rows, control)
    by_trap = Counter(e["trap"] for e, _, _ in rows)
    print("\nper trap: failures to read, and writes in them the control never made on "
          "those tasks")
    for trap, n in sorted(by_trap.items(), key=lambda kv: (-kv[1], kv[0])):
        new = ", ".join(f"{k} {v}" for k, v in fresh[trap].most_common()) or "-"
        print(f"  {trap:40s} {n:3d}   {new}")
    photos = sum(e["photos"] for e, _, _ in rows)
    if photos:
        print(f"\n  {photos} of them after the customer said she sent photos she was told "
              f"she does not have (WHAT_FAILED #43): the verdict there is hers")
    vouch = sum(e["voucher"] for e, _, _ in rows)
    if vouch:
        print(f"\n  {vouch} of them escalated after the customer was offered the voucher and "
              f"answered (WHAT_FAILED #46): read her answer -- where she declined it for her "
              f"money or a human, P6.3 and P10 ask for the escalation the check fails")

    w = a.width
    for e, p, n in rows:
        print("\n" + "=" * min(w, 100))
        print(f"{e['id']}  [{e['trap']}, {e['language']}]  control {p}/{n}  stop={e['stop']}"
              + ("  CUSTOMER CLAIMED PHOTOS" if e["photos"] else "")
              + ("  ESCALATED AFTER THE VOUCHER OFFER" if e["voucher"] else ""))
        made = [_write(nm, args, w) + ("" if ok else " (refused)")
                for nm, args, ok in e["writes"]]
        print("  writes: " + ("; ".join(made) if made else "none"))
        for f in e["failures"]:
            print("  failed: " + _short(f, w - 10))
        if a.no_conversation:
            continue
        for line in e["lines"]:
            if line[0] == "C":
                print("  C: " + _short(line[1], w - 5))
            else:
                text, calls = line[1], line[2]
                shown = _short(text, w - 5) if text else ""
                if calls:
                    shown += ("  >> " if shown else ">> ") + "; ".join(
                        _write(nm, args, 80) + ("" if ok else " (refused)")
                        for nm, args, ok in calls)
                print("  A: " + shown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
