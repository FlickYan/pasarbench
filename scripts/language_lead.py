"""
The duplicate-refund trap by language (docs/RUNBOOK.md, 1r).

P10 says to escalate a claim for a refund already paid "even if you have
explained the existing refund and the customer accepts it". The trap fails far
more often outside English (1j, 1m). This reads every episode under the given
trace roots and asks of each conversation -- the closing phase left out: it
comes after the customer has gone -- whether the agent escalated while the
customer was there and, in this trap, when: in the turn it found the refund
(its first get_order, whose result lists the refund already paid), before the
customer said anything more, or only after she had.

For each agent model, over the run cells that ran the trap in English and in
another language both, it prints the escalation rate
  in each cell, English and Singlish against the other languages
  in each language, those cells together
  of each local-language task against its English twin -- the same order, the
  same refund, the same story told in English -- in the cells that ran both,
  with a sign test over the twins
  English against the rest again within each thing that differs before the
  decision: the path to the lookup (customer turns before it, identity checked
  first or not), the customer simulator, the tool exposure
then the same twin comparison for every trap the policy answers with an
escalation -- whether a language costs escalation in general, or this one rule
-- and for every trap on the verdicts the runs recorded, in the cells without a
closing phase: the table by trap that a comparison averaged over the suite
cannot show (WHAT_FAILED #40).

    python scripts/language_lead.py traces
    python scripts/language_lead.py traces/Q-off traces/Q-check

Read the twins first. A language low everywhere could be one hard world; a
local twin below its English twin in world after world is the language.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAP = "duplicate_refund_escalate"
LOOKUP = "get_order"         # its result lists the refund already paid (existing_refunds)
ENGLISH = ("en", "sg-en")
BEFORE, AFTER = "before she said more", "after she said more"

Pair = tuple[str, str, tuple[int, int], tuple[int, int]]


def conversation(recs: list[dict]) -> dict[str, Any]:
    """What the agent did while the customer was there: whether it escalated,
    and when against its first lookup of the order; and the path to that
    lookup -- the customer turns before it, and whether the agent checked her
    identity first."""
    turn, look, esc, checked = 0, None, None, False
    for r in recs:
        if r.get("type") == "event" and r.get("kind") == "user_turn":
            turn += 1
        if r.get("type") != "step" or r.get("closing"):
            continue
        for tc, tr in zip(r.get("tool_calls") or [], r.get("tool_results") or []):
            if not tr.get("ok"):
                continue
            name = tc.get("name")
            if name == "verify_identity" and look is None:
                checked = True
            if name == LOOKUP and look is None:
                look = turn
            if name == "escalate_to_human" and esc is None:
                esc = turn
    return {"escalated": esc is not None,
            "when": None if esc is None else BEFORE if look is None or esc <= look else AFTER,
            "path": None if look is None else (look, checked)}


def scan(roots: list[Path]) -> list[dict]:
    """One row per episode: model, cell, task, trap, language, simulator,
    exposure, closing mode, the verdict it recorded, and what its
    conversation did."""
    files = sorted({f for r in roots for f in ([r] if r.is_file() else r.rglob("*.jsonl"))})
    rows = []
    for f in files:
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not recs or recs[0].get("type") != "header":
            continue
        h = recs[0]
        foot = recs[-1] if recs[-1].get("type") == "footer" else {}
        rows.append({"model": h.get("requested_model") or "?",
                     "cell": f"{f.parent.parent.name}/{f.parent.name}",
                     "task": h.get("task_id") or f.stem.rsplit("__", 1)[0],
                     "trap": h.get("trap") or "?", "language": h.get("language") or "?",
                     "simulator": h.get("simulator") or "?",
                     "exposure": h.get("exposure") or "default",
                     "closing": h.get("closing") or "off", "passed": foot.get("passed"),
                     **conversation(recs)})
    return rows


def english(r: dict) -> bool:
    return r["language"] in ENGLISH


def rate(rows: list[dict], key: str = "escalated") -> tuple[int, int]:
    """(episodes where `key` holds -- escalated while the customer was there,
    by default -- of how many)."""
    return sum(bool(r[key]) for r in rows), len(rows)


def twins(rows: list[dict], key: str = "escalated") -> list[Pair]:
    """(English task, local task, the local task's rate, its English twin's)
    for every local-language task, over the cells that ran both."""
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[r["task"], r["cell"]].append(r)
    cells: dict[str, set[str]] = defaultdict(set)
    for task, cell in by:
        cells[task].add(cell)
    out = []
    for task in sorted(cells):
        world = task.split(".")[0]
        shared = sorted(cells[task] & cells.get(world, set()))
        if task != world and shared:
            out.append((world, task, rate([r for c in shared for r in by[task, c]], key),
                        rate([r for c in shared for r in by[world, c]], key)))
    return out


def gap(pairs: list[Pair]) -> float:
    """The mean of local twin minus English twin, as a fraction."""
    return sum(_frac(l) - _frac(e) for *_, l, e in pairs) / len(pairs) if pairs else 0.0


def signs(pairs: list[Pair]) -> str:
    from pasarbench.diagnose import _sign_p

    lower = sum(_frac(l) < _frac(e) for *_, l, e in pairs)
    higher = sum(_frac(l) > _frac(e) for *_, l, e in pairs)
    return (f"below their English twin {lower} of {len(pairs)}, above {higher}, level "
            f"{len(pairs) - lower - higher}; mean gap {round(100 * gap(pairs)):+d} points; "
            f"sign test p={_sign_p(higher, lower):.4f}")


def escalation_traps() -> list[str]:
    """Every trap whose check requires an escalation."""
    from pasarbench.rescore import all_tasks

    return sorted({t.trap for t in all_tasks().values()
                   for a in t.checks.required_actions if a.tool == "escalate_to_human"})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+", help="trace directories (or files)")
    a = ap.parse_args(argv)

    rows = scan([Path(r) for r in a.roots])
    if not any(r["trap"] == TRAP for r in rows):
        raise SystemExit(f"no episodes of {TRAP} under {' '.join(a.roots)}")
    for model in sorted({r["model"] for r in rows if r["trap"] == TRAP}):
        mine = [r for r in rows if r["model"] == model]
        trap = [r for r in mine if r["trap"] == TRAP]
        cells = sorted({r["cell"] for r in trap if english(r)}
                       & {r["cell"] for r in trap if not english(r)})
        both = [r for r in trap if r["cell"] in cells]
        print(f"\n== {model}: {len(trap)} episodes of the duplicate-refund trap in "
              f"{len({r['cell'] for r in trap})} cells, {len(both)} of them in the "
              f"{len(cells)} cells that ran it in English and another language both")
        alone = sorted({r["cell"] for r in trap} - set(cells))
        if alone:
            print(f"  left out, one side only: {', '.join(alone)}")
        if not both:
            continue
        print("escalated while the customer was there:")
        print(f"  {'run/cell':36s} {'English and Singlish':>22s} {'the other languages':>22s}")
        for cell in cells:
            here = [r for r in both if r["cell"] == cell]
            print(f"  {cell:36s} {_pct(rate([r for r in here if english(r)])):>22s} "
                  f"{_pct(rate([r for r in here if not english(r)])):>22s}")
        print(f"  by language, those cells together -- and how many escalated {BEFORE}, "
              f"in the turn it found the refund:")
        langs = {r["language"] for r in both}
        for lang in sorted(langs, key=lambda l: (-_frac(rate([r for r in both
                                                                if r["language"] == l])), l)):
            these = [r for r in both if r["language"] == lang]
            print(f"    {lang:8s} {_pct(rate(these)):>14s}   "
                  f"{sum(r['when'] == BEFORE for r in these)} of them {BEFORE}")
        pairs = twins(both)
        if pairs:
            print("  by world -- each local-language task against its English twin, the same "
                  "order and refund, in the cells that ran both:")
            for world, task, local, en in pairs:
                print(f"    {task:20s} {_pct(local):>14s}   English twin {world}: {_pct(en)}")
            print(f"  {signs(pairs)}")
        print("  English and Singlish against the rest, within each thing that differs "
              "before the decision:")
        _split(both, lambda r: _path(r["path"]), "the path to the lookup")
        _split(both, lambda r: r["simulator"], "the customer simulator")
        _split(both, lambda r: r["exposure"], "the tool exposure")

        print("  every trap the policy answers with an escalation -- each local-language "
              "task against its English twin, in the cells that ran both:")
        for name in escalation_traps():
            pairs = twins([r for r in mine if r["trap"] == name])
            print(f"    {name:38s} {signs(pairs) if pairs else '-'}")
        judged = [r for r in mine if r["closing"] == "off" and r["passed"] is not None]
        table = [(name, twins([r for r in judged if r["trap"] == name], "passed"))
                 for name in sorted({r["trap"] for r in judged})]
        table = sorted([(n, p) for n, p in table if p], key=lambda t: (gap(t[1]), t[0]))
        if table:
            print(f"  every trap, passed -- on the verdicts the runs recorded, in the cells "
                  f"without a closing phase; with {len(table)} traps, a gap stands on its own "
                  f"at p < {0.05 / len(table):.4f}:")
            for name, pairs in table:
                print(f"    {name:38s} {signs(pairs)}")
    return 0


def _split(rows: list[dict], key: Callable[[dict], Any], label: str) -> None:
    print(f"    {label + ':':60s} {'English':>14s} {'the rest':>14s}")
    for k in sorted({key(r) for r in rows} - {None}, key=str):
        these = [r for r in rows if key(r) == k]
        print(f"      {str(k):58s} {_pct(rate([r for r in these if english(r)])):>14s} "
              f"{_pct(rate([r for r in these if not english(r)])):>14s}")


def _path(path: tuple[int, bool] | None) -> str | None:
    if path is None:
        return None
    turns, checked = path
    return (f"{min(turns, 2)}{'+' if turns >= 2 else ''} customer turn"
            f"{'' if turns == 1 else 's'} before it, identity "
            f"{'checked' if checked else 'not checked'} first")


def _frac(r: tuple[int, int]) -> float:
    return r[0] / r[1] if r[1] else 0.0


def _pct(r: tuple[int, int]) -> str:
    return f"{r[0]}/{r[1]} {_frac(r):4.0%}" if r[1] else "-"


if __name__ == "__main__":
    raise SystemExit(main())
