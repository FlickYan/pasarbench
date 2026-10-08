"""
One experiment arm against its control, paired by task.

    python scripts/compare_cells.py traces/J-control/full+search-300 \
                                    traces/J-named/full+search-300

Pass rates on today's checks (as RESULTS.md), the paired sign test over the
tasks both cells ran, cost, the traps that moved -- and, for each tool named
with --tools, how often an episode whose task REQUIRES that tool called it
successfully. That last line is the one the tool-naming experiment turns on
(docs/RUNBOOK.md, 1g): behind search the agent never looked for the tools the
policy describes but does not name, so if naming them is the mechanism, the
named arm calls them where the control did not.

Read the paired line before anything else. 32 tasks see large effects only,
and a gap the sign test does not resolve is not a result. Then read why the
remaining failures failed: `python scripts/audit_tool_arms.py <run>` says, for
each tool an episode needed and never used, whether it searched for it.

For the claim guardrail (docs/RUNBOOK.md, 1i) it also counts the replies the
customer got that claim a write no call had done, in both cells, two ways: by
the guardrail's own reading, and by the audit's, which reads phrasings the
guardrail does not and so can see what it misses. Grade a guardrail by the
second -- graded by its own reading it passes by construction (WHAT_FAILED
#34). For the guarded cell it lists every reply held back and what the agent
did next. Read the held replies: one that claims nothing, or claims what
happened before the conversation, is a false alarm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_TOOLS = "escalate_to_human,get_shipment,issue_goodwill_voucher"


def _calls(cell: Path) -> dict[str, list[tuple[str, bool, str]]]:
    """transcript_id -> [(tool, ok, error)] in order."""
    out = {}
    for f in sorted(cell.glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        out[f.stem] = [(tc["name"], bool(tr.get("ok")), tr.get("error") or "")
                       for st in recs if st.get("type") == "step"
                       for tc, tr in zip(st.get("tool_calls") or [],
                                         st.get("tool_results") or [])]
    return out


def _head(cell: Path) -> dict:
    f = next(iter(sorted(cell.glob("*.jsonl"))), None)
    if f is None:
        return {}
    return next((json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l).get("type") == "header"), {})


def _reviews(cell: Path, ids: set[str]) -> dict[str, dict]:
    """transcript_id -> guardrail.review of its steps, for finished episodes."""
    from pasarbench.harness.guardrail import review
    out = {}
    for f in sorted(cell.glob("*.jsonl")):
        if f.stem not in ids:
            continue
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        out[f.stem] = review([r for r in recs if r.get("type") == "step"])
    return out


def _audit_claims(cell: Path, eps: list[dict], tasks: dict) -> tuple[int, int]:
    """(failed English episodes read, of them claiming a required action the
    database never saw) -- scripts/audit_tool_arms.py's section 3, which reads
    passives and "we" for every action and so is not the guardrail's reading."""
    from pasarbench.claims import ENGLISH, false_claims
    read = claimed = 0
    for e in eps:
        task = tasks.get(e["task_id"])
        if e["passed"] or e.get("language") not in ENGLISH or task is None:
            continue
        recs = [json.loads(l) for l in (cell / f"{e['transcript_id']}.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        steps = [r for r in recs if r.get("type") == "step"]
        texts = [st["model_content"] for st in steps
                 if (st.get("model_content") or "").strip() and not st.get("guardrail")]
        calls = [(tc["name"], bool(tr.get("ok"))) for st in steps
                 for tc, tr in zip(st.get("tool_calls") or [], st.get("tool_results") or [])]
        read += 1
        claimed += bool(false_claims({"task": task, "texts": texts, "calls": calls}))
    return read, claimed


def summary(eps: list[dict]) -> dict:
    by: dict[str, list[bool]] = defaultdict(list)
    for e in eps:
        by[e["task_id"]].append(e["passed"])
    n = len(eps) or 1
    return {"episodes": len(eps), "tasks": len(by),
            "pass1": sum(e["passed"] for e in eps) / n,
            "passk": sum(all(v) for v in by.values()) / (len(by) or 1),
            "tokens": sum(e["tokens"] for e in eps) / n,
            "steps": sum(e["steps"] for e in eps) / n}


def tool_use(eps: list[dict], calls: dict, tool: str, tasks: dict) -> tuple[int, int, int]:
    """(episodes whose task requires `tool`, of them called it successfully,
    of them first tried it as an unknown tool)."""
    need = [e for e in eps if tasks.get(e["task_id"]) is not None and any(
        tool in a.tools for a in tasks[e["task_id"]].checks.required_actions)]
    used = sum(any(n == tool and ok for n, ok, _ in calls.get(e["transcript_id"], []))
               for e in need)
    refused = sum(any(n == tool and "unknown tool" in err
                      for n, _, err in calls.get(e["transcript_id"], [])) for e in need)
    return len(need), used, refused


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("control", help="the cell to compare against, e.g. "
                                    "traces/J-control/full+search-300")
    ap.add_argument("treatment", help="the experiment arm, e.g. "
                                      "traces/J-named/full+search-300")
    ap.add_argument("--tools", default=DEFAULT_TOOLS,
                    help="tools whose use to count among episodes that need them")
    ap.add_argument("--checker", choices=["current", "recorded"], default="current",
                    help="today's checks on the replayed episodes (default, as "
                         "RESULTS.md) or the verdicts the runs recorded")
    a = ap.parse_args(argv)
    from pasarbench.diagnose import load_episodes, paired_episodes
    from pasarbench.rescore import all_tasks

    cells = [Path(a.control), Path(a.treatment)]
    for c in cells:
        if not c.is_dir() or not any(c.glob("*.jsonl")):
            raise SystemExit(f"no traces in {c} (run from the repo root)")
    ea, eb = (load_episodes(c, checker=a.checker) for c in cells)
    tasks = all_tasks()
    heads = [_head(c) for c in cells]
    names = [f"{c.parent.name}/{c.name}" for c in cells]

    print(f"{'':12s} {names[0]:>34s} {names[1]:>34s}")
    for key in ("policy_mode", "exposure", "guardrail", "requested_model", "simulator"):
        default = "off" if key == "guardrail" else None
        print(f"{key:12s} {str(heads[0].get(key, default)):>34s} "
              f"{str(heads[1].get(key, default)):>34s}")
    sa, sb = summary(ea), summary(eb)
    for key, fmt in (("episodes", "{:d}"), ("tasks", "{:d}"), ("pass1", "{:.3f}"),
                     ("passk", "{:.3f}"), ("tokens", "{:,.0f}"), ("steps", "{:.1f}")):
        label = {"pass1": "pass^1", "passk": "pass^k", "tokens": "tokens/ep",
                 "steps": "steps/ep"}.get(key, key)
        print(f"{label:12s} {fmt.format(sa[key]):>34s} {fmt.format(sb[key]):>34s}")

    g = paired_episodes(ea, eb)
    print()
    if g is None:
        print("paired: fewer than 5 shared tasks -- not testable")
    else:
        verdict = (("BETTER" if g["diff"] > 0 else "WORSE") if g["resolved"]
                   else "INCONCLUSIVE")
        print(f"paired by task ({g['n']} shared): {g['diff']:+.3f}; {g['better']} tasks "
              f"better, {g['worse']} worse, sign test p={g['p']:.3f}  ->  {verdict}")
    t = paired_episodes(ea, eb, metric="tokens")
    if t:
        print(f"tokens, paired: {t['diff']:+,.0f} per episode; {t['better']} tasks "
              f"costlier, {t['worse']} cheaper, p={t['p']:.3f}")

    ca, cb = _calls(cells[0]), _calls(cells[1])
    print("\ncalled successfully, of the episodes whose task requires the tool "
          "(first tried as an unknown tool):")
    for tool in [x.strip() for x in a.tools.split(",") if x.strip()]:
        na, ua, ra = tool_use(ea, ca, tool, tasks)
        nb, ub, rb = tool_use(eb, cb, tool, tasks)
        print(f"  {tool:28s} {f'{ua}/{na} ({ra})':>18s} {f'{ub}/{nb} ({rb})':>18s}")

    va, vb = (_reviews(c, {e["transcript_id"] for e in eps}) for c, eps in zip(cells, (ea, eb)))
    told = [sum(bool(r["delivered"]) for r in rv.values()) for rv in (va, vb)]
    (na, ca_), (nb, cb_) = (_audit_claims(c, eps, tasks) for c, eps in zip(cells, (ea, eb)))
    print("\nreplies the customer got that claim a write no call had done (English):")
    print(f"  {'guardrail reading, episodes':28s} {f'{told[0]}/{len(va)}':>18s} "
          f"{f'{told[1]}/{len(vb)}':>18s}")
    print(f"  {'audit reading, failures read':28s} {f'{ca_}/{na}':>18s} {f'{cb_}/{nb}':>18s}")
    for name, cell, rv in zip(names, cells, (va, vb)):
        held = [(tid, t) for tid, r in rv.items() for t in r["held"]]
        if not held:
            continue
        outs = Counter(o for r in rv.values() for o in r["outcomes"])
        print(f"\nguardrail in {name}: held back {len(held)} replies in "
              f"{len({tid for tid, _ in held})} episodes. After the note, for each "
              f"action a held reply claimed:")
        for o in ("made the call", "reworded", "claimed again", "ended"):
            print(f"  {o:28s} {outs.get(o, 0):>6d}")
        print("  the held replies -- read them; one that claims nothing, or what "
              "happened before the conversation, is a false alarm:")
        for tid in sorted({tid for tid, _ in held})[:40]:
            recs = [json.loads(l) for l in (cell / f"{tid}.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]
            for st in recs:
                if st.get("type") != "step":
                    continue
                for tool, quote in (st.get("guardrail") or {}).get("claims", []):
                    print(f"    {tid:24s} {tool}: \"{quote[:110]}\"")

    def per_trap(eps):
        d: dict[str, list[bool]] = defaultdict(list)
        for e in eps:
            d[e["trap"]].append(e["passed"])
        return {k: sum(v) / len(v) for k, v in d.items()}
    pa, pb = per_trap(ea), per_trap(eb)
    moved = sorted((pb[k] - pa[k], k) for k in pa if k in pb and abs(pb[k] - pa[k]) > 1e-9)
    print("\ntraps that moved (leads to read in the traces, not results):")
    for d, k in moved or [(0.0, "(none)")]:
        if k == "(none)":
            print("  none")
            break
        print(f"  {k:40s} {pa[k]:.2f} -> {pb[k]:.2f}  ({d:+.2f})")
    print(f"\nwhy each remaining failure failed: python scripts/audit_tool_arms.py "
          f"{cells[1].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
