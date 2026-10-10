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

For the closing check (docs/RUNBOOK.md, 1j) it says what the agent did after
the customer left: the required actions first made then -- the mechanism --
and the forbidden calls first made then -- the cost -- read off the database's
action log as the verifier reads it, and how many closing phases were cut
short. Since 1j it also counts the two costs that run found and its rule had
not: a write made a second time, and a verification tried with digits the
customer never gave. Since 1k, each episode's verdict as the customer left it
beside its verdict at the end -- the phase's effect with the conversation held
fixed -- the paired comparison with every case as she left it, which is the
conversations' difference alone, the writes no check scores either way, and
under each episode listed what she last said. Read all of them before the
pass rate: a rescue after "yes, go ahead, thanks" is the turn the episode's
end never gave the agent (WHAT_FAILED #38), not a second look.

For every cell it lists the failures the customer ended -- the agent's last
reply, then her last words, or that she said nothing more -- and with
--endings every episode she ended, passes too. Under the first simulated
customer a yes there was a turn the agent never had; the second is told to
wait for what she agreed to (v31, WHAT_FAILED #38), and these lists are how a
run shows whether she did. Under tool checks 3 it counts what each
cancellation did to the payment and the refunds the tools refused as already
settled (#37). Two cells with different customers, or different tool
versions, are flagged: a difference between them is partly the setup's.
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


def _closings(cell: Path, eps: list[dict], tasks: dict) -> dict[str, dict]:
    """transcript_id -> closing.review of its trace (harness/closing.py), with
    whether the episode passed, for finished episodes. The verdicts before and
    after the note are today's checks on the replayed calls, so they are kept
    only where the episode's own verdict is too ("rescored"): beside the runs'
    recorded verdicts (--checker recorded), or an episode rescore.py leaves
    alone, they would be two scorings in one line."""
    from pasarbench.harness.closing import review
    out = {}
    for e in eps:
        recs = [json.loads(l) for l in (cell / f"{e['transcript_id']}.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        r = review(recs, tasks.get(e["task_id"]))
        if e.get("scored") != "rescored":
            r = {**r, "at_note": None, "at_end": None}
        out[e["transcript_id"]] = {**r, "passed": e["passed"], "trap": e["trap"]}
    return out


def _endings(cell: Path, eps: list[dict], every: bool = False
             ) -> list[tuple[str, str, bool, str, str]]:
    """(transcript, trap, passed, the agent's last reply, her last words) for
    each failed episode -- or, with `every`, each episode -- that the customer
    ended. Under the first simulated customer a yes there, to something the
    agent had offered or asked to confirm, was a turn the agent never had: she
    agreed and left in one message, and the episode ended before the agent read
    it (WHAT_FAILED #38). The second is told to wait -- and to end with
    ###END### alone, which leaves no words: an ending with none is listed too,
    since one right after an open offer is her leaving without an answer.
    Nothing here decides what is a yes -- an offer can end without a question
    mark, in any language -- so every ending is listed, to be read. Episodes
    the agent, or a budget, ended are not hers, and are not listed. The reply
    is the last she saw: not one the guardrail held back, nor one written
    after she left, nor text beside a tool call, which the simulated customer
    is never shown (simulator.py)."""
    from pasarbench.harness.trace import customer_saw
    out = []
    for e in eps:
        if e["passed"] and not every:
            continue
        recs = [json.loads(l) for l in (cell / f"{e['transcript_id']}.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        turns = [r for r in recs if r.get("type") == "event" and r.get("kind") == "user_turn"]
        last = turns[-1] if turns else {}
        if not last.get("ended"):
            continue
        words = " ".join(str(last.get("text") or "").split())
        said = [st.get("model_content") for st in recs if st.get("type") == "step"
                and customer_saw(st) and not st.get("tool_calls")
                and (st.get("model_content") or "").strip()]
        reply = " ".join(said[-1].split())[-200:] if said else ""
        out.append((e["transcript_id"], e["trap"], bool(e["passed"]), reply, words))
    return out


def _print_endings(name: str, rows: list[tuple[str, str, bool, str, str]],
                   every: bool = False) -> None:
    what = "episodes" if every else "failures"
    failed = sum(not passed for _, _, passed, _, _ in rows)
    print(f"\n{what} in {name} that the customer ended: {len(rows)}"
          + (f" ({failed} failed)" if every else "")
          + (" -- read them: a yes to something the agent had offered is a turn the "
             "agent never had, and an end without an answer to it is her leaving it "
             "(WHAT_FAILED #38)" if rows else ""))
    for tid, trap, passed, reply, words in rows:
        print(f"    {tid:24s} {trap:34s} {'pass' if passed else 'FAIL'}  after: "
              f"\"...{reply[-80:]}\"")
        print(f"      she said: \"{words[:100]}{'...' if len(words) > 100 else ''}\"" if words
              else "      she said nothing more")


def _settlements(cell: Path, eps: list[dict]) -> tuple[Counter, int, int]:
    """What cancellations did to payments (tool checks 3), counted from what
    each trace recorded of them, and the refunds or credit the tools refused
    as already settled -- in all, and in a step after the cancellation's, when
    the agent had read the tool's note (one response can hold both calls, and
    then the refusal comes before the note is read). Over the episodes the
    rest of the comparison counts: an unfinished one is not among them."""
    from pasarbench.tools import SETTLED
    done: Counter = Counter()
    refused = later = 0
    for f in (cell / f"{e['transcript_id']}.jsonl" for e in eps):
        cancelled_at = None
        for i, st in enumerate(json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
                               if l.strip()):
            if st.get("type") != "step":
                continue
            for tr in st.get("tool_results") or []:
                if tr.get("name") == "cancel_order" and tr.get("payment"):
                    done[tr["payment"]] += 1
                    cancelled_at = i if cancelled_at is None else cancelled_at
                elif str(tr.get("error") or "").startswith(SETTLED):
                    refused += 1
                    later += cancelled_at is not None and cancelled_at < i
    return done, refused, later


def _print_settlements(name: str, done: Counter, refused: int, later: int) -> None:
    if not done:
        return
    known = ("refunded", "released", "nothing collected")
    print(f"\nwhat cancellations in {name} did to the payment: "
          + ", ".join(f"{k} {done[k]}" for k in (*known, *sorted(set(done) - set(known)))
                      if done[k])
          + f"; refunds or credit refused as already settled: {refused}"
          + (f", {later} of them in a step after the cancellation -- an agent that had "
             f"read the tool's note and tried to pay again; read them" if refused else ""))


def _as_left(eps: list[dict], cv: dict[str, dict]) -> list[dict]:
    """The episodes, each with its verdict as the customer left the case
    (closing.review's at_note) -- what the arm would have scored without the
    note, its conversations unchanged. An episode the note reached but whose
    case could not be scored as she left it is left out: its verdict at the
    end could count what the note did."""
    out = []
    for e in eps:
        r = cv.get(e["transcript_id"], {})
        if r.get("at_note") is not None:
            out.append({**e, "passed": r["at_note"]})
        elif not r.get("noted"):
            out.append(e)
    return out


def _print_closing(name: str, cv: dict[str, dict], show: int = 40) -> None:
    """What the closing check did, read before the pass rate: the mechanism is
    a required call first made after the customer left, and the cost is a
    forbidden call first made then -- and, since 1j, a write made a second
    time and a verification tried with digits nobody gave. Every episode with
    a cost is listed, and every case the phase turned from a fail into a pass
    some other way; the rest up to `show`. Each with what the customer last
    said: a rescue after her "yes, go ahead" is a turn the agent never had
    (WHAT_FAILED #38), not a second look."""
    noted = [t for t, r in cv.items() if r["noted"]]
    if not noted:
        return
    wrote = [t for t in noted if cv[t]["writes"]]
    calls = Counter(tool for t in noted for tool, ok in cv[t]["writes"] if ok)
    first = [t for t in noted if cv[t]["first"]]
    cost = [t for t in noted if cv[t]["forbidden"]]
    again = [t for t in noted if cv[t]["repeated"]]
    guess = [t for t in noted if cv[t]["guessed"]]
    cut = Counter(why for t in noted for why in cv[t]["cut"])
    print(f"\nclosing check in {name}: the customer left and the agent got the note in "
          f"{len(noted)} episodes; it made a write after that in {len(wrote)}.")
    if cut:
        print(f"  cut short before the agent was done: {sum(cut.values())} episodes ("
              + ", ".join(f"{n} by {why}" for why, n in cut.most_common()) + ")")
    print("  successful writes after the customer left: "
          + (", ".join(f"{t} {n}" for t, n in calls.most_common()) or "none"))
    print(f"  {'a required action first made then':42s} {len(first):>4d} episodes, "
          f"{sum(cv[t]['passed'] for t in first)} of them passed -- the mechanism")
    scored = [t for t in noted if cv[t]["at_note"] is not None]
    fixed = sorted(t for t in scored if not cv[t]["at_note"] and cv[t]["at_end"])
    broke = sorted(t for t in scored if cv[t]["at_note"] and not cv[t]["at_end"])
    if scored:
        print(f"  {'the case as the customer left it -> at end':42s} {len(fixed):>4d} failed -> "
              f"passed, {len(broke)} passed -> failed, of {len(scored)}"
              f"{' -- read every one it broke' if broke else ''}")
    else:
        print(f"  {'the case as the customer left it -> at end':42s} not scored: the "
              f"verdicts here are not today's checks on replayed calls")
    lost = Counter(cv[t]["not_replayed"] for t in noted if cv[t].get("not_replayed"))
    if lost:
        print(f"  {'not read: the calls do not replay':42s} {sum(lost.values()):>4d} episodes ("
              + "; ".join(f"{n}: {why[:60]}" for why, n in lost.most_common(2)) + ")")
    print(f"  {'a forbidden call first made then':42s} {len(cost):>4d} episodes"
          f"{' -- its cost; read them' if cost else ''}")
    print(f"  {'a write repeating one already made':42s} {len(again):>4d} episodes"
          f"{' -- one case done twice; read them' if again else ''}")
    print(f"  {'a verification with digits never given':42s} {len(guess):>4d} episodes"
          f"{' -- read them' if guess else ''}")
    blind = [t for t in noted if cv[t]["unscored"]]
    print(f"  {'a write no check asks for or forbids':42s} {len(blind):>4d} episodes"
          f"{' -- the checks cannot say; read them' if blind else ''}")
    # A fail turned into a pass by no required action -- a write a database
    # check needed -- is the mechanism too, and nothing else names it.
    other = [t for t in fixed if not cv[t]["first"]]
    paid = sorted(set(cost) | set(again) | set(guess) | set(broke) | set(blind) | set(other))
    rest = sorted(set(first) - set(paid))
    for t in paid + rest[:show]:
        r = cv[t]
        what = ([f"first: {', '.join(r['first'])}"] if r["first"] else []) + \
               ([f"FORBIDDEN: {', '.join(tool for tool, _ in r['forbidden'])}"]
                if r["forbidden"] else []) + \
               ([f"REPEATED: {', '.join(tool for tool, _ in r['repeated'])}"]
                if r["repeated"] else []) + \
               ([f"GUESSED: verify_identity {', '.join(r['guessed'])}"] if r["guessed"] else []) + \
               ([f"UNSCORED: {', '.join(tool for tool, _ in r['unscored'])}"]
                if r["unscored"] else []) + \
               (["BROKE A PASS"] if r["at_note"] and r["at_end"] is False else []) + \
               (["FIXED A FAIL"] if r["at_note"] is False and r["at_end"] and not r["first"]
                else [])
        print(f"    {t:24s} {r['trap']:34s} {'pass' if r['passed'] else 'FAIL'}  "
              f"{'; '.join(what)}")
        if r.get("left_saying") is not None:
            words = " ".join(r["left_saying"].split())
            print(f"      she left saying: \"{words[:100]}{'...' if len(words) > 100 else ''}\""
                  if words else "      she left saying nothing")
    if len(rest) > show:
        print(f"    ... and {len(rest) - show} more with a required action first made then")


def _audit_claims(cell: Path, eps: list[dict], tasks: dict) -> tuple[int, int]:
    """(failed English episodes read, of them claiming a required action the
    database never saw) -- scripts/audit_tool_arms.py's section 3, which reads
    passives and "we" for every action and so is not the guardrail's reading."""
    from pasarbench.claims import ENGLISH, false_claims
    from pasarbench.harness.trace import customer_saw
    read = claimed = 0
    for e in eps:
        task = tasks.get(e["task_id"])
        if e["passed"] or e.get("language") not in ENGLISH or task is None:
            continue
        recs = [json.loads(l) for l in (cell / f"{e['transcript_id']}.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        steps = [r for r in recs if r.get("type") == "step"]
        texts = [st["model_content"] for st in steps
                 if (st.get("model_content") or "").strip() and customer_saw(st)]
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
    ap.add_argument("--endings", action="store_true",
                    help="list every episode the customer ended, passes too, not only "
                         "the failures (RUNBOOK 1l: does the simulated customer wait?)")
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
    # tool_checks: absent before v27, when the tools took any argument value
    # (WHAT_FAILED #35) -- two cells that differ here are not one experiment.
    defaults = {"guardrail": "off", "closing": "off", "tool_checks": 1}
    for key in ("policy_mode", "exposure", "guardrail", "closing", "tool_checks",
                "requested_model", "simulator"):
        default = defaults.get(key)
        print(f"{key:12s} {str(heads[0].get(key, default)):>34s} "
              f"{str(heads[1].get(key, default)):>34s}")
    if heads[0].get("tool_checks", 1) != heads[1].get("tool_checks", 1):
        print("!! the two cells ran under different tool checks (WHAT_FAILED #35, #37): "
              "a difference between them is partly the tools'. Re-run the older one.")
    if heads[0].get("simulator") != heads[1].get("simulator"):
        print("!! the two cells had different simulated customers -- another model, "
              "gating, or the rules v31 changed (WHAT_FAILED #38): a difference between "
              "them is partly the customer's. Read it as an experiment on the customer, "
              "or re-run one cell with the other's.")
    sa, sb = summary(ea), summary(eb)
    for key, fmt in (("episodes", "{:d}"), ("tasks", "{:d}"), ("pass1", "{:.3f}"),
                     ("passk", "{:.3f}"), ("tokens", "{:,.0f}"), ("steps", "{:.1f}")):
        label = {"pass1": "pass^1", "passk": "pass^k", "tokens": "tokens/ep",
                 "steps": "steps/ep"}.get(key, key)
        print(f"{label:12s} {fmt.format(sa[key]):>34s} {fmt.format(sb[key]):>34s}")

    g = paired_episodes(ea, eb)
    cvs = [_closings(c, eps, tasks) for c, eps in zip(cells, (ea, eb))]
    print()
    if g is None:
        print("paired: fewer than 5 shared tasks -- not testable")
    else:
        verdict = (("BETTER" if g["diff"] > 0 else "WORSE") if g["resolved"]
                   else "INCONCLUSIVE")
        print(f"paired by task ({g['n']} shared): {g['diff']:+.3f}; {g['better']} tasks "
              f"better, {g['worse']} worse, sign test p={g['p']:.3f}  ->  {verdict}")
        if any(r["at_note"] is not None for cv in cvs for r in cv.values()):
            # Every closing cell's cases as each customer left them, against
            # the other cell's the same way: what the conversations did alone.
            la, lb = _as_left(ea, cvs[0]), _as_left(eb, cvs[1])
            out = len(ea) + len(eb) - len(la) - len(lb)
            gl = paired_episodes(la, lb)
            print("  the same, each case as its customer left it: " + (
                f"{gl['diff']:+.3f}; {gl['better']} tasks better, {gl['worse']} worse, "
                f"p={gl['p']:.3f} -- the conversations alone, before any closing note"
                if gl else "fewer than 5 shared tasks -- not testable")
                + (f" ({out} episodes left out: not scored as the customer left them)"
                   if out else ""))
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

    for name, cv in zip(names, cvs):
        _print_closing(name, cv)
    for name, cell, eps in zip(names, cells, (ea, eb)):
        _print_endings(name, _endings(cell, eps, every=a.endings), every=a.endings)
        _print_settlements(name, *_settlements(cell, eps))

    def per_trap(eps):
        d: dict[str, list[bool]] = defaultdict(list)
        for e in eps:
            d[e["trap"]].append(e["passed"])
        return {k: sum(v) / len(v) for k, v in d.items()}
    pa, pb = per_trap(ea), per_trap(eb)
    moved = sorted((pb[k] - pa[k], k) for k in pa if k in pb and abs(pb[k] - pa[k]) > 1e-9)
    def by_language(eps, trap):
        d: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for e in eps:
            if e["trap"] == trap:
                d[e.get("language", "?")][0] += e["passed"]
                d[e.get("language", "?")][1] += 1
        return d
    print("\ntraps that moved (leads to read in the traces, not results):")
    for d, k in moved or [(0.0, "(none)")]:
        if k == "(none)":
            print("  none")
            break
        print(f"  {k:40s} {pa[k]:.2f} -> {pb[k]:.2f}  ({d:+.2f})")
        la, lb = by_language(ea, k), by_language(eb, k)
        print("      by language, passed: " + ", ".join(
            f"{lang} {la[lang][0]}/{la[lang][1]} -> {lb[lang][0]}/{lb[lang][1]}"
            for lang in sorted(set(la) | set(lb))))
    print(f"\nwhy each remaining failure failed: python scripts/audit_tool_arms.py "
          f"{cells[1].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
