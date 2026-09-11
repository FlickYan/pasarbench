"""
Context strategy + ablation analysis tests.

Every strategy here is a chance to silently corrupt a conversation. The tests
that matter are the ones checking a strategy did NOT quietly break something
the model then gets blamed for.

Run: python -m tests.test_context
"""

from __future__ import annotations

import json

from pasarbench.analyze import (Point, degradation_matrix, frontier_report,
                                markdown_report, pareto_frontier)
from pasarbench.db import Database
from pasarbench.harness.backends import ScriptedBackend
from pasarbench.harness.context import (NoteTaking, Summarize, ToolResultTrim,
                                        SlidingWindow, FullContext,
                                        make_strategy, note_discipline, to_units)
from pasarbench.harness.loop import run_episode
from pasarbench.harness.simulator import ScriptedUser
from pasarbench.harness.types import Message, ModelResponse, Usage
from pasarbench.run import SOLUTIONS
from pasarbench.tasks import BY_ID
from tests.test_harness import pairing_ok

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))


class FakeSummarizer:
    """Returns a valid schema JSON. Counts calls so we can prove the running
    summary is incremental rather than redone from scratch each step."""
    name, reports_usage = "fake-summarizer", False

    def __init__(self, payload=None, raise_on=None):
        self.calls, self.raise_on = 0, raise_on
        self.payload = payload or {
            "customer_goal": "return a blouse",
            "order_id": "O1003",
            "identity_verified": "yes",
            "facts_established": ["delivered 2 days ago", "eligible for return"],
            "actions_taken": ["check_return_eligibility -> eligible"],
            "outstanding": "issue the refund",
        }

    def chat(self, messages, tools):
        self.calls += 1
        if self.raise_on and self.calls >= self.raise_on:
            raise ConnectionError("summarizer down")
        return ModelResponse(content=json.dumps(self.payload), usage=Usage(50, 50))


class BadJSONSummarizer(FakeSummarizer):
    def chat(self, messages, tools):
        self.calls += 1
        return ModelResponse(content="Sure! Here's a summary: they want a refund.",
                             usage=Usage(50, 50))


def long_episode(strategy=None, extra_script=None):
    task = BY_ID["T01"]
    db = Database.fresh(task.db_patch)
    script = (extra_script or []) + SOLUTIONS["T01"]
    return task, run_episode(task, db, ScriptedBackend(script),
                             simulator=ScriptedUser(["ok", "and one more thing", "no thanks"]),
                             context=strategy or FullContext())


def test_pairing_preserved():
    print("\n=== new strategies never orphan a tool result ===")
    _, res = long_episode()
    for strat in (Summarize(FakeSummarizer(), keep_recent=2, trigger_units=3),
                  NoteTaking(2), NoteTaking(1)):
        built = strat.build(res.state)
        ok, why = pairing_ok(built)
        check(f"{strat.name}: pairing intact", ok, why)


def test_summarize_schema():
    print("\n=== summarisation is structured, incremental and fault-tolerant ===")
    _, res = long_episode()
    fake = FakeSummarizer()
    strat = Summarize(fake, keep_recent=2, trigger_units=3)

    built = strat.build(res.state)
    text = "\n".join(m.content for m in built)
    check("summary is injected into context", "[summary of the conversation so far]" in text)
    for field in ("identity_verified", "order_id", "actions_taken", "outstanding"):
        check(f"summary retains `{field}`", field in text)
    check("identity_verified survives compaction with its value",
          "identity_verified: yes" in text, text[:200])

    calls_after_first = fake.calls
    strat.build(res.state)
    check("running summary is incremental, not redone every step",
          fake.calls == calls_after_first, f"{fake.calls} vs {calls_after_first}")
    check("summarized_upto advances", res.state.summarized_upto > 0,
          str(res.state.summarized_upto))

    _, res2 = long_episode()
    bad = Summarize(BadJSONSummarizer(), keep_recent=2, trigger_units=3)
    out = bad.build(res2.state)
    check("malformed summary JSON degrades instead of crashing", bool(out))
    ok, why = pairing_ok(out)
    check("malformed summary still yields valid context", ok, why)

    _, res3 = long_episode()
    dead = Summarize(FakeSummarizer(raise_on=1), keep_recent=2, trigger_units=3)
    out3 = dead.build(res3.state)
    check("summarizer outage does not kill the episode", bool(out3))

    _, res4 = long_episode()
    short = Summarize(FakeSummarizer(), keep_recent=2, trigger_units=999)
    before = short.backend.calls
    short.build(res4.state)
    check("no summarisation below the trigger threshold",
          short.backend.calls == before)


def test_notes():
    print("\n=== note-taking, and the failure mode of not taking notes ===")
    task, res = long_episode(strategy=NoteTaking(2))
    d = note_discipline(res.state.messages)
    check("an agent that writes nothing is detected", d["wrote_any"] == 0, str(d))
    built = NoteTaking(2).build(res.state)
    check("with no notes, nothing is compacted",
          len(built) == len(res.state.messages),
          f"{len(built)} vs {len(res.state.messages)}")

    note = [("write_note", {"content": "order O1003, identity verified, eligible"})]
    task2, res2 = long_episode(strategy=NoteTaking(2), extra_script=note)
    d2 = note_discipline(res2.state.messages)
    check("notes written are counted", d2["notes_written"] == 1, str(d2))
    built2 = NoteTaking(2).build(res2.state)
    text = "\n".join(m.content for m in built2)
    check("the note board is pinned into context", "[your notes]" in text)
    check("the note content survives", "identity verified" in text)
    check("with notes, history IS compacted",
          len(built2) < len(res2.state.messages),
          f"{len(built2)} vs {len(res2.state.messages)}")
    ok, why = pairing_ok(built2)
    check("compacted note context is still valid", ok, why)


def test_tool_exposure():
    print("\n=== only the notes strategy sees the scratchpad ===")
    task = BY_ID["T01"]
    for name, expect in (("full", False), ("trim3", False), ("notes4", True)):
        db = Database.fresh(task.db_patch)
        strat = make_strategy(name)
        res = run_episode(task, db, ScriptedBackend(SOLUTIONS["T01"]), context=strat)
        # ScriptedBackend ignores tools, so inspect what the loop offered
        from pasarbench.tools import DEFAULT_TOOLS, schemas
        names = list(DEFAULT_TOOLS) + list(getattr(strat, "extra_tools", []))
        has = "write_note" in names
        check(f"{name}: write_note exposed == {expect}", has == expect)
    check("default tool set excludes the scratchpad",
          "write_note" not in __import__("pasarbench.tools", fromlist=["DEFAULT_TOOLS"]).DEFAULT_TOOLS)


def test_pareto():
    print("\n=== Pareto frontier ===")
    pts = [Point("full", 10000, 0.80), Point("trim3", 6000, 0.79),
           Point("window4", 4000, 0.70), Point("bad", 9000, 0.60),
           Point("window8", 7000, 0.74)]
    front = [p.label for p in pareto_frontier(pts)]
    check("dominated points are excluded", "bad" not in front, str(front))
    check("cheapest point is on the frontier", "window4" in front, str(front))
    check("best-quality point is on the frontier", "full" in front, str(front))
    check("frontier is sorted by cost",
          front == sorted(front, key=lambda l: next(p.cost for p in pts if p.label == l)),
          str(front))
    check("window8 is dominated by trim3", "window8" not in front, str(front))


def test_reports():
    print("\n=== degradation matrix + report ===")
    rows = [
        {"strategy": "full", "pass^1": 0.80, "pass^k": 0.62, "mean_tokens": 10000,
         "per_trap": {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.8},
         "per_language": {"en": 0.85, "th": 0.70}},
        {"strategy": "window4", "pass^1": 0.70, "pass^k": 0.50, "mean_tokens": 4000,
         "per_trap": {"a": 0.9, "b": 0.3, "c": 0.2, "d": 0.8},
         "per_language": {"en": 0.76, "th": 0.55}},
        {"strategy": "trim3", "pass^1": 0.79, "pass^k": 0.60, "mean_tokens": 6000,
         "per_trap": {"a": 0.9, "b": 0.75, "c": 0.7, "d": 0.8},
         "per_language": {"en": 0.84, "th": 0.69}},
    ]
    deg = degradation_matrix(rows)
    check("regressions are identified per trap",
          set(deg["window4"]["regressions"]) == {"b", "c"},
          str(deg["window4"]["regressions"]))
    check("a concentrated loss is called concentrated",
          "concentrated" in deg["window4"]["verdict"], deg["window4"]["verdict"])
    check("a strategy with no regression is described as such",
          deg["trim3"]["n_traps_hurt"] == 0, str(deg["trim3"]["regressions"]))

    f = frontier_report(rows)
    check("frontier report names the baseline comparison",
          any(t["strategy"] == "trim3" for t in f["vs_baseline"]), str(f))
    saving = next(t for t in f["vs_baseline"] if t["strategy"] == "trim3")
    check("token saving is computed", abs(saving["token_saving"] - 0.4) < 1e-6,
          str(saving))

    md = markdown_report(rows)
    check("report renders a table", "| strategy |" in md)
    check("report surfaces the language gap", "Pass rate by language" in md)
    check("report explains where each strategy loses", "Where each strategy loses" in md)
    print("\n" + "\n".join(md.splitlines()[:9]))


def main() -> int:
    test_pairing_preserved()
    test_summarize_schema()
    test_notes()
    test_tool_exposure()
    test_pareto()
    test_reports()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
