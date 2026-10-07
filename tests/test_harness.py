"""
Harness tests.

Each of these exists because the failure it catches is one that silently
corrupts results rather than crashing. A harness bug that inflates scores is
far worse than one that throws.

Run: python -m tests.test_harness
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pasarbench.agents import scripted_harness_agent
from pasarbench.db import Database
from pasarbench.harness import (Budget, FullContext, Message, SlidingWindow,
                                StopReason, ToolCall, ToolResultTrim,
                                TraceWriter, run_episode, restore, snapshot,
                                summarise_run, to_units)
from pasarbench.harness.backends import (ConfusedBackend, FailingBackend,
                                         MuteBackend, ScriptedBackend)
from pasarbench.harness.simulator import ScriptedUser
from pasarbench.run import SOLUTIONS, run_all
from pasarbench.tasks import BY_ID, TASKS
from pasarbench.verifier import verify

PASS, FAIL = [], []


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))


# --------------------------------------------------------------------------

def test_reference_through_loop():
    print("\n=== the loop actually works ===")
    out = run_all(scripted_harness_agent(SOLUTIONS))
    check("reference solutions score 1.0 through the harness",
          out["summary"]["pass^1"] == 1.0, str(out["summary"]))

    out = run_all(__import__("pasarbench.agents", fromlist=["HarnessAgent"]).HarnessAgent(
        backend_factory=lambda t: MuteBackend(), label="mute"))
    check("mute backend scores 0.0", out["summary"]["pass^1"] == 0.0,
          str(out["summary"]))


def test_budgets_bite():
    print("\n=== budgets terminate runaway agents ===")
    task = BY_ID["T01"]

    db = Database.fresh(task.db_patch)
    res = run_episode(task, db, ConfusedBackend(),
                      budget=Budget(max_steps=5, max_tool_calls=99))
    check("step cap terminates an infinite tool loop",
          res.stop_reason == StopReason.MAX_STEPS and res.budget["steps"] == 5,
          f"{res.stop_reason} steps={res.budget['steps']}")

    db = Database.fresh(task.db_patch)
    res = run_episode(task, db, ConfusedBackend(),
                      budget=Budget(max_steps=99, max_tool_calls=3))
    check("tool-call cap terminates",
          res.stop_reason == StopReason.MAX_TOOL_CALLS, str(res.stop_reason))

    db = Database.fresh(task.db_patch)
    res = run_episode(task, db, ConfusedBackend(),
                      budget=Budget(max_steps=99, max_tool_calls=2))
    check("exhausted tool budget still answers every call (no orphans)",
          pairing_ok(res.state.messages)[0], str(pairing_ok(res.state.messages)))

    db = Database.fresh(task.db_patch)
    res = run_episode(task, db, ConfusedBackend(), budget=Budget(max_tokens=500))
    check("token cap terminates", res.stop_reason == StopReason.MAX_TOKENS,
          str(res.stop_reason))


def test_error_degrades():
    print("\n=== a bad provider call ends one episode, not the sweep ===")
    task = BY_ID["T01"]
    db = Database.fresh(task.db_patch)
    try:
        res = run_episode(task, db, FailingBackend(fail_on=2))
        raised = False
    except Exception as e:  # noqa: BLE001
        raised, res = True, None
    check("backend exception does not propagate", not raised)
    if res:
        check("stop reason is BACKEND_ERROR",
              res.stop_reason == StopReason.BACKEND_ERROR, str(res.stop_reason))
        check("error text is captured for triage", bool(res.error), str(res.error))

    db = Database.fresh(task.db_patch)
    bad = ScriptedBackend([("get_order", {"__malformed__": "{order_id: O100"})])
    res = run_episode(task, db, bad)
    tool_msgs = [m for m in res.state.messages if m.role == "tool"]
    check("malformed arguments come back as a recoverable tool error",
          any("not valid JSON" in m.content for m in tool_msgs))


def test_interrupt_resume():
    print("\n=== interrupt and resume produce an identical outcome ===")
    task = BY_ID["T08"]          # the flagship, 5 tool calls
    script = SOLUTIONS["T08"]

    db_a = Database.fresh(task.db_patch)
    run_episode(task, db_a, ScriptedBackend(script))
    verdict_a = verify(task, db_a)

    backend = ScriptedBackend(script)
    db_b = Database.fresh(task.db_patch)
    first = run_episode(task, db_b, backend, interrupt_after_steps=2)
    check("interrupt stops at the requested step",
          first.stop_reason == StopReason.INTERRUPTED and first.budget["steps"] == 2,
          f"{first.stop_reason} steps={first.budget['steps']}")

    blob = snapshot(first.state, db_b)
    state2, db_c = restore(blob)
    restored_step = state2.step          # run_episode mutates state2 in place
    restored_msgs = len(state2.messages)
    run_episode(task, db_c, backend, state=state2)
    verdict_b = verify(task, db_c)

    check("resumed episode reaches the same verdict",
          verdict_a.passed and verdict_b.passed,
          f"a={verdict_a.passed} b={verdict_b.passed}")
    check("resumed episode produces the same final DB",
          db_a.tables == db_c.tables)
    check("snapshot round-trips through JSON",
          len(blob) > 100 and restored_step == 2 and restored_msgs > 4,
          f"step={restored_step} msgs={restored_msgs}")


def test_context_never_orphans():
    print("\n=== context strategies preserve tool-call pairing ===")
    task = BY_ID["T01"]
    db = Database.fresh(task.db_patch)
    res = run_episode(task, db, ScriptedBackend(SOLUTIONS["T01"]),
                      simulator=ScriptedUser(["ok thanks", "anything else?", "no"]))
    msgs = res.state.messages
    check("built a multi-unit conversation", len(to_units(msgs)) >= 6,
          f"{len(to_units(msgs))} units")

    for strat in (FullContext(), SlidingWindow(2), SlidingWindow(1),
                  SlidingWindow(3, pin_opening=False), ToolResultTrim(1)):
        built = strat.build(res.state)
        ok, why = pairing_ok(built)
        check(f"{strat.name}: no orphaned tool result", ok, why)

    pinned = SlidingWindow(1).build(res.state)
    check("sliding window pins the opening complaint",
          any(task.opening in m.content for m in pinned))

    full_chars = sum(len(m.content) for m in FullContext().build(res.state))
    trim_chars = sum(len(m.content) for m in ToolResultTrim(1).build(res.state))
    check("trim actually reduces payload size", trim_chars < full_chars,
          f"{trim_chars} vs {full_chars}")


def test_traces():
    print("\n=== traces are written and foldable ===")
    with tempfile.TemporaryDirectory() as tmp:
        tw = TraceWriter(root=tmp, run_id="testrun")
        for tid in ("T01", "T08", "T13"):
            task = BY_ID[tid]
            db = Database.fresh(task.db_patch)
            res = run_episode(task, db, ScriptedBackend(SOLUTIONS[tid]), trace=tw)
            v = verify(task, db)
            tw.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
        files = sorted(Path(tmp, "testrun").glob("*.jsonl"))
        check("one trace file per episode", len(files) == 3, str(files))
        s = summarise_run(Path(tmp, "testrun"))
        check("run folds to headline numbers",
              s["episodes"] == 3 and s["passed"] == 3, str(s))
        check("token accounting is non-zero", s["mean_tokens"] > 0, str(s))


def test_prompt_modes():
    print("\n=== policy modes differ as intended ===")
    from pasarbench.harness.prompts import system_prompt
    pre = system_prompt("U001", "MY", "preload")
    jit = system_prompt("U001", "MY", "jit")
    check("preload embeds the policy", "P7.2" in pre, f"{len(pre)} chars")
    check("jit does not embed the policy", "P7.2" not in jit, f"{len(jit)} chars")
    check("jit is much smaller", len(jit) < len(pre) / 2,
          f"jit={len(jit)} preload={len(pre)}")


# --------------------------------------------------------------------------

def pairing_ok(messages: list[Message]) -> tuple[bool, str]:
    """Every tool result must answer a tool call that precedes it, and every
    tool call must be answered. Violating either is a 400 from most APIs."""
    open_calls: dict[str, str] = {}
    answered: set[str] = set()
    for m in messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                open_calls[tc.id] = tc.name
        elif m.role == "tool":
            if m.tool_call_id not in open_calls:
                return False, f"orphaned tool result {m.tool_call_id}"
            answered.add(m.tool_call_id)
    missing = set(open_calls) - answered
    if missing:
        return False, f"unanswered tool call(s): {[open_calls[i] for i in missing]}"
    return True, ""


def test_claim_guardrail():
    """docs/RUNBOOK.md 1i. Behind search the agent told customers it had
    escalated when no escalation existed. The guardrail holds such a reply
    back -- the customer must never see it, nor the note that replaces it --
    and must not fire on what happened before the conversation."""
    print("\n=== the claim guardrail holds back a claim no call backs ===")
    import subprocess
    import sys

    from pasarbench.claims import own_claims
    from pasarbench.harness import guardrail as g
    from pasarbench.harness.backends import OpenAICompatBackend
    from pasarbench.harness.simulator import LLMUser
    from pasarbench.harness.types import ModelResponse, Usage

    cases = [
        ("I've escalated your case with the `customs_hold` category", ["escalate_to_human"]),
        ("so I'm escalating this to our team for investigation", ["escalate_to_human"]),
        ("I've issued a goodwill voucher of **THB 300** to your account", ["issue_goodwill_voucher"]),
        ("I'll issue the goodwill voucher now. The item value is THB 1,500.00", ["issue_goodwill_voucher"]),
        ("I’ve escalated your case", ["escalate_to_human"]),
        ("I can see that a full refund of PHP 2,630.00 was already issued on 9 November", []),
        ("We've already issued a full refund on 9 November", []),
        ("Your refund was issued on 9 November.", []),
        ("Your case has been escalated to a specialist", []),
        ("I have not escalated this yet", []),
        ("I'll escalate if it reaches 10 days", []),
        ("Would you like me to escalate this?", []),
    ]
    wrong = [(t, want, got) for t, want in cases
             if (got := [x for x, _ in own_claims(t)]) != want]
    check("it reads the agent's own claims -- \"I've escalated\", \"I'm escalating\", "
          "\"I'll issue it now\" -- and not \"we've already refunded\", \"your refund was "
          "issued\", a negation, a conditional or a question", not wrong, str(wrong))

    class Customer:
        """An LLM customer's backend that records what the customer is shown."""
        name = "seen"
        reports_usage = False

        def __init__(self):
            self.views: list[list[str]] = []

        def chat(self, messages, tools):
            self.views.append([m.content for m in messages[1:]])
            return ModelResponse(content="ok ###END###" if len(self.views) > 1 else "thanks",
                                 usage=Usage(1, 1))

    class Rec:
        def __init__(self):
            self.meta, self.steps = {}, []

        def open_episode(self, task_id, run_index=None, meta=None):
            self.meta = meta or {}

        def step(self, record):
            self.steps.append(record)

        def event(self, *a, **kw):
            pass

    task = BY_ID["T14"]
    esc = ("escalate_to_human", {"order_id": "O1009", "category": "duplicate_refund",
                                 "reason": "refund already issued for this order"})
    claim = "I've escalated your case to a specialist."

    def run(plan, guard):
        cust, rec = Customer(), Rec()
        db = Database.fresh(task.db_patch)
        res = run_episode(task, db, ScriptedBackend(plan), trace=rec, guardrail=guard,
                          simulator=LLMUser(cust, persona="a customer", facts={}))
        return res, cust, rec, db

    _, cust, rec, _ = run([claim], "off")
    check("off: the claim reaches the customer, and the trace is as it always was",
          claim in cust.views[0] and "guardrail" not in rec.meta
          and not any("guardrail" in s for s in rec.steps), str(cust.views))

    res, cust, rec, db = run([claim, esc, claim], "claims")
    held = [s for s in rec.steps if s.get("guardrail")]
    last = cust.views[-1]
    check("on: the unbacked claim is held back, and the customer sees neither it nor the "
          "note -- only the reply that came after the call",
          len(held) == 1 and held[0]["guardrail"]["claims"][0][0] == "escalate_to_human"
          and last.count(claim) == 1 and not any(c.startswith("[") for c in last)
          and rec.meta.get("guardrail") == "claims", str(cust.views))
    note = res.state.messages[res.state.messages.index(
        next(m for m in res.state.messages if m.hidden)) + 1]
    r = g.review(rec.steps)
    check("…the agent got a note naming the claim and the tool that would back it, made "
          "the call, and the trace reads back that way",
          note.role == "user" and note.content.startswith(g.NOTE_HEAD)
          and "`escalate_to_human`" in note.content and r["outcomes"] == ["made the call"]
          and not r["delivered"] and any(a.tool == "escalate_to_human" and a.ok
                                         for a in db.action_log), str((note.content, r)))

    _, cust, rec, _ = run([claim] * 4, "claims")
    r = g.review(rec.steps)
    check(f"an agent that insists is held back {g.MAX_NOTES} times and then let through, "
          f"so it cannot loop -- and the trace says it claimed again",
          len(r["held"]) == g.MAX_NOTES and r["outcomes"] == ["claimed again"]
          and r["delivered"], str(r))

    before = ("I can see a full refund of PHP 2,630.00 was already issued on 9 November, "
              "so the money is on its way.")
    _, cust, rec, _ = run([before], "claims")
    check("a refund issued before the conversation is not taken for a claim -- held back, "
          "it would have sent the agent towards a second refund",
          not any(s.get("guardrail") for s in rec.steps) and before in cust.views[0])

    from pasarbench.db import Action
    refund = "I've processed your refund: MYR 189.00 is now in your account."
    check("a refund claim after store credit is backed -- for a COD order that is the "
          "policy's refund (P4.2) -- and with no remedy at all it is not",
          not g.unbacked(refund, [Action("issue_store_credit", {}, ok=True)])
          and [t for t, _ in g.unbacked(refund, [])] == ["issue_refund"]
          and g.unbacked(refund, [Action("issue_refund", {}, ok=False)]))

    m = Message("assistant", claim, hidden=True)
    check("a held reply stays held through snapshot and resume, and never reaches a "
          "provider's wire format", Message.from_dict(m.to_dict()).hidden
          and "hidden" not in OpenAICompatBackend._msg(m)
          and "hidden" not in Message("assistant", claim).to_dict())
    try:
        run_episode(task, Database.fresh(task.db_patch), ScriptedBackend([]), guardrail="on")
        bad_mode = False
    except ValueError:
        bad_mode = True
    check("an unknown guardrail mode is refused, not ignored", bad_mode)

    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "-m", "pasarbench.sweep", "--backend", "scripted",
                            "--suite", "core", "--tasks", "T14", "--strategies", "full",
                            "--guardrail", "claims", "--run-id", "g", "--trace-root", d],
                           capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parent.parent)
        heads = [__import__("json").loads(f.read_text().splitlines()[0])
                 for f in Path(d).glob("g/*/*.jsonl")]
        check("the sweep's --guardrail reaches every episode's trace header",
              r.returncode == 0 and heads and all(h.get("guardrail") == "claims" for h in heads),
              r.stderr[-500:] or str(heads))


def main() -> int:
    test_reference_through_loop()
    test_budgets_bite()
    test_error_degrades()
    test_interrupt_resume()
    test_context_never_orphans()
    test_traces()
    test_prompt_modes()
    test_claim_guardrail()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
