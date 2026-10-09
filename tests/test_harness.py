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
        ("The refund has been processed to your original payment method", []),
        ("I have not escalated this yet", []),
        ("I'll escalate if it reaches 10 days", []),
        ("Would you like me to escalate this?", []),
        ("I can offer you a goodwill voucher worth THB 300 if you like", []),
        ("the voucher you used was issued by the seller last year", []),
        # No voucher or escalation can predate the conversation, so for those
        # two a passive or "we" is a claim -- v24 missed these (WHAT_FAILED #34).
        ("Your voucher has been issued. Is there anything else?", ["issue_goodwill_voucher"]),
        ("Your goodwill voucher of VND 220,000 has been issued.", ["issue_goodwill_voucher"]),
        ("Your voucher of THB 300.00 has been arranged", ["issue_goodwill_voucher"]),
        ("Your case has been escalated to a specialist", ["escalate_to_human"]),
        ("We've escalated your case", ["escalate_to_human"]),
    ]
    wrong = [(t, want, got) for t, want in cases
             if (got := [x for x, _ in own_claims(t)]) != want]
    check("it reads the agent's own claims -- \"I've escalated\", \"I'm escalating\", "
          "\"I'll issue it now\", and for vouchers and escalations \"your voucher has been "
          "issued\" -- and not \"we've already refunded\", \"your refund was issued\", an "
          "offer, a negation, a conditional or a question", not wrong, str(wrong))

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


def test_tool_input_checks():
    """WHAT_FAILED #35. escalate_to_human listed eight categories and took any
    string, and took "unknown" for an order; the agent was told it had
    escalated and the check then failed it. The tools now refuse what their
    schemas do not allow -- and a run recorded before that must still replay
    under the checks it ran under, or re-scoring would rebuild a state the
    episode never ended in."""
    print("\n=== the tools refuse what their schemas do not allow ===")
    import hashlib
    import json
    import subprocess
    import sys

    from pasarbench.harness.replay import replay
    from pasarbench.tasks import world_digest
    from pasarbench.tools import CHECKS, call

    task = BY_ID["T14"]

    def fresh() -> Database:
        return Database.fresh(task.db_patch)

    good = {"order_id": "O1009", "category": "duplicate_refund", "reason": "already refunded"}
    db = fresh()
    r = call(db, "escalate_to_human", {**good, "category": "duplicate_refund_claim"})
    check("a category the schema does not list is refused, the error lists the allowed "
          "ones, nothing is written, and the attempt is logged as a failed call",
          not r["ok"] and r["error"].startswith("invalid value: category must be one of")
          and "'duplicate_refund'" in r["error"] and not db.t("escalations")
          and [(a.tool, a.ok) for a in db.action_log] == [("escalate_to_human", False)], str(r))
    r = call(fresh(), "escalate_to_human", {**good, "category": "remboursement dupliqué"})
    check("…the value is quoted the same on every interpreter (ascii, not repr), and the "
          "error is not counted as malformed JSON",
          r.get("error", "").endswith("got 'remboursement dupliqu\\xe9'")
          and "bad arguments" not in r.get("error", ""), str(r))
    db = fresh()
    r = call(db, "escalate_to_human", {**good, "category": None})
    check("a required argument sent as null is refused, not stored -- the same failure "
          "as an unlisted category",
          r.get("error") == "bad arguments: category must be a string, not null"
          and not db.t("escalations"), str(r))
    for bad_args in (["order_id"], [], "", 5):
        try:
            r = call(fresh(), "get_order", bad_args)
            ok = r.get("error") == "bad arguments: the arguments must be a JSON object"
        except Exception as e:  # noqa: BLE001
            ok, r = False, f"raised {type(e).__name__}: {e}"
        if not ok:
            break
    check("arguments that are valid JSON but not an object are refused, never raised",
          ok, str(r))
    t06 = BY_ID["T06"]
    cod = ("issue_refund", {"order_id": "O1004", "amount_minor": 1000, "method": "cod",
                            "reason": "customer insists"})
    for checks_ in (1, 2):
        db6 = Database.fresh(t06.db_patch)
        for name, a in [SOLUTIONS["T06"][0], cod, *SOLUTIONS["T06"][1:]]:
            call(db6, name, a, checks=checks_)
        v = verify(t06, db6)
        if checks_ == 1:
            v1_failures = v.failures
    check("a forbidden attempt the schema refuses is still an attempt: refunding a COD order "
          "with method 'cod' fails T06 under the checks, as it did before them",
          not v.passed and any("forbidden" in f and "issue_refund" in f for f in v.failures)
          and v1_failures == v.failures, str((v1_failures, v.failures)))
    db = fresh()
    r = call(db, "escalate_to_human", {**good, "order_id": "unknown"})
    check("an escalation naming no order is answered 'no such order', as every other "
          "write answers it, and logged as a failed call",
          not r["ok"] and r["error"] == "no such order" and not db.t("escalations")
          and [(a.tool, a.ok) for a in db.action_log] == [("escalate_to_human", False)],
          str(r))
    check("…a blank order too",
          call(fresh(), "escalate_to_human", {**good, "order_id": ""}).get("error")
          == "no such order")
    db = fresh()
    check("a valid escalation still goes through",
          call(db, "escalate_to_human", good)["ok"] and len(db.t("escalations")) == 1)
    refund = {"order_id": "O1009", "amount_minor": "4700", "method": "card", "reason": "r"}
    r = call(fresh(), "issue_refund", refund)
    check("an amount sent as text is refused for its type before the tool runs",
          r.get("error") == "bad arguments: amount_minor must be an integer, not a string",
          str(r))
    r = call(fresh(), "issue_refund", {**refund, "amount_minor": True})
    check("…and true is not an integer",
          r.get("error") == "bad arguments: amount_minor must be an integer, not true or false",
          str(r))
    db = fresh()
    call(db, "verify_identity", {"user_id": task.user_id,
                                 "phone_last4": task.hidden_facts["phone_last4"]})
    whole = call(db, "issue_refund", {**refund, "amount_minor": 4700.0})
    part = call(fresh(), "issue_refund", {**refund, "amount_minor": 4700.5})
    stored = [r["amount_minor"] for r in db.t("refunds").values() if r["refund_id"] != "REF9999"]
    check("…but no stricter than JSON Schema: 4700.0 is an integer, and the tool is given "
          "4700; 4700.5 is not an integer",
          whole.get("ok") and stored == [4700] and type(stored[0]) is int
          and part.get("error") == "bad arguments: amount_minor must be an integer, not a number",
          str((whole, part, stored)))
    u = task.user_id
    check("an empty optional filter is still 'not given', as the tool reads it; a status "
          "the schema does not list is refused",
          call(fresh(), "list_user_orders", {"user_id": u, "status": ""})["ok"]
          and call(fresh(), "list_user_orders", {"user_id": u, "status": None})["ok"]
          and not call(fresh(), "list_user_orders", {"user_id": u, "status": "lost"})["ok"])
    db = fresh()
    bad = {**good, "order_id": "unknown", "category": "duplicate_refund_claim"}
    check("under version 1, what every run before v27 ran under, both were taken",
          CHECKS == 2 and call(db, "escalate_to_human", bad, checks=1)["ok"]
          and len(db.t("escalations")) == 1)

    # A run recorded before v27: no tool_checks in its header, and an
    # escalation the old tools took. The payload is the one v26 returned,
    # written out, so a drift in version 1 itself would show here.
    args = {"order_id": "unknown", "category": "duplicate_refund", "reason": "r"}
    payload = '{"ok": true, "escalation_id": "ESC0001"}'
    check("version 1 still answers exactly what v26 answered",
          json.dumps(call(fresh(), "escalate_to_human", args, checks=1),
                     ensure_ascii=False, default=str) == payload)
    head = {"type": "header", "task_id": task.task_id, "world_digest": world_digest(task)}
    step = {"type": "step", "step": 1,
            "tool_calls": [{"name": "escalate_to_human", "arguments": args}],
            "tool_results": [{"name": "escalate_to_human", "ok": True, "error": None,
                              "result_chars": len(payload),
                              "result_sha1": hashlib.sha1(payload.encode()).hexdigest()[:12]}]}
    old = replay([head, step], task)
    new = replay([{**head, "tool_checks": 2}, step], task)
    check("a run recorded before v27 replays under the checks it ran under, and rebuilds "
          "the state it ended in",
          not old.diverged and old.verified == 1 and len(old.db.t("escalations")) == 1,
          str(old.diverged))
    check("…and one that recorded today's checks replays under them",
          len(new.diverged) == 1 and "no such order" in new.diverged[0]
          and not new.db.t("escalations"), str(new.diverged))

    class Rec:
        meta: dict = {}

        def open_episode(self, task_id, run_index=None, meta=None):
            self.meta = meta or {}

        def step(self, record):
            pass

        def event(self, *a, **kw):
            pass

    rec = Rec()
    run_episode(task, fresh(), ScriptedBackend(["Let me look into that."]), trace=rec)
    check("every new run records the checks it ran under in its trace header",
          rec.meta.get("tool_checks") == CHECKS, str(rec.meta))
    try:
        res = run_episode(task, fresh(), ScriptedBackend([("get_order", 5), ("get_order", [])]))
        tool_msgs = [m.content for m in res.state.messages if m.role == "tool"]
        ok = (res.stop_reason == StopReason.DONE and len(tool_msgs) == 2
              and all("must be a JSON object" in c for c in tool_msgs))
        detail = str(tool_msgs)
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"raised {type(e).__name__}: {e}"
    check("a model that sends a bare number or a list as its arguments gets an error back, "
          "and the episode goes on", ok, detail)

    with tempfile.TemporaryDirectory() as d:
        sweep = [sys.executable, "-m", "pasarbench.sweep", "--backend", "scripted",
                 "--suite", "core", "--tasks", "T14", "--strategies", "full",
                 "--run-id", "c", "--trace-root", d]
        cwd = Path(__file__).resolve().parent.parent
        first = subprocess.run(sweep, capture_output=True, text=True, cwd=cwd)
        f = next(Path(d).glob("c/*/*.jsonl"))
        lines = f.read_text().splitlines()
        h = json.loads(lines[0])
        h.pop("tool_checks", None)
        f.write_text("\n".join([json.dumps(h), *lines[1:]]) + "\n")
        again = subprocess.run(sweep + ["--resume"], capture_output=True, text=True, cwd=cwd)
        check("--resume refuses to mix a cell's episodes from before v27 with new ones",
              first.returncode == 0 and again.returncode != 0
              and "tool checks v1" in again.stdout + again.stderr,
              (again.stdout + again.stderr)[-400:])


def test_closing_check():
    """docs/RUNBOOK.md 1j. When the customer leaves, the agent gets one note to
    go over the case against the policy. The customer never sees it; the calls
    the agent makes then are real, the text it writes is a case note no one
    reads; the phase is capped, survives interrupt and resume, and the trace
    says what it did -- including any forbidden call, which is its cost."""
    print("\n=== the closing check: one look at the case after the customer leaves ===")
    import json
    import subprocess
    import sys

    from pasarbench.harness import closing as cl
    from pasarbench.harness import guardrail as g
    from pasarbench.harness.simulator import LLMUser
    from pasarbench.harness.trace import customer_saw
    from pasarbench.harness.types import ModelResponse, Usage
    from pasarbench.tools import DEFAULT_TOOLS

    bye = "Oh, I didn't notice that. Thanks, bye"

    class Customer:
        """Accepts the first explanation and leaves, as the duplicate-refund
        customer does."""
        name = "leaves"
        model = "c"
        reports_usage = False

        def __init__(self):
            self.views: list[list[str]] = []

        def chat(self, messages, tools):
            self.views.append([m.content for m in messages[1:]])
            return ModelResponse(content=f"{bye} ###END###", usage=Usage(1, 1))

    class Rec:
        """The trace, as the records a reader gets from the file."""
        def __init__(self):
            self.meta, self.steps, self.events, self.recs = {}, [], [], []

        def open_episode(self, task_id, run_index=None, meta=None):
            self.meta = meta or {}
            self.recs.append({"type": "header", "task_id": task_id, **self.meta})

        def step(self, record):
            self.steps.append(record)
            self.recs.append({"type": "step", **json.loads(json.dumps(record, default=str))})

        def event(self, kind, **kw):
            self.events.append(kind)
            self.recs.append({"type": "event", "kind": kind, **kw})

    class Seeing(ScriptedBackend):
        """Remembers every context it was sent."""
        def __init__(self, plan):
            super().__init__(plan)
            self.seen: list[list[tuple[str, str]]] = []

        def chat(self, messages, tools):
            self.seen.append([(m.role, m.content) for m in messages])
            return super().chat(messages, tools)

    task = BY_ID["T14"]
    look = ("get_order", {"order_id": "O1009"})
    explain = "I can see a full refund of SGD 47.00 was already issued on 8 November."
    esc = ("escalate_to_human", {"order_id": "O1009", "category": "duplicate_refund",
                                 "reason": "refund already issued for this order"})
    note = "Escalated: duplicate refund request; the refund had already been issued."
    plan = [look, explain, esc, note]

    def run(plan, mode, task=task, **kw):
        cust, rec = Customer(), Rec()
        db = Database.fresh(task.db_patch)
        backend = Seeing(list(plan))
        res = run_episode(task, db, backend, trace=rec, closing=mode,
                          simulator=LLMUser(cust, persona="a customer", facts={}), **kw)
        return res, cust, rec, db, backend

    res, cust, rec, db, _ = run(plan, "off")
    check("off: the episode ends when the customer leaves, the escalation P10 asks for is "
          "never made, and the trace is as it always was",
          res.stop_reason == StopReason.DONE and not verify(task, db).passed
          and "closing" not in rec.meta and not any("closing" in s for s in rec.steps))

    res, cust, rec, db, seen = run(plan, "check")
    r = cl.review(rec.recs, task)
    done_db = db
    check("check: after the customer leaves the agent gets the note, makes the call the "
          "policy requires, and the case passes -- and the trace says so",
          res.stop_reason == StopReason.DONE and verify(task, db).passed
          and rec.meta.get("closing") == "check"
          and [s["step"] for s in rec.steps if s.get("closing")] == [3, 4]
          and r["noted"] and r["ran"] and not r["cut"]
          and r["first"] == ["escalate_to_human"] and r["writes"] == [("escalate_to_human", True)]
          and not r["forbidden"] and "closing_check" in rec.events, str((r, rec.events)))
    check("…the agent reads the customer's last words, then the note",
          seen.seen[2][-2:] == [("user", bye), ("user", cl.NOTE)], str(seen.seen[2][-3:]))
    check("…the customer is asked for nothing after leaving and never sees the note; the "
          "agent's last words are a case note, hidden from any customer view",
          len(cust.views) == 1 and not any(cl.NOTE in v for vs in cust.views for v in vs)
          and res.state.messages[-1].hidden and res.state.messages[-1].content == note
          and not customer_saw(rec.steps[-1]))

    claim = "I've escalated your case to a specialist."
    res, _, rec, _, _ = run([look, explain, claim], "check", guardrail="claims")
    check("a case note is not a reply the customer got: the claim guardrail lets it be, "
          "and its readers do not count it",
          res.stop_reason == StopReason.DONE and not any(s.get("guardrail") for s in rec.steps)
          and rec.steps[-1].get("closing") and rec.steps[-1]["model_content"] == claim
          and not g.review(rec.steps)["delivered"], str(g.review(rec.steps)))

    ver = ("verify_identity", {"user_id": task.user_id,
                               "phone_last4": task.hidden_facts["phone_last4"]})
    refund = ("issue_refund", {"order_id": "O1009", "amount_minor": 4700, "method": "card",
                               "reason": "customer asked"})
    _, _, rec, db, _ = run([look, explain, ver, refund, "Refunded again."], "check")
    r = cl.review(rec.recs, task)
    check("a forbidden call made after the customer left is the check's cost: the case "
          "fails and the review names the call",
          [t for t, _ in r["forbidden"]] == ["issue_refund"] and not verify(task, db).passed,
          str(r))

    _, _, rec, _, _ = run([look, esc, explain, esc, "Escalated again."], "check")
    r = cl.review(rec.recs, task)
    check("an action the conversation had already made is not one the phase rescued",
          r["first"] == [] and r["writes"] == [("escalate_to_human", True)], str(r))

    # T06 forbids even an attempt at a refund (must_succeed=False), as 24 of
    # 1j's 57 tasks do -- the place a call that never reached the tools, or one
    # the conversation had already made, would be miscounted as the cost.
    t06 = BY_ID["T06"]
    ver6 = ("verify_identity", {"user_id": t06.user_id,
                                "phone_last4": t06.hidden_facts["phone_last4"]})
    look6 = ("get_order", {"order_id": "O1004"})
    esc6 = ("escalate_to_human", {"order_id": "O1004", "category": "out_of_window_dispute",
                                  "reason": "customer disputes the window"})
    refund6 = ("issue_refund", {"order_id": "O1004", "amount_minor": 34000,
                                "method": "card", "reason": "asked"})
    hidden = [n for n in DEFAULT_TOOLS if n != "issue_refund"]
    _, _, rec, db, _ = run([ver6, look6, "It is out of the window.", esc6, refund6, "Done."],
                           "check", task=t06, tool_names=hidden)
    r = cl.review(rec.recs, t06)
    check("a call to a tool the arm did not show never reached the database: no write, "
          "no cost -- the verifier passes it, and so does the review",
          verify(t06, db).passed and r["first"] == ["escalate_to_human"]
          and not r["forbidden"] and r["writes"] == [("escalate_to_human", True)], str(r))
    _, _, rec, db, _ = run([ver6, look6, refund6, "It is out of the window.", refund6, "Done."],
                           "check", task=t06)
    r = cl.review(rec.recs, t06)
    check("a forbidden call the conversation had already made is its failure, not the "
          "phase's cost", not verify(t06, db).passed and not r["forbidden"], str(r))
    _, _, rec, _, _ = run([ver6, look6, "It is out of the window.", refund6, "Done."],
                          "check", task=t06)
    r = cl.review(rec.recs, t06)
    check("…while one first made after the customer left is",
          [t for t, _ in r["forbidden"]] == ["issue_refund"], str(r))

    t03 = BY_ID["T03"]
    wrong = ("verify_identity", {"user_id": t03.user_id, "phone_last4": "0000"})
    _, _, rec, db, _ = run([("get_order", {"order_id": "O1002"}), "Let me check.", wrong,
                            "Could not verify."], "check", task=t03)
    r = cl.review(rec.recs, t03)
    check("a verification the tool denied is not a verification the phase made",
          r["ran"] and "verify_identity" not in r["first"]
          and any("denied" in f for f in verify(t03, db).failures), str(r))

    res, _, rec, _, _ = run([look, explain] + [look] * 10, "check")
    r = cl.review(rec.recs, task)
    check(f"the closing phase is capped at {cl.MAX_STEPS} model calls, the episode ends "
          "as the conversation did, and the review says it was cut short",
          res.stop_reason == StopReason.DONE and "closing_cut_short" in rec.events
          and sum(1 for s in rec.steps if s.get("closing")) == cl.MAX_STEPS
          and r["cut"] == ["closing.MAX_STEPS"], str(r))
    res, _, rec, _, _ = run(plan, "check", budget=Budget(max_steps=2))
    r = cl.review(rec.recs, task)
    check("…when the episode's budget runs out inside it, that is not a budget stop: the "
          "conversation had ended, and the review counts the note given, nothing done",
          res.stop_reason == StopReason.DONE and r["noted"] and not r["ran"]
          and r["cut"] == ["max_steps"], str(r))
    res, _, rec, db, _ = run(plan, "check", budget=Budget(max_user_turns=1))
    check("…and the cap on customer turns does not end it: no customer is left to take one",
          res.stop_reason == StopReason.DONE and "closing_cut_short" not in rec.events
          and verify(task, db).passed, str(rec.events))

    backend, cust = ScriptedBackend(list(plan)), Customer()
    sim = LLMUser(cust, persona="a customer", facts={})
    db = Database.fresh(task.db_patch)
    first = run_episode(task, db, backend, simulator=sim, closing="check",
                        interrupt_after_steps=3)
    state2, db2 = restore(snapshot(first.state, db))
    second = run_episode(task, db2, backend, simulator=sim, closing="check", state=state2)
    check("an episode interrupted in its closing phase resumes in it, and ends with the "
          "database an uninterrupted one ends with",
          first.stop_reason == StopReason.INTERRUPTED and second.stop_reason == StopReason.DONE
          and db2.t("escalations") == done_db.t("escalations") and verify(task, db2).passed)

    try:
        run_episode(task, Database.fresh(task.db_patch), ScriptedBackend([]), closing="on")
        bad_mode = False
    except ValueError:
        bad_mode = True
    check("an unknown closing mode is refused, not ignored", bad_mode)

    with tempfile.TemporaryDirectory() as d:
        sweep = [sys.executable, "-m", "pasarbench.sweep", "--backend", "scripted",
                 "--suite", "core", "--tasks", "T14", "--strategies", "full",
                 "--run-id", "z", "--trace-root", d]
        cwd = Path(__file__).resolve().parent.parent
        r = subprocess.run(sweep + ["--closing", "check"], capture_output=True, text=True, cwd=cwd)
        heads = [json.loads(f.read_text().splitlines()[0]) for f in Path(d).glob("z/*/*.jsonl")]
        check("the sweep's --closing reaches every episode's trace header",
              r.returncode == 0 and heads and all(h.get("closing") == "check" for h in heads),
              r.stderr[-500:] or str(heads))
        again = subprocess.run(sweep + ["--resume"], capture_output=True, text=True, cwd=cwd)
        check("--resume will not add episodes without the closing check to a cell run with it",
              again.returncode != 0 and "closing 'check'" in again.stdout + again.stderr,
              (again.stdout + again.stderr)[-400:])


def test_context_keeps_the_closing_note():
    """The note is the agent's whole instruction once the customer has left,
    and the phase can outlast a short window: window4 dropped it after four
    tool calls, summarize4 folded it away, notes4 too once notes existed."""
    print("\n=== every context strategy keeps the closing note ===")
    from pasarbench.harness import closing as cl
    from pasarbench.harness.context import NoteTaking, Summarize
    from pasarbench.harness.types import EpisodeState, ModelResponse, Usage

    def cycle(i, name="get_order"):
        args = {"content": f"fact {i}"} if name == "write_note" else {"order_id": "O1009"}
        return [Message("assistant", "", tool_calls=[ToolCall(name=name, arguments=args,
                                                              id=f"c{i}")]),
                Message("tool", '{"ok": true}', name=name, tool_call_id=f"c{i}")]

    class Summary:
        def chat(self, messages, tools):
            return ModelResponse(content='{"customer_goal": "a refund"}', usage=Usage(1, 1))

    msgs = [Message("system", "policy"), Message("user", "I want a refund for O1009")]
    msgs += cycle(0, "write_note") + cycle(1)
    msgs += [Message("assistant", "A refund was already issued."), Message("user", "Bye"),
             Message("user", cl.NOTE)]
    for i in range(2, 8):
        msgs += cycle(i)
    strategies = [FullContext(), SlidingWindow(4), SlidingWindow(8), ToolResultTrim(3),
                  Summarize(Summary(), keep_recent=4, trigger_units=8), NoteTaking(4)]
    built = {}
    for strat in strategies:
        state = EpisodeState(task_id="T14")
        state.messages = list(msgs)
        built[strat.name] = strat.build(state)
    kept = {n: sum(m.content == cl.NOTE for m in b) for n, b in built.items()}
    check("after six calls in the closing phase, every strategy still sends the note, once",
          all(v == 1 for v in kept.values()), str(kept))
    whole = {n: all(any(p.role == "assistant" and any(tc.id == m.tool_call_id
                                                      for tc in p.tool_calls)
                        for p in b[:i]) for i, m in enumerate(b) if m.role == "tool")
             for n, b in built.items()}
    check("…without orphaning a tool result", all(whole.values()), str(whole))
    order = {n: [m.content for m in b].index(cl.NOTE) < max(
        i for i, m in enumerate(b) if m.role == "tool") for n, b in built.items()}
    check("…and before the calls that answered it", all(order.values()), str(order))


def main() -> int:
    test_reference_through_loop()
    test_budgets_bite()
    test_error_degrades()
    test_interrupt_resume()
    test_context_never_orphans()
    test_traces()
    test_prompt_modes()
    test_claim_guardrail()
    test_tool_input_checks()
    test_closing_check()
    test_context_keeps_the_closing_note()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
