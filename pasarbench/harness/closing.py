"""
The closing check: when the customer leaves, the agent gets one note -- the
customer never sees it -- asking it to go over the case against the policy
before it closes, make any call the policy requires that it has not made, and
make no other.

With the tools named (docs/RUNBOOK.md, 1h), the largest group of remaining
failures is neither an agent that could not find a tool nor one that claimed
an action it never took (the claim guardrail, 1i). It is an agent that knew
the rule and let it go: it explained the earlier refund, the customer said
"Oh, I didn't notice, thanks", and it closed -- though P10 says, in bold, to
escalate even when the customer accepts the explanation. Sixteen of the named
arm's 34 failures. A check on what the agent says cannot reach that. A moment
to review the case before it closes might.

It reads nothing about the task -- not its trap, its checks or its answer --
so a deployed agent could run it: work after the customer has gone is ordinary
in support, and the note only points the agent at the policy it already has.

What it changes is recorded: the trace header carries `closing`, the agent's
steps after the note carry `closing: true`, and the note itself is an event
(`closing_check`), as is a phase cut short (`closing_cut_short`). Any write
made in the closing phase is a real write -- the verifier reads the database
the episode ended with -- but the text written then is a case note no customer
reads, so readers of what the customer saw skip it (`trace.customer_saw`), and
the judges are told where the customer left. Forbidden writes it causes are
its cost, and `review` reports them.

The phase runs inside the episode's budget for steps, tool calls, tokens and
time, with MAX_STEPS of its own; the cap on customer turns does not apply, as
no customer is left to take one. Context strategies that drop old messages
keep the note (context.py): it is the agent's whole instruction for the phase.

That budget is what the conversation left of it, and on the longest cases it
left nothing: in 1q the episode's 120,000 tokens cut 27 of 456 closing phases
short, 17 of them before the agent took a step -- each step's prompt is the
whole conversation by then. With `closing_room` (v49, opt-in, in the trace
header) the phase is bounded by its own steps alone: the episode's steps, tool
calls and tokens end with the conversation (ROOM), its time does not, and the
conversation itself runs exactly as before.

Two notes. "check" is 1j's, unchanged since the run that fixed it. Its one
cost in every run since has been a call made again -- 22 escalations in 1m's
186 episodes, of cases the conversation had already escalated -- and 1k named
the fix: have the agent read its own calls before it makes one. Asking it to
list them would end the phase, which ends on the agent's first reply without
a call. So "record" (v35, RUNBOOK 1n) carries the list itself: the case
record, every identity check and every action the conversation attempted,
each with what came of it, read off the database's action log as a support
system's case history would show it -- then 1j's note, with one sentence
more. It reads nothing about the task either. The trace's closing_check
event carries the note it gave.

It did not stop the call made again (1n): nearly every second call, in every
run, was the agent's first answer to the note -- 91 of 93 -- made with the
first call in front of it. A note cannot reach that; a refusal can. "guard"
(v37, RUNBOOK 1o) gives 1j's note unchanged and, in the closing phase only,
does not run a write that repeats one already made, by the review's own
definition (`repeat_of`): the agent is told it was already done and that
nothing new was filed (`refusal`), and the trace records a closing_refused
event. The call never reaches the tools, so a replay reproduces it from the
recorded error (replay.NO_DISPATCH), as it does an unknown tool.

Where it was not built for (1p) it refused a correction: the shipping credit
P4.6 asks for after a credit for the item, read as the same credit twice,
because a target without the amount cannot tell the two apart (WHAT_FAILED
#39). Since v41 a payment's amount and currency are part of its target, for
the review's count and the guard alike (GUARD_VERSION 2, in the trace header
as closing_guard): the same payment again is still a repeat, and refused; one
of another amount is not, and is listed to be read (`another_amount`).
"""

from __future__ import annotations

from typing import Any, Iterable

MODES = ("off", "check", "record", "guard")
# The closing phase's own cap, inside the episode's budget: enough to look a
# tool up, call it and write the note, not enough to start the case again.
MAX_STEPS = 6
# The episode's limits that stop at the conversation when the phase has room
# of its own (loop.run_episode's closing_room): its steps, tool calls and
# tokens. Not its wall clock, nor MAX_STEPS above.
ROOM = ("max_steps", "max_tool_calls", "max_tokens")
# The first line of every closing note, and how a note is told from any other
# message -- by a resumed episode, and by a context strategy that must keep it.
HEAD = ("[Support system: the customer has left the conversation and will not see "
        "anything you write from now on.]")
# Fixed before any run (RUNBOOK 1j). A reworded note is a new intervention and
# needs tasks it was not tuned on.
NOTE = (HEAD + "\n"
        "Before this case closes, check what you did against the policy. If the policy "
        "requires an action in this case that no tool call has made -- including one you "
        "told the customer was done or under way -- make that call now. Take no action "
        "the policy does not require. When you are finished, write a one-line note for "
        "the case file.")
# Fixed before any run (RUNBOOK 1n): the case record, then 1j's note with the
# record in place of "no tool call" and one sentence more.
RECORD_NOTE = (HEAD + "\n"
               "The case record -- every identity check and every action attempted in "
               "this conversation, in order:\n"
               "{record}\n"
               "Before this case closes, check what you did against the policy. If the "
               "policy requires an action in this case that no call in the record made -- "
               "including one you told the customer was done or under way -- make that "
               "call now. A call the record shows as done is done: do not make it again. "
               "Take no action the policy does not require. When you are finished, write a "
               "one-line note for the case file.")
NO_RECORD = "(nothing: no identity check or action was attempted)"
# A value longer than this is cut in the record: a reason can run to a
# paragraph, and the record is for telling one call from another.
SHOWN = 60


def note(mode: str, log: Iterable[Any] = ()) -> str:
    """The note the agent gets when the customer leaves, in this mode: 1j's,
    or the case record (`record`) and the note after it."""
    if mode in ("check", "guard"):
        return NOTE
    if mode == "record":
        return RECORD_NOTE.replace("{record}", "\n".join(record(log)) or NO_RECORD)
    raise ValueError(f"no closing note in mode {mode!r}")


def record(log: Iterable[Any]) -> list[str]:
    """The case record: every identity check and every write the conversation
    attempted, in order, each with what came of it -- from the database's
    action log, which every tool call that reached the tools wrote to. A read
    changes nothing and is left out; so is a call that never reached the
    tools (an unknown tool, arguments that were not JSON), which did nothing.
    What the agent did, and nothing about the task."""
    from ..tools import WRITE_TOOLS
    out = []
    for a in log:
        if a.tool not in WRITE_TOOLS and a.tool != "verify_identity":
            continue
        args = ", ".join(f"{k}={_shown(v)}" for k, v in (a.args or {}).items() if v is not None)
        if not a.ok:
            what = f"not done: {_cut(str(a.error or 'refused'), 2 * SHOWN)}"
        elif a.tool == "verify_identity":
            what = "not verified: the digits did not match" if a.denied else "verified"
        else:
            what = "done"
        out.append(f"{len(out) + 1}. {a.tool}({args}) -- {what}")
    return out


def _cut(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _shown(value: Any) -> str:
    import json
    if isinstance(value, str):
        value = _cut(value, SHOWN)
    return json.dumps(value, ensure_ascii=False, default=str)


# How the guard's refusal begins: replay.NO_DISPATCH knows it, so a replay
# reproduces the refused call from its recorded error instead of running it.
REFUSED = "already done in this case: "
# What the guard compares, by version: 1 (v37-v40) a payment without its
# amount; 2 (v41) with it. Recorded in the trace header, so a run under one is
# not resumed under the other.
GUARD_VERSION = 2


def repeat_of(log: Iterable[Any], tool: str, args: Any) -> Any:
    """The successful write in the log that a call of `tool` with `args`
    would repeat -- the same tool on the same order, item or customer and, for
    an escalation, the same category, for a payment the same amount and
    currency, as `review` counts a repeat -- or None. The call's target is
    read from the keys the tool would log: every write logs all its
    parameters, an optional one as None when it is not given; an amount sent
    as 4700.0 is the 4700 the tool would log."""
    from ..tools import TOOLS, WRITE_TOOLS, _integers
    if tool not in WRITE_TOOLS or not isinstance(args, dict):
        return None
    params = TOOLS[tool]["schema"]["function"]["parameters"]["properties"]
    args = _integers(params, args)
    target = (tool,) + tuple((k, args.get(k)) for k in _keys(tool) if k in params)
    return next((a for a in reversed(list(log))
                 if a.ok and a.tool in WRITE_TOOLS and _target(a) == target), None)


def would_refuse(recs: list[dict[str, Any]], task) -> list[tuple[str, Any]] | None:
    """The calls the guard would not have run in a closing phase recorded
    without it: the episode replayed to the note, then its closing calls
    walked in order, each write asked `repeat_of` before it runs as it ran. A
    call the loop never dispatched (an unknown tool, say) is skipped, as it
    never reached the tools. None without a closing phase, or when the
    conversation does not replay (scripts/closing_guard_dry_run.py)."""
    from ..tools import WRITE_TOOLS, call
    from .replay import NO_DISPATCH, replay
    head = [r for r in recs if r.get("type") == "header"][:1]
    steps = [r for r in recs if r.get("type") == "step"]
    at = next((i for i, st in enumerate(steps) if st.get("closing")), None)
    if at is None or not head or task is None:
        return None
    rebuilt = replay(head + steps[:at], task)
    if rebuilt.diverged or rebuilt.mismatched:
        return None
    db, checks = rebuilt.db, int(head[0].get("tool_checks") or 1)
    out = []
    for st in steps[at:]:
        for tc, tr in zip(st.get("tool_calls") or [], st.get("tool_results") or []):
            if any((tr.get("error") or "").startswith(p) for p in NO_DISPATCH):
                continue
            name, args = tc.get("name"), tc.get("arguments")
            if name in WRITE_TOOLS and repeat_of(db.action_log, name, args) is not None:
                out.append((name, args))
            if name in WRITE_TOOLS or name == "verify_identity":
                call(db, name, args if isinstance(args, dict) else {}, checks=checks)
    return out


def refusal(done: Any) -> str:
    """What the agent is told when the guard does not run a repeat."""
    args = ", ".join(f"{k}={_shown(v)}" for k, v in (done.args or {}).items() if v is not None)
    return (f"{REFUSED}{done.tool}({args}) succeeded earlier, so this call would make it "
            f"a second time. It was not run, and nothing new was filed.")


def is_note(content: str | None) -> bool:
    """Whether a message is a closing note, in either mode."""
    return (content or "").startswith(HEAD + "\n")


def started(messages: Iterable[Any]) -> int | None:
    """Index of the closing note in the conversation, or None. Read from the
    messages, not kept in a variable, so an episode interrupted in its closing
    phase resumes in it."""
    for i, m in enumerate(messages):
        if getattr(m, "role", None) == "user" and is_note(getattr(m, "content", "")):
            return i
    return None


def steps_since(messages: list[Any], at: int) -> int:
    """Model calls the agent has made since the note."""
    return sum(1 for m in messages[at + 1:] if getattr(m, "role", None) == "assistant")


def review(recs: list[dict[str, Any]], task=None) -> dict[str, Any]:
    """What the closing phase did in one episode, from its trace records
    (header, steps, events).

    noted      the customer left and the agent was given the note
    ran        the agent made a model call after it
    cut        why the phase was stopped before the agent finished, if it was:
               "closing.MAX_STEPS", or the budget that ran out
    writes     [(tool, ok)] write attempts that reached the tools after the note
    first      required actions first made after the note -- what it rescued,
               if the episode passed
    forbidden  [(tool, args)] forbidden actions first made after the note --
               its cost
    repeated   [(tool, args)] writes after the note that repeat one already
               made: the same tool on the same order, item or customer and,
               for an escalation, the same category, for a payment the same
               amount and currency (since v41, WHAT_FAILED #39) -- a second
               ticket for one case, a payment made twice. No check forbids
               it; a person handles it twice.
    another_amount  [(tool, args)] payments after the note on an order, item or
               customer already paid, of another amount or currency: the
               shipping P4.6 refunds after the item, say -- a correction the
               policy asks for, or a second payment it does not. Not a
               repeat, not counted: listed, to be read.
    repeated_at  [step] for each of those, the step of the closing phase that
               made it: 1 is the agent's first answer to the note. None where
               the calls and the log do not pair up. 1n found nearly every
               repeat at step 1, under either note: made before anything the
               note said could change it.
    refused    [(tool, args)] calls the guard did not run as repeats (mode
               "guard", RUNBOOK 1o), and refused_at the step of each, as above.
               Not writes: none reached the tools.
    substituted  [(tool, args)] writes after the first refusal that no check
               asks for -- not required, and forbidden or seen by no check:
               the same order escalated under another category, say. The
               repeat in other words, and counted as a cost.
    guessed    [digits] identity verifications after the note with digits the
               customer never gave -- there was no one left to ask.
    unscored   [(tool, args)] writes after the note that the checks do not see:
               not required, not forbidden, not a repeat, and the episode
               replayed without the write ends with the same verdict and the
               same failures. A write a database check needs -- a return the
               check counts, though no required action names it -- is seen,
               and is not here. These are read, not counted.
    at_note    whether the case passed as the customer left it -- the verdict
               the episode would have had without the note
    at_end     whether it passed at the end. at_note -> at_end is the closing
               phase's whole effect on the verdict, episode by episode, with
               the conversation held fixed: a pass it broke shows here however
               it broke it, which "forbidden" alone does not see.
    left_saying  the customer's last words, as she left ("" if she said
               nothing). A rescue after "yes, go ahead, thanks" is not a
               second look: it is the turn the agent never had, because
               without the note the episode ends on those words before the
               agent reads them (WHAT_FAILED #38). Read it beside "first".
    not_replayed  why the rest could not be read, if it could not: the task
               changed since the run, or a call replays differently. Then the
               lists are empty and the verdicts None -- as rescore.py leaves
               such an episode's recorded verdict alone rather than guess.

    The 1j run found "repeated" and "guessed", which its rule had not counted
    (RUNBOOK 1j): 15 escalations made a second time and 2 verifications tried
    with "0000", in 171 episodes; neither ever happened without the note. 1k
    added "unscored", the verdicts before and after, and "left_saying": what
    it did that no check scores, what it changed, and what the customer last
    said -- 4 of 1k's 6 rescues followed a yes the agent never got to act on.

    All but the first three read what the verifier reads: the database's
    action log, rebuilt by replaying the episode's calls (replay.py) and split
    where the note came. So a call that never reached the tools -- an unknown
    tool, arguments that were not JSON, a spent budget -- is not an action,
    and a verification the tool denied is not a verification. A forbidden
    action the conversation had already taken is the conversation's failure,
    not the closing phase's: "forbidden", like "first", counts what the phase
    did first. Without the task they are empty, and the verdicts None.
    """
    steps = [r for r in recs if r.get("type") == "step"]
    events = [r for r in recs if r.get("type") == "event"]
    out: dict[str, Any] = {
        "noted": any(e.get("kind") == "closing_check" for e in events),
        "ran": any(st.get("closing") for st in steps),
        "cut": [e.get("reason") for e in events if e.get("kind") == "closing_cut_short"],
        "writes": [], "first": [], "forbidden": [], "repeated": [], "repeated_at": [],
        "another_amount": [], "guessed": [], "refused": [], "refused_at": [],
        "substituted": [],
        "unscored": [], "at_note": None, "at_end": None,
        "left_saying": next((str(e.get("text") or "").strip() for e in reversed(events)
                             if e.get("kind") == "user_turn" and e.get("ended")), None),
        "not_replayed": None}
    refusals = [e for e in events if e.get("kind") == "closing_refused"]
    opened = next((st.get("step") for st in steps if st.get("closing")), None)
    out["refused"] = [(e.get("tool"), e.get("args")) for e in refusals]
    out["refused_at"] = [e["step"] - opened + 1 if opened is not None and e.get("step")
                         else None for e in refusals]
    if task is None or not out["noted"]:
        return out
    from ..tasks import task_digest
    from ..tools import WRITE_TOOLS
    from ..verifier import verify
    from .replay import replay

    head = [r for r in recs if r.get("type") == "header"][:1]
    if head and head[0].get("task_digest") and head[0]["task_digest"] != task_digest(task):
        out["not_replayed"] = "the task changed since the run"
        return out
    at = next((i for i, st in enumerate(steps) if st.get("closing")), len(steps))
    end = replay(head + steps, task)
    if end.diverged or end.mismatched:
        out["not_replayed"] = (end.diverged or end.mismatched)[0]
        return out
    db_note, db_end = replay(head + steps[:at], task).db, end.db
    v_end = verify(task, db_end)
    out["at_note"] = verify(task, db_note).passed
    out["at_end"] = v_end.passed
    if not out["ran"]:
        return out
    n_before = len(db_note.action_log)
    log = db_end.action_log
    before, during = log[:n_before], log[n_before:]
    out["writes"] = [(a.tool, a.ok) for a in during if a.tool in WRITE_TOOLS]
    for spec in task.checks.required_actions:
        if (not any(spec.matches(a) for a in before)
                and any(spec.matches(a) for a in during)):
            out["first"].append(spec.tool)
    for spec in task.checks.forbidden_actions:
        hit = next((a for a in during if spec.matches(a)), None)
        if hit is not None and not any(spec.matches(a) for a in before):
            out["forbidden"].append((hit.tool, hit.args))
    # The phase's successful write calls, in order: each logged exactly one
    # successful action, so the k-th of them is the k-th successful write in
    # the log after the note (the replay reproduced every call's success).
    calls = [(i, j) for i in range(at, len(steps))
             for j, (tc, tr) in enumerate(zip(steps[i].get("tool_calls") or [],
                                              steps[i].get("tool_results") or []))
             if tc.get("name") in WRITE_TOOLS and tr.get("ok")]
    wrote = [a for a in during if a.tool in WRITE_TOOLS and a.ok]
    paired = len(calls) == len(wrote)

    def seen(k: int) -> bool:
        """Whether the checks see the k-th write: replayed without it, the
        episode ends with another verdict or other failures. If the calls and
        the log do not pair up, it cannot be told -- and a write is not called
        one no check sees on a guess."""
        if not paired:
            return True
        i, j = calls[k]
        st = steps[i]
        cut = {**st, "tool_calls": [c for n, c in enumerate(st.get("tool_calls") or []) if n != j],
               "tool_results": [c for n, c in enumerate(st.get("tool_results") or []) if n != j]}
        v = verify(task, replay(head + steps[:i] + [cut] + steps[i + 1:], task).db)
        return (v.passed, sorted(v.failures)) != (v_end.passed, sorted(v_end.failures))

    made = {_target(a) for a in before if a.tool in WRITE_TOOLS and a.ok}
    paid = {_payee(a) for a in before if a.tool in _PAYMENTS and a.ok}
    checks = task.checks
    for k, a in enumerate(wrote):
        if _target(a) in made:
            out["repeated"].append((a.tool, a.args))
            out["repeated_at"].append(calls[k][0] - at + 1 if paired else None)
        elif a.tool in _PAYMENTS and _payee(a) in paid:
            out["another_amount"].append((a.tool, a.args))
        elif not any(spec.via(a.tool, a.args) for spec in checks.required_actions) \
                and not any(spec.matches(a) for spec in checks.forbidden_actions) \
                and not seen(k):
            out["unscored"].append((a.tool, a.args))
        made.add(_target(a))
        if a.tool in _PAYMENTS:
            paid.add(_payee(a))
    if refusals and paired:
        # After the first refusal, a write no check asks for -- one that is
        # forbidden, or that no check sees -- is the same ticket made another way.
        no = (refusals[0].get("step") or 0, refusals[0].get("index") or 0)
        for k, a in enumerate(wrote):
            i, j = calls[k]
            if (steps[i].get("step") or 0, j) > no and (
                    (a.tool, a.args) in out["unscored"]
                    or (a.tool, a.args) in out["forbidden"]):
                out["substituted"].append((a.tool, a.args))
    said = _digits(" ".join([task.opening] + [str(e.get("text") or "") for e in events
                                               if e.get("kind") == "user_turn"]))
    out["guessed"] = [str(a.args.get("phone_last4")) for a in during
                      if a.tool == "verify_identity"
                      and _digits(str(a.args.get("phone_last4"))) not in said]
    return out


# What a write acts on. An escalation's category is part of it: escalating a
# case again under the category the policy names is a correction, not a copy.
_TARGET = ("order_id", "order_item_id", "user_id", "category")


# A payment's amount and currency are part of what it is (v41, WHAT_FAILED
# #39): a credit for the shipping after one for the item is the rest of what is
# owed, not the same credit twice. The same amount again still is.
_PAYMENTS = ("issue_refund", "issue_store_credit", "issue_goodwill_voucher")
_AMOUNT = ("amount_minor", "currency")


def _keys(tool: str) -> tuple[str, ...]:
    return _TARGET + (_AMOUNT if tool in _PAYMENTS else ())


def _target(a) -> tuple:
    return (a.tool,) + tuple((k, a.args.get(k)) for k in _keys(a.tool) if k in a.args)


def _payee(a) -> tuple:
    """Who and what a payment is for, without how much: the target before v41."""
    return (a.tool,) + tuple((k, a.args.get(k)) for k in _TARGET if k in a.args)


def paid_before(log: Iterable[Any], tool: str, args: Any) -> Any:
    """The successful payment in the log on the same order, item or customer
    as a payment of `tool` with `args`, of another amount or currency -- what
    a payment `review` lists as `another_amount` came after -- or None."""
    from ..tools import TOOLS, _integers
    if tool not in _PAYMENTS or not isinstance(args, dict):
        return None
    params = TOOLS[tool]["schema"]["function"]["parameters"]["properties"]
    args = _integers(params, args)
    payee = (tool,) + tuple((k, args.get(k)) for k in _TARGET if k in params)
    amount = tuple(args.get(k) for k in _AMOUNT if k in params)
    return next((a for a in log if a.ok and a.tool == tool and _payee(a) == payee
                 and tuple(a.args.get(k) for k in _AMOUNT if k in a.args) != amount), None)


_CJK = dict(zip("〇零一二三四五六七八九", "00123456789"))


def _digits(text: str) -> str:
    """Every digit in the text, in order, as ASCII: Thai and full-width digits
    and Chinese numerals included, so digits a customer gave in her own script
    are not mistaken for a guess."""
    import unicodedata
    out = []
    for ch in text:
        if ch.isdigit():
            try:
                out.append(str(unicodedata.digit(ch)))
            except ValueError:
                pass
        elif ch in _CJK:
            out.append(_CJK[ch])
    return "".join(out)
