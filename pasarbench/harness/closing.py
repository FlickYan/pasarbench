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
"""

from __future__ import annotations

from typing import Any, Iterable

MODES = ("off", "check")
# The closing phase's own cap, inside the episode's budget: enough to look a
# tool up, call it and write the note, not enough to start the case again.
MAX_STEPS = 6
# Fixed before any run (RUNBOOK 1j). A reworded note is a new intervention and
# needs tasks it was not tuned on.
NOTE = ("[Support system: the customer has left the conversation and will not see "
        "anything you write from now on.]\n"
        "Before this case closes, check what you did against the policy. If the policy "
        "requires an action in this case that no tool call has made -- including one you "
        "told the customer was done or under way -- make that call now. Take no action "
        "the policy does not require. When you are finished, write a one-line note for "
        "the case file.")


def started(messages: Iterable[Any]) -> int | None:
    """Index of the closing note in the conversation, or None. Read from the
    messages, not kept in a variable, so an episode interrupted in its closing
    phase resumes in it."""
    for i, m in enumerate(messages):
        if getattr(m, "role", None) == "user" and (getattr(m, "content", "") or "") == NOTE:
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
               for an escalation, the same category -- a second ticket for
               one case. No check forbids it; a person handles it twice.
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
        "writes": [], "first": [], "forbidden": [], "repeated": [], "guessed": [],
        "unscored": [], "at_note": None, "at_end": None,
        "left_saying": next((str(e.get("text") or "").strip() for e in reversed(events)
                             if e.get("kind") == "user_turn" and e.get("ended")), None),
        "not_replayed": None}
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
    checks = task.checks
    for k, a in enumerate(wrote):
        if _target(a) in made:
            out["repeated"].append((a.tool, a.args))
        elif not any(spec.via(a.tool, a.args) for spec in checks.required_actions) \
                and not any(spec.matches(a) for spec in checks.forbidden_actions) \
                and not seen(k):
            out["unscored"].append((a.tool, a.args))
        made.add(_target(a))
    said = _digits(" ".join([task.opening] + [str(e.get("text") or "") for e in events
                                               if e.get("kind") == "user_turn"]))
    out["guessed"] = [str(a.args.get("phone_last4")) for a in during
                      if a.tool == "verify_identity"
                      and _digits(str(a.args.get("phone_last4"))) not in said]
    return out


# What a write acts on. An escalation's category is part of it: escalating a
# case again under the category the policy names is a correction, not a copy.
_TARGET = ("order_id", "order_item_id", "user_id", "category")


def _target(a) -> tuple:
    return (a.tool,) + tuple((k, a.args.get(k)) for k in _TARGET if k in a.args)


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
