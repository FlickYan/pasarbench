"""
Context strategies.

This is the seam the week-4 ablation runs through. Every strategy takes the
full episode history and returns the messages actually sent to the model.
Swapping one for another must change nothing else.

Shipped now:
    FullContext      -- baseline, send everything
    SlidingWindow    -- keep the last N units
    ToolResultTrim   -- keep all turns, truncate stale tool payloads

Week 4 adds Summarize, NoteTaking and JITRetrieval behind the same interface.

The non-obvious part is UNIT GROUPING. An assistant message carrying tool_calls
and the tool messages answering them are ONE atomic unit. Truncating between
them orphans a tool_result, and every major API rejects that with a 400. Most
homegrown sliding windows have this bug and it shows up as a mysterious error
rate that people blame on the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from .types import EpisodeState, Message


@dataclass
class Unit:
    """An atomic slice of history that must never be split."""
    messages: list[Message]
    kind: str          # system | user | assistant_text | tool_cycle

    @property
    def chars(self) -> int:
        return sum(len(m.content or "") for m in self.messages)


def to_units(messages: list[Message]) -> list[Unit]:
    units: list[Unit] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "system":
            units.append(Unit([m], "system"))
            i += 1
        elif m.role == "user":
            units.append(Unit([m], "user"))
            i += 1
        elif m.role == "assistant" and m.tool_calls:
            group = [m]
            j = i + 1
            while j < len(messages) and messages[j].role == "tool":
                group.append(messages[j])
                j += 1
            units.append(Unit(group, "tool_cycle"))
            i = j
        else:
            units.append(Unit([m], "assistant_text"))
            i += 1
    return units


def flatten(units: list[Unit]) -> list[Message]:
    return [m for u in units for m in u.messages]


class ContextStrategy(Protocol):
    name: str
    extra_tools: list[str]      # tools this strategy needs exposed, e.g. notes

    def build(self, state: EpisodeState) -> list[Message]:
        ...


class FullContext:
    """Send everything. The baseline every other strategy is measured against."""
    name = "full"
    extra_tools: list[str] = []

    def build(self, state: EpisodeState) -> list[Message]:
        return list(state.messages)


class SlidingWindow:
    """Keep the system prompt, the opening user message, and the last N units.

    The opening message is pinned because it usually carries the customer's
    actual complaint. Dropping it is the classic sliding-window failure: the
    agent forgets what it is solving and starts asking the customer to repeat
    themselves around turn 15.
    """

    extra_tools: list[str] = []

    def __init__(self, keep_units: int = 8, pin_opening: bool = True):
        self.keep_units = keep_units
        self.pin_opening = pin_opening
        self.name = f"window{keep_units}{'+pin' if pin_opening else ''}"

    def build(self, state: EpisodeState) -> list[Message]:
        units = to_units(state.messages)
        system = [u for u in units if u.kind == "system"]
        rest = [u for u in units if u.kind != "system"]

        pinned: list[Unit] = []
        if self.pin_opening and rest and rest[0].kind == "user":
            pinned = [rest[0]]
            rest = rest[1:]

        tail = rest[-self.keep_units:] if self.keep_units > 0 else []
        if pinned and pinned[0] in tail:
            pinned = []

        dropped = len(rest) - len(tail)
        out = system + pinned
        if dropped > 0:
            out.append(Unit([Message(role="user",
                                     content=f"[{dropped} earlier exchanges omitted]")],
                            "user"))
        out += tail
        return flatten(out)


class ToolResultTrim:
    """Keep every turn but shrink old tool payloads to a stub.

    Conversation structure is usually cheap; tool JSON is what actually fills
    the window. This is the strategy that most often wins on the cost axis in
    practice, which is why it is worth having in the ablation from day one.
    """

    extra_tools: list[str] = []

    def __init__(self, keep_full: int = 3, stub_chars: int = 160):
        self.keep_full = keep_full
        self.stub_chars = stub_chars
        self.name = f"trim{keep_full}"

    def build(self, state: EpisodeState) -> list[Message]:
        units = to_units(state.messages)
        cycles = [i for i, u in enumerate(units) if u.kind == "tool_cycle"]
        recent = set(cycles[-self.keep_full:]) if self.keep_full > 0 else set()

        out: list[Unit] = []
        for i, u in enumerate(units):
            if u.kind != "tool_cycle" or i in recent:
                out.append(u)
                continue
            trimmed = []
            for m in u.messages:
                if m.role == "tool" and len(m.content) > self.stub_chars:
                    trimmed.append(Message(role="tool", name=m.name,
                                           tool_call_id=m.tool_call_id,
                                           content=_stub(m.content, self.stub_chars)))
                else:
                    trimmed.append(m)
            out.append(Unit(trimmed, "tool_cycle"))
        return flatten(out)


def _stub(content: str, limit: int) -> str:
    """Keep the ok/error verdict, drop the payload. The verdict is what the
    agent needs later; the full row rarely is."""
    try:
        d = json.loads(content)
        head = {"ok": d.get("ok")}
        if not d.get("ok"):
            head["error"] = d.get("error")
        return json.dumps(head) + f" [payload elided, {len(content)} chars]"
    except (json.JSONDecodeError, AttributeError):
        return content[:limit] + f"... [elided, {len(content)} chars]"


SUMMARY_SYSTEM = """You compress a customer-service conversation so another agent can \
continue it with no loss of anything that matters.

Return ONLY a JSON object with exactly these keys:
  customer_goal      one sentence
  order_id           the order id, or "unknown"
  identity_verified  "yes", "no", or "unknown"
  facts_established  list of short strings: what tools have confirmed
  actions_taken      list of short strings: "tool_name -> outcome"
  outstanding        what still needs doing, or what the agent is waiting on

No prose, no markdown fences."""

SUMMARY_FIELDS = ("customer_goal", "order_id", "identity_verified",
                  "facts_established", "actions_taken", "outstanding")


class Summarize:
    """Fold old units into a RUNNING, STRUCTURED summary.

    Two decisions here matter more than the compression ratio.

    STRUCTURED, NOT FREE-TEXT. A generic "summarise this" prompt reliably drops
    exactly the things the policy gates on: whether identity was verified, and
    what the eligibility check actually returned. The agent then re-runs a
    write action it already ran, or acts without verifying. Pinning a schema
    with `identity_verified` and `facts_established` as required keys is what
    makes this strategy competitive instead of catastrophic.

    RUNNING, NOT REDONE. The summary lives in EpisodeState and only NEW units
    are folded in, so cost is O(conversation) rather than O(conversation^2) and
    it survives interrupt/resume. Re-summarising the whole history every step
    is the naive implementation and it is more expensive than sending the raw
    context it was meant to replace.

    What it LOSES is the interesting measurement -- run the ablation, then diff
    per-trap pass rates against `full` and read the traces where they differ.
    """
    extra_tools: list[str] = []

    def __init__(self, backend, keep_recent: int = 4, trigger_units: int = 8):
        self.backend = backend
        self.keep_recent = keep_recent
        self.trigger_units = trigger_units
        self.name = f"summarize{keep_recent}"

    def build(self, state: EpisodeState) -> list[Message]:
        units = to_units(state.messages)
        system = [u for u in units if u.kind == "system"]
        rest = [u for u in units if u.kind != "system"]
        if len(rest) <= self.trigger_units:
            return flatten(units)

        cutoff = len(rest) - self.keep_recent
        if cutoff > state.summarized_upto:
            new = rest[state.summarized_upto:cutoff]
            state.summary = self._fold(state.summary, new)
            state.summarized_upto = cutoff

        head = Message(role="user",
                       content=f"[summary of the conversation so far]\n{state.summary}")
        return flatten(system) + [head] + flatten(rest[state.summarized_upto:])

    def _fold(self, previous: str, new_units: list[Unit]) -> str:
        transcript = []
        for u in new_units:
            for m in u.messages:
                body = m.content or ""
                if m.tool_calls:
                    body += " CALLS: " + ", ".join(
                        f"{tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})"
                        for tc in m.tool_calls)
                transcript.append(f"{m.role}: {body[:600]}")
        prompt = (f"Existing summary (may be empty):\n{previous or '(none)'}\n\n"
                  f"New exchanges to fold in:\n" + "\n".join(transcript))
        try:
            resp = self.backend.chat(
                [Message("system", SUMMARY_SYSTEM), Message("user", prompt)], tools=[])
            return self._normalise(resp.content or "", previous)
        except Exception:            # noqa: BLE001 - never kill an episode here
            return previous or "(summarisation unavailable)"

    @staticmethod
    def _normalise(raw: str, previous: str) -> str:
        text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            # A malformed summary is better than a lost one, but it means the
            # schema did not hold -- count these, they predict where the
            # strategy fails.
            return (previous + "\n" + text.strip())[-4000:] if previous else text.strip()
        lines = []
        for k in SUMMARY_FIELDS:
            v = d.get(k)
            if isinstance(v, list):
                v = "; ".join(str(x) for x in v) or "(none)"
            lines.append(f"{k}: {v if v not in (None, '') else 'unknown'}")
        return "\n".join(lines)


class NoteTaking:
    """Agentic memory: the agent keeps its own scratchpad and old tool cycles
    are dropped once it has written something down.

    THE FAILURE MODE IS THE AGENT SIMPLY NOT WRITING NOTES. Then this degrades
    to an aggressive sliding window with no pinned facts, and it does so
    silently. `note_discipline` measures it; report that number next to the
    pass rate or the strategy's result is uninterpretable.
    """
    extra_tools = ["write_note", "read_notes"]

    def __init__(self, keep_recent: int = 4):
        self.keep_recent = keep_recent
        self.name = f"notes{keep_recent}"

    def build(self, state: EpisodeState) -> list[Message]:
        units = to_units(state.messages)
        system = [u for u in units if u.kind == "system"]
        rest = [u for u in units if u.kind != "system"]

        notes = self._notes_from(state.messages)
        # Nothing written down yet: do not drop anything. Compacting a
        # conversation the agent has not recorded is how this strategy loses
        # tasks it should win.
        if not notes or len(rest) <= self.keep_recent:
            return flatten(units)

        pinned = [rest[0]] if rest and rest[0].kind == "user" else []
        tail = rest[-self.keep_recent:]
        board = Message(role="user",
                        content="[your notes]\n" + "\n".join(f"- {n}" for n in notes))
        return flatten(system + pinned) + [board] + flatten(tail)

    @staticmethod
    def _notes_from(messages: list[Message]) -> list[str]:
        out = []
        for m in messages:
            for tc in m.tool_calls:
                if tc.name == "write_note":
                    c = tc.arguments.get("content")
                    if c:
                        out.append(str(c))
        return out


def note_discipline(messages: list[Message]) -> dict[str, int]:
    """Did the agent actually use the scratchpad? Read this before believing
    any NoteTaking result."""
    notes = NoteTaking._notes_from(messages)
    tool_cycles = sum(1 for u in to_units(messages) if u.kind == "tool_cycle")
    return {"notes_written": len(notes), "tool_cycles": tool_cycles,
            "wrote_any": int(bool(notes))}


# NOTE ON FACTORISATION
# ---------------------
# JIT policy retrieval is NOT a context strategy. It is an orthogonal axis
# (`policy_mode` in prompts.py) governing what goes into the SYSTEM prompt,
# while these strategies govern what happens to CONVERSATION HISTORY. Folding
# them together is why many published context ablations cannot be read: a
# single "JIT" row conflates a smaller system prompt with a different history
# policy. Sweep them as a grid instead -- 5 strategies x 2 policy modes.

BASE_STRATEGIES: dict[str, ContextStrategy] = {
    "full": FullContext(),
    "window8": SlidingWindow(8),
    "window4": SlidingWindow(4),
    "trim3": ToolResultTrim(3),
    "notes4": NoteTaking(4),
}

STRATEGIES = BASE_STRATEGIES        # back-compat for existing callers


def make_strategy(name: str, summarizer=None) -> ContextStrategy:
    """`summarize4` needs a model; everything else is free."""
    if name.startswith("summarize"):
        if summarizer is None:
            raise ValueError("the summarize strategy needs a summarizer backend "
                             "(pass --summarizer to the sweep)")
        keep = int(name.removeprefix("summarize") or 4)
        return Summarize(summarizer, keep_recent=keep)
    if name not in BASE_STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; have "
                         f"{list(BASE_STRATEGIES)} + summarize<N>")
    return BASE_STRATEGIES[name]
