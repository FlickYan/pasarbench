"""
The language swap (docs/RUNBOOK.md, 1t): the run the language finding needs.

1r found that deepseek-v4-pro escalates a refund already paid far less often
outside English, and that it decides in the turn it reads the refund. What
recorded runs cannot separate is the language from everything else that
differs between twin conversations -- the customer's words, the agent's own
earlier replies. This holds a recorded conversation fixed at that moment and
changes only its language.

A PREFIX is a recorded duplicate-refund conversation up to and including the
agent's first lookup of the order, whose result lists the refund already paid:
the system prompt, every message the agent saw, every tool payload (rebuilt by
replaying the recorded calls, each checked against the digest the run
recorded), and the world in the state those calls left it.

Each prefix runs in three ARMS, the agent continuing from it until it next
speaks to the customer:

    orig   the conversation as it was recorded
    swap   every customer turn and agent reply translated into the other
           language -- English for a local-language conversation, the
           market's language for an English one
    back   translated there and back: the same language as the recording,
           carrying whatever the translator changes

swap and back have both been through the translator and differ only in their
language, so their comparison is the test; orig against back says what the
translator alone does. The system prompt, the policy, the tools and their
payloads, the order of the messages and the world are the same in all three.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .db import Database
from .locales import LANGUAGE_NAMES, LANGUAGES_FOR_MARKET
from .tasks import task_digest, world_digest

TRAP = "duplicate_refund_escalate"
LOOKUP = "get_order"              # its result lists the refund already paid
ESCALATE = "escalate_to_human"
ENGLISH = ("en", "sg-en")
ARMS = ("orig", "swap", "back")


@dataclass
class Prefix:
    """One recorded conversation held at the moment the agent read the refund."""
    pid: str                      # <run>.<cell>.<transcript>: unique across runs
    source: str                   # the trace it came from
    task_id: str
    market: str
    language: str                 # the conversation's language, as recorded
    target: str                   # the other one: en, or the market's language
    head: dict[str, Any]          # the source's setup: model, temperature, exposure, ...
    messages: list[dict[str, Any]]
    steps: int                    # agent steps in it, the lookup's included
    turns: int                    # customer turns in it, the opening not counted
    world: dict[str, Any]         # Database.to_dict() after the prefix's calls
    escalated: bool               # in the recording: escalated before its next reply
    escalated_later: bool         # ...or later, while the customer was there

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prefix":
        return cls(**d)


def target_language(task, language: str, pid: str) -> str | None:
    """The language a prefix is swapped into: English for a local-language
    conversation; for an English or Singlish one, a language of its market
    other than English, picked by the prefix's id so a market's languages
    share its conversations (Malaysia's between Malay and Chinese). None where
    the market has no other language (PH)."""
    if language not in ENGLISH:
        return "en"
    local = [x for x in LANGUAGES_FOR_MARKET.get(task.market, []) if x not in ENGLISH]
    if not local:
        return None
    return local[int(hashlib.sha1(pid.encode()).hexdigest(), 16) % len(local)]


def local_language(prefix: "Prefix") -> str:
    """The language other than English a prefix is about: the one it was
    recorded in, or the one it is swapped into."""
    return prefix.target if prefix.language in ENGLISH else prefix.language


def decision_prefix(recs: list[dict[str, Any]], task, *, pid: str = "",
                    source: str = "") -> tuple[Prefix | None, str]:
    """The prefix of one recorded episode, or (None, why not).

    Only what can be rebuilt exactly: an unmodified agent (the full context),
    no claim guardrail before the lookup (its held replies and notes are not
    the conversation), the task and world the run recorded, and every payload
    reproduced by its digest."""
    from .harness.prompts import system_prompt
    from .harness.replay import replay

    head = next((r for r in recs if r.get("type") == "header"), None)
    if head is None:
        return None, "no header"
    if head.get("trap") != TRAP:
        return None, "not the duplicate refund"
    if head.get("context") != "full":
        return None, "context strategy not full"
    if head.get("guardrail", "off") != "off":
        return None, "claim guardrail"
    if head.get("task_digest") != task_digest(task) or head.get("world_digest") != world_digest(task):
        return None, "task or world changed since the run"
    steps = [r for r in recs if r.get("type") == "step" and not r.get("closing")]
    events = {e["turn"]: e for e in recs
              if e.get("type") == "event" and e.get("kind") == "user_turn"}
    order = task.hidden_facts.get("order_id")
    at = next((i for i, st in enumerate(steps)
               for tc, tr in zip(st.get("tool_calls") or [], st.get("tool_results") or [])
               if tc.get("name") == LOOKUP and tr.get("ok")
               and (tc.get("arguments") or {}).get("order_id") == order), None)
    if at is None:
        return None, "the order never looked up"
    if any(tc.get("name") == ESCALATE for tc in steps[at].get("tool_calls") or []):
        return None, "escalated in the lookup's own step"
    rep = replay([head] + steps[:at + 1], task)
    if rep.diverged or rep.unverified or rep.by != "digest":
        return None, "replay did not reproduce the payloads"

    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(task.user_id, task.market,
                                                    head.get("policy_mode", "preload"))},
        {"role": "user", "content": task.opening}]
    k, turn = 0, 1
    for st in steps[:at + 1]:
        calls = st.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": st.get("model_content") or "",
                     "tool_calls": [{"id": tc["id"], "name": tc["name"],
                                     "arguments": tc.get("arguments")} for tc in calls]})
        for tc in calls:
            msgs.append({"role": "tool", "name": tc["name"], "tool_call_id": tc["id"],
                         "content": rep.payloads[k]})
            k += 1
        if not calls:
            ev = events.get(turn)
            if ev is None or ev.get("ended"):
                return None, "the customer left before the lookup"
            msgs.append({"role": "user", "content": ev.get("text") or ""})
            turn += 1

    after = steps[at + 1:]
    nxt = next((i for i, st in enumerate(after) if not st.get("tool_calls")), len(after))

    def escalates(sts: list[dict[str, Any]]) -> bool:
        return any(tc.get("name") == ESCALATE and tr.get("ok")
                   for st in sts for tc, tr in zip(st.get("tool_calls") or [],
                                                   st.get("tool_results") or []))
    target = target_language(task, task.language, pid or task.task_id)
    if target is None:
        return None, "no other language in its market"
    keep = ("requested_model", "agent_temperature", "agent_extra_body", "exposure",
            "policy_mode", "simulator", "closing", "tool_checks", "task_digest",
            "world_digest")
    return Prefix(pid=pid or task.task_id, source=source, task_id=task.task_id,
                  market=task.market, language=task.language, target=target,
                  head={k: head.get(k) for k in keep}, messages=msgs, steps=at + 1,
                  turns=turn - 1, world=rep.db.to_dict(),
                  escalated=escalates(after[:nxt]),
                  escalated_later=escalates(after[nxt:])), ""


# --------------------------------------------------------------------------
# translation

TRANSLATE = (
    "You translate one message from a customer-service chat between an online "
    "marketplace's support agent and a customer. Translate it into {language}. Keep "
    "every order id, number, amount, currency, date, name and code exactly as "
    "written, and keep its register: a customer's informal message stays informal, the "
    "agent's polite reply stays polite. If it is already in {language}, return it "
    "unchanged. Reply with the translation only.")


def translatable(m: dict[str, Any]) -> bool:
    """Words someone said in the conversation: the customer's turns and the
    agent's replies. Not the system prompt, not a tool's payload."""
    return m.get("role") in ("user", "assistant") and bool((m.get("content") or "").strip())


class Translator:
    """A model that translates, behind a cache: every arm of every prefix reads
    a message's translation from one place, so the same text is translated
    once and a rerun spends nothing."""

    def __init__(self, backend, cache: str | Path | None = None):
        self.backend = backend
        self.model = getattr(backend, "model", getattr(backend, "name", "?"))
        self.path = Path(cache) if cache else None
        self.cache: dict[str, str] = (json.loads(self.path.read_text(encoding="utf-8"))
                                      if self.path and self.path.exists() else {})
        # One lock a text: two threads asking for the same translation at once
        # wait for one answer, so an arm never reads a translation the cache
        # does not keep.
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def key(self, text: str, language: str) -> str:
        return hashlib.sha1(json.dumps([self.model, language, text], ensure_ascii=False)
                            .encode("utf-8")).hexdigest()

    def __call__(self, text: str, language: str) -> str:
        from .harness.types import Message
        k = self.key(text, language)
        with self._guard:
            lock = self._locks.setdefault(k, threading.Lock())
        with lock:
            if k not in self.cache:
                out = self.backend.chat(
                    [Message("system", TRANSLATE.format(language=LANGUAGE_NAMES[language])),
                     Message("user", text)], [])
                got = (out.content or "").strip()
                if not got:
                    raise RuntimeError(f"the translator returned nothing for {text[:60]!r}")
                self.cache[k] = got
            return self.cache[k]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0),
                                 encoding="utf-8")


def arm_language(prefix: Prefix, arm: str) -> str:
    return prefix.target if arm == "swap" else prefix.language


def arm_messages(prefix: Prefix, arm: str, translate: Callable[[str, str], str]
                 ) -> list[dict[str, Any]]:
    """The prefix's messages in an arm: as recorded, translated, or translated
    and back. Only the words change."""
    if arm not in ARMS:
        raise ValueError(f"arm is one of {ARMS}, not {arm!r}")
    if arm == "orig":
        return [dict(m) for m in prefix.messages]
    out = []
    for m in prefix.messages:
        m = dict(m)
        if translatable(m):
            text = translate(m["content"], prefix.target)
            if arm == "back":
                text = translate(text, prefix.language)
            m["content"] = text
        out.append(m)
    return out


# --------------------------------------------------------------------------
# the agent, from the prefix on

@dataclass
class Outcome:
    escalated: bool                    # escalate_to_human, run, before its next reply
    category: str | None               # the first escalation's category
    calls: list[str] = field(default_factory=list)
    reply: str = ""                    # what it then said to the customer
    stop: str = ""
    budget: dict[str, Any] = field(default_factory=dict)


def continue_agent(prefix: Prefix, messages: list[dict[str, Any]], task, backend, *,
                   trace=None, run_index: int | None = None, max_steps: int = 8,
                   exposure=None) -> Outcome:
    """The agent from the end of the prefix until it next speaks to the
    customer -- no customer reply, no closing phase: the decision 1r measured
    is the one it makes in this turn. The world is the prefix's, the tools the
    source run's (its exposure; with tool search, what it had found so far is
    rebuilt from the world's action log), and the policy mode its own."""
    from .harness.loop import run_episode
    from .harness.simulator import SilentUser
    from .harness.types import Budget, EpisodeState, Message, Usage

    db = Database.from_dict(prefix.world)
    state = EpisodeState(task_id=task.task_id,
                         messages=[Message.from_dict(m) for m in messages],
                         step=prefix.steps, turn=prefix.turns, usage=Usage(),
                         simulator_cursor=prefix.turns)
    budget = Budget(max_steps=prefix.steps + max_steps, max_tool_calls=10_000,
                    max_tokens=10**9, max_wall_seconds=600.0,
                    max_user_turns=prefix.turns + 1)
    res = run_episode(task, db, backend, simulator=SilentUser(), budget=budget,
                      trace=trace, policy_mode=prefix.head.get("policy_mode") or "preload",
                      state=state, exposure=exposure, run_index=run_index)
    esc = [(tc, tr) for st in res.steps for tc, tr in zip(st.tool_calls, st.tool_results)
           if tc["name"] == ESCALATE and tr.get("ok")]
    last = res.steps[-1] if res.steps else None
    return Outcome(escalated=bool(esc),
                   category=(esc[0][0].get("arguments") or {}).get("category") if esc else None,
                   calls=[tc["name"] for st in res.steps for tc in st.tool_calls],
                   reply=(last.model_content or "") if last and not last.tool_calls else "",
                   stop=res.stop_reason.value, budget=res.budget)
