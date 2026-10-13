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

1t found the language moves the decision. Two runs follow it (v62):

    1v  the policy in the conversation's language. The same prefixes in
        three POLICY_ARMS -- `local`, the conversation in the language other
        than English under the policy in English (1t's arm in that language);
        `policy`, the same conversation under the policy translated into its
        language, in the system prompt and from search_policy alike;
        `english`, the conversation in English (1t's other arm). local
        against policy is the test: only the policy's words differ.
    1w  other agents: 1t's three arms on each agent's own recorded
        conversations -- the prefixes are any agent model's.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
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
POLICY_ARMS = ("local", "policy", "english")


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


# The policy's language, named for a written policy -- not a chat's register:
# Singapore's conversations are asked for code-mixed Chinese (LANGUAGE_NAMES),
# its policy is not. Chinese is named in its own script too: asked for
# "Simplified Chinese, as written in Malaysia", the translator wrote one section
# in Malay (WHAT_FAILED #51).
POLICY_LANGUAGES = {
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Malaysia",
    "th": "Thai",
    "vi": "Vietnamese",
    "zh-MY": "Simplified Chinese (简体中文) as written in Malaysia",
    "zh-SG": "Simplified Chinese (简体中文) as written in Singapore",
}

# The third wording (v66). The first left a section of Malaysian Chinese in
# Malay; the second, told to stress in bold, where a script has no capitals, a
# word the English writes in them, set in bold nearly every prohibition in
# Chinese and Thai (WHAT_FAILED #51). Each wording's translations stay in the
# cache under it, and nothing ran on them.
TRANSLATE_POLICY = (
    "You translate one section of the customer-service policy that an online "
    "marketplace's support agents must follow exactly. Translate it into {language}, in "
    "the register of a written policy, adding nothing and leaving nothing out, and write "
    "every sentence of it in {language}: none in English, none in another language. Keep "
    "exactly as written: the markdown (headings, list markers, and bold: on the words the "
    "English sets in bold, and on no others), every section number such as P10 or P6.3, "
    "every text in backticks, and every number, amount, currency and date. Reply with the "
    "translation only.")

# The policy is translated in the pieces between its lines of ---: its title
# and preamble, then one section each, P1 to P11.
POLICY_BREAK = "\n---\n"
_CODE = re.compile(r"`([^`\n]+)`")
# A section number where it stands, whatever touches it: Chinese and Thai
# write words against it with no space, which \b does not see as a boundary.
_SECTION = re.compile(r"(?<![A-Za-z0-9])P\d+(?:\.\d+)?(?![0-9])")
_DIGITS = re.compile(r"\d+")
# A word in capitals outside backticks: a code the policy keeps (SGD, COD, SLA),
# or a word the English stresses (NOT, ANY, BEFORE, IS).
_CAPS = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,}(?![A-Za-z0-9])")
_WORD = re.compile(r"[A-Za-z]{3,}")
# The letters a language is written in, where letters can tell it: Chinese in
# its characters, Thai in its script, Vietnamese in its own letters among the
# Latin ones. Malay and Indonesian share theirs with each other.
_SCRIPTS = {"zh-MY": ("Chinese characters", lambda c: "\u4e00" <= c <= "\u9fff", 0.5),
            "zh-SG": ("Chinese characters", lambda c: "\u4e00" <= c <= "\u9fff", 0.5),
            "th": ("Thai letters", lambda c: "\u0e00" <= c <= "\u0e7f", 0.5),
            "vi": ("Vietnamese letters", lambda c: not c.isascii(), 0.05)}


def _shape(text: str) -> dict[str, int]:
    """What a piece is made of, beside its words: headings, list items, bold."""
    lines = text.splitlines()
    return {"heading": sum(line.startswith("#") for line in lines),
            "list item": sum(line.lstrip().startswith("- ") for line in lines),
            "bold": text.count("**") // 2}


def policy_missing(english: str, translated: str, language: str | None = None) -> list[str]:
    """What a translated piece of the policy lost, if anything. Kept, each:
    a text in backticks (a status, a category, a tool's name); a section
    number; every number as often as the English has it -- as a number, so
    2026-11-01 written 2026年11月1日 keeps it, and "5 days" written "15" does
    not; and its headings, list items and bold, as many as the English has --
    so a rule left out, or a "**not**", is missed. A script without capitals
    stresses in bold what the English stresses in capitals, so the bold may
    be one more for each word the English writes in capitals and the
    translation does not (WHAT_FAILED #51). Given its language, the piece must
    also be written in it (_off_language). What it says is read, not checked
    here: P10 above all (RUNBOOK 1v)."""
    lost = [f"`{c}`" for c in dict.fromkeys(_CODE.findall(english)) if c not in translated]
    lost += [x for x in dict.fromkeys(_SECTION.findall(english))
             if not re.search(rf"(?<![A-Za-z0-9]){re.escape(x)}(?![0-9])", translated)]
    have = Counter(int(d) for d in _DIGITS.findall(translated))
    need = Counter(int(d) for d in _DIGITS.findall(english))
    lost += [f"{n} (x{k}, {have[n]} in the translation)" if have[n] else str(n)
             for n, k in need.items() if have[n] < k]
    en, tr = _shape(english), _shape(translated)
    restressed = sum(not re.search(rf"(?<![A-Za-z0-9]){w}(?![A-Za-z0-9])", translated)
                     for w in _CAPS.findall(_CODE.sub(" ", english)))
    for k in en:
        if k == "bold" and en[k] <= tr[k] <= en[k] + restressed:
            continue
        if en[k] != tr[k]:
            lost.append(f"{en[k]} {k}(s), {tr[k]} in the translation" + (
                f" ({restressed} more allowed, for words in capitals)"
                if k == "bold" and restressed else ""))
    off = _off_language(english, translated, language) if language else None
    return lost + ([off] if off else [])


def _off_language(english: str, translated: str, language: str) -> str | None:
    """How a translated piece is not in its language, as far as its letters
    tell, or None. What the policy keeps as written -- its backticks, section
    numbers and words in capitals -- is taken out first. Then: Chinese written
    in Chinese characters and Thai in Thai, for at least half of the letters
    left; Vietnamese with its own letters among them; and in every language,
    no more than half of its words the English's own. Malay written for
    Indonesian, or the reverse, it cannot see: that is read (WHAT_FAILED #51)."""
    rest = _CAPS.sub(" ", _SECTION.sub(" ", _CODE.sub(" ", translated)))
    letters = [c for c in rest if c.isalpha()]
    if language in _SCRIPTS and letters:
        name, mine, least = _SCRIPTS[language]
        share = sum(map(mine, letters)) / len(letters)
        if share < least:
            return f"not in its language: {share:.0%} of its letters are {name}"
    own = {w.lower() for w in _WORD.findall(_CAPS.sub(" ", _CODE.sub(" ", english)))}
    words = [w.lower() for w in _WORD.findall(rest)]
    same = sum(w in own for w in words) / len(words) if len(words) >= 5 else 0.0
    if same > 0.5:
        return f"not translated: {same:.0%} of its words are the English's"
    return None


def _unfenced(text: str) -> str:
    m = re.match(r"^```[A-Za-z]*\n(.*)\n```$", text.strip(), re.S)
    return m.group(1) if m else text


def translate_policy(text: str, language: str, translate: Callable[[str, str], str]
                     ) -> tuple[str, list[tuple[int, list[str]]]]:
    """The policy in `language`, piece by piece, its lines of --- and the
    space around each piece kept; and [(piece, what it lost)] for every piece
    that lost something or is not in `language` (policy_missing). A run uses
    it only if none did."""
    out, bad = [], []
    for i, chunk in enumerate(text.split(POLICY_BREAK)):
        core = chunk.strip()
        lead, trail = chunk[:len(chunk) - len(chunk.lstrip())], chunk[len(chunk.rstrip()):]
        got = _unfenced(translate(core, language)).strip() if core else ""
        lost = policy_missing(core, got, language)
        if lost:
            bad.append((i, lost))
        out.append(lead + got + trail)
    return POLICY_BREAK.join(out), bad


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

    def key(self, text: str, language: str, kind: tuple[str, ...] = ()) -> str:
        """A conversation's text keyed as 1t keyed it; a piece of the policy
        under its own kind, so the two never share an entry."""
        return hashlib.sha1(json.dumps([self.model, *kind, language, text], ensure_ascii=False)
                            .encode("utf-8")).hexdigest()

    def __call__(self, text: str, language: str) -> str:
        return self._ask(text, language,
                         TRANSLATE.format(language=LANGUAGE_NAMES[language]), ())

    def policy(self, text: str, language: str) -> str:
        """A piece of the policy in `language`, in a written policy's register
        (RUNBOOK 1v), from the same cache -- keyed by the prompt too, so a
        reworded prompt is a new translation, not the old one read back."""
        prompt = TRANSLATE_POLICY.format(language=POLICY_LANGUAGES[language])
        return self._ask(text, language, prompt,
                         ("policy", hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]))

    def _ask(self, text: str, language: str, system: str, kind: tuple[str, ...]) -> str:
        from .harness.types import Message
        k = self.key(text, language, kind)
        with self._guard:
            lock = self._locks.setdefault(k, threading.Lock())
        with lock:
            if k not in self.cache:
                out = self.backend.chat([Message("system", system), Message("user", text)], [])
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


def prefix_policy(prefix: Prefix) -> str:
    """The English policy the prefix's system prompt carries -- with the tools
    named in it or not, by its policy mode."""
    from .harness.prompts import policy_text
    mode = prefix.head.get("policy_mode") or "preload"
    if mode not in ("preload", "preload-named"):
        raise ValueError(f"{prefix.pid}: policy mode {mode!r} puts no policy in the prompt")
    return policy_text(named=mode == "preload-named")


def policy_arm_language(prefix: Prefix, arm: str) -> str:
    """The language of the conversation in a 1v arm."""
    if arm == "english":
        return prefix.language if prefix.language in ENGLISH else prefix.target
    return local_language(prefix)


def policy_arm_messages(prefix: Prefix, arm: str, translate: Callable[[str, str], str],
                        policy: str | None = None) -> list[dict[str, Any]]:
    """1v's arms. `local` and `english`: the conversation in the language
    other than English or in English -- 1t's translated words, its `swap` or
    `back` -- under the policy in English. `policy`: `local`'s conversation
    with `policy`, the policy in that language, in the system prompt in place
    of the English; the instructions around it stay as they were. local and
    policy differ in the policy's words and nothing else."""
    if arm not in POLICY_ARMS:
        raise ValueError(f"arm is one of {POLICY_ARMS}, not {arm!r}")
    first_english = prefix.language in ENGLISH
    if arm == "english":
        which = "back" if first_english else "swap"
    else:
        which = "swap" if first_english else "back"
    msgs = arm_messages(prefix, which, translate)
    if arm == "policy":
        if not policy:
            raise ValueError("the policy arm needs the policy in its language")
        why = policy_unchangeable(prefix)
        if why:
            raise ValueError(f"{prefix.pid}: {why}")
        msgs[0] = {**msgs[0], "content": msgs[0]["content"].replace(prefix_policy(prefix),
                                                                    policy)}
    return msgs


def policy_unchangeable(prefix: Prefix) -> str:
    """Why the policy a prefix's agent read cannot be put in another language
    by changing its prompt, or "" if it can: a mode with no policy in the
    prompt; a prompt without today's policy in it, once; or the policy
    already read in English from search_policy before the lookup, a payload
    that stays as recorded."""
    try:
        english = prefix_policy(prefix)
    except ValueError as e:
        return str(e).split(": ", 1)[-1]
    first = prefix.messages[0] if prefix.messages else {}
    if first.get("role") != "system" or (first.get("content") or "").count(english) != 1:
        return "its system prompt does not carry today's policy once, as written"
    if any(m.get("role") == "tool" and m.get("name") == "search_policy"
           for m in prefix.messages):
        return "it read the policy from search_policy before the lookup, in English"
    return ""


# --------------------------------------------------------------------------
# the agent, from the prefix on

@dataclass
class Outcome:
    escalated: bool                    # escalate_to_human, run, before its next reply
    category: str | None               # the first escalation's category
    calls: list[str] = field(default_factory=list)
    reply: str = ""                    # what it then said to the customer
    searched_nothing: int = 0          # search_policy calls that found no section
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
    nothing = 0
    for m in res.state.messages[len(messages):]:
        if m.role == "tool" and m.name == "search_policy":
            try:
                nothing += not json.loads(m.content).get("results")
            except (ValueError, AttributeError):
                pass
    return Outcome(escalated=bool(esc),
                   category=(esc[0][0].get("arguments") or {}).get("category") if esc else None,
                   calls=[tc["name"] for st in res.steps for tc in st.tool_calls],
                   reply=(last.model_content or "") if last and not last.tool_calls else "",
                   stop=res.stop_reason.value, budget=res.budget, searched_nothing=nothing)
