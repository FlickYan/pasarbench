"""
Simulator QA.

Nobody validates their user simulator, and it is half the benchmark.

The failure is quiet and it inflates everything. A simulator that volunteers
the order id in its opening turn converts a multi-turn information-gathering
task into single-turn instruction following. Scores go up, the benchmark stops
measuring what it claims to measure, and nothing in the results looks wrong.

Two things are checked here:

  LEAKAGE   -- did a hidden fact appear in a user turn before the agent asked
               for it? A fact revealed on request is correct behaviour. The
               same fact volunteered unprompted is a leak.

  COMPLIANCE -- did the simulator stay in character? Breaking the fourth wall
               ("as an AI", "the correct answer is"), inventing order ids that
               do not exist, or coaching the agent all corrupt the episode.

Run this over a sample of transcripts EVERY TIME you change the persona prompt
or the simulator model. Report the leak rate next to your pass rates; a
benchmark with a 30% leak rate is reporting a different, easier benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .harness.types import Message

# Signals that the agent asked for a given kind of fact.
#
# A pattern that misses how the agent actually asks does not fail loudly: the
# customer's correct answer is counted as a LEAK. Both errors are therefore
# costly -- too broad and real leaks read as answers, too narrow and answers read
# as leaks -- but only the second one concentrates in whichever languages the
# list covers worst, which is exactly the axis this suite measures (#26).
#
# The first list was English plus one textbook phrase per language. The agent
# writes "4 digit terakhir", "nomor order", "เลขออเดอร์", "订单号", and every one of
# its requests in those words was scored as the customer volunteering the fact.
# The phrasings below are the ones the agent used in 2,005 episodes (runs C and
# D); with them, no reveal outside English is unprompted in any of the four runs
# whose traces still match today's tasks.
# A new agent model can ask differently: read the contexts `inspect_trace.py`
# prints under each flagged leak before believing a per-language rate. It did:
# Qwen3.8-27B's first 1,075 episodes flagged 20 leaks, all in Indonesian and
# Thai, and every one followed a request -- "ID order", a misspelt "nomor
# pesannya", and Thai written with tone marks and vowels dropped or misplaced
# ("หมายเลขโทรศัพท" for "หมายเลขโทรศัพท์"). Thai is therefore matched with its
# combining marks folded out, on both sides (_fold). Its fine-tune asked one
# more way, "哪个订单" (which order), in three Chinese episodes.
ASK_PATTERNS: dict[str, list[str]] = {
    "order_id": [r"order\s*(id|number|no)", r"which order", r"order.*\bref",
                 # id / ms
                 r"nomor pesanan", r"nombor pesanan", r"nomor order", r"nombor order",
                 r"id pesanan", r"no\.?\s*pesanan", r"\bid\s*order", r"nomor pesan",
                 # vi / th
                 r"mã đơn", r"số đơn", r"หมายเลขคำสั่ง", r"เลขออเดอร์", r"หมายเลขออเดอร์",
                 r"เลขที่คำสั่งซื้อ", r"เลขคำสั่งซื้อ",
                 # zh
                 r"订单号", r"订单编号", r"訂單號", r"单号", r"哪(个|一个|笔)订单"],
    "phone_last4": [r"last\s*(4|four)", r"digits", r"verify", r"phone number",
                    # id / ms
                    r"4 angka", r"\b4\s*digit", r"empat digit", r"digit terakhir",
                    r"angka terakhir", r"nomor (telepon|hp)", r"nombor telefon",
                    # vi / th
                    r"số điện thoại", r"4 số cuối", r"เบอร์โทร", r"หมายเลขโทรศัพท์",
                    # "last digits" and "4 digits" -- but not ตัวทายาท (heir),
                    # or หลักฐาน / หลักเกณฑ์ / หลักการ (evidence, criteria,
                    # principle) and หลีกเลี่ยง (avoid), which fold to the same
                    # letters as หลัก once the marks are gone.
                    r"ยืนยันตัวตน", r"ตัว(เลข)?(สุด)?ท้าย(?!าท)",
                    r"(?:[4๔]|สี่)\s*หลัก(?!ฐาน|เกณฑ์|การ|เลี่ยง)",
                    # zh
                    r"手机号", r"电话号码", r"[后末]\s*[4四]\s*位"],
    "new_address": [r"new address", r"address.*change", r"where.*deliver",
                    r"alamat baru", r"ที่อยู่ใหม่", r"địa chỉ mới", r"新地址", r"新的地址"],
}

# Thai vowel and tone marks written above or below a consonant (Mn). A model
# that drops or misplaces them still asks the question; matching without them
# reads it either way. No other script in the suite uses these code points.
_THAI_MARKS = dict.fromkeys([0x0E31, *range(0x0E34, 0x0E3B), *range(0x0E47, 0x0E4F)])


def _fold(text: str) -> str:
    """Thai combining marks out. Case is left to re.IGNORECASE: lowercasing a
    PATTERN would turn \\S or \\D into \\s or \\d."""
    return text.translate(_THAI_MARKS)

# The first list, kept so #26 can be reproduced: run the leak audit with it
# (`inspect_trace.py --ask-patterns v1`) and the "leaks" come back.
ASK_PATTERNS_V1: dict[str, list[str]] = {
    "order_id": [r"order\s*(id|number|no)", r"which order", r"order.*\bref",
                 r"nomor pesanan", r"nombor pesanan", r"mã đơn", r"หมายเลขคำสั่ง"],
    "phone_last4": [r"last\s*(4|four)", r"digits", r"verify", r"phone number",
                    r"4 angka", r"nombor telefon", r"số điện thoại", r"เบอร์โทร"],
    "new_address": [r"new address", r"address.*change", r"where.*deliver"],
}

BREAK_CHARACTER = [
    r"\bas an AI\b", r"\bas a language model\b", r"\bI am simulating\b",
    r"\bthe correct answer\b", r"\bper the policy\b", r"\bsection P\d",
    r"\byou should call\b", r"\bthe tool\b", r"###",
]


@dataclass
class LeakReport:
    task_id: str
    leaked: list[tuple[str, int]] = field(default_factory=list)     # (fact, turn)
    revealed_on_request: list[tuple[str, int]] = field(default_factory=list)
    broke_character: list[tuple[str, int]] = field(default_factory=list)
    invented_ids: list[tuple[str, int]] = field(default_factory=list)
    user_turns: int = 0
    # (fact, turn, the agent's last message before it) for every leak -- the
    # line a reader needs to tell a volunteered fact from a question the
    # patterns could not read.
    leak_context: list[tuple[str, int, str]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.leaked or self.broke_character or self.invented_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "clean": self.clean,
                "leaked": self.leaked, "revealed_on_request": self.revealed_on_request,
                "leak_context": self.leak_context,
                "broke_character": self.broke_character,
                "invented_ids": self.invented_ids, "user_turns": self.user_turns}


def _asked_for(fact: str, agent_text: str,
               patterns: dict[str, list[str]] | None = None) -> bool:
    pats = (ASK_PATTERNS if patterns is None else patterns).get(fact)
    if not pats:
        return True          # unknown fact kind: do not accuse
    # The first list is matched as it was, so #26 reproduces to the episode.
    if patterns is ASK_PATTERNS_V1:
        low = agent_text.lower()
        return any(re.search(p, low) for p in pats)
    text = _fold(agent_text)
    return any(re.search(_fold(p), text, re.IGNORECASE) for p in pats)


def leak_report(task, messages: list[Message], known_ids: set[str] | None = None,
                patterns: dict[str, list[str]] | None = None) -> LeakReport:
    """Scan one transcript. `messages[0]` is the system prompt, `messages[1]`
    the opening -- the opening counts as turn 0 and a fact appearing there is
    always a leak, since the agent has said nothing yet."""
    rep = LeakReport(task_id=task.task_id)
    facts = {k: str(v) for k, v in task.hidden_facts.items() if v}
    agent_so_far = ""
    last_agent = ""
    turn = 0

    for m in messages:
        if m.role == "assistant" and m.content:
            agent_so_far += "\n" + m.content
            last_agent = m.content
            continue
        if m.role != "user":
            continue
        text = m.content or ""
        if text.startswith("["):          # harness-inserted elision marker
            continue
        rep.user_turns += 1

        for fact, value in facts.items():
            if value.lower() not in text.lower():
                continue
            if _asked_for(fact, agent_so_far, patterns):
                rep.revealed_on_request.append((fact, turn))
            else:
                rep.leaked.append((fact, turn))
                rep.leak_context.append((fact, turn, last_agent[-200:]))

        for pat in BREAK_CHARACTER:
            if re.search(pat, text, re.I):
                rep.broke_character.append((pat, turn))

        if known_ids:
            for cand in re.findall(r"\b[A-Z]{1,3}[O0-9][-A-Z0-9]{3,}\b", text):
                if cand not in known_ids and cand not in facts.values():
                    rep.invented_ids.append((cand, turn))
        turn += 1

    return rep


def stall_report(task, messages: list[Message],
                 patterns: dict[str, list[str]] | None = None) -> tuple[int, int]:
    """(requests, unanswered): how often the agent asked for a fact the customer
    holds and has not given yet, and how many of those requests the very next
    customer turn left unanswered.

    The mirror image of a leak. An honest simulator answers at a similar rate in
    every language; a gated one whose patterns cannot read a language's requests
    withholds the fact and stalls there (#26: run G's Indonesian and Chinese)."""
    pats = ASK_PATTERNS if patterns is None else patterns
    facts = {k: str(v) for k, v in task.hidden_facts.items() if v and k in pats}
    given = ""
    pending: set[str] = set()
    asked = unanswered = 0
    for m in messages:
        if m.role == "assistant" and m.content:
            pending |= {k for k, v in facts.items()
                        if v.lower() not in given and _asked_for(k, m.content, patterns)}
            continue
        if m.role != "user":
            continue
        text = (m.content or "").lower()
        if text.startswith("["):
            continue
        for k in pending:
            asked += 1
            unanswered += facts[k].lower() not in text
        pending = set()
        given += "\n" + text
    return asked, unanswered


def audit(reports: list[LeakReport]) -> dict[str, Any]:
    """Fold reports into the numbers to publish alongside pass rates."""
    n = len(reports) or 1
    leaked = sum(1 for r in reports if r.leaked)
    broke = sum(1 for r in reports if r.broke_character)
    invented = sum(1 for r in reports if r.invented_ids)
    by_fact: dict[str, int] = {}
    for r in reports:
        for fact, _ in r.leaked:
            by_fact[fact] = by_fact.get(fact, 0) + 1
    return {
        "transcripts": len(reports),
        "leak_rate": round(leaked / n, 4),
        "break_character_rate": round(broke / n, 4),
        "invented_id_rate": round(invented / n, 4),
        "clean_rate": round(sum(r.clean for r in reports) / n, 4),
        "leaks_by_fact": by_fact,
        "mean_user_turns": round(sum(r.user_turns for r in reports) / n, 2),
    }


VERDICT = """
HOW TO READ THIS

  leak_rate > 0.15         Either the persona prompt is not holding, or the
                           detector cannot read how the agent asks. Tell them
                           apart FIRST: read the agent line printed under each
                           leak. If it asks for the fact, the pattern list is
                           short, not the simulator (#26 -- every "leak" outside
                           English in four runs was this). If it does not, the
                           prompt is not holding: strengthen "reveal ONLY when
                           asked" or move to a stronger simulator model, and do
                           not report pass rates collected under it.

  mean_user_turns < 2      Episodes are effectively single-turn. Either the
                           agent is resolving everything in one shot (check the
                           traces) or the simulator is ending too eagerly.

  invented_id_rate > 0.05  The simulator is hallucinating order ids, which
                           sends the agent chasing records that do not exist
                           and produces failures that are the simulator's
                           fault, not the policy's.

  break_character_rate > 0 Fix immediately. A simulator that coaches the agent
                           is grading its own exam.
"""
