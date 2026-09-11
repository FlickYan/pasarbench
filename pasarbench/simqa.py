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

# Signals that the agent asked for a given kind of fact. Deliberately broad:
# a false "the agent asked" reading makes the leak detector too lenient, so
# these are kept tight and the ambiguous cases are reported separately.
ASK_PATTERNS: dict[str, list[str]] = {
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

    @property
    def clean(self) -> bool:
        return not (self.leaked or self.broke_character or self.invented_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "clean": self.clean,
                "leaked": self.leaked, "revealed_on_request": self.revealed_on_request,
                "broke_character": self.broke_character,
                "invented_ids": self.invented_ids, "user_turns": self.user_turns}


def _asked_for(fact: str, agent_text: str) -> bool:
    pats = ASK_PATTERNS.get(fact)
    if not pats:
        return True          # unknown fact kind: do not accuse
    low = agent_text.lower()
    return any(re.search(p, low) for p in pats)


def leak_report(task, messages: list[Message], known_ids: set[str] | None = None
                ) -> LeakReport:
    """Scan one transcript. `messages[0]` is the system prompt, `messages[1]`
    the opening -- the opening counts as turn 0 and a fact appearing there is
    always a leak, since the agent has said nothing yet."""
    rep = LeakReport(task_id=task.task_id)
    facts = {k: str(v) for k, v in task.hidden_facts.items() if v}
    agent_so_far = ""
    turn = 0

    for m in messages:
        if m.role == "assistant" and m.content:
            agent_so_far += "\n" + m.content
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
            if _asked_for(fact, agent_so_far):
                rep.revealed_on_request.append((fact, turn))
            else:
                rep.leaked.append((fact, turn))

        for pat in BREAK_CHARACTER:
            if re.search(pat, text, re.I):
                rep.broke_character.append((pat, turn))

        if known_ids:
            for cand in re.findall(r"\b[A-Z]{1,3}[O0-9][-A-Z0-9]{3,}\b", text):
                if cand not in known_ids and cand not in facts.values():
                    rep.invented_ids.append((cand, turn))
        turn += 1

    return rep


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

  leak_rate > 0.15         The persona prompt is not holding. Strengthen the
                           "reveal ONLY when asked" instruction, or move to a
                           stronger simulator model. Do NOT report pass rates
                           collected under a leaky simulator -- they measure an
                           easier benchmark than the one you describe.

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
