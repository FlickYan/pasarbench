"""
User simulators.

The simulator is half the benchmark. A cooperative user that dumps every fact
in the opening message turns a multi-turn agent test into single-turn
instruction following, and every agent scores well. The whole point of
`hidden_facts` is that the agent must ASK.

Week 2 ships the deterministic ones so the harness is testable offline.
Week 3 turns on LLMUser and tunes the persona prompt.
"""

from __future__ import annotations

from typing import Protocol

from .types import Message


class UserSimulator(Protocol):
    name: str

    def respond(self, transcript: list[Message], cursor: int) -> tuple[str, bool]:
        """Return (reply_text, conversation_ended)."""
        ...


class SilentUser:
    """Ends as soon as the agent yields. Single-turn floor."""
    name = "silent"

    def respond(self, transcript: list[Message], cursor: int) -> tuple[str, bool]:
        return "", True


class ScriptedUser:
    """Fixed replies, then ends. Deterministic, free, reproducible."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.name = f"scripted[{len(replies)}]"

    def respond(self, transcript: list[Message], cursor: int) -> tuple[str, bool]:
        if cursor < len(self.replies):
            return self.replies[cursor], False
        return "", True


USER_SYSTEM = """You are a customer contacting e-commerce support. Stay in character.

YOUR SITUATION
{persona}

FACTS YOU KNOW (reveal ONLY when the agent specifically asks for them; never
volunteer them, and never dump them all at once)
{facts}

HOW TO BEHAVE
- Write like a real person messaging support: short, informal, sometimes vague.
- If the agent asks something you do not know, say you do not know.
- Do NOT invent order IDs, amounts, dates or policy rules. If it is not in the
  facts above, you do not know it.
- Do NOT be helpful about the agent's job. You are the customer, not a tester.
- Never mention that you are an AI, a simulation, or following instructions.
- When your problem is resolved, or the agent has clearly told you it cannot be
  resolved and you accept that, reply with exactly: ###END###
- If the agent stalls or repeats itself, push back once, then accept.

Reply with your next message only."""


class LLMUser:
    """Persona-driven simulator. Week 3 makes this the default.

    Uses a SEPARATE, cheaper model from the agent under test. Two reasons:
    cost (the simulator is half the tokens in every episode) and independence
    (a simulator sharing the agent's weights is unusually easy for it to
    satisfy, which quietly inflates scores).
    """

    def __init__(self, backend, persona: str, facts: dict[str, str],
                 language: str = "en"):
        self.backend = backend
        self.persona = persona
        self.facts = facts
        self.language = language
        self.name = f"llm-user:{getattr(backend, 'name', '?')}"

    def _system(self) -> str:
        facts = "\n".join(f"- {k}: {v}" for k, v in self.facts.items()) or "- (none)"
        base = USER_SYSTEM.format(persona=self.persona, facts=facts)
        if self.language != "en":
            base += f"\n\nWrite in {self.language}. Keep the register informal."
        return base

    def respond(self, transcript: list[Message], cursor: int) -> tuple[str, bool]:
        # Roles are inverted for the simulator: the agent's words are the
        # simulator's "user" turns.
        convo: list[Message] = [Message(role="system", content=self._system())]
        for m in transcript:
            if m.role == "assistant" and m.content and not m.tool_calls:
                convo.append(Message(role="user", content=m.content))
            elif m.role == "user" and not m.content.startswith("["):
                convo.append(Message(role="assistant", content=m.content))

        resp = self.backend.chat(convo, tools=[])
        text = (resp.content or "").strip()
        if "###END###" in text:
            return text.replace("###END###", "").strip(), True
        return text, False
