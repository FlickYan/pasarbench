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

from .types import Message, Usage


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

FACTS YOU KNOW
{facts}

HOW TO USE THOSE FACTS -- THIS IS THE MOST IMPORTANT RULE
- Reveal AT MOST ONE fact per message, and ONLY the exact fact the agent just
  asked for in its previous message.
- If the agent has not asked for a fact, do not mention it. Not as context, not
  as "by the way", not to be helpful.
- Never state your order number unless the agent asks for the order number.
- Never state your phone digits unless the agent asks to verify your identity.
- A real customer does not recite their account details unprompted. Neither do
  you.

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
                 language: str = "en", gate_facts: bool = False,
                 release_after_turn: int = 6):
        self.backend = backend
        self.persona = persona
        self.facts = facts
        self.language = language
        # gate_facts: withhold each fact from the simulator's own prompt until
        # the agent has asked for it, as judged by simqa.ASK_PATTERNS.
        # It was built to stop leaks in id/ms/th/zh that were never there: the
        # patterns could not read how the agent asks in those languages, so its
        # answers were scored as volunteered facts (WHAT_FAILED #26). The gate
        # inherited the same blind spot and turned it into a real handicap --
        # the fact is never released, the customer stonewalls until the
        # release_after_turn fallback, and the episode fails. Run G's Chinese
        # and Indonesian "gaps" were this, not the agent.
        # Gating is therefore allowed only in languages where the patterns have
        # been checked against real requests: on an UNGATED run, every reveal
        # the detector flags has been read and none was a question it missed.
        # That check was done for these, with deepseek-v4-pro as the agent, and
        # again with Qwen3.8-27B, whose Indonesian and Thai needed more
        # phrasings (#26). A different agent can phrase its requests
        # differently: check an ungated run of it first.
        if gate_facts:
            checked = {"en", "sg-en", "id", "ms", "th", "vi", "zh-MY", "zh-SG"}
            if language not in checked:
                raise ValueError(
                    f"--gate-facts: the ask-patterns have not been checked against "
                    f"real requests in {language!r}. Gating it would stall the "
                    f"customer on every request the patterns miss, and the result "
                    f"would be a harness artefact, not a language effect. Run "
                    f"ungated, read the flagged leaks with scripts/inspect_trace.py, "
                    f"extend simqa.ASK_PATTERNS, then add the language here.")
        self.gate_facts = gate_facts
        self.release_after_turn = release_after_turn
        self.name = (f"llm-user:{getattr(backend, 'name', '?')}"
                     + ("+gated" if gate_facts else ""))
        # Accumulated across the episode. Without this every cost projection is
        # agent-only, which on a multi-turn benchmark understates the bill by
        # roughly half -- and the simulator is usually on a DIFFERENT provider
        # at a different price, so you cannot just scale the agent number.
        self.usage = Usage()

    def _visible_facts(self, transcript) -> dict[str, str]:
        """Facts the simulator is allowed to KNOW right now."""
        if not self.gate_facts:
            return self.facts
        from ..simqa import _asked_for
        agent_text = "\n".join(m.content for m in transcript
                               if m.role == "assistant" and m.content)
        turns = sum(1 for m in transcript if m.role == "user")
        out = {}
        for k, v in self.facts.items():
            # Released once asked for, or after release_after_turn as a
            # fallback so a missed pattern cannot stall the episode forever.
            if _asked_for(k, agent_text) or turns >= self.release_after_turn:
                out[k] = v
        return out

    def _system(self, visible: dict[str, str] | None = None) -> str:
        src = self.facts if visible is None else visible
        facts = "\n".join(f"- {k}: {v}" for k, v in src.items()) or (
            "- (you cannot recall any details right now; if the agent asks for "
            "something specific, say you will look it up)")
        base = USER_SYSTEM.format(persona=self.persona, facts=facts)
        if self.language != "en":
            base += f"\n\nWrite in {self.language}. Keep the register informal."
            # Restate the hold-back rule in the target language. It was the
            # fix for leaks that turned out to be the detector's (#26) and has
            # no measurable effect; it stays so the prompt is stable across runs.
            from ..locales import hold_back_rule
            rule = hold_back_rule(self.language)
            if rule:
                base += f"\n\n{rule}"
        return base

    def respond(self, transcript: list[Message], cursor: int) -> tuple[str, bool]:
        # Roles are inverted for the simulator: the agent's words are the
        # simulator's "user" turns.
        convo: list[Message] = [
            Message(role="system",
                    content=self._system(self._visible_facts(transcript)))]
        for m in transcript:
            if m.role == "assistant" and m.content and not m.tool_calls:
                convo.append(Message(role="user", content=m.content))
            elif m.role == "user" and not m.content.startswith("["):
                convo.append(Message(role="assistant", content=m.content))

        resp = self.backend.chat(convo, tools=[])
        self.usage = self.usage + resp.usage
        text = (resp.content or "").strip()
        if "###END###" in text:
            return text.replace("###END###", "").strip(), True
        return text, False
