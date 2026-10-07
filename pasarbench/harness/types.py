"""
Harness core types.

Deliberately backend-neutral. Messages here are OUR representation; each
backend adapter translates to and from its own wire format. That translation
boundary is the whole point -- it is what makes the model a swappable part
rather than the thing the harness is built around.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Message:
    role: str                                   # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None             # set on role="tool"
    name: str | None = None                     # tool name, on role="tool"
    # An agent reply the claim guardrail held back (harness/guardrail.py): it
    # stays in the agent's context, and the simulated customer never sees it.
    # Ours only -- the backends build their wire messages field by field.
    hidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.hidden:
            d["hidden"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content", "") or "",
            tool_calls=[ToolCall(**tc) for tc in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            hidden=bool(d.get("hidden", False)),
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0      # prompt tokens served from the provider's cache

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Free prefix-caching signal. DeepSeek and several others report this
        in `usage`, which means the week-7 caching result can be measured
        against an API with no GPU involved. The ~2.2k-token policy prefix is
        identical on every call, so this should be high -- if it is not, the
        prompt is being perturbed somewhere it should not be."""
        return round(self.cached_tokens / self.prompt_tokens, 4) if self.prompt_tokens else 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens,
                     self.completion_tokens + other.completion_tokens,
                     self.cached_tokens + other.cached_tokens)


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    # What the provider says it actually ran. NOT the same as what you asked
    # for: providers silently route retired aliases to newer models. A result
    # recorded against an alias is unreproducible once that alias re-points.
    served_model: str | None = None

    def as_message(self) -> Message:
        return Message(role="assistant", content=self.content, tool_calls=self.tool_calls)


# --------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------

class StopReason(str, Enum):
    DONE = "done"                       # simulator ended the conversation
    AGENT_YIELDED = "agent_yielded"     # agent produced text, no user left to reply
    MAX_STEPS = "max_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_TOKENS = "max_tokens"
    MAX_WALL = "max_wall_seconds"
    MAX_TURNS = "max_user_turns"
    BACKEND_ERROR = "backend_error"
    INTERRUPTED = "interrupted"


@dataclass
class Budget:
    """Hard caps. An agent that cannot finish inside these has failed the task.

    Set these deliberately: a generous budget hides the cost differences you
    are trying to measure in the week-4 context ablation.
    """
    max_steps: int = 30
    max_tool_calls: int = 40
    max_tokens: int = 120_000
    max_wall_seconds: float = 180.0
    max_user_turns: int = 20

    def tracker(self) -> "BudgetTracker":
        return BudgetTracker(self)


@dataclass
class BudgetTracker:
    budget: Budget
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    turns: int = 0
    started: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def exceeded(self) -> StopReason | None:
        b = self.budget
        if self.steps >= b.max_steps:
            return StopReason.MAX_STEPS
        if self.tool_calls >= b.max_tool_calls:
            return StopReason.MAX_TOOL_CALLS
        if self.tokens >= b.max_tokens:
            return StopReason.MAX_TOKENS
        if self.elapsed >= b.max_wall_seconds:
            return StopReason.MAX_WALL
        if self.turns >= b.max_user_turns:
            return StopReason.MAX_TURNS
        return None

    def snapshot(self) -> dict[str, Any]:
        return {"steps": self.steps, "tool_calls": self.tool_calls,
                "tokens": self.tokens, "turns": self.turns,
                "elapsed_s": round(self.elapsed, 3)}


# --------------------------------------------------------------------------
# Episode state
# --------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One model call and its consequences. This is the unit of the trace."""
    step: int
    turn: int
    context_messages: int
    context_strategy: str
    model_content: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    usage: dict[str, int]
    latency_ms: int
    budget: dict[str, Any]
    n_tools: int = 0             # schemas injected on THIS call
    schema_tokens: int = 0       # approximate cost of injecting them
    # The names behind n_tools. A training example has to be rendered with the
    # same tool list the model was shown, and with progressive disclosure that
    # list changes from call to call.
    tool_names: list[str] = field(default_factory=list)
    # Set when the claim guardrail held this reply back: {"claims": [[tool,
    # quote], ...], "note": the note the agent got instead of a customer turn}.
    guardrail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["guardrail"] is None:
            del d["guardrail"]       # traces without the guardrail stay as they were
        return d


@dataclass
class EpisodeState:
    """Everything needed to stop an episode and pick it up again later."""
    task_id: str
    messages: list[Message] = field(default_factory=list)
    step: int = 0
    turn: int = 0
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason | None = None
    simulator_cursor: int = 0
    served_model: str | None = None   # what the provider actually ran
    summary: str = ""            # running summary, for the Summarize strategy
    summarized_upto: int = 0     # index of the last unit folded into `summary`

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "messages": [m.to_dict() for m in self.messages],
            "step": self.step,
            "turn": self.turn,
            "usage": asdict(self.usage),
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "simulator_cursor": self.simulator_cursor,
            "served_model": self.served_model,
            "summary": self.summary,
            "summarized_upto": self.summarized_upto,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeState":
        return cls(
            task_id=d["task_id"],
            messages=[Message.from_dict(m) for m in d["messages"]],
            step=d["step"],
            turn=d["turn"],
            usage=Usage(**d["usage"]),
            stop_reason=StopReason(d["stop_reason"]) if d["stop_reason"] else None,
            simulator_cursor=d.get("simulator_cursor", 0),
            served_model=d.get("served_model"),
            summary=d.get("summary", ""),
            summarized_upto=d.get("summarized_upto", 0),
        )

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class EpisodeResult:
    state: EpisodeState
    stop_reason: StopReason
    steps: list[StepRecord]
    budget: dict[str, Any]
    error: str | None = None


def approx_tokens(text: str) -> int:
    """Cheap fallback when a backend reports no usage.

    Four characters per token is roughly right for English and roughly WRONG
    for Thai and Vietnamese -- non-Latin scripts tokenise far worse. Never
    report a token number that came from here as if it were measured; the
    trace records which backend supplied it.
    """
    return max(1, len(text) // 4)
