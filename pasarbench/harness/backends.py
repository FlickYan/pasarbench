"""
Model backends.

The harness never talks to a provider directly. It talks to a Backend, which
owns the wire format. Adding vLLM, a new provider, or a fake for testing is
one class -- nothing in loop.py changes.

Real HTTP is done with urllib so the harness itself stays dependency-free.
Swap in the official SDKs later if you want retries and streaming; the
interface is the same.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

from .types import Message, ModelResponse, ToolCall, Usage, approx_tokens


class Backend(Protocol):
    name: str
    reports_usage: bool

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        ...


# ==========================================================================
# Test backends -- these make the whole harness runnable with zero API spend
# ==========================================================================

class ScriptedBackend:
    """Replays a fixed plan. One tool call per step, then a closing message.

    Used to prove the loop, the tool dispatch, the message threading and the
    trace all work, without a model in the way. If the reference solutions
    score 1.0 through this backend, the harness is correct.
    """
    reports_usage = False

    def __init__(self, script: list[tuple[str, dict[str, Any]]],
                 closing: str = "All done. Is there anything else I can help with?"):
        self.name = "scripted"
        self.script = list(script)
        self.closing = closing
        self.i = 0

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        prompt_tokens = sum(approx_tokens(m.content) for m in messages)
        if self.i < len(self.script):
            name, args = self.script[self.i]
            self.i += 1
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(name=name, arguments=args)],
                usage=Usage(prompt_tokens, 20),
            )
        return ModelResponse(content=self.closing, usage=Usage(prompt_tokens, 15))


class ConfusedBackend:
    """Always calls the first available tool with empty args, forever.

    Exists to prove budgets actually bite. Without a backend like this you
    have no evidence your step cap works.
    """
    reports_usage = False
    name = "confused"

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        first = tools[0]["function"]["name"] if tools else "get_order"
        return ModelResponse(
            content="",
            tool_calls=[ToolCall(name=first, arguments={})],
            usage=Usage(sum(approx_tokens(m.content) for m in messages), 10),
        )


class MuteBackend:
    """Never calls a tool. Yields immediately. The floor of the benchmark."""
    reports_usage = False
    name = "mute"

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        return ModelResponse(content="Sorry, I am unable to help with that.",
                             usage=Usage(10, 10))


class FailingBackend:
    """Raises on the Nth call. Proves the loop degrades instead of crashing."""
    reports_usage = False
    name = "failing"

    def __init__(self, fail_on: int = 2):
        self.fail_on = fail_on
        self.calls = 0

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        self.calls += 1
        if self.calls >= self.fail_on:
            raise ConnectionError("simulated provider outage")
        return ModelResponse(content="", tool_calls=[ToolCall("get_order", {"order_id": "O1001"})],
                             usage=Usage(10, 10))


# ==========================================================================
# Real backends
# ==========================================================================

def _post(url: str, payload: dict, headers: dict, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:500]
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e


class OpenAICompatBackend:
    """Works with OpenAI, vLLM's server, Together, DeepSeek, Modal-hosted vLLM.

    For a local vLLM served on Modal:
        OpenAICompatBackend(model="Qwen/Qwen3-8B",
                            base_url="https://<your-app>.modal.run/v1",
                            api_key="EMPTY")
    """
    reports_usage = True

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 2048,
                 extra_body: dict[str, Any] | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Provider-specific body fields. The one that matters here:
        # DeepSeek V4 has THINKING ENABLED BY DEFAULT and reasoning counts
        # toward max_tokens, so without {"thinking": {"type": "disabled"}} the
        # token axis of the context ablation measures reasoning verbosity
        # rather than context strategy, and the multilingual tokens-per-char
        # mechanism becomes unmeasurable. Pass it explicitly; never rely on a
        # provider default you did not choose.
        self.extra_body = dict(extra_body or {})
        self.name = f"openai-compat:{model}"

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": [self._msg(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update(self.extra_body)

        data = _post(f"{self.base_url}/chat/completions", payload,
                     {"Content-Type": "application/json",
                      "Authorization": f"Bearer {self.api_key}"})

        choice = data["choices"][0]["message"]
        calls = []
        for tc in choice.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                # Malformed tool arguments are a REAL failure mode, especially
                # for smaller models and non-Latin scripts. Surface it as a
                # recoverable tool error rather than crashing the episode.
                args = {"__malformed__": raw}
            calls.append(ToolCall(name=tc["function"]["name"], arguments=args, id=tc["id"]))

        u = data.get("usage") or {}
        # Cache accounting differs by provider: OpenAI nests it under
        # prompt_tokens_details.cached_tokens, DeepSeek reports
        # prompt_cache_hit_tokens at the top level. Take whichever is present.
        cached = (u.get("prompt_cache_hit_tokens")
                  or (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                  or 0)
        content = choice.get("content") or ""
        if not content and choice.get("reasoning_content"):
            # Thinking mode produced reasoning but no answer -- almost always
            # means max_tokens was exhausted by the reasoning trace. Surface it
            # rather than returning a silent empty turn.
            content = ""
        return ModelResponse(
            content=content,
            tool_calls=calls,
            usage=Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                        int(cached)),
            stop_reason=data["choices"][0].get("finish_reason"),
            served_model=data.get("model"),
        )

    @staticmethod
    def _msg(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {"role": "tool", "tool_call_id": m.tool_call_id,
                    "name": m.name, "content": m.content}
        d: dict[str, Any] = {"role": m.role, "content": m.content or None}
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name,
                              "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                for tc in m.tool_calls
            ]
        return d


class AnthropicBackend:
    """Anthropic Messages API. Converts our OpenAI-shaped schemas on the way out."""
    reports_usage = True

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 2048):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._convert(messages),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t["function"]["name"],
                 "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]}
                for t in tools
            ]

        data = _post("https://api.anthropic.com/v1/messages", payload,
                     {"Content-Type": "application/json",
                      "x-api-key": self.api_key,
                      "anthropic-version": "2023-06-01"})

        text, calls = "", []
        for block in data.get("content", []):
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                calls.append(ToolCall(name=block["name"], arguments=block["input"],
                                      id=block["id"]))
        u = data.get("usage") or {}
        return ModelResponse(content=text, tool_calls=calls,
                             usage=Usage(u.get("input_tokens", 0), u.get("output_tokens", 0)),
                             stop_reason=data.get("stop_reason"),
                             served_model=data.get("model"))

    @staticmethod
    def _convert(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                         "content": m.content}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id,
                                   "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "..."}]})
            else:
                out.append({"role": "user", "content": m.content})
        return out
