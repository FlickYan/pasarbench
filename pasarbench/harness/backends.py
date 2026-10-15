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

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
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
    score 1.0 through this backend, the harness is correct. A plain string in
    the plan is a reply to the customer at that point, so a test can script
    what the agent says between calls -- a claim the guardrail should catch.
    """
    reports_usage = False

    def __init__(self, script: list[tuple[str, dict[str, Any]] | str],
                 closing: str = "All done. Is there anything else I can help with?"):
        self.name = "scripted"
        self.script = list(script)
        self.closing = closing
        self.i = 0

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        prompt_tokens = sum(approx_tokens(m.content) for m in messages)
        if self.i < len(self.script) and isinstance(self.script[self.i], str):
            self.i += 1
            return ModelResponse(content=self.script[self.i - 1], usage=Usage(prompt_tokens, 15))
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

def _host(url: str) -> str:
    return (urllib.parse.urlparse(url).hostname or "").lower()


# Gateways that sell a plan with a usage limit (v73): OpenCode Go. Once one
# answers that the plan's limit is reached, every later request to it in this
# process ends at once, unsent -- the rest of a run's episodes end as server
# errors without a burst of refused requests, and run again with --resume
# once the limit resets. Every other provider's errors end one episode each,
# as before.
PLAN_HOSTS: tuple[str, ...] = ("opencode.ai",)
_LIMIT_WORDS = re.compile(r"limit|quota|usage|exceed|balance|subscription|credit", re.I)
_HALTED: dict[str, str] = {}


def _plan_host(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in PLAN_HOSTS)


def _post(url: str, payload: dict, headers: dict, timeout: float = 120.0) -> dict:
    host = _host(url)
    if host in _HALTED:
        raise RuntimeError(f"not sent: {_HALTED[host]} -- reached earlier in this run; run "
                           f"it again once the limit resets")
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:500]
        if e.code in (402, 429) and _plan_host(host) and _LIMIT_WORDS.search(detail):
            _HALTED[host] = f"the usage limit (HTTP {e.code} from {host}: {detail[:200]})"
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e


def conversation_id(messages: list[dict[str, Any]]) -> str:
    """One id for every request of a conversation, from what does not change
    in it: its system prompt and the customer's first message. Nothing is
    kept between requests, so parallel conversations need no bookkeeping; two
    with the same start (one task's repeats) share a server, and a cache."""
    head = json.dumps(messages[:2], sort_keys=True, ensure_ascii=False)
    return "pasarbench-" + hashlib.sha256(head.encode()).hexdigest()[:24]


# What this harness calls itself to a gateway that asks (v73).
USER_AGENT = "pasarbench/1.0"


def gateway(url: str) -> str | None:
    """The gateway an endpoint is behind, for the trace header: OpenCode's
    Go plan or its pay-as-you-go Zen; None for a provider's own endpoint."""
    host = _host(url)
    if host != "opencode.ai" and not host.endswith(".opencode.ai"):
        return None
    return "opencode-go" if "/go/" in urllib.parse.urlparse(url).path + "/" else "opencode-zen"


def gateway_headers(url: str, start: list[dict[str, Any]]) -> dict[str, str]:
    """What a gateway asks every client to send (v73). OpenCode Go: a user
    agent of the client's own, not a generic library's, and one session id
    per conversation (x-opencode-session), for its routing and its prompt
    cache -- the conversation's id, from its first two messages (`start`).
    Every other endpoint: nothing, so what each provider is sent stays as it
    was."""
    if not gateway(url):
        return {}
    return {"User-Agent": USER_AGENT, "x-opencode-session": conversation_id(start)}


def _start(messages: list[Message]) -> list[dict[str, Any]]:
    """A conversation's first two messages, as the id is made from them."""
    return [{"role": m.role, "content": m.content} for m in messages[:2]]


def _settings(extra_body: dict[str, Any] | None, temperature: float | None) -> tuple[dict, float | None]:
    """Request settings, and the temperature they leave (v73). A temperature
    among the settings is the one sent, and the trace header records it; a
    setting given as null is left out of the request, so `{"temperature":
    null}` runs a model that refuses one at its provider's default --
    recorded as None, not as the 0 it was not sent."""
    extra = dict(extra_body or {})
    if "temperature" in extra:
        temperature = extra["temperature"]
    return extra, temperature


def _without_nulls(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


class OpenAICompatBackend:
    """Works with OpenAI, SGLang's and vLLM's servers, Together, DeepSeek,
    Modal-hosted vLLM.

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
        # OPENAI_API_KEY is OpenAI's: without a key given, it is used for
        # api.openai.com and sent nowhere else (v55).
        host = (urllib.parse.urlparse(self.base_url).hostname or "").lower()
        openai = host == "api.openai.com"
        self.api_key = api_key or (os.environ.get("OPENAI_API_KEY") if openai else None) or "EMPTY"
        # xAI keeps its prompt cache on each server, and sends the requests of
        # one conversation to one server when they carry the same
        # x-grok-conv-id (v56). Without it each step may land where the steps
        # before it are not cached -- the bill of a run several times over.
        self.sticky = host == "x.ai" or host.endswith(".x.ai")
        # Z.ai (and Zhipu's own platform) documents an assistant message's content
        # as a string and a tool result without a name: sent so (v59). Every other
        # provider gets the messages exactly as before.
        self.documented = (host == "z.ai" or host.endswith(".z.ai")
                           or host == "bigmodel.cn" or host.endswith(".bigmodel.cn"))
        self.max_tokens = max_tokens
        # Provider-specific body fields. The one that matters here:
        # DeepSeek V4 has THINKING ENABLED BY DEFAULT and reasoning counts
        # toward max_tokens, so without {"thinking": {"type": "disabled"}} the
        # token axis of the context ablation measures reasoning verbosity
        # rather than context strategy, and the multilingual tokens-per-char
        # mechanism becomes unmeasurable. Pass it explicitly; never rely on a
        # provider default you did not choose.
        self.extra_body, self.temperature = _settings(extra_body, temperature)
        # The gateway the endpoint is behind, if any (v73): the trace header
        # names it, and it is sent the headers it asks for.
        self.via = gateway(self.base_url)
        self.name = f"openai-compat:{model}"

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": [self._msg(m, self.documented) for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update(self.extra_body)
        payload = _without_nulls(payload)

        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        if self.sticky:
            headers["x-grok-conv-id"] = conversation_id(payload["messages"])
        headers.update(gateway_headers(self.base_url, _start(messages)))
        data = _post(f"{self.base_url}/chat/completions", payload, headers)

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
    def _msg(m: Message, documented: bool = False) -> dict[str, Any]:
        """One message on the wire. `documented`: as Z.ai's schema has it -- an
        empty content as "" rather than null, a tool result without its tool's
        name."""
        if m.role == "tool":
            if documented:
                return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
            return {"role": "tool", "tool_call_id": m.tool_call_id,
                    "name": m.name, "content": m.content}
        d: dict[str, Any] = {"role": m.role,
                             "content": m.content or ("" if documented else None)}
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name,
                              "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                for tc in m.tool_calls
            ]
        return d


class AnthropicBackend:
    """Anthropic Messages API. Converts our OpenAI-shaped schemas on the way out.

    At api.anthropic.com by default; since v73 at any endpoint that speaks it
    -- OpenCode Go serves Claude, MiniMax and Qwen this way, at its own
    /messages. ANTHROPIC_API_KEY is Anthropic's: without a key given it is
    used for api.anthropic.com and sent nowhere else."""
    reports_usage = True

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 base_url: str = "https://api.anthropic.com/v1",
                 extra_body: dict[str, Any] | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        own = _host(self.base_url) == "api.anthropic.com"
        self.api_key = api_key or (os.environ.get("ANTHROPIC_API_KEY", "") if own else "")
        self.max_tokens = max_tokens
        self.extra_body, self.temperature = _settings(extra_body, temperature)
        self.via = gateway(self.base_url)
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
        payload.update(self.extra_body)
        payload = _without_nulls(payload)

        headers = {"Content-Type": "application/json",
                   "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}
        headers.update(gateway_headers(self.base_url, _start(messages)))
        data = _post(f"{self.base_url}/messages", payload, headers)

        text, calls = "", []
        for block in data.get("content", []):
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                calls.append(ToolCall(name=block["name"], arguments=block["input"],
                                      id=block["id"]))
        u = data.get("usage") or {}
        # Anthropic counts the prompt in three parts: tokens read from its
        # cache, tokens written to it, and the rest (input_tokens). The prompt
        # is all three, as every other backend reports it (v73).
        read = u.get("cache_read_input_tokens") or 0
        prompt = u.get("input_tokens", 0) + read + (u.get("cache_creation_input_tokens") or 0)
        return ModelResponse(content=text, tool_calls=calls,
                             usage=Usage(prompt, u.get("output_tokens", 0), int(read)),
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


class ResponsesBackend:
    """OpenAI's Responses API (POST /responses), v73: the way OpenCode Go
    serves GPT, Grok and Muse Spark models, and OpenAI's own newer endpoint.

    No state is kept on the server (store: false): every request carries the
    conversation, as every other backend's does. The system prompt goes as
    `instructions`; a turn of the customer's or the agent's as a message
    item; a tool call the agent made as a function_call item and its result
    as a function_call_output item, matched by the call's id. The model's
    reasoning, when it reasons, is not sent back -- the chat backends do not
    send it either."""
    reports_usage = True

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 2048, extra_body: dict[str, Any] | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        own = _host(self.base_url) == "api.openai.com"
        self.api_key = api_key or (os.environ.get("OPENAI_API_KEY") if own else None) or "EMPTY"
        self.max_tokens = max_tokens
        self.extra_body, self.temperature = _settings(extra_body, temperature)
        self.via = gateway(self.base_url)
        self.name = f"responses:{model}"

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self._input(messages),
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "store": False,
        }
        if system:
            payload["instructions"] = system
        if tools:
            payload["tools"] = [{"type": "function", "name": t["function"]["name"],
                                 "description": t["function"]["description"],
                                 "parameters": t["function"]["parameters"]} for t in tools]
            payload["tool_choice"] = "auto"
        payload.update(self.extra_body)
        payload = _without_nulls(payload)

        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        headers.update(gateway_headers(self.base_url, _start(messages)))
        data = _post(f"{self.base_url}/responses", payload, headers)

        text, calls = "", []
        for item in data.get("output") or []:
            kind = item.get("type")
            if kind == "message":
                for part in item.get("content") or []:
                    if part.get("type") in ("output_text", "text"):
                        text += part.get("text") or ""
                    elif part.get("type") == "refusal":
                        text += part.get("refusal") or ""
            elif kind == "function_call":
                raw = item.get("arguments") or "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    # As the chat backend does: a recoverable tool error.
                    args = {"__malformed__": raw}
                calls.append(ToolCall(name=item["name"], arguments=args,
                                      id=item.get("call_id") or item.get("id")))
        u = data.get("usage") or {}
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
        status = data.get("status")
        stop = ("length" if status == "incomplete"
                and (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
                else "tool_calls" if calls else status)
        return ModelResponse(content=text, tool_calls=calls,
                             usage=Usage(u.get("input_tokens", 0), u.get("output_tokens", 0),
                                         int(cached)),
                             stop_reason=stop, served_model=data.get("model"))

    @staticmethod
    def _input(messages: list[Message]) -> list[dict[str, Any]]:
        """Our messages as the Responses API's input items."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                out.append({"type": "function_call_output", "call_id": m.tool_call_id,
                            "output": m.content})
            elif m.role == "assistant":
                if m.content:
                    out.append({"role": "assistant", "content": m.content})
                for tc in m.tool_calls:
                    out.append({"type": "function_call", "call_id": tc.id, "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False)})
            else:
                out.append({"role": "user", "content": m.content})
        return out
