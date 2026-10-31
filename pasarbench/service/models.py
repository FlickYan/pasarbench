"""
The models a conversation can run on, and the backends that call them.

    mock                 the service's own agent, rule-based: no key, no network,
                         the same answer to the same conversation every time. It
                         reads the customer's messages for an order id, the four
                         digits of her phone and what she asks for -- the status,
                         tracking, a cancellation, an address change, a voucher, a
                         human -- calls the tools a careful agent would, in the
                         order the policy wants (identity first), and answers from
                         what they returned. Load tests run on it: real tool calls,
                         real writes, real locks, the model's latency simulated
                         (PASAR_MOCK_LATENCY_MS) and its tokens estimated and priced
                         as DeepSeek-V4.1-Flash, so the ledger sees realistic sums.
    scripted:<task_id>   the task's reference solution, step by step: what the
                         parity test runs to show the service and the benchmark
                         reach the same verdict on every task.
    deepseek-v4-pro      DeepSeek's API as RUNBOOK 1s runs it: temperature 0,
    deepseek-flash       thinking off, 2,048 tokens a step.
    PASAR_MODELS_FILE    more, as JSON (deploy/models.example.json): a name, the
                         API it speaks, the endpoint, the request settings and the
                         price-book row it is charged at.

A model needs a price: a conversation is never opened on a model the price
book cannot charge for. A real model needs its provider's key, read from that
provider's own variables only (pasarbench.sweep.key_for, without the run-wide
PASARBENCH_API_KEY): a key never goes to another provider's endpoint.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..harness.types import Message, ModelResponse, ToolCall, Usage, approx_tokens


class ModelError(Exception):
    """A model a conversation cannot be opened on: unknown, not allowed, or
    not ready (no key)."""


@dataclass(frozen=True)
class ModelSpec:
    name: str                       # what a conversation asks for
    provider: str                   # the provider, as the price book and the gates know it
    model: str                      # the provider's own name for it
    api: str                        # mock | scripted | chat | messages | responses
    base_url: str = ""
    extra_body: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    temperature: float | None = 0.0
    max_tokens: int = 2048
    price_model: str = ""           # the price-book row, if not `model`

    def settings(self) -> dict[str, Any]:
        """What a conversation records about the model it ran on."""
        return {"provider": self.provider, "model": self.model, "api": self.api,
                "base_url": self.base_url or None, "extra_body": dict(self.extra_body),
                "temperature": self.temperature, "max_tokens": self.max_tokens}

    @property
    def priced_as(self) -> tuple[str, str]:
        return self.provider, self.price_model or self.model


DEEPSEEK = "https://api.deepseek.com/v1"
NO_THINKING = {"thinking": {"type": "disabled"}}
BUILTIN: dict[str, ModelSpec] = {
    "mock": ModelSpec("mock", "local", "mock", "mock", temperature=None),
    "deepseek-v4-pro": ModelSpec("deepseek-v4-pro", "deepseek", "deepseek-v4-pro", "chat",
                                 DEEPSEEK, NO_THINKING),
    "deepseek-flash": ModelSpec("deepseek-flash", "deepseek", "deepseek-flash", "chat",
                                DEEPSEEK, NO_THINKING),
}
APIS = ("chat", "messages", "responses")


def _solutions() -> dict[str, list]:
    from ..generate import generate
    from ..run import SOLUTIONS
    out = dict(SOLUTIONS)
    for edition in (1, 2, 3):
        out.update(generate(edition=edition)[1] if edition > 1 else generate()[1])
    return out


_SOLUTIONS: dict[str, list] | None = None


def solutions() -> dict[str, list]:
    global _SOLUTIONS
    if _SOLUTIONS is None:
        _SOLUTIONS = _solutions()
    return _SOLUTIONS


class Registry:
    """The models this deployment offers (PASAR_MODELS) and how to call each."""

    def __init__(self, allowed: tuple[str, ...] = ("mock", "scripted:*"), models_file: str = "",
                 mock_latency_ms: int = 0):
        self.allowed = tuple(allowed)
        self.mock_latency_ms = mock_latency_ms
        self.presets: dict[str, ModelSpec] = dict(BUILTIN)
        if models_file:
            for spec in load_models_file(models_file):
                self.presets[spec.name] = spec

    def is_allowed(self, name: str) -> bool:
        return name in self.allowed or (name.startswith("scripted:") and "scripted:*" in self.allowed)

    def resolve(self, name: str) -> ModelSpec:
        if not self.is_allowed(name):
            raise ModelError(f"model {name!r} is not offered here (PASAR_MODELS: "
                             f"{', '.join(self.allowed)})")
        if name.startswith("scripted:"):
            if name[9:] not in solutions():
                raise ModelError(f"no reference solution for task {name[9:]!r}")
            return ModelSpec(name, "local", "scripted", "scripted", temperature=None)
        spec = self.presets.get(name)
        if spec is None:
            raise ModelError(f"model {name!r} is allowed but not defined (built in: "
                             f"{', '.join(BUILTIN)}; more in PASAR_MODELS_FILE)")
        return spec

    @staticmethod
    def key(spec: ModelSpec) -> str:
        if spec.api not in APIS:
            return ""
        from ..sweep import key_for          # the provider's own variables, never another's
        return key_for(spec.base_url, override=None)

    def check_ready(self, spec: ModelSpec) -> None:
        from ..sweep import _local          # localhost, 127.0.0.1: a local server needs no key
        if spec.api in APIS and self.key(spec) in ("", "EMPTY") and not _local(spec.base_url):
            from ..sweep import key_hint
            raise ModelError(f"model {spec.name!r} has no API key on this server: "
                             f"{key_hint(spec.base_url, override=None)}")

    def backend(self, spec: ModelSpec) -> Any:
        """A backend for one turn. Real ones are the benchmark's own
        (harness/backends.py): the same requests the runs sent."""
        if spec.api == "mock":
            return MockAgent(latency_ms=self.mock_latency_ms)
        if spec.api == "scripted":
            return ScriptedAgent(solutions()[spec.name[9:]])
        from ..harness import backends as B
        common = dict(model=spec.model, api_key=self.key(spec), temperature=spec.temperature,
                      max_tokens=spec.max_tokens, extra_body=dict(spec.extra_body))
        if spec.api == "chat":
            inner = B.OpenAICompatBackend(base_url=spec.base_url, **common)
        elif spec.api == "messages":
            inner = B.AnthropicBackend(base_url=spec.base_url, **common)
        else:
            inner = B.ResponsesBackend(base_url=spec.base_url, **common)
        return _Unhalted(inner, B)


class _Unhalted:
    """A benchmark run stops sending to a plan's gateway for good once it says
    the plan's limit is reached (backends.PLAN_HOSTS). A service runs for
    months: here the circuit breaker decides when to try again."""

    def __init__(self, inner: Any, backends: Any):
        self.inner, self._b = inner, backends
        self.name = inner.name
        self.model = inner.model
        self.reports_usage = inner.reports_usage

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        self._b._HALTED.pop(self._b._host(self.inner.base_url), None)
        return self.inner.chat(messages, tools)


def load_models_file(path: str) -> list[ModelSpec]:
    raw = json.loads(Path(path).read_text())
    out = []
    for i, d in enumerate(raw.get("models", raw) if isinstance(raw, dict) else raw):
        try:
            if d["api"] not in APIS:
                raise ValueError(f"api is one of {APIS}")
            if not str(d["base_url"]).startswith(("https://", "http://localhost",
                                                  "http://127.0.0.1")):
                raise ValueError("base_url must be https (or a local server)")
            out.append(ModelSpec(name=d["name"], provider=d["provider"], model=d["model"],
                                 api=d["api"], base_url=d["base_url"].rstrip("/"),
                                 extra_body=dict(d.get("extra_body") or {}),
                                 temperature=d.get("temperature", 0.0),
                                 max_tokens=int(d.get("max_tokens", 2048)),
                                 price_model=d.get("price_model", "")))
        except (KeyError, TypeError, ValueError) as e:
            raise ModelError(f"{path}: model {i}: {e}") from e
    return out


# ---------------------------------------------------------------------------
# The reference solutions, as a model
# ---------------------------------------------------------------------------

CLOSING = "All done. Is there anything else I can help with?"


class ScriptedAgent:
    """A task's reference solution, one step a model call. Where it is in the
    script is read from the conversation -- the steps it already holds -- so
    nothing is kept between calls and a turn can run in any process."""
    name = "scripted"
    model = "scripted"
    reports_usage = False

    def __init__(self, script: list):
        self.script = list(script)

    def position(self, messages: list[Message]) -> int:
        i = 0
        for m in messages:
            if m.role != "assistant" or i >= len(self.script):
                continue
            step = self.script[i]
            if isinstance(step, str):
                if m.content == step and not m.tool_calls:
                    i += 1
            elif m.tool_calls and m.tool_calls[0].name == step[0] \
                    and m.tool_calls[0].arguments == step[1]:
                i += 1
        return i

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        i = self.position(messages)
        if i >= len(self.script):
            return ModelResponse(content=CLOSING)
        step = self.script[i]
        if isinstance(step, str):
            return ModelResponse(content=step)
        return ModelResponse(tool_calls=[ToolCall(name=step[0], arguments=dict(step[1]),
                                                  id=f"call_s{i:03d}")])


# ---------------------------------------------------------------------------
# The mock agent
# ---------------------------------------------------------------------------

ORDER = re.compile(r"\bO\d{4}(?:-C\d{4})?\b")
DIGITS = re.compile(r"(?<![\w-])(\d{4})(?!\w)")      # not an order number's
USER = re.compile(r"user_id \*\*(\S+?)\*\*")
ADDRESS = re.compile(r"\bto\s*[:\-]?\s*(.{12,200})$", re.I | re.S)
INTENTS = (
    ("human", re.compile(r"\b(human|real person|supervisor|escalate|manusia|nhân viên|"
                         r"เจ้าหน้าที่)\b", re.I)),
    ("cancel", re.compile(r"\b(cancel|batal|hủy|ยกเลิก)\w*", re.I)),
    ("address", re.compile(r"\b(address|alamat|địa chỉ|ที่อยู่)\b", re.I)),
    ("voucher", re.compile(r"\b(voucher|goodwill|compensat\w*|baucar)\b", re.I)),
    ("track", re.compile(r"\b(where|track\w*|deliver\w*|shipment|parcel|ship\w*)\b", re.I)),
)


def _tool_results(messages: list[Message]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """(tool, arguments, result) for every call the conversation holds."""
    calls: dict[str, ToolCall] = {}
    out = []
    for m in messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                calls[tc.id] = tc
        elif m.role == "tool" and m.tool_call_id in calls:
            tc = calls[m.tool_call_id]
            try:
                res = json.loads(m.content)
            except ValueError:
                res = {}
            out.append((tc.name, tc.arguments if isinstance(tc.arguments, dict) else {},
                        res if isinstance(res, dict) else {}))
    return out


class MockAgent:
    """A careful, literal-minded agent, written as rules: what a load test
    needs from a model -- tool calls and writes in a plausible order -- and
    nothing it does not (a judgment, a language)."""
    name = "mock"
    model = "mock"
    reports_usage = True

    def __init__(self, latency_ms: int = 0):
        self.latency_ms = latency_ms

    def plan(self, messages: list[Message], turn_start: int
             ) -> tuple[list[tuple[str, dict[str, Any]]], str]:
        """The calls this turn needs, in order, and the reply once they are
        made: decided from what the conversation held when the turn began, so
        the plan stays the same while its calls are made."""
        system = next((m.content for m in messages if m.role == "system"), "")
        user_id = (USER.search(system) or [None, None])[1]
        said = [m.content for m in messages[:turn_start + 1] if m.role == "user"]
        last = said[-1] if said else ""
        history = _tool_results(messages[:turn_start])
        verified = any(t == "verify_identity" and r.get("verified") for t, _a, r in history)
        order_id = next((o for text in reversed(said) for o in ORDER.findall(text)[-1:]), None)
        digits = next((d for text in reversed(said) for d in DIGITS.findall(text)[-1:]), None)
        intent = next((name for name, rx in INTENTS if rx.search(last)), "status")
        calls: list[tuple[str, dict[str, Any]]] = []
        if not verified and digits and user_id:
            calls.append(("verify_identity", {"user_id": user_id, "phone_last4": digits}))
        if order_id is None:
            if user_id:
                calls.append(("list_user_orders", {"user_id": user_id}))
            return calls, "Which order is this about? Please give me its order number (it starts with O)."
        calls.append(("get_order", {"order_id": order_id}))
        if intent == "track":
            calls.append(("get_shipment", {"order_id": order_id}))
        writes = {"cancel", "address", "voucher", "human"}
        if intent in writes and not verified and not digits:
            return calls, ("I can help with that. First, please tell me the last 4 digits of the "
                           "phone number on your account so I can verify it's you.")
        if intent == "cancel":
            calls.append(("cancel_order", {"order_id": order_id,
                                           "reason": "customer requested cancellation"}))
        elif intent == "address":
            m = ADDRESS.search(last)
            if not m:
                return calls, "What is the new delivery address, with its postcode?"
            calls.append(("modify_shipping_address", {"order_id": order_id,
                                                      "new_address": m.group(1).strip()}))
        elif intent == "voucher" and user_id:
            calls.append(("issue_goodwill_voucher", {"user_id": user_id, "amount_minor": 500,
                                                     "currency": "{currency}",
                                                     "reason": "goodwill for the inconvenience"}))
        elif intent == "human":
            calls.append(("escalate_to_human", {"order_id": order_id,
                                                "category": "customer_request",
                                                "reason": "customer asked for a human agent"}))
        return calls, ""

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        turn_start = max(i for i, m in enumerate(messages) if m.role == "user")
        done = _tool_results(messages[turn_start:])
        calls, ask = self.plan(messages, turn_start)
        failed = next(((t, r) for t, _a, r in done if not r.get("ok", True)
                       or (t == "verify_identity" and r.get("verified") is False)), None)
        resp: ModelResponse
        if failed is None and len(done) < len(calls):
            name, args = calls[len(done)]
            if args.get("currency") == "{currency}":
                order = next((r.get("order") for t, _a, r in _tool_results(messages)
                              if t == "get_order" and r.get("order")), None) or {}
                args = {**args, "currency": order.get("currency", "SGD")}
                if args["currency"] in ("IDR", "VND"):
                    args["amount_minor"] = {"IDR": 50000, "VND": 80000}[args["currency"]]
            resp = ModelResponse(tool_calls=[ToolCall(name=name, arguments=args,
                                                      id=f"call_m{len(messages):04d}")])
        else:
            resp = ModelResponse(content=ask or self.answer(done, failed))
        return self._usage(messages, tools, resp)

    @staticmethod
    def answer(done: list[tuple[str, dict[str, Any], dict[str, Any]]],
               failed: tuple[str, dict[str, Any]] | None) -> str:
        if failed is not None and failed[0] == "verify_identity":
            return ("Those digits don't match the phone number on the account, so I can't "
                    "make changes to it. Could you check them?")
        if failed is not None:
            return (f"I'm sorry, I couldn't complete that: {failed[1].get('error', 'it failed')}. "
                    f"Is there anything else I can do?")
        said = []
        for tool, _args, r in done:
            if tool == "verify_identity":
                said.append("Thank you, you're verified." if r.get("verified") else
                            "Those digits don't match the phone number on the account, so I "
                            "can't make changes to it.")
            elif tool == "get_order" and r.get("order"):
                o = r["order"]
                said.append(f"Order {o['order_id']} is {o['status']} "
                            f"(total {o.get('total_display', o.get('total_minor'))}).")
            elif tool == "get_shipment" and r.get("ok"):
                s = r.get("shipment")
                said.append("It hasn't been dispatched yet." if not s else
                            f"Its shipment is {s.get('status')} with {s.get('courier')}"
                            + (f", tracking number {s['tracking_no']}." if s.get("tracking_no")
                               else "."))
            elif tool == "cancel_order":
                said.append(f"I've cancelled order {r.get('order_id')}. {r.get('note', '')}".strip())
            elif tool == "modify_shipping_address":
                said.append("I've updated the delivery address.")
            elif tool == "issue_goodwill_voucher":
                said.append(f"I've added a goodwill voucher of {r.get('display')} to your account.")
            elif tool == "escalate_to_human":
                said.append(f"I've passed this to a colleague (ticket {r.get('escalation_id')}); "
                            f"they'll be in touch.")
            elif tool == "list_user_orders":
                ids = [o.get("order_id") for o in r.get("orders", [])][:5]
                said.append("Your recent orders: " + ", ".join(ids) + "." if ids
                            else "I can't find any orders on your account.")
        return " ".join(said) or "How can I help you today?"

    @staticmethod
    def _usage(messages: list[Message], tools: list[dict[str, Any]], resp: ModelResponse
               ) -> ModelResponse:
        """Tokens as a provider would count them, estimated: the whole context
        each call, the part before the last reply served from cache."""
        schema = approx_tokens(json.dumps(tools))
        per = [approx_tokens(m.content) + 4 for m in messages]
        last_reply = max((i for i, m in enumerate(messages) if m.role == "assistant"), default=0)
        prompt = schema + sum(per)
        cached = schema + sum(per[:last_reply]) if last_reply else 0
        out = approx_tokens(resp.content) + 24 * len(resp.tool_calls)
        resp.usage = Usage(prompt_tokens=prompt, completion_tokens=max(out, 1),
                           cached_tokens=min(cached, prompt))
        return resp
