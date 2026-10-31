"""
The benchmark's episode, with the agent behind the service's HTTP API.

harness/loop.run_episode runs an agent in process: a model call, its tool
calls, the customer's turn, again. Here the service runs the agent's side of
every turn -- its model, its tools, its transactions, its ledger -- and this
module plays the benchmark's part around it: it makes a tenant whose world is
the task's, opens the conversation for the task's customer, posts her opening,
hands each reply to the simulated customer the sweep would use, posts what she
answers, and stops where the benchmark's episode stops -- she ends it, or a cap
is reached (pasarbench.harness.types.Budget). The service is configured to
apply the same caps inside a turn (scripts/service_sweep.py); between turns
they are checked here, as the loop checks them before each step.

Then the task's verifier reads the tenant's world and action log from the
service's database, as the parity test does. The episode is written as a trace
in the benchmark's own format, its steps rebuilt from what the service recorded
-- the messages, each model call's event, each tool's answer -- so the readers
of traces (scripts/compare_cells.py, rescore.py, inspect_trace.py) read an
episode run through the service like any other, and re-scoring it replays the
calls the service ran through the benchmark's own tools.

Nothing here holds a key: the server process has the agent's, and the
simulated customer's backend its own (pasarbench.sweep.make_simulator).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..db import Database
from ..harness.exposure import schema_tokens
from ..harness.prompts import system_prompt
from ..harness.replay import NO_DISPATCH
from ..harness.trace import TraceWriter
from ..harness.types import Budget, Message, StopReason
from ..tasks import task_digest, world_digest
from ..tools import CHECKS as TOOL_CHECKS
from ..tools import DEFAULT_TOOLS, schemas
from ..verifier import verify

VIA = "pasarbench-service"          # the trace header's "via": how the agent was reached
TOOL_NAMES = list(DEFAULT_TOOLS)
N_SCHEMA_TOKENS = schema_tokens(TOOL_NAMES)


def digest(text: str) -> str:
    """As the service fingerprints a conversation's prompt and tools (runtime.digest)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


TOOLS_DIGEST = digest(json.dumps(schemas(TOOL_NAMES), sort_keys=True))

# A turn the service ended without the agent's reply, as the benchmark's
# episode would have ended there. A spent conversation budget is the token cap
# when the tokens are what ran out (the server holds both).
STOPS = {"max_steps": StopReason.MAX_STEPS, "max_turn_seconds": StopReason.MAX_WALL,
         "provider_error": StopReason.BACKEND_ERROR,
         "provider_unavailable": StopReason.BACKEND_ERROR}


@dataclass
class Agent:
    """The agent as a trace header names it. `model` is the service's name for
    it ("deepseek-flash", "scripted" for each task's reference solution)."""
    model: str
    backend: str                     # e.g. openai-compat:deepseek-flash
    requested_model: str | None      # the provider's own name, as the sweep records it
    temperature: float | None = None
    extra_body: dict[str, Any] | None = None

    def for_task(self, task) -> str:
        return f"scripted:{task.task_id}" if self.model == "scripted" else self.model


@dataclass
class Outcome:
    task_id: str
    seed: int
    tenant_id: str
    conversation_id: str | None = None
    stop: str = StopReason.BACKEND_ERROR.value
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    turns: list[dict[str, Any]] = field(default_factory=list)
    cost_picousd: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    customer_usage: dict[str, int] | None = None
    prompt_ok: bool | None = None    # the model saw the benchmark's own system prompt...
    tools_ok: bool | None = None     # ...and its own tool schemas
    gaps: int = 0                    # model calls a trace step could not be rebuilt for
    error: str | None = None
    trace: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _messages(rows: list[dict[str, Any]]) -> list[Message]:
    """The service's transcript as the harness's messages, the system prompt left out."""
    return [Message.from_dict({k: v for k, v in r.items()
                               if k in ("role", "content", "tool_calls", "tool_call_id", "name")})
            for r in rows if r["role"] != "system"]


def _step_records(rows: list[dict[str, Any]], events: list[dict[str, Any]], turn: int,
                  tracker) -> tuple[list[dict[str, Any]], int]:
    """The steps of one turn, as harness/loop.py records them: each model call
    (its event) with the message it wrote, and each of its tool calls with the
    answer the agent was given -- whose digest and length a replay checks."""
    calls = [e for e in events if e["type"] == "model.call" and e["turn"] == turn]
    said = [m for m in rows if m["turn"] == turn and m["role"] == "assistant"
            and not m.get("platform")]
    answers = {m["tool_call_id"]: m for m in rows if m["role"] == "tool" and m["turn"] == turn
               and not m.get("platform")}
    out = []
    for ev, msg in zip(calls, said):
        d = ev["data"]
        tracker.steps += 1
        usage = {"prompt": int(d.get("prompt_tokens") or 0),
                 "completion": int(d.get("completion_tokens") or 0),
                 "cached": int(d.get("cached_tokens") or 0)}
        tracker.tokens += usage["prompt"] + usage["completion"]
        results = []
        for tc in msg.get("tool_calls") or []:
            answer = answers.get(tc["id"])
            payload = answer["content"] if answer is not None else ""
            try:
                res = json.loads(payload) if payload else {}
            except ValueError:
                res = {}
            res = res if isinstance(res, dict) else {}
            err = res.get("error")
            if answer is not None and not any(str(err or "").startswith(p) for p in NO_DISPATCH):
                tracker.tool_calls += 1      # the loop counts only the calls the tools ran
            results.append({"name": tc["name"], "args": tc.get("arguments"), "ok": res.get("ok"),
                            "error": err, "result_chars": len(payload),
                            "result_sha1": hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12],
                            **({"payment": res["payment"]}
                               if tc["name"] == "cancel_order" and res.get("payment") else {})})
        out.append({
            "step": tracker.steps, "turn": turn - 1,
            "context_messages": sum(1 for m in rows if m["seq"] < msg["seq"]),
            "context_strategy": "full", "model_content": msg.get("content") or "",
            "tool_calls": [{"id": tc["id"], "name": tc["name"], "arguments": tc.get("arguments")}
                           for tc in msg.get("tool_calls") or []],
            "tool_results": results, "usage": usage,
            "latency_ms": int(d.get("latency_ms") or 0), "budget": tracker.snapshot(),
            "n_tools": len(TOOL_NAMES), "schema_tokens": N_SCHEMA_TOKENS,
            "tool_names": TOOL_NAMES})
    return out, abs(len(calls) - len(said))


def _get(client, path: str, headers: dict[str, str]) -> Any:
    r = client.get(path, headers=headers)
    r.raise_for_status()
    return r.json()


def _events(client, cid: str, headers: dict[str, str], after: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while True:
        page = _get(client, f"/v1/conversations/{cid}/events?after={after}&limit=1000", headers)
        out += page
        if len(page) < 1000:
            return out
        after = page[-1]["id"]


def post_turn(client, cid: str, headers: dict[str, str], content: str, key: str,
              attempts: int = 6):
    """The customer's message, under an Idempotency-Key: a request the network
    lost is sent again and answered once (409 while the first still runs, 503
    when the server is busy: both say when to try again). The client's own
    timeout must outlast a turn (the kit's: 600 seconds)."""
    import httpx
    for attempt in range(attempts):
        t0 = time.monotonic()
        try:
            r = client.post(f"/v1/conversations/{cid}/messages", json={"content": content},
                            headers={**headers, "Idempotency-Key": key})
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 10))
            continue
        ms = (time.monotonic() - t0) * 1000
        if r.status_code in (409, 503) and attempt < attempts - 1:
            time.sleep(min(float(r.headers.get("Retry-After") or 1), 10))
            continue
        return r, ms
    raise RuntimeError("unreachable")


def run(task, seed: int, *, client, operator: dict[str, str], engine, agent: Agent,
        simulator, budget: Budget, writer: TraceWriter, tenant_id: str,
        policy_mode: str = "preload") -> Outcome:
    """One episode of `task` through the service at `client` (an httpx client
    on the server, or FastAPI's TestClient), traced by `writer`."""
    from .store import conversation_actions, tenant_world

    out = Outcome(task.task_id, seed, tenant_id)
    tracker = budget.tracker()
    r = client.post("/v1/admin/tenants", headers=operator,
                    json={"tenant_id": tenant_id, "name": task.task_id[:64],
                          "world": f"task:{task.task_id}"})
    r.raise_for_status()
    headers = {"Authorization": f"Bearer {r.json()['api_key']}"}
    r = client.post("/v1/conversations", headers=headers,
                    json={"user_id": task.user_id, "model": agent.for_task(task),
                          "market": task.market})
    r.raise_for_status()
    conv = r.json()
    cid = out.conversation_id = conv["conversation_id"]
    prov = conv.get("provenance") or {}
    out.prompt_ok = prov.get("prompt_digest") == digest(
        system_prompt(task.user_id, task.market, policy_mode))
    out.tools_ok = prov.get("tools_digest") == TOOLS_DIGEST

    writer.open_episode(task.task_id, run_index=seed, meta={
        "backend": agent.backend, "simulator": getattr(simulator, "name", "?"),
        "context": "full", "policy_mode": policy_mode, "exposure": "default",
        "resumed": False, "requested_model": agent.requested_model, "trap": task.trap,
        "market": task.market, "language": task.language, "task_digest": task_digest(task),
        "world_digest": world_digest(task), "agent_extra_body": agent.extra_body or None,
        "agent_temperature": agent.temperature, "via": VIA, "tool_checks": TOOL_CHECKS,
        "service": {"model": agent.for_task(task), "tenant_id": tenant_id,
                    "conversation_id": cid, "provenance": prov}})
    out.trace = str(writer.dir / f"{task.task_id}__r{seed}.jsonl")   # as open_episode names it

    text, last_event, cursor, stop = task.opening, 0, 0, None
    turn = 0
    while stop is None:
        turn += 1
        try:
            resp, http_ms = post_turn(client, cid, headers, text, f"{cid}-t{turn}")
        except Exception as e:  # noqa: BLE001 -- a network that never answers ends the episode
            out.error = f"{type(e).__name__}: {e}"[:500]
            writer.event("backend_error", error=out.error, step=tracker.steps)
            stop = StopReason.BACKEND_ERROR
            break
        if resp.status_code != 200:
            body = resp.json() if resp.headers.get("content-type", "").startswith(
                ("application/json", "application/problem+json")) else {}
            out.error = f"HTTP {resp.status_code}: {body.get('title') or body.get('detail') or ''}"
            writer.event("backend_error", error=out.error, step=tracker.steps)
            stop = StopReason.BACKEND_ERROR
            break
        result = resp.json()
        events = _events(client, cid, headers, last_event)
        if events:
            last_event = events[-1]["id"]
        rows = _get(client, f"/v1/conversations/{cid}/messages?include_system=true", headers)
        steps, gap = _step_records(rows, events, result["turn"], tracker)
        out.gaps += gap
        for st in steps:
            writer.step(st)
        model_ms = sum(int(e["data"].get("latency_ms") or 0) for e in events
                       if e["type"] == "model.call")
        tool_ms = sum(int(e["data"].get("latency_ms") or 0) for e in events
                      if e["type"] == "tool.result")
        u = result["usage"]
        out.turns.append({"turn": result["turn"], "http_ms": round(http_ms, 1),
                          "server_ms": result["latency_ms"], "model_ms": model_ms,
                          "tool_ms": tool_ms, "model_calls": u["model_calls"],
                          "tool_calls": len(result["tool_calls"]),
                          "cost_picousd": result["cost_picousd"],
                          "stop_reason": result["stop_reason"]})
        writer.event("service_turn", **out.turns[-1])
        out.cost_picousd += result["cost_picousd"]
        out.prompt_tokens += u["prompt_tokens"]
        out.cached_tokens += u["cached_tokens"]
        out.completion_tokens += u["completion_tokens"]
        out.model_calls += u["model_calls"]
        out.tool_calls += len(result["tool_calls"])

        if result["stop_reason"] != "replied":
            reason = result["stop_reason"]
            if reason == "conversation_budget":
                stop = (StopReason.MAX_TOKENS if tracker.tokens >= budget.max_tokens
                        else StopReason.INTERRUPTED)
            else:
                stop = STOPS.get(reason, StopReason.INTERRUPTED)
            out.error = f"the service ended the turn: {reason}" + (
                f" ({result['error']})" if result.get("error") else "")
            if stop is StopReason.BACKEND_ERROR:
                writer.event("backend_error", error=out.error, step=tracker.steps)
            break

        # The customer's turn, on everything the agent said so far.
        try:
            reply, ended = simulator.respond(_messages(rows), cursor)
        except Exception as e:  # noqa: BLE001 -- the customer's provider, as the loop treats it
            out.error = f"customer: {type(e).__name__}: {e}"[:500]
            writer.event("backend_error", error=out.error, step=tracker.steps)
            stop = StopReason.BACKEND_ERROR
            break
        cursor += 1
        tracker.turns += 1
        su = getattr(simulator, "usage", None)
        writer.event("user_turn", turn=tracker.turns, ended=ended, text=reply,
                     sim_usage=({"prompt": su.prompt_tokens, "completion": su.completion_tokens,
                                 "cached": su.cached_tokens} if su else None))
        if ended:
            stop = StopReason.DONE
            break
        stop = tracker.exceeded()
        text = reply
    if su := getattr(simulator, "usage", None):
        out.customer_usage = {"prompt": su.prompt_tokens, "completion": su.completion_tokens,
                              "cached": su.cached_tokens}

    # The verdict, from the world and the action log the service left.
    conv = _get(client, f"/v1/conversations/{cid}", headers)
    with engine.connect() as conn:
        tables = tenant_world(conn, tenant_id)
        actions = conversation_actions(conn, cid)
    db = Database(tables=tables, action_log=actions,
                  verified={task.user_id} if conv["identity_verified"] else set())
    v = verify(task, db, n_turns=tracker.steps, tokens=tracker.tokens)
    out.stop, out.passed, out.failures = stop.value, v.passed, v.failures
    out.budget = tracker.snapshot()
    writer.event("service_episode", cost_picousd=out.cost_picousd, prompt_tokens=out.prompt_tokens,
                 cached_tokens=out.cached_tokens, completion_tokens=out.completion_tokens,
                 model_calls=out.model_calls, tool_calls=out.tool_calls,
                 prompt_ok=out.prompt_ok, tools_ok=out.tools_ok, gaps=out.gaps)
    writer.close_episode(out.stop, v.passed, v.failures, out.budget)
    writer.close()
    return out
