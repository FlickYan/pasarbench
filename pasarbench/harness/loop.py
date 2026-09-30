"""
The agent loop.

No framework. ~120 lines of actual control flow, and every branch here exists
because something breaks without it:

  * A tool_call with no matching tool_result is a 400 from every major API.
    So when the tool-call budget runs out mid-batch we still emit an error
    result for each remaining call rather than skipping them. Skipping is the
    obvious implementation and it corrupts the conversation.

  * Backend exceptions end the episode with BACKEND_ERROR instead of raising.
    A sweep of 150 tasks x 5 strategies x 8 seeds must not die on one 529.

  * Malformed tool arguments come back from the backend as a payload, not an
    exception, and are handed to the model as a normal recoverable tool error.
    Small models emit broken JSON constantly, especially on non-Latin scripts.

  * Interrupt and resume are real: an episode serialises to JSON (state + DB)
    and continues from the same point. Tested for identical outcomes.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

_ALIAS_WARNED: set[tuple[str, str]] = set()

from ..db import Database
from ..tools import DEFAULT_TOOLS
from ..tools import call as tool_call
from ..tools import schemas
from .exposure import schema_tokens
from .context import ContextStrategy, FullContext
from ..tasks import task_digest, world_digest
from .prompts import system_prompt
from .simulator import SilentUser, UserSimulator
from .trace import NullTrace
from .types import (Budget, EpisodeResult, EpisodeState, Message, StepRecord,
                    StopReason, Usage, approx_tokens)


def run_episode(
    task,
    db: Database,
    backend,
    *,
    simulator: UserSimulator | None = None,
    context: ContextStrategy | None = None,
    budget: Budget | None = None,
    trace=None,
    tool_names: list[str] | None = None,
    policy_mode: str = "preload",
    state: EpisodeState | None = None,
    interrupt_after_steps: int | None = None,
    exposure=None,
    run_index: int | None = None,
) -> EpisodeResult:
    simulator = simulator or SilentUser()
    context = context or FullContext()
    budget = budget or Budget()
    trace = trace or NullTrace()
    def visible_tools() -> list[str]:
        """Recomputed EVERY step, because progressive disclosure means the set
        grows as the agent searches. Computing it once was fine until tool
        search existed; it is a silent bug the moment it does."""
        if exposure is not None:
            names = list(exposure.tools_for(state, db, task))
        elif tool_names is not None:
            names = list(tool_names)
        else:
            names = list(DEFAULT_TOOLS)
        for extra in getattr(context, "extra_tools", []):
            if extra not in names:
                names.append(extra)
        return names
    tracker = budget.tracker()

    resuming = state is not None
    if state is None:
        state = EpisodeState(task_id=task.task_id)
        state.messages = [
            Message(role="system",
                    content=system_prompt(task.user_id, task.market, policy_mode)),
            Message(role="user", content=task.opening),
        ]
    else:
        tracker.steps = state.step
        tracker.turns = state.turn
        tracker.tokens = state.usage.total

    trace.open_episode(task.task_id, run_index=run_index, meta={
        "backend": getattr(backend, "name", "?"),
        "simulator": getattr(simulator, "name", "?"),
        "context": context.name,
        "policy_mode": policy_mode,
        "exposure": getattr(exposure, "name", "default"),
        "resumed": resuming,
        "requested_model": getattr(backend, "model", None),
        "trap": task.trap,
        "market": task.market,
        "language": task.language,
        # lets an audit that regenerates tasks check it is reading the same one
        "task_digest": task_digest(task),
        # ...and a replay check that the tools would read the same world
        "world_digest": world_digest(task),
        # Request flags that change what the model sees, e.g. Qwen's
        # chat_template_kwargs {"enable_thinking": false}. Training data has to
        # be rendered with the same ones or it teaches a different prompt.
        "agent_extra_body": dict(getattr(backend, "extra_body", None) or {}) or None,
        # Collection samples at temperature 1.0 and evaluation at 0.0; a report
        # that compares runs has to be able to tell which one it is reading.
        "agent_temperature": getattr(backend, "temperature", None),
    })

    steps: list[StepRecord] = []
    stop: StopReason | None = None
    error: str | None = None

    while True:
        if (hit := tracker.exceeded()) is not None:
            stop = hit
            break
        if interrupt_after_steps is not None and tracker.steps >= interrupt_after_steps:
            stop = StopReason.INTERRUPTED
            break

        ctx = context.build(state)
        names = visible_tools()
        tools = schemas(names)
        t0 = time.monotonic()
        try:
            resp = backend.chat(ctx, tools)
        except Exception as e:  # noqa: BLE001 -- one bad call must not kill a sweep
            error = f"{type(e).__name__}: {e}"
            trace.event("backend_error", error=error, step=tracker.steps)
            stop = StopReason.BACKEND_ERROR
            break
        latency_ms = int((time.monotonic() - t0) * 1000)

        # A provider that silently routes a retired alias to a newer model
        # makes your results unreproducible: re-running the same command later
        # can hit different weights with no signal that anything changed.
        # Record what was actually served, and say so once.
        requested = getattr(backend, "model", None)
        served = resp.served_model
        if served and requested and served != requested:
            key = (requested, served)
            if key not in _ALIAS_WARNED:
                _ALIAS_WARNED.add(key)
                msg = (f"requested model {requested!r} but the provider served "
                       f"{served!r} -- an alias was re-pointed. Pin the served "
                       f"name and record it in your writeup.")
                print(f"\n!! {msg}\n", flush=True)
                trace.event("model_alias_mismatch", requested=requested, served=served)
            state.served_model = served
        elif served:
            state.served_model = served

        usage = resp.usage
        if not getattr(backend, "reports_usage", False) and usage.total == 0:
            usage = Usage(sum(approx_tokens(m.content) for m in ctx),
                          approx_tokens(resp.content))
        tracker.tokens += usage.total
        state.usage = state.usage + usage
        tracker.steps += 1
        state.step = tracker.steps

        state.messages.append(resp.as_message())

        results: list[dict[str, Any]] = []
        for tc in resp.tool_calls:
            if "__malformed__" in tc.arguments:
                out = {"ok": False,
                       "error": "arguments were not valid JSON; re-emit them as a JSON object"}
            elif tc.name not in names:
                # A tool the arm did not show does not exist for the agent.
                # Without this check every exposure arm leaked: the model saw
                # only the visible schemas, but any registered tool ran if its
                # name was guessed -- and `issue_refund` is easy to guess. The
                # message says "unknown", not "hidden", so it leaks nothing.
                out = {"ok": False, "error": f"unknown tool {tc.name!r}"}
                trace.event("hidden_tool_call", tool=tc.name, step=tracker.steps)
            elif tracker.tool_calls >= budget.max_tool_calls:
                # Still answer the call. An orphaned tool_call breaks the next request.
                out = {"ok": False, "error": "tool call budget exhausted for this episode"}
            else:
                out = tool_call(db, tc.name, tc.arguments)
                tracker.tool_calls += 1
            payload = json.dumps(out, ensure_ascii=False, default=str)
            state.messages.append(Message(role="tool", name=tc.name,
                                          tool_call_id=tc.id, content=payload))
            # The length alone cannot tell two payloads of the same size apart:
            # a salted hash once changed a tracking number between processes
            # and every replay still verified (WHAT_FAILED #33). The digest can.
            results.append({"name": tc.name, "args": tc.arguments,
                            "ok": out.get("ok"), "error": out.get("error"),
                            "result_chars": len(payload),
                            "result_sha1": hashlib.sha1(
                                payload.encode("utf-8")).hexdigest()[:12]})

        steps.append(StepRecord(
            step=tracker.steps, turn=state.turn, context_messages=len(ctx),
            context_strategy=context.name, model_content=resp.content,
            tool_calls=[tc.to_dict() for tc in resp.tool_calls],
            tool_results=results,
            usage={"prompt": usage.prompt_tokens,
                   "completion": usage.completion_tokens,
                   "cached": usage.cached_tokens},
            latency_ms=latency_ms, budget=tracker.snapshot(),
            n_tools=len(names), schema_tokens=schema_tokens(names),
            tool_names=list(names),
        ))
        trace.step(steps[-1].to_dict())

        if resp.tool_calls:
            continue

        # No tool calls: the agent has spoken to the customer. Hand over.
        reply, ended = simulator.respond(state.messages, state.simulator_cursor)
        state.simulator_cursor += 1
        tracker.turns += 1
        state.turn = tracker.turns
        su = getattr(simulator, "usage", None)
        trace.event("user_turn", turn=state.turn, ended=ended, text=reply,
                    sim_usage=({"prompt": su.prompt_tokens,
                                "completion": su.completion_tokens,
                                "cached": su.cached_tokens} if su else None))
        if ended:
            stop = StopReason.DONE
            break
        state.messages.append(Message(role="user", content=reply))

    state.stop_reason = stop or StopReason.AGENT_YIELDED
    return EpisodeResult(state=state, stop_reason=state.stop_reason,
                         steps=steps, budget=tracker.snapshot(), error=error)


# --------------------------------------------------------------------------
# Interrupt / resume
# --------------------------------------------------------------------------

def snapshot(state: EpisodeState, db: Database) -> str:
    return json.dumps({"state": state.to_dict(), "db": db.to_dict()},
                      ensure_ascii=False, default=str)


def restore(blob: str) -> tuple[EpisodeState, Database]:
    d = json.loads(blob)
    return EpisodeState.from_dict(d["state"]), Database.from_dict(d["db"])
