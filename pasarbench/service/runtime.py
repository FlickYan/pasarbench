"""
A conversation turn: the customer's message in, the agent's reply out.

    Runtime.open_conversation   binds a conversation to a customer and a model,
                                and records what it will be shown: the system
                                prompt, the policy and the tool schemas, by digest
    Runtime.begin_turn          admits the turn (or 503), checks the budgets
                                (or 429), and takes the conversation's lease, so
                                one turn runs at a time in it, across processes
                                (or 409)
    Turn.run                    the loop the benchmark's harness runs, as a
                                service: a model call -- outside any transaction,
                                through the provider's gate -- and its usage to
                                the ledger; its tool calls, each one transaction
                                (store.run_tool) that also writes the tool's
                                message and the audit rows; until the agent
                                replies, or a cap, a budget or the provider stops
                                it, and the customer gets a fallback reply rather
                                than silence

Every step commits before the next begins, and every event a stream sends is
the row that step wrote. A process that dies mid-turn leaves a conversation
whose last committed step is true: a tool call either ran and was recorded, or
neither, and the next turn answers any call the crash left unanswered
("interrupted: this call was not run") before it starts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import Connection, Engine, bindparam, or_, select, update
from sqlalchemy import insert as sql_insert

from ..db import NOW
from ..harness.prompts import policy_text, system_prompt
from ..harness.types import Message, ToolCall, Usage, approx_tokens
from ..tools import CHECKS, DEFAULT_TOOLS, schemas
from . import costs
from . import errors as E
from . import schema as S
from .config import Settings
from .costs import Price
from .models import ModelError, ModelSpec, Registry
from .resilience import CircuitOpen, Gates, Overloaded
from .store import (ToolOutcome, conversation_actions, result_digest, run_tool, storable,
                    transaction)

log = logging.getLogger("pasar.runtime")

TOOL_NAMES = list(DEFAULT_TOOLS)
TOOL_SCHEMAS = schemas(TOOL_NAMES)
FALLBACK = ("I'm sorry, I wasn't able to finish this just now. Please try again in a moment, "
            "or ask me to pass you to a human agent.")
INTERRUPTED = {"ok": False, "error": "interrupted: this call was not run"}
Emit = Callable[[dict[str, Any]], None]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


POLICY_DIGEST = digest(policy_text())
TOOLS_DIGEST = digest(json.dumps(TOOL_SCHEMAS, sort_keys=True))


def utcnow() -> datetime:
    """Every time the service writes is UTC (SQLite keeps a time without its zone)."""
    return datetime.now(timezone.utc)


def seconds_to_utc_midnight() -> float:
    now = utcnow()
    return ((now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            - now).total_seconds()


class LeaseLost(Exception):
    """The turn outlived its lease and another took the conversation."""


class _Heartbeat:
    """Renews a running turn's lease -- and whatever else should live as long as
    the turn (its Idempotency-Key: `keepalive`) -- every third of the lease,
    from a thread of its own, so the lease outlives any one step however long
    a model call takes (a gate's wait, three tries, each a slow provider), and
    a process that dies frees its conversation within one lease."""

    def __init__(self, turn: "Turn", keepalive: Callable[[], None] | None):
        self.turn, self.keepalive = turn, keepalive
        self.every = max(0.2, turn.rt.settings.lease_seconds / 3)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pasar-lease", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            try:
                if not self.turn._renew():
                    return              # the lease is another's now: the turn finds out at its next write
                if self.keepalive is not None:
                    self.keepalive()
            except Exception:  # noqa: BLE001 -- the next beat tries again
                log.exception("lease heartbeat failed", extra={"conversation_id": self.turn.cid})


# The statements a turn runs, built once with named parameters (store.py says
# why: built for each call, one cost more CPU than its round trip).
_cv, _m, _e = S.conversations, S.messages, S.events
_Q_CONVERSATION = select(_cv).where(_cv.c.conversation_id == bindparam("cid"),
                                    _cv.c.tenant_id == bindparam("tenant"))
_Q_LEASE = (update(_cv).where(
    _cv.c.conversation_id == bindparam("cid"), _cv.c.tenant_id == bindparam("tenant"),
    _cv.c.status == "open",
    or_(_cv.c.lease_owner.is_(None), _cv.c.lease_expires_at < bindparam("now")))
    .values(lease_owner=bindparam("owner"), lease_expires_at=bindparam("expires"))
    .returning(*_cv.c))
_Q_DAILY = select(S.tenants.c.daily_budget_picousd).where(
    S.tenants.c.tenant_id == bindparam("tenant"))
_Q_MESSAGES = select(_m).where(_m.c.conversation_id == bindparam("cid")).order_by(_m.c.seq)
_HELD = (_cv.c.conversation_id == bindparam("cid")) & (_cv.c.lease_owner == bindparam("owner"))
_Q_RENEW = update(_cv).where(_HELD).values(lease_expires_at=bindparam("expires"))
_Q_RELEASE = update(_cv).where(_HELD).values(lease_owner=None, lease_expires_at=None)
_Q_STEP = update(_cv).where(_HELD).values(
    steps=_cv.c.steps + 1, n_messages=_cv.c.n_messages + 1,
    prompt_tokens=_cv.c.prompt_tokens + bindparam("d_prompt"),
    cached_tokens=_cv.c.cached_tokens + bindparam("d_cached"),
    completion_tokens=_cv.c.completion_tokens + bindparam("d_completion"),
    cost_picousd=_cv.c.cost_picousd + bindparam("d_cost"),
    updated_at=bindparam("now"), lease_expires_at=bindparam("expires"))
_Q_FINISH = {fallback: update(_cv).where(_HELD).values(
    last_stop=bindparam("stop"), updated_at=bindparam("now"), lease_owner=None,
    lease_expires_at=None, **({"n_messages": _cv.c.n_messages + 1} if fallback else {}))
    for fallback in (False, True)}
_Q_MESSAGE = sql_insert(_m)
_Q_EVENT = sql_insert(_e).returning(_e.c.event_id)
_Q_LEDGER = sql_insert(S.usage_ledger)
_Q_TOOL_CALL = sql_insert(S.tool_calls)
_Q_ACTION = sql_insert(S.actions)
_HOLDER: dict[tuple[tuple[str, ...], tuple[str, ...]], Any] = {}
_HOLDER_LOCK = threading.Lock()


def _holder(set_: tuple[str, ...], add: tuple[str, ...]) -> Any:
    """The conversation's row updated as the turn holding it, setting `set_`
    and adding to `add`: one statement per shape, built the first time."""
    key = (set_, add)
    stmt = _HOLDER.get(key)
    if stmt is None:
        values = {"updated_at": bindparam("now"), "lease_expires_at": bindparam("expires")}
        values.update({k: bindparam("v_" + k) for k in set_})
        values.update({k: _cv.c[k] + bindparam("d_" + k) for k in add})
        stmt = update(_cv).where(_HELD).values(**values)
        with _HOLDER_LOCK:
            _HOLDER.setdefault(key, stmt)
    return stmt


def _message(r: Any) -> Message:
    return Message(role=r.role, content=r.content or "",
                   tool_calls=[ToolCall(name=t["name"], arguments=t["arguments"], id=t["id"])
                               for t in (r.tool_calls or [])],
                   tool_call_id=r.tool_call_id, name=r.name)


def _event(conn: Connection, cid: str, turn: int, kind: str, data: dict[str, Any]) -> dict:
    eid = conn.execute(_Q_EVENT, {"conversation_id": cid, "turn": turn, "type": kind,
                                  "payload": data, "created_at": utcnow()}).scalar_one()
    return {"id": int(eid), "type": kind, "turn": turn, "data": data}


def iso(t: datetime | None) -> str | None:
    return None if t is None else _aware(t).isoformat().replace("+00:00", "Z")


def public_conversation(r: Any) -> dict[str, Any]:
    return {
        "conversation_id": r.conversation_id, "user_id": r.user_id, "market": r.market,
        "status": r.status, "model": r.model, "policy_mode": r.policy_mode,
        "provenance": {"prompt_digest": r.prompt_digest, "policy_digest": r.policy_digest,
                       "tools_digest": r.tools_digest, "tool_checks": r.tool_checks,
                       "strict_money": bool(r.strict_money), "world_clock": r.world_clock,
                       "model_settings": r.model_settings},
        "identity_verified": r.user_id in (r.verified or []),
        "turns": r.turns, "model_calls": r.steps, "tool_calls": r.tool_calls,
        "usage": {"prompt_tokens": int(r.prompt_tokens), "cached_tokens": int(r.cached_tokens),
                  "completion_tokens": int(r.completion_tokens)},
        "cost_picousd": int(r.cost_picousd), "cost_usd": costs.usd(int(r.cost_picousd)),
        "last_stop": r.last_stop,
        "turn_in_progress": r.lease_owner is not None,
        "created_at": iso(r.created_at), "updated_at": iso(r.updated_at),
        "closed_at": iso(r.closed_at),
    }


class Runtime:
    """What every turn in this process shares: the database, the models, the
    provider gates, the admission limit and the metrics."""

    def __init__(self, engine: Engine, settings: Settings, registry: Registry | None = None,
                 gates: Gates | None = None, metrics: Any = None):
        self.engine = engine
        self.settings = settings
        self.registry = registry or Registry(settings.models, settings.models_file,
                                             settings.mock_latency_ms)
        self.gates = gates or Gates(settings.provider_concurrency,
                                    settings.provider_acquire_timeout)
        self.metrics = metrics
        self.admission = threading.BoundedSemaphore(settings.max_concurrent_turns)
        self.node = f"{socket.gethostname()[:24]}:{os.getpid()}"
        self._prices: dict[tuple[str, str], tuple[float, Price | None]] = {}
        self._lock = threading.Lock()
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        """Turns admitted and not yet finished, in this process."""
        return self._in_flight

    def _admitted(self, delta: int) -> None:
        with self._lock:
            self._in_flight += delta
        if self.metrics:
            self.metrics.turns_in_flight.set(self._in_flight)

    # --- prices -----------------------------------------------------------
    def price(self, conn: Connection | None, spec: ModelSpec) -> Price | None:
        """The model's price, from a 60-second cache; `conn` or a new
        connection when it must be read."""
        key = spec.priced_as
        with self._lock:
            hit = self._prices.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        if conn is None:
            with self.engine.connect() as c:
                p = costs.price_for(c, *key)
        else:
            p = costs.price_for(conn, *key)
        with self._lock:
            self._prices[key] = (time.monotonic() + 60, p)
        return p

    # --- conversations ----------------------------------------------------
    def _conversation(self, conn: Connection, tenant_id: str, cid: str) -> Any:
        row = conn.execute(_Q_CONVERSATION, {"cid": cid, "tenant": tenant_id}).first()
        if row is None:
            raise E.NotFound(f"no conversation {cid!r}")
        return row

    def open_conversation(self, tenant_id: str, user_id: str, model: str | None = None,
                          market: str | None = None) -> dict[str, Any]:
        """`market`: the storefront's market the customer is shopping in, which
        the agent is told (the system prompt) -- by default her own country.
        A customer from Malaysia on the Indonesian site is in the Indonesian
        market, as the benchmark's task T09 has it (WHAT_FAILED #67)."""
        from ..generate import MARKETS
        if market is not None and market not in MARKETS:
            raise E.BadRequest(f"no market {market!r}: one of {', '.join(sorted(MARKETS))}")
        name = model or self.settings.default_model
        try:
            spec = self.registry.resolve(name)
            self.registry.check_ready(spec)
        except ModelError as e:
            raise E.BadRequest(str(e)) from e
        with transaction(self.engine, write=True) as conn:
            if self.price(conn, spec) is None:
                raise E.BadRequest(f"model {name!r} has no row in the price book: a "
                                   f"conversation is never run on a model it cannot charge for")
            cu = S.customers
            cust = conn.execute(select(cu.c.country).where(cu.c.tenant_id == tenant_id,
                                                           cu.c.user_id == user_id)).first()
            if cust is None:
                raise E.NotFound(f"no customer {user_id!r}")
            market = market or cust.country
            prompt = system_prompt(user_id, market, self.settings.policy_mode)
            cid = "cv_" + uuid.uuid4().hex[:24]
            now = utcnow()
            conn.execute(sql_insert(S.conversations).values(
                conversation_id=cid, tenant_id=tenant_id, user_id=user_id,
                market=market, status="open", model=spec.name,
                model_settings=spec.settings(), policy_mode=self.settings.policy_mode,
                prompt_digest=digest(prompt), policy_digest=POLICY_DIGEST,
                tools_digest=TOOLS_DIGEST, tool_checks=CHECKS,
                strict_money=self.settings.strict_money, world_clock=NOW.isoformat(),
                verified=[], n_messages=1, created_at=now, updated_at=now))
            conn.execute(sql_insert(S.messages).values(
                conversation_id=cid, seq=0, turn=0, role="system", content=prompt,
                platform=True, created_at=now))
            _event(conn, cid, 0, "conversation.opened", {"model": spec.name})
        return self.get_conversation(tenant_id, cid)

    def get_conversation(self, tenant_id: str, cid: str) -> dict[str, Any]:
        with self.engine.connect() as conn:
            return public_conversation(self._conversation(conn, tenant_id, cid))

    def list_conversations(self, tenant_id: str, user_id: str | None = None, limit: int = 50,
                           before: str | None = None) -> dict[str, Any]:
        """Newest first, a page at a time: `before` is the last id of the page
        before (keyset, so a page is as fast at the millionth row as the first)."""
        c = S.conversations
        q = select(c).where(c.c.tenant_id == tenant_id)
        if user_id:
            q = q.where(c.c.user_id == user_id)
        with self.engine.connect() as conn:
            if before:
                ref = self._conversation(conn, tenant_id, before)
                q = q.where(or_(c.c.created_at < ref.created_at,
                                (c.c.created_at == ref.created_at)
                                & (c.c.conversation_id < ref.conversation_id)))
            rows = conn.execute(q.order_by(c.c.created_at.desc(), c.c.conversation_id.desc())
                                .limit(limit + 1)).all()
        page = [public_conversation(r) for r in rows[:limit]]
        return {"data": page,
                "next": page[-1]["conversation_id"] if len(rows) > limit else None}

    def messages(self, tenant_id: str, cid: str, include_system: bool = False) -> list[dict]:
        m = S.messages
        with self.engine.connect() as conn:
            self._conversation(conn, tenant_id, cid)
            rows = conn.execute(select(m).where(m.c.conversation_id == cid).order_by(m.c.seq))
            out = []
            for r in rows:
                if r.role == "system" and not include_system:
                    continue
                d: dict[str, Any] = {"seq": r.seq, "turn": r.turn, "role": r.role,
                                     "content": r.content}
                if r.tool_calls:
                    d["tool_calls"] = r.tool_calls
                if r.tool_call_id:
                    d["tool_call_id"], d["name"] = r.tool_call_id, r.name
                if r.platform:
                    d["platform"] = True
                out.append(d)
            return out

    def events(self, tenant_id: str, cid: str, after: int = 0, limit: int = 500) -> list[dict]:
        e = S.events
        with self.engine.connect() as conn:
            self._conversation(conn, tenant_id, cid)
            rows = conn.execute(select(e).where(e.c.conversation_id == cid, e.c.event_id > after)
                                .order_by(e.c.event_id).limit(limit))
            return [{"id": int(r.event_id), "type": r.type, "turn": r.turn, "data": r.payload,
                     "at": iso(r.created_at)} for r in rows]

    def tool_calls(self, tenant_id: str, cid: str) -> list[dict]:
        t = S.tool_calls
        with self.engine.connect() as conn:
            self._conversation(conn, tenant_id, cid)
            rows = conn.execute(select(t).where(t.c.conversation_id == cid).order_by(t.c.seq))
            return [{"seq": r.seq, "turn": r.turn, "step": r.step, "tool": r.tool,
                     "args": r.args, "ok": r.ok, "error": r.error, "is_write": r.is_write,
                     "dispatched": r.dispatched, "world_seq": r.world_seq,
                     "customer_seq": r.customer_seq, "allocated_ids": r.allocated_ids,
                     "attempts": r.attempts, "latency_ms": r.latency_ms} for r in rows]

    def close(self, tenant_id: str, cid: str) -> dict[str, Any]:
        c = S.conversations
        with transaction(self.engine, write=True) as conn:
            row = self._conversation(conn, tenant_id, cid)
            now = utcnow()
            if row.lease_owner is not None and row.lease_expires_at is not None \
                    and _aware(row.lease_expires_at) > now:
                raise E.TurnInProgress("a turn is running; close the conversation after it")
            if row.status == "open":
                conn.execute(update(c).where(c.c.conversation_id == cid)
                             .values(status="closed", closed_at=now, updated_at=now))
                _event(conn, cid, row.turns, "conversation.closed", {})
        return self.get_conversation(tenant_id, cid)

    # --- turns ------------------------------------------------------------
    def begin_turn(self, tenant_id: str, cid: str) -> "Turn":
        """Admit a turn and take its conversation's lease. Raises what the
        client is told: 503 at capacity, 404, 409 (closed, turn running, out of
        turns), 429 (a budget spent), 400 (its model withdrawn)."""
        if not self.admission.acquire(timeout=self.settings.admission_timeout):
            if self.metrics:
                self.metrics.rejections.labels("overloaded").inc()
            raise E.Overloaded("too many turns are running here; try again shortly",
                               retry_after=2)
        try:
            owner = f"{self.node}:{uuid.uuid4().hex[:8]}"
            s = self.settings
            with transaction(self.engine, write=True) as conn:
                # The lease first, and the conversation as the lease finds it:
                # the UPDATE waits out a turn that is finishing, and RETURNING
                # gives the row that turn left -- a row read before the lease
                # could be one it has since changed (its tool calls, who it
                # verified). A closed conversation takes no lease.
                now = utcnow()
                row = conn.execute(_Q_LEASE, {
                    "cid": cid, "tenant": tenant_id, "now": now, "owner": owner,
                    "expires": now + timedelta(seconds=s.lease_seconds)}).first()
                if row is None:
                    if self._conversation(conn, tenant_id, cid).status != "open":
                        raise E.ConversationClosed()
                    if self.metrics:
                        self.metrics.rejections.labels("turn_in_progress").inc()
                    raise E.TurnInProgress(retry_after=2)
                # Any refusal below rolls the lease back with the transaction.
                if row.turns >= s.max_turns:
                    raise E.Conflict(f"the conversation has had its {s.max_turns} turns; "
                                     f"open a new one")
                if (row.cost_picousd >= s.conversation_budget_picousd or
                        row.prompt_tokens + row.completion_tokens >= s.conversation_max_tokens):
                    raise E.BudgetExhausted("this conversation has spent its budget; open a "
                                            "new one", scope="conversation")
                daily = conn.execute(_Q_DAILY, {"tenant": tenant_id}).scalar_one()
                if daily is not None and costs.spent_today(conn, tenant_id) >= daily:
                    raise E.BudgetExhausted("the daily spending limit is reached",
                                            retry_after=seconds_to_utc_midnight(),
                                            scope="tenant")
                try:
                    spec = self.registry.resolve(row.model)
                except ModelError as e:
                    raise E.BadRequest(f"this conversation's model is no longer offered: {e}")
            turn = Turn(self, tenant_id, row, owner, spec, daily)
        except BaseException:
            self.admission.release()
            raise
        self._admitted(+1)
        return turn


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


class Turn:
    """One turn, admitted and holding its conversation's lease until run()
    returns (or release() is called without running it)."""

    def __init__(self, rt: Runtime, tenant_id: str, row: Any, owner: str, spec: ModelSpec,
                 daily_budget: int | None):
        self.rt, self.tenant_id, self.row, self.owner, self.spec = rt, tenant_id, row, owner, spec
        self.cid, self.user_id = row.conversation_id, row.user_id
        self.daily_budget = daily_budget
        self.number = row.turns + 1
        self._released = False
        self._lease_cleared = False     # the turn's last transaction gave the lease back

    # --- the lease --------------------------------------------------------
    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            if self._lease_cleared:
                return
            with transaction(self.rt.engine, write=True) as conn:
                conn.execute(_Q_RELEASE, {"cid": self.cid, "owner": self.owner})
        except Exception:  # noqa: BLE001 -- the lease expires by itself
            log.exception("lease release failed", extra={"conversation_id": self.cid})
        finally:
            self.rt.admission.release()
            self.rt._admitted(-1)

    def _renew(self) -> bool:
        """The lease renewed, if this turn still holds it."""
        expires = utcnow() + timedelta(seconds=self.rt.settings.lease_seconds)
        with transaction(self.rt.engine, write=True) as conn:
            return conn.execute(_Q_RENEW, {"cid": self.cid, "owner": self.owner,
                                           "expires": expires}).rowcount == 1

    def _as_holder(self, conn: Connection, set_: dict[str, Any] | None = None,
                   add: dict[str, int] | None = None) -> None:
        """Update the conversation's row as this turn -- setting `set_`, adding
        to `add` -- only while it holds the lease, which every write renews.
        Every write a turn makes goes through here or checks the same, in its
        own transaction -- so a turn that outlived its lease writes nothing
        more, not even its fallback reply."""
        set_, add = set_ or {}, add or {}
        now = utcnow()
        params = {"cid": self.cid, "owner": self.owner, "now": now,
                  "expires": now + timedelta(seconds=self.rt.settings.lease_seconds),
                  **{"v_" + k: v for k, v in set_.items()},
                  **{"d_" + k: v for k, v in add.items()}}
        if conn.execute(_holder(tuple(set_), tuple(add)), params).rowcount != 1:
            raise LeaseLost(f"conversation {self.cid}: the turn outlived its lease")

    # --- the loop ---------------------------------------------------------
    def run(self, text: str, emit: Emit | None = None,
            keepalive: Callable[[], None] | None = None) -> dict[str, Any]:
        beat = _Heartbeat(self, keepalive)
        beat.start()
        try:
            return self._run(text, emit or (lambda _e: None))
        finally:
            beat.stop()
            self.release()

    def _run(self, text: str, emit: Emit) -> dict[str, Any]:
        rt, s, spec = self.rt, self.rt.settings, self.spec
        t0 = time.monotonic()
        out_events: list[dict] = []

        # 1. The customer's message, after answering what a crash left unanswered.
        with transaction(rt.engine, write=True) as conn:
            rows = conn.execute(_Q_MESSAGES, {"cid": self.cid}).all()
            self.messages = [_message(r) for r in rows]
            self.next_seq = rows[-1].seq + 1 if rows else 0
            # Every tool_calls row counts itself on the conversation (record(),
            # _answer()), numbered from 0: the next is the count.
            self.call_seq = int(self.row.tool_calls)
            self.actions = conversation_actions(conn, self.cid)
            self.action_seq = len(self.actions)
            self.verified = list(self.row.verified or [])
            answered = {x.tool_call_id for x in self.messages if x.role == "tool"}
            dangling = [tc for x in self.messages if x.role == "assistant"
                        for tc in x.tool_calls if tc.id not in answered]
            for tc in dangling:
                out_events.append(self._answer(conn, tc, INTERRUPTED, step=0,
                                               turn=self.row.turns, platform=True))
            now = utcnow()
            conn.execute(_Q_MESSAGE, {"conversation_id": self.cid, "seq": self.next_seq,
                                      "turn": self.number, "role": "user", "content": text,
                                      "created_at": now})
            self.messages.append(Message(role="user", content=text))
            self.next_seq += 1
            self._as_holder(conn, {"turns": self.number}, {"n_messages": 1})
            out_events.append(_event(conn, self.cid, self.number, "turn.started",
                                     {"repaired": len(dangling)}))
        for ev in out_events:
            emit(ev)

        price = rt.price(None, spec)
        gate = rt.gates.get(spec.provider)
        backend = rt.registry.backend(spec)
        steps = dispatched = 0
        usage, cost = Usage(), 0
        calls: list[dict[str, Any]] = []
        stop, reply, error = None, "", None
        spent = int(self.row.cost_picousd)
        tokens = int(self.row.prompt_tokens + self.row.completion_tokens)

        while True:
            if steps >= s.max_steps_per_turn or (
                    s.conversation_max_steps
                    and int(self.row.steps) + steps >= s.conversation_max_steps):
                stop = "max_steps"
                break
            if time.monotonic() - t0 >= s.max_turn_seconds:
                stop = "max_turn_seconds"
                break
            if spent >= s.conversation_budget_picousd or tokens >= s.conversation_max_tokens:
                stop = "conversation_budget"
                break
            if self.daily_budget is not None:
                with rt.engine.connect() as conn:
                    if costs.spent_today(conn, self.tenant_id) >= self.daily_budget:
                        stop = "tenant_budget"
                        break

            t1 = time.monotonic()
            try:
                resp = gate.call(lambda: backend.chat(list(self.messages), TOOL_SCHEMAS))
            except (Overloaded, CircuitOpen) as e:
                stop, error = "provider_unavailable", str(e)
            except Exception as e:  # noqa: BLE001 -- the provider's failure ends the turn, not the server
                stop, error = "provider_error", f"{type(e).__name__}: {e}"[:500]
            seconds = time.monotonic() - t1
            if stop is not None:
                if rt.metrics:
                    rt.metrics.model_call(spec.provider, spec.model, seconds, 0, 0, 0, 0,
                                          outcome=stop)
                with transaction(rt.engine, write=True) as conn:
                    self._as_holder(conn)
                    ev = _event(conn, self.cid, self.number, "model.error",
                                {"step": steps + 1, "error": error, "stop": stop})
                emit(ev)
                break

            u, estimated = resp.usage, spec.api in ("mock", "scripted")
            if u.total == 0:
                # No usage reported -- a scripted step, or a provider that sent
                # none: estimated, and marked so, never charged as free.
                u = Usage(sum(approx_tokens(x.content) for x in self.messages),
                          approx_tokens(resp.content) + 24 * len(resp.tool_calls))
                estimated = True
            # What the model sent, as a database can keep it, before anything
            # reads it: 1e400 parses as infinity, and a NUL character is valid
            # JSON ("\u0000") that PostgreSQL's text refuses (store.storable).
            resp.content = storable(resp.content or "")
            resp.served_model = storable(resp.served_model)
            for tc in resp.tool_calls:
                tc.name, tc.id = storable(tc.name), storable(tc.id)
                tc.arguments = storable(tc.arguments)
            call_cost = costs.cost_picousd(u.prompt_tokens, u.cached_tokens,
                                           u.completion_tokens, price)
            steps += 1
            msg = resp.as_message()
            with transaction(rt.engine, write=True) as conn:
                # The money first: a call that was made is charged, whatever else happens.
                conn.execute(_Q_LEDGER, {
                    "tenant_id": self.tenant_id, "conversation_id": self.cid, "turn": self.number,
                    "step": steps, "provider": spec.provider, "model": spec.model,
                    "price_id": price.price_id if price else None,
                    "prompt_tokens": u.prompt_tokens, "cached_tokens": u.cached_tokens,
                    "completion_tokens": u.completion_tokens, "estimated": estimated,
                    "cost_picousd": call_cost, "latency_ms": int(seconds * 1000),
                    "served_model": resp.served_model, "created_at": utcnow()})
                # The step's counters, and the lease renewed, in one statement:
                # no row means the lease is no longer this turn's.
                now = utcnow()
                lost = conn.execute(_Q_STEP, {
                    "cid": self.cid, "owner": self.owner, "d_prompt": u.prompt_tokens,
                    "d_cached": u.cached_tokens, "d_completion": u.completion_tokens,
                    "d_cost": call_cost, "now": now,
                    "expires": now + timedelta(seconds=s.lease_seconds)}).rowcount != 1
                if not lost:
                    conn.execute(_Q_MESSAGE, {
                        "conversation_id": self.cid, "seq": self.next_seq, "turn": self.number,
                        "role": "assistant", "content": msg.content or "",
                        "tool_calls": [tc.to_dict() for tc in msg.tool_calls] or None,
                        "created_at": now})
                    ev = _event(conn, self.cid, self.number, "model.call", {
                        "step": steps, "latency_ms": int(seconds * 1000),
                        "prompt_tokens": u.prompt_tokens, "cached_tokens": u.cached_tokens,
                        "completion_tokens": u.completion_tokens, "estimated": estimated,
                        "cost_picousd": call_cost,
                        "tool_calls": [tc.name for tc in msg.tool_calls],
                        **({"content": msg.content} if msg.content else {})})
            if rt.metrics:
                rt.metrics.model_call(spec.provider, spec.model, seconds, u.prompt_tokens,
                                      u.cached_tokens, u.completion_tokens, call_cost)
            if lost:
                raise LeaseLost(f"conversation {self.cid}: the turn outlived its lease")
            emit(ev)
            self.messages.append(msg)
            self.next_seq += 1
            usage, cost = usage + u, cost + call_cost
            spent, tokens = spent + call_cost, tokens + u.total

            if not msg.tool_calls:
                stop, reply = "replied", msg.content or ""
                break
            for tc in msg.tool_calls:
                calls.append(self._tool(tc, steps, dispatched, emit))
                if calls[-1]["dispatched"]:
                    dispatched += 1

        fallback = stop != "replied"
        latency_ms = int((time.monotonic() - t0) * 1000)
        result = {"conversation_id": self.cid, "turn": self.number,
                  "reply": FALLBACK if fallback else reply, "fallback": fallback,
                  "stop_reason": stop,
                  "tool_calls": [{k: v for k, v in x.items() if k != "dispatched"}
                                 for x in calls],
                  "usage": {"model_calls": steps, "prompt_tokens": usage.prompt_tokens,
                            "cached_tokens": usage.cached_tokens,
                            "completion_tokens": usage.completion_tokens},
                  "cost_picousd": cost, "cost_usd": costs.usd(cost), "latency_ms": latency_ms}
        if error:
            result["error"] = error
        with transaction(rt.engine, write=True) as conn:
            if conn.execute(_Q_FINISH[fallback], {"cid": self.cid, "owner": self.owner,
                                                  "stop": stop, "now": utcnow()}).rowcount != 1:
                raise LeaseLost(f"conversation {self.cid}: the turn outlived its lease")
            if fallback:
                conn.execute(_Q_MESSAGE, {
                    "conversation_id": self.cid, "seq": self.next_seq, "turn": self.number,
                    "role": "assistant", "content": FALLBACK, "platform": True,
                    "created_at": utcnow()})
                self.next_seq += 1
            ev = _event(conn, self.cid, self.number, "turn.completed", result)
        self._lease_cleared = True
        emit(ev)
        if rt.metrics:
            rt.metrics.turn(stop, latency_ms / 1000)
        log.info("turn", extra={"tenant_id": self.tenant_id, "conversation_id": self.cid,
                                "turn": self.number, "stop_reason": stop, "steps": steps,
                                "tool_calls": len(calls), "cost_picousd": cost,
                                "ms": latency_ms})
        return result

    # --- tool calls -------------------------------------------------------
    def _tool(self, tc: ToolCall, step: int, dispatched: int, emit: Emit) -> dict[str, Any]:
        """One of the model's tool calls: run, or answered without running."""
        rt, s = self.rt, self.rt.settings
        args = tc.arguments
        refusal = None
        if isinstance(args, dict) and "__malformed__" in args:
            refusal = "arguments were not valid JSON; re-emit them as a JSON object"
        elif tc.name not in TOOL_NAMES:
            refusal = f"unknown tool {tc.name!r}"       # says nothing of tools it is not shown
        elif dispatched >= s.max_tool_calls_per_turn:
            refusal = "tool call budget exhausted for this turn"
        if refusal is not None:
            with transaction(rt.engine, write=True) as conn:
                ev = self._answer(conn, tc, {"ok": False, "error": refusal}, step, self.number)
            emit(ev)
            if rt.metrics:
                rt.metrics.tool_call(tc.name if tc.name in TOOL_NAMES else "unknown", "refused",
                                     False, 0.0, 1)
            return {"tool": tc.name, "args": args, "ok": False, "error": refusal,
                    "dispatched": False}

        verified_before = list(self.verified)
        call_seq, msg_seq, action_seq = self.call_seq, self.next_seq, self.action_seq
        box: dict[str, Any] = {}

        def record(conn: Connection, out: ToolOutcome) -> None:
            payload = json.dumps(out.result, ensure_ascii=False, default=str)
            conn.execute(_Q_TOOL_CALL, {
                "conversation_id": self.cid, "seq": call_seq, "tenant_id": self.tenant_id,
                "user_id": self.user_id, "turn": self.number, "step": step, "call_id": tc.id,
                "tool": tc.name, "args": args, "ok": bool(out.result.get("ok")),
                "error": out.result.get("error"), "result_sha": out.result_sha,
                "is_write": out.is_write, "dispatched": True, "world_seq": out.world_seq,
                "verified_before": verified_before, "customer_seq": out.customer_seq,
                "changes": out.changes, "allocated_ids": out.allocated or None,
                "attempts": out.attempts, "latency_ms": int(out.seconds * 1000),
                "created_at": utcnow()})
            if out.actions:
                conn.execute(_Q_ACTION, [
                    {"conversation_id": self.cid, "seq": action_seq + i, "call_seq": call_seq,
                     "tool": a.tool, "args": a.args, "ok": a.ok, "error": a.error,
                     "denied": bool(a.denied), "created_at": utcnow()}
                    for i, a in enumerate(out.actions)])
            conn.execute(_Q_MESSAGE, {
                "conversation_id": self.cid, "seq": msg_seq, "turn": self.number, "role": "tool",
                "content": payload, "tool_call_id": tc.id, "name": tc.name,
                "created_at": utcnow()})
            # A turn that lost its lease raises here, and the call's change to
            # the world rolls back with its record: neither is written.
            self._as_holder(conn, {"verified": out.verified}
                            if out.verified != sorted(verified_before) else None,
                            {"tool_calls": 1, "n_actions": len(out.actions), "n_messages": 1})
            box["payload"] = payload
            box["event"] = _event(conn, self.cid, self.number, "tool.result", {
                "step": step, "tool": tc.name, "args": args, "ok": out.result.get("ok"),
                "error": out.result.get("error"), "is_write": out.is_write,
                "customer_seq": out.customer_seq, "latency_ms": int(out.seconds * 1000)})

        try:
            out = run_tool(rt.engine, tenant_id=self.tenant_id, user_id=self.user_id,
                           name=tc.name, args=args, verified=verified_before,
                           prior_actions=self.actions, strict_money=bool(self.row.strict_money),
                           record=record)
        except LeaseLost:
            raise
        except Exception as e:  # noqa: BLE001 -- one call fails, not the turn
            # Nothing was written (the call's transaction rolled back): it is
            # answered as one the platform could not carry out, and the turn
            # goes on -- a model's arguments must never take the server down.
            log.exception("tool call failed", extra={"conversation_id": self.cid})
            failure = {"ok": False, "error": f"the platform could not complete this call "
                                             f"({type(e).__name__}); nothing was changed"}
            with transaction(rt.engine, write=True) as conn:
                ev = self._answer(conn, tc, failure, step, self.number)
            emit(ev)
            return {"tool": tc.name, "args": args, "ok": False, "error": failure["error"],
                    "dispatched": False}
        self.messages.append(Message(role="tool", name=tc.name, tool_call_id=tc.id,
                                     content=box["payload"]))
        self.actions.extend(out.actions)
        self.verified = out.verified
        self.call_seq += 1
        self.next_seq += 1
        self.action_seq += len(out.actions)
        emit(box["event"])
        if rt.metrics:
            outcome = "ok" if out.result.get("ok") else (
                "refused" if "refused by the platform" in str(out.result.get("error")) else "error")
            rt.metrics.tool_call(tc.name, outcome, out.is_write, out.seconds, out.attempts)
        return {"tool": tc.name, "args": args, "ok": out.result.get("ok"),
                "error": out.result.get("error"), "dispatched": True}

    def _answer(self, conn: Connection, tc: ToolCall, result: dict[str, Any], step: int,
                turn: int, platform: bool = False) -> dict:
        """A tool call answered without reaching the tools, recorded as one."""
        payload = json.dumps(result, ensure_ascii=False, default=str)
        conn.execute(_Q_TOOL_CALL, {
            "conversation_id": self.cid, "seq": self.call_seq, "tenant_id": self.tenant_id,
            "user_id": self.user_id, "turn": turn, "step": step, "call_id": tc.id,
            "tool": tc.name, "args": tc.arguments, "ok": False, "error": result.get("error"),
            "result_sha": result_digest(result), "is_write": False, "dispatched": False,
            "world_seq": None, "verified_before": list(self.verified), "latency_ms": 0,
            "created_at": utcnow()})
        conn.execute(_Q_MESSAGE, {
            "conversation_id": self.cid, "seq": self.next_seq, "turn": turn, "role": "tool",
            "content": payload, "tool_call_id": tc.id, "name": tc.name, "platform": platform,
            "created_at": utcnow()})
        self._as_holder(conn, None, {"tool_calls": 1, "n_messages": 1})
        self.messages.append(Message(role="tool", name=tc.name, tool_call_id=tc.id,
                                     content=payload))
        self.call_seq += 1
        self.next_seq += 1
        return _event(conn, self.cid, turn, "tool.result", {
            "step": step, "tool": tc.name, "args": tc.arguments, "ok": False,
            "error": result.get("error"), "is_write": False, "dispatched": False})
