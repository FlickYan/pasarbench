"""
A step's writes in one statement, on PostgreSQL (v84).

A model call's step was six round trips to the database -- BEGIN, its
ledger row, the conversation's counters and lease, the assistant's message,
its event, COMMIT -- and a tool call's record five statements inside the
call's transaction. On PostgreSQL each is one statement now: data-modifying
CTEs, in which the conversation's UPDATE is the gate -- no row, and the lease
is not this turn's -- and a statement that stands alone runs in autocommit,
with no BEGIN or COMMIT of its own. What is written is what was: the same
rows, with the same values, in one round trip.

    STEP          a model call: the ledger row, always; the counters and the
                  lease renewed; the assistant's message and model.call, if
                  the lease is the turn's
    FINISH[fb]    the turn's end: the lease given back, the fallback reply if
                  there is one (fb), turn.completed -- all or nothing
    RECORD[v]     a tool call's record, in the call's transaction: its row,
                  its actions, its message, the conversation's counters (and
                  who is verified, v), tool.result
    OPEN          a turn's start, after its reads: the customer's message,
                  the turn counted, turn.started

SQLite has no data-modifying CTEs and no round trips to save; it keeps the
statements one at a time (runtime.py). Each statement returns its event's id,
or no row if the lease was not the turn's: everything but the ledger row is
written only through the gate, so a statement whose lease is another's draws
nothing -- not even an event id, which an insert that runs and is rolled back
would use up.

Every parameter is named apart from every column of the tables a statement
writes: SQLAlchemy takes a parameter named after a column of a DML statement
for that column's value -- inside a CTE too -- and adds it to the SET or the
VALUES (tests/test_service.py checks each statement for it).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, String, bindparam, func, insert, null, select
from sqlalchemy import column as col
from sqlalchemy import literal_column as sql
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import JSONB

from . import schema as S

_cv, _m, _e = S.conversations, S.messages, S.events
_HELD = (_cv.c.conversation_id == bindparam("cid")) & (_cv.c.lease_owner == bindparam("owner"))
_NOW = DateTime(timezone=True)


def _p(name: str, type_: Any) -> Any:
    return bindparam(name, type_=type_)


def _gate(values: dict[str, Any]) -> Any:
    """The conversation's row, updated as the turn holding its lease."""
    return update(_cv).where(_HELD).values(**values).returning(_cv.c.conversation_id).cte("held")


def _gated(table: Any, names: list[str], held: Any, *values: Any, cte: str) -> Any:
    """Rows inserted only if the gate matched: one per row the gate returned."""
    return insert(table).from_select(names, select(held.c.conversation_id, *values)).cte(cte)


def _step() -> Any:
    lg = S.usage_ledger
    ledger = insert(lg).values(
        tenant_id=_p("l_tenant", String), conversation_id=_p("cid", String),
        turn=_p("l_turn", Integer), step=_p("l_step", Integer),
        provider=_p("l_provider", String), model=_p("l_model", String),
        price_id=_p("l_price", String), prompt_tokens=_p("d_prompt", lg.c.prompt_tokens.type),
        cached_tokens=_p("d_cached", lg.c.cached_tokens.type),
        completion_tokens=_p("d_completion", lg.c.completion_tokens.type),
        estimated=_p("l_estimated", Boolean), cost_picousd=_p("d_cost", lg.c.cost_picousd.type),
        latency_ms=_p("l_latency", Integer), served_model=_p("l_served", String),
        created_at=_p("now", _NOW)).cte("ledger")
    held = _gate({
        "steps": _cv.c.steps + sql("1"), "n_messages": _cv.c.n_messages + sql("1"),
        "prompt_tokens": _cv.c.prompt_tokens + _p("d_prompt", lg.c.prompt_tokens.type),
        "cached_tokens": _cv.c.cached_tokens + _p("d_cached", lg.c.cached_tokens.type),
        "completion_tokens": _cv.c.completion_tokens + _p("d_completion",
                                                          lg.c.completion_tokens.type),
        "cost_picousd": _cv.c.cost_picousd + _p("d_cost", lg.c.cost_picousd.type),
        "updated_at": _p("now", _NOW), "lease_expires_at": _p("expires", _NOW)})
    msg = _gated(_m, ["conversation_id", "seq", "turn", "role", "content", "tool_calls",
                      "created_at"], held,
                 _p("m_seq", Integer), _p("l_turn", Integer), sql("'assistant'", String),
                 _p("m_content", String), _p("m_tool_calls", _m.c.tool_calls.type),
                 _p("now", _NOW), cte="msg")
    ev = insert(_e).from_select(
        ["conversation_id", "turn", "type", "payload", "created_at"],
        select(held.c.conversation_id, _p("l_turn", Integer), sql("'model.call'", String),
               _p("e_payload", _e.c.payload.type), _p("now", _NOW))
    ).returning(_e.c.event_id).cte("ev")
    return select(ev.c.event_id).add_cte(ledger).add_cte(msg)


def _finish(fallback: bool) -> Any:
    values: dict[str, Any] = {"last_stop": _p("f_stop", String), "updated_at": _p("now", _NOW),
                              "lease_owner": null(), "lease_expires_at": null()}
    if fallback:
        values["n_messages"] = _cv.c.n_messages + sql("1")
    held = _gate(values)
    ev = insert(_e).from_select(
        ["conversation_id", "turn", "type", "payload", "created_at"],
        select(held.c.conversation_id, _p("m_turn", Integer), sql("'turn.completed'", String),
               _p("e_payload", _e.c.payload.type), _p("now", _NOW))
    ).returning(_e.c.event_id).cte("ev")
    stmt = select(ev.c.event_id)
    if fallback:
        stmt = stmt.add_cte(_gated(
            _m, ["conversation_id", "seq", "turn", "role", "content", "platform", "created_at"],
            held, _p("m_seq", Integer), _p("m_turn", Integer), sql("'assistant'", String),
            _p("m_content", String), sql("true", Boolean), _p("now", _NOW), cte="msg"))
    return stmt


def _holder(set_: tuple[str, ...], add: tuple[str, ...]) -> Any:
    values: dict[str, Any] = {"updated_at": _p("now", _NOW), "lease_expires_at": _p("expires", _NOW)}
    values.update({k: bindparam("v_" + k, type_=_cv.c[k].type) for k in set_})
    values.update({k: _cv.c[k] + bindparam("d_" + k, type_=_cv.c[k].type) for k in add})
    return _gate(values)


def _record(verified: bool) -> Any:
    tc_, ac = S.tool_calls, S.actions
    held = _holder(("verified",) if verified else (), ("tool_calls", "n_actions", "n_messages"))
    tc = _gated(tc_, [
        "conversation_id", "seq", "tenant_id", "user_id", "turn", "step", "call_id", "tool", "args",
        "ok", "error", "result_sha", "is_write", "dispatched", "world_seq", "verified_before",
        "customer_seq", "changes", "allocated_ids", "attempts", "latency_ms", "created_at"], held,
        _p("t_seq", Integer), _p("t_tenant", String), _p("t_user", String),
        _p("t_turn", Integer), _p("t_step", Integer), _p("t_call", String), _p("t_tool", String),
        _p("t_args", tc_.c.args.type), _p("t_ok", Boolean), _p("t_error", String),
        _p("t_sha", String), _p("t_write", Boolean), sql("true", Boolean),
        _p("t_world", tc_.c.world_seq.type), _p("t_verified", tc_.c.verified_before.type),
        _p("t_customer", tc_.c.customer_seq.type), _p("t_changes", tc_.c.changes.type),
        _p("t_allocated", tc_.c.allocated_ids.type), _p("t_attempts", Integer),
        _p("t_latency", Integer), _p("now", _NOW), cte="tc")
    rows = func.jsonb_to_recordset(_p("a_rows", JSONB)).table_valued(
        col("seq", Integer), col("tool", String), col("args", JSONB), col("ok", Boolean),
        col("error", String), col("denied", Boolean)).render_derived(name="a", with_types=True)
    acts = insert(ac).from_select(
        ["conversation_id", "seq", "call_seq", "tool", "args", "ok", "error", "denied",
         "created_at"],
        select(held.c.conversation_id, rows.c.seq, _p("t_seq", Integer), rows.c.tool,
               rows.c.args, rows.c.ok, rows.c.error, rows.c.denied, _p("now", _NOW))
        .select_from(held).join(rows, sql("true"))
    ).cte("acts")
    msg = _gated(_m, ["conversation_id", "seq", "turn", "role", "content", "tool_call_id", "name",
                      "created_at"], held,
                 _p("m_seq", Integer), _p("t_turn", Integer), sql("'tool'", String),
                 _p("m_content", String), _p("t_call", String), _p("t_tool", String),
                 _p("now", _NOW), cte="msg")
    ev = insert(_e).from_select(
        ["conversation_id", "turn", "type", "payload", "created_at"],
        select(held.c.conversation_id, _p("t_turn", Integer), sql("'tool.result'", String),
               _p("e_payload", _e.c.payload.type), _p("now", _NOW))
    ).returning(_e.c.event_id).cte("ev")
    return select(ev.c.event_id).add_cte(tc).add_cte(acts).add_cte(msg)


def _open() -> Any:
    held = _holder(("turns",), ("n_messages",))
    msg = _gated(_m, ["conversation_id", "seq", "turn", "role", "content", "created_at"], held,
                 _p("m_seq", Integer), _p("m_turn", Integer), sql("'user'", String),
                 _p("m_content", String), _p("now", _NOW), cte="msg")
    ev = insert(_e).from_select(
        ["conversation_id", "turn", "type", "payload", "created_at"],
        select(held.c.conversation_id, _p("m_turn", Integer), sql("'turn.started'", String),
               _p("e_payload", _e.c.payload.type), _p("now", _NOW))
    ).returning(_e.c.event_id).cte("ev")
    return select(ev.c.event_id).add_cte(msg)


STEP = _step()
FINISH = {fb: _finish(fb) for fb in (False, True)}
RECORD = {v: _record(v) for v in (False, True)}
OPEN = _open()
ALL = {"step": STEP, "finish": FINISH[False], "finish, fallback": FINISH[True],
       "record": RECORD[False], "record, verified": RECORD[True], "open": OPEN}
