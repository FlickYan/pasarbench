"""
Run every tool call a tenant's conversations made again, against its starting
world rebuilt from its name, and compare: each result, each write, and the
world they leave.

    python -m pasarbench.service replay --tenant acme

What makes it possible is what the service records of every call it runs
(schema.tool_calls): the state of the customer's world it read (world_seq,
the writes committed to it before), who the conversation had verified, the
action log as it stood (actions.call_seq), the ids it was given, what it
answered (result_sha) and what it changed. The tools are deterministic
functions of those -- the clock is frozen, the ids are given back -- so a
call run again must answer and change exactly what it did.

Calls are run customer by customer, in the order of the states they read:
every call that read state k, then the write that made state k+1. Two
customers never share a row a tool writes, so customers replay independently.
The log and the live world are read in one snapshot, so a tenant can be
replayed while it takes traffic.

A mismatch means one of: the tools or the policy changed since (compare the
conversation's digests and tool_checks), the world builder changed (the
starting world's digest says so first), a row was edited outside the service,
or a write was made without the customer's lock -- which is how the tests show
that it is load-bearing.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select

from ..db import Action
from . import schema as S
from .store import (StoreError, create_all, make_engine, run_tool, seed_tenant, tenant_world,
                    transaction, world_tables)


def _norm(x: Any) -> Any:
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def world_differences(a: dict[str, dict[str, dict[str, Any]]],
                      b: dict[str, dict[str, dict[str, Any]]], limit: int = 20) -> list[str]:
    out: list[str] = []
    for table in sorted(set(a) | set(b)):
        ra, rb = a.get(table, {}), b.get(table, {})
        for rid in sorted(set(ra) | set(rb)):
            if rid not in ra or rid not in rb:
                out.append(f"{table} {rid}: only in the {'database' if rid in rb else 'replay'}")
            elif _norm(ra[rid]) != _norm(rb[rid]):
                fields = sorted(k for k in set(ra[rid]) | set(rb[rid])
                                if _norm(ra[rid].get(k)) != _norm(rb[rid].get(k)))
                out.append(f"{table} {rid}: {', '.join(fields)} differ")
            if len(out) >= limit:
                return out
    return out


def replay_tenant(engine: Engine, tenant_id: str) -> dict[str, Any]:
    t0 = time.monotonic()
    tc, a, c = S.tool_calls, S.actions, S.conversations
    # One snapshot for the log and the world it is compared with: on PostgreSQL
    # a statement apart from the others would see writes committed between them.
    with transaction(engine, write=False, snapshot=True) as conn:
        tenant = conn.execute(select(S.tenants).where(S.tenants.c.tenant_id == tenant_id)).first()
        if tenant is None:
            raise StoreError(f"no tenant {tenant_id!r}")
        strict = {r.conversation_id: bool(r.strict_money) for r in conn.execute(
            select(c.c.conversation_id, c.c.strict_money).where(c.c.tenant_id == tenant_id))}
        calls = conn.execute(select(tc).where(tc.c.tenant_id == tenant_id,
                                              tc.c.dispatched.is_(True))).all()
        logs: dict[str, list[tuple[int, Action]]] = defaultdict(list)
        for r in conn.execute(select(a).join(c, c.c.conversation_id == a.c.conversation_id)
                              .where(c.c.tenant_id == tenant_id).order_by(a.c.seq)):
            logs[r.conversation_id].append((r.call_seq, Action(
                tool=r.tool, args=dict(r.args or {}), ok=bool(r.ok), error=r.error,
                denied=bool(r.denied))))
        live = tenant_world(conn, tenant_id)

    report: dict[str, Any] = {"tenant_id": tenant_id, "world": tenant.world, "calls": len(calls),
                              "writes": sum(1 for r in calls if r.customer_seq is not None),
                              "identical": 0, "mismatches": []}
    with tempfile.TemporaryDirectory(prefix="pasar-replay-") as tmp:
        scratch = make_engine(f"sqlite:///{Path(tmp) / 'replay.db'}")
        try:
            create_all(scratch)
            digest = seed_tenant(scratch, tenant_id, tenant.name, tenant.world)
            report["world_digest_ok"] = digest == tenant.world_digest
            order = sorted(calls, key=lambda r: (r.user_id, r.world_seq,
                                                 r.customer_seq is not None,
                                                 r.conversation_id, r.seq))
            for r in order:
                prior = [act for call_seq, act in logs[r.conversation_id] if call_seq < r.seq]
                problem = None
                try:
                    out = run_tool(scratch, tenant_id=tenant_id, user_id=r.user_id, name=r.tool,
                                   args=r.args, verified=list(r.verified_before or []),
                                   prior_actions=prior,
                                   strict_money=strict.get(r.conversation_id, True),
                                   replay_ids=list(r.allocated_ids or []), attempts=1)
                except StoreError as e:
                    problem = f"could not run: {e}"
                else:
                    if out.world_seq != r.world_seq:
                        problem = (f"read state {out.world_seq} of the customer's world, "
                                   f"not {r.world_seq}")
                    elif out.result_sha != r.result_sha:
                        problem = "answered differently"
                    elif (out.customer_seq is None) != (r.customer_seq is None):
                        problem = ("changed the world" if r.customer_seq is None
                                   else "changed nothing")
                    elif out.customer_seq is not None and (
                            out.customer_seq != r.customer_seq
                            or _norm(out.changes) != _norm(r.changes)):
                        problem = "changed the world differently"
                if problem is None:
                    report["identical"] += 1
                elif len(report["mismatches"]) < 50:
                    report["mismatches"].append({"conversation_id": r.conversation_id,
                                                 "seq": r.seq, "tool": r.tool,
                                                 "problem": problem})
                else:
                    report["mismatches_truncated"] = True
            with scratch.connect() as conn:
                rebuilt = tenant_world(conn, tenant_id)
        finally:
            scratch.dispose()
    diffs = world_differences(rebuilt, live)
    report["world_identical"] = not diffs
    report["world_differences"] = diffs
    report["ok"] = (report["world_digest_ok"] and report["world_identical"]
                    and report["identical"] == report["calls"])
    report["seconds"] = round(time.monotonic() - t0, 3)
    return report


def starting_world_digest(world: str) -> str:
    from .store import world_digest
    return world_digest(world_tables(world))
