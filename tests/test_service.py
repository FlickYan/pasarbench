"""
The agent service (pasarbench/service, docs/SERVICE.md).

What these checks hold it to:

  parity        every reference solution, run through the HTTP API, passes its
                task -- and leaves the world and the action log exactly as the
                benchmark's own episode leaves them: what the benchmark scores
                is what the service runs
  replay        every tool call, run again against the rebuilt starting world,
                answers and changes the same; a row edited behind the service's
                back is found
  concurrency   eight refunds of one order at once never pay back more than was
                paid. On PostgreSQL, without the customer's lock, two refunds
                both pass the money guard -- the lock is load-bearing -- and the
                version check alone still catches a stale read-modify-write
  protocol      idempotency keys, the turn lease, Server-Sent Events, problem
                details, keys and tenants, rate limits, admission control,
                budgets, a failing provider and its breaker, a crash mid-turn
  schema        an empty database migrated to head is exactly schema.py, on
                SQLite and on PostgreSQL; the API has exactly its routes

Needs requirements-service.txt: without it every check is skipped (the
benchmark itself needs nothing). PASAR_TEST_POSTGRES_URL -- a database these
tests may wipe -- runs the storage checks on PostgreSQL too.

Run: python -m tests.test_service
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import warnings

PASS: list[str] = []
FAIL: list[str] = []
SKIP: list[str] = []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))


def skip(label, why):
    SKIP.append(label)
    print(f"  [--] {label}  (skipped: {why})")


# The service's dependencies (requirements-service.txt): the first one missing, if any.
MISSING = next((m for m in ("fastapi", "sqlalchemy", "alembic", "prometheus_client", "httpx")
                if importlib.util.find_spec(m) is None), None)

PG = os.environ.get("PASAR_TEST_POSTGRES_URL", "").strip()
TMP = tempfile.mkdtemp(prefix="pasar-test-")
OP = {"Authorization": "Bearer op-token"}
warnings.filterwarnings("ignore", message=".*httpx.*starlette.testclient.*")
warnings.filterwarnings("ignore", message="Using `httpx` with")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sqlite_url(name: str) -> str:
    return f"sqlite:///{TMP}/{name}.db"


def wipe_postgres() -> None:
    from sqlalchemy import create_engine, text
    e = create_engine(PG)
    with e.begin() as c:
        c.execute(text("DROP SCHEMA public CASCADE"))
        c.execute(text("CREATE SCHEMA public"))
    e.dispose()


def database(kind: str, name: str):
    """A new, empty database with the schema and the price book: (engine, url)."""
    from pasarbench.service.costs import seed_prices
    from pasarbench.service.store import create_all, make_engine, transaction
    if kind == "postgres":
        wipe_postgres()
        url = PG
    else:
        url = sqlite_url(name)
    engine = make_engine(url)
    create_all(engine)
    with transaction(engine, write=True) as conn:
        seed_prices(conn)
    return engine, url


def make_app(engine, url, **settings):
    from pasarbench.service.app import create_app
    from pasarbench.service.config import Settings
    base = dict(database_url=url, admin_token="op-token", log_level="WARNING", rate=1000.0,
                burst=1000)
    base.update(settings)
    return create_app(Settings(**base), engine)


def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def tenant(c, tenant_id: str, world: str = "demo", budget: str | None = None) -> dict:
    r = c.post("/v1/admin/tenants", headers=OP,
               json={"tenant_id": tenant_id, "name": tenant_id, "world": world,
                     **({"daily_budget_usd": budget} if budget else {})})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['api_key']}"}


def norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def until(cond, timeout: float = 10.0) -> bool:
    """Wait for a condition instead of guessing how long it takes."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Plan:
    """A model that makes the calls it is given, one a step, then replies."""
    name = model = "plan"
    reports_usage = True

    def __init__(self, calls, usage=(120, 12), before=None):
        self.calls, self.usage, self.before, self.n = list(calls), usage, before, 0

    def chat(self, messages, tools):
        from pasarbench.harness.types import ModelResponse, ToolCall, Usage
        self.n += 1
        if self.before:
            self.before(self.n)
        start = max(i for i, m in enumerate(messages) if m.role == "user")
        done = sum(1 for m in messages[start:] if m.role == "assistant")
        u = Usage(*self.usage)
        if done < len(self.calls):
            name, args = self.calls[done]
            return ModelResponse(tool_calls=[ToolCall(name=name, arguments=args,
                                                      id=f"c{len(messages)}")], usage=u)
        return ModelResponse(content="done", usage=u)


def demo_orders():
    """From the seed: a captured card order, a processing order, their owners."""
    from pasarbench.db import Database
    t = Database.fresh().tables
    pays = {p["order_id"]: p for p in t["payments"].values()}
    paid = next(o for o in t["orders"].values()
                if pays.get(o["order_id"], {}).get("status") == "captured"
                and pays[o["order_id"]]["method"] in ("card", "ewallet"))
    proc = next(o for o in t["orders"].values() if o["status"] == "processing")
    return paid, pays[paid["order_id"]], proc, t["users"]


# ---------------------------------------------------------------------------
# Costs and resilience: pure functions
# ---------------------------------------------------------------------------

def test_costs():
    print("costs")
    from pasarbench.service.costs import PRICE_BOOK, Price, cost_picousd, micro, price_for, usd
    check("prices are whole microdollars a million tokens: MiMo's cached $0.003625 is 3625",
          micro("0.003625") == 3625 and micro("1.32") == 1_320_000)
    try:
        micro("0.0000001")
        refused = False
    except ValueError:
        refused = True
    check("a price finer than that is refused, not rounded", refused)
    check("every row of the price book is exact", all(micro(x) >= 0 for r in PRICE_BOOK
                                                      for x in r[2:5]))
    flash = Price("p", "deepseek", "deepseek-flash", micro("0.30"), micro("0.006"), micro("1.20"))
    cost = cost_picousd(1_000_000, 900_000, 10_000, flash)
    check("a call's cost: uncached input, cached input and output at their prices, in "
          "picodollars", cost == 100_000 * 300_000 + 900_000 * 6_000 + 10_000 * 1_200_000, cost)
    check("…shown to six places", usd(cost) == "$0.047400", usd(cost))
    calls = [(1234 + 17 * i, 999 + 11 * i, 77 + i) for i in range(500)]
    whole = cost_picousd(sum(c[0] for c in calls), sum(c[1] for c in calls),
                         sum(c[2] for c in calls), flash)
    check("costs add exactly: 500 calls priced one by one sum to the price of their total",
          sum(cost_picousd(*c, flash) for c in calls) == whole)
    check("cached tokens past the prompt count as the prompt", cost_picousd(10, 50, 0, flash)
          == 10 * flash.cached)

    from sqlalchemy import insert
    from pasarbench.service import schema as S
    from pasarbench.service.store import transaction
    engine, _ = database("sqlite", "prices")
    with transaction(engine, write=True) as conn:
        conn.execute(insert(S.price_book).values(
            price_id="deepseek:deepseek-flash:2026-12-01", provider="deepseek",
            model="deepseek-flash", input_microusd_per_mtok=150_000,
            cached_microusd_per_mtok=3_000, output_microusd_per_mtok=600_000,
            effective_from="2026-12-01", source="test"))
    with engine.connect() as conn:
        before = price_for(conn, "deepseek", "deepseek-flash", on="2026-11-30")
        after = price_for(conn, "deepseek", "deepseek-flash", on="2026-12-01")
        none = price_for(conn, "deepseek", "deepseek-flash", on="2026-01-01")
    check("a new price applies from its date, the old one until then",
          before.input == 300_000 and after.input == 150_000 and none is None,
          (before, after, none))
    engine.dispose()


def test_resilience():
    print("resilience")
    from pasarbench.service.resilience import (CircuitBreaker, Overloaded, ProviderGate,
                                               TokenBucket, backoff, is_transient, retry)
    check("429s, 5xx and timeouts are worth another try; a 400 is not",
          is_transient(RuntimeError("HTTP 429 from https://x: slow down"))
          and is_transient(RuntimeError("HTTP 503 from https://x"))
          and is_transient(TimeoutError("The read operation timed out"))
          and not is_transient(RuntimeError("HTTP 400 from https://x: bad request")))
    slept, calls = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("HTTP 503 from https://provider: overloaded")
        return "answer"

    out = retry(flaky, attempts=3, sleep=slept.append, rand=lambda: 0.5)
    check("a transient failure is tried again, with growing waits",
          out == "answer" and len(calls) == 3 and slept == [0.25, 0.5], slept)
    calls.clear()

    def bad():
        calls.append(1)
        raise RuntimeError("HTTP 401 from https://provider: bad key")

    try:
        retry(bad, attempts=3, sleep=slept.append)
        raised = False
    except RuntimeError:
        raised = True
    check("…and anything else raises at once", raised and len(calls) == 1)
    check("full jitter: every wait under its cap", all(
        backoff(n, rand=lambda: 0.9999) < min(8.0, 0.5 * 2 ** n) for n in range(8)))

    now = [0.0]
    b = CircuitBreaker(threshold=3, reset_s=30, clock=lambda: now[0])
    for _ in range(3):
        b.record(False)
    opened = b.state == "open" and b.allow() is not None
    now[0] = 31
    trial = b.allow() is None and b.allow() is not None     # one trial call at a time
    b.record(True)
    check("the breaker opens after three failures, lets one trial through after 30s, and "
          "closes when it works", opened and trial and b.state == "closed")

    gate = ProviderGate("p", concurrency=1, acquire_timeout=0.05)
    held, release = threading.Event(), threading.Event()

    def hold():
        gate.call(lambda: (held.set(), release.wait(2)))

    t = threading.Thread(target=hold)
    t.start()
    held.wait(2)
    try:
        gate.call(lambda: None)
        overloaded = False
    except Overloaded as e:
        overloaded = e.retry_after >= 1
    release.set()
    t.join()
    check("a provider at its concurrency limit turns a call away, with a Retry-After, "
          "instead of queueing it", overloaded)

    clock = [0.0]
    bucket = TokenBucket(rate=2, burst=3, clock=lambda: clock[0])
    burst = [bucket.take() for _ in range(4)]
    clock[0] = 0.5
    check("the token bucket: a burst of 3, then 2 a second",
          burst[:3] == [None] * 3 and burst[3] is not None and abs(burst[3] - 0.5) < 1e-9
          and bucket.take() is None, burst)


# ---------------------------------------------------------------------------
# Schema and migrations
# ---------------------------------------------------------------------------

def test_migrations(kind: str):
    print(f"migrations [{kind}]")
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine, inspect
    from pasarbench.service.__main__ import alembic_config
    from pasarbench.service.schema import metadata
    if kind == "postgres":
        wipe_postgres()
        url = PG
    else:
        url = sqlite_url("migrations")
    cfg = alembic_config(url)
    check(f"[{kind}] one line of revisions", len(ScriptDirectory.from_config(cfg).get_heads()) == 1)
    command.upgrade(cfg, "head")
    engine = create_engine(url)
    with engine.connect() as conn:
        drift = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}),
                                 metadata)
    check(f"[{kind}] an empty database migrated to head is exactly schema.py", drift == [],
          drift[:3])
    command.downgrade(cfg, "base")
    with engine.connect() as conn:
        left = set(inspect(conn).get_table_names()) - {"alembic_version"}
    command.upgrade(cfg, "head")
    check(f"[{kind}] …and downgrades to nothing, and up again", not left, left)
    engine.dispose()


# ---------------------------------------------------------------------------
# Parity: the reference solutions through the API, against the benchmark
# ---------------------------------------------------------------------------

def test_parity(kind: str, every: int = 1):
    print(f"parity [{kind}]")
    from pasarbench.db import Database
    from pasarbench.harness.backends import ScriptedBackend
    from pasarbench.harness.loop import run_episode
    from pasarbench.harness.prompts import system_prompt
    from pasarbench.rescore import all_tasks
    from pasarbench.service.models import solutions
    from pasarbench.service.replay import replay_tenant
    from pasarbench.service.store import conversation_actions, tenant_world
    from pasarbench.verifier import verify

    engine, url = database(kind, "parity")
    app = make_app(engine, url)
    tasks = list(all_tasks().items())[::every]
    passed = same = prompts = 0
    bad: list = []
    replays = []
    t0 = time.monotonic()
    with client(app) as c:
        for i, (tid, task) in enumerate(tasks):
            ten = f"p{i:03d}"
            h = tenant(c, ten, f"task:{tid}")
            cid = c.post("/v1/conversations", headers=h, json={
                "user_id": task.user_id, "model": f"scripted:{tid}",
                "market": task.market}).json()["conversation_id"]
            # What a real model reads first: the scripted one never does (#67).
            shown = c.get(f"/v1/conversations/{cid}/messages?include_system=true",
                          headers=h).json()[0]["content"]
            prompts += shown == system_prompt(task.user_id, task.market, "preload")
            out = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                         json={"content": task.opening}).json()
            conv = c.get(f"/v1/conversations/{cid}", headers=h).json()
            with engine.connect() as conn:
                tables = tenant_world(conn, ten)
                acts = conversation_actions(conn, cid)
            db = Database(tables=tables, action_log=acts,
                          verified={task.user_id} if conv["identity_verified"] else set())
            ok = verify(task, db).passed
            ref = Database.fresh(task.db_patch)
            run_episode(task, ref, ScriptedBackend(solutions()[tid]))
            identical = (norm({k: v for k, v in ref.tables.items() if k != "notes"})
                         == norm({k: v for k, v in tables.items() if k != "notes"})
                         and [a.to_dict() for a in ref.action_log] == [a.to_dict() for a in acts])
            passed += ok
            same += identical
            if not (ok and identical and out["stop_reason"] == "replied"):
                bad.append((tid, ok, identical, out["stop_reason"]))
            if i % 4 == 0:
                replays.append(replay_tenant(engine, ten))
    n = len(tasks)
    scope = "every task" if every == 1 else f"every {every}th task"
    check(f"[{kind}] {scope}'s reference solution passes through the HTTP API "
          f"({passed}/{n}, {time.monotonic() - t0:.0f}s)", passed == n, bad[:3])
    check(f"[{kind}] …leaving the world and the action log exactly as the benchmark's own "
          f"episode does ({same}/{n})", same == n, bad[:3])
    check(f"[{kind}] …the model shown the benchmark's own system prompt, byte for byte, "
          f"on every task ({prompts}/{n}; T09's customer is from MY, its market ID)",
          prompts == n, prompts)
    check(f"[{kind}] replay: every tool call of {len(replays)} of those tenants run again "
          f"answers and changes the same, and rebuilds the same world",
          all(r["ok"] for r in replays) and sum(r["calls"] for r in replays) > 0,
          [r for r in replays if not r["ok"]][:1])
    engine.dispose()


def test_episodes(kind: str, every: int = 8):
    """The benchmark's episode with the agent behind the API (service/episodes.py,
    what scripts/service_sweep.py runs on a real model), traced as the
    benchmark traces one."""
    print(f"episodes through the API [{kind}]")
    from pasarbench.db import Database
    from pasarbench.harness.backends import ScriptedBackend
    from pasarbench.harness.loop import run_episode
    from pasarbench.harness.simulator import SilentUser
    from pasarbench.harness.trace import TraceWriter, read_episode
    from pasarbench.harness.types import Budget
    from pasarbench.rescore import all_tasks, rescore_file
    from pasarbench.service import episodes
    from pasarbench.service.models import solutions

    engine, url = database(kind, "episodes")
    tasks = list(all_tasks().values())[::every]
    agent = episodes.Agent("scripted", "scripted", None)
    root = os.path.join(TMP, f"episodes-{kind}")
    keys = ("step", "turn", "context_messages", "model_content")
    passed = shown = same = agree = 0
    bad: list = []
    with client(make_app(engine, url)) as c:
        for i, task in enumerate(tasks):
            out = episodes.run(task, 0, client=c, operator=OP, engine=engine, agent=agent,
                               simulator=SilentUser(), budget=Budget(),
                               writer=TraceWriter(root=root, run_id="service/full"),
                               tenant_id=f"e{i:04d}")
            ref = TraceWriter(root=root, run_id="benchmark/full")
            run_episode(task, Database.fresh(task.db_patch),
                        ScriptedBackend(solutions()[task.task_id]), trace=ref, run_index=0)
            ref.close()

            def view(recs):
                return ([({k: s[k] for k in keys},
                          [(tc["name"], tc["arguments"]) for tc in s["tool_calls"]],
                          [{k: r.get(k) for k in ("name", "args", "ok", "error", "result_chars",
                                                  "result_sha1", "payment")}
                           for r in s["tool_results"]]) for s in recs if s["type"] == "step"],
                        [(e["turn"], e["ended"], e["text"]) for e in recs
                         if e["type"] == "event" and e["kind"] == "user_turn"])
            name = f"{task.task_id}__r0.jsonl"
            mine = view(read_episode(os.path.join(root, "service", "full", name)))
            theirs = view(read_episode(os.path.join(root, "benchmark", "full", name)))
            rs = rescore_file(out.trace)
            passed += out.passed and out.stop == "done"
            shown += bool(out.prompt_ok and out.tools_ok)
            same += mine == theirs
            agree += rs.status == "rescored" and rs.current == out.passed
            if mine != theirs or not out.passed:
                bad.append((task.task_id, out.stop, out.failures[:1]))
    n = len(tasks)
    check(f"[{kind}] {n} episodes through the API, each task's reference solution: every "
          f"one passes, the model shown the benchmark's own prompt and tools ({passed}, "
          f"{shown})", passed == shown == n, bad[:3])
    check(f"[{kind}] …each traced step for step as the benchmark traces the same episode: "
          f"its calls and arguments, the answers' digests and lengths, its replies, the "
          f"context's length, the customer's turns ({same}/{n})", same == n, bad[:3])
    check(f"[{kind}] …and each trace re-scores to the service's verdict, the benchmark's "
          f"tools replaying the calls the service ran ({agree}/{n})", agree == n)
    engine.dispose()


# ---------------------------------------------------------------------------
# Storage under concurrency
# ---------------------------------------------------------------------------

def test_concurrency(kind: str):
    print(f"concurrency [{kind}]")
    from sqlalchemy import select
    from pasarbench.service import schema as S
    from pasarbench.service.store import run_tool, seed_tenant

    engine, _ = database(kind, "concurrency")
    seed_tenant(engine, "c1", "c1", "demo")
    order, pay, proc, users = demo_orders()
    user, total = order["user_id"], order["total_minor"]
    args = {"order_id": order["order_id"], "amount_minor": int(total * 0.3),
            "method": pay["method"], "reason": "test"}
    gate = threading.Barrier(8)
    outs: list = [None] * 8

    def refund(i):
        gate.wait()
        outs[i] = run_tool(engine, tenant_id="c1", user_id=user, name="issue_refund",
                           args=args, verified=[user], prior_actions=[])

    threads = [threading.Thread(target=refund, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with engine.connect() as conn:
        back = sum(r.amount_minor for r in conn.execute(select(S.refunds).where(
            S.refunds.c.tenant_id == "c1", S.refunds.c.order_id == order["order_id"])))
        seq = conn.execute(select(S.customers.c.write_seq).where(
            S.customers.c.tenant_id == "c1", S.customers.c.user_id == user)).scalar_one()
    ok = [o for o in outs if o.result.get("ok")]
    refused = [o for o in outs if "refused by the platform" in str(o.result.get("error"))]
    check(f"[{kind}] eight refunds of 30% of one order at once: three paid, five refused, "
          f"never more than was paid ({back} of {total})",
          len(ok) == 3 and len(refused) == 5 and back <= total, (len(ok), len(refused), back))
    check(f"[{kind}] …the customer's writes numbered 1, 2, 3, no gaps, no repeats",
          seq == 3 and sorted(o.customer_seq for o in ok) == [1, 2, 3])

    if kind != "postgres":
        skip(f"[{kind}] the lock and the version check, raced", "SQLite has one writer at a "
             "time: a write transaction holds the whole database (BEGIN IMMEDIATE)")
        engine.dispose()
        return

    # Two 60% refunds, raced: the second reads before the first commits.
    def race(lock: bool, tenant_id: str, name: str, a1: dict, a2: dict, user_id: str):
        seed_tenant(engine, tenant_id, tenant_id, "demo")
        t1_wrote, t2_read = threading.Event(), threading.Event()
        res: dict = {}

        def first():
            def hold():
                t1_wrote.set()
                t2_read.wait(1.5)
                time.sleep(0.2)
            res[1] = run_tool(engine, tenant_id=tenant_id, user_id=user_id, name=name[0], args=a1,
                              verified=[user_id], prior_actions=[], lock=lock,
                              hooks={"before_commit": hold})

        def second():
            t1_wrote.wait(5)
            res[2] = run_tool(engine, tenant_id=tenant_id, user_id=user_id, name=name[1], args=a2,
                              verified=[user_id], prior_actions=[], lock=lock,
                              hooks={"after_read": t2_read.set})

        ts = [threading.Thread(target=first), threading.Thread(target=second)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return res

    big = {**args, "amount_minor": int(total * 0.6)}
    paid_back = []
    for lock, ten in ((False, "race-nolock"), (True, "race-lock")):
        res = race(lock, ten, ("issue_refund", "issue_refund"), big, big, user)
        with engine.connect() as conn:
            paid_back.append(sum(r.amount_minor for r in conn.execute(select(S.refunds).where(
                S.refunds.c.tenant_id == ten, S.refunds.c.order_id == order["order_id"]))))
    check(f"[{kind}] without the customer's lock, two raced refunds both pass the money guard "
          f"-- {paid_back[0]} paid back of {total}: the lock is load-bearing",
          paid_back[0] > total, paid_back)
    check(f"[{kind}] with it, the second waits, reads the first, and is refused "
          f"({paid_back[1]} of {total})", paid_back[1] <= total, paid_back)

    owner = proc["user_id"]
    move = {"order_id": proc["order_id"], "new_address": "9 Jalan Baru, 50450 Kuala Lumpur"}
    cancel = {"order_id": proc["order_id"], "reason": "changed my mind"}
    attempts = {}
    for lock, ten in ((False, "stale-nolock"), (True, "stale-lock")):
        res = race(lock, ten, ("modify_shipping_address", "cancel_order"), move, cancel, owner)
        with engine.connect() as conn:
            o = conn.execute(select(S.orders).where(S.orders.c.tenant_id == ten,
                                                    S.orders.c.order_id == proc["order_id"])).first()
        attempts[lock] = (res[2].attempts, res[2].result.get("ok"), o.status,
                          o.shipping_address == move["new_address"], o.version)
    check(f"[{kind}] without the lock, a cancellation that read the order before an address "
          f"change committed is caught by the version check and run again on the new row: "
          f"both changes kept", attempts[False] == (2, True, "cancelled", True, 3), attempts)
    check(f"[{kind}] with the lock it waits instead, and runs once",
          attempts[True] == (1, True, "cancelled", True, 3), attempts)
    engine.dispose()


def test_isolation():
    print("isolation")
    from sqlalchemy import select
    from pasarbench.service import schema as S
    from pasarbench.service.store import run_tool, seed_tenant
    engine, _ = database("sqlite", "isolation")
    seed_tenant(engine, "i1", "i1", "demo")
    seed_tenant(engine, "i2", "i2", "demo")
    from pasarbench.db import Database
    t = Database.fresh().tables
    mine = next(o for o in t["orders"].values() if o["user_id"] == "U001")
    theirs = next(o for o in t["orders"].values() if o["user_id"] != "U001")

    def call(name, args, tenant_id="i1"):
        return run_tool(engine, tenant_id=tenant_id, user_id="U001", name=name, args=args,
                        verified=["U001"], prior_actions=[]).result

    check("another customer's order does not exist for the tools",
          call("get_order", {"order_id": theirs["order_id"]}).get("error") == "no such order")
    check("…nor can it be escalated", call("escalate_to_human", {
        "order_id": theirs["order_id"], "category": "other", "reason": "x"}).get("error")
          == "no such order")
    got = call("issue_store_credit", {"user_id": "U001", "amount_minor": 100, "currency": "MYR",
                                      "reason": "x", "order_id": theirs["order_id"]})
    with engine.connect() as conn:
        grants = conn.execute(select(S.store_credit_grants)).all()
    check("store credit against another customer's order is refused by the platform, and "
          "nothing is written", "refused by the platform" in str(got.get("error"))
          and grants == [], got)
    made = call("escalate_to_human", {"order_id": mine["order_id"], "category": "other",
                                      "reason": "x"}, tenant_id="i1")
    with engine.connect() as conn:
        where = sorted(r.tenant_id for r in conn.execute(select(S.escalations).where(
            S.escalations.c.order_id == mine["order_id"])))
    check("two tenants with the same order ids never touch each other's rows",
          made.get("ok") and where == ["i1"], where)
    engine.dispose()


# ---------------------------------------------------------------------------
# The HTTP API
# ---------------------------------------------------------------------------

ROUTES = {
    ("GET", "/healthz"), ("GET", "/readyz"),
    ("POST", "/v1/conversations"), ("GET", "/v1/conversations"),
    ("GET", "/v1/conversations/{cid}"), ("POST", "/v1/conversations/{cid}/close"),
    ("GET", "/v1/conversations/{cid}/messages"), ("POST", "/v1/conversations/{cid}/messages"),
    ("GET", "/v1/conversations/{cid}/events"), ("GET", "/v1/conversations/{cid}/tool-calls"),
    ("GET", "/v1/usage"),
    ("POST", "/v1/admin/tenants"), ("POST", "/v1/admin/tenants/{tenant_id}/keys"),
    ("POST", "/v1/admin/keys/{key_id}/revoke"), ("GET", "/v1/admin/tenants/{tenant_id}/replay"),
    ("GET", "/v1/admin/outbox"), ("POST", "/v1/admin/outbox/ack"), ("GET", "/v1/admin/gates"),
}


def sse(text: str) -> list[dict]:
    out = []
    for block in text.strip().split("\n\n"):
        ev = {}
        for line in block.splitlines():
            k, _, v = line.partition(": ")
            ev[k] = v
        if ev:
            out.append(ev)
    return out


def test_api():
    print("API")
    engine, url = database("sqlite", "api")
    app = make_app(engine, url, mock_latency_ms=0)
    with client(app) as c:
        paths = {(m.upper(), p) for p, ops in app.openapi()["paths"].items() for m in ops}
        check("the API has exactly its routes", paths == ROUTES,
              sorted(paths ^ ROUTES))
        check("every /v1 operation documents its problem responses",
              all("409" in op.get("responses", {}) for p, ops in app.openapi()["paths"].items()
                  if p.startswith("/v1") for op in ops.values()))

        h = tenant(c, "acme")
        h2 = tenant(c, "other")
        r = c.get("/v1/conversations")
        check("no key: 401, as problem details", r.status_code == 401
              and r.headers["content-type"].startswith("application/problem+json")
              and r.json()["type"] == "/problems/unauthorized")
        check("a wrong key: 401", c.get("/v1/conversations", headers={
            "Authorization": "Bearer psk_nope"}).status_code == 401)
        k = c.post("/v1/admin/tenants/acme/keys", headers=OP).json()
        c.post(f"/v1/admin/keys/{k['key_id']}/revoke", headers=OP)
        check("a revoked key: 401", c.get("/v1/conversations", headers={
            "Authorization": f"Bearer {k['api_key']}"}).status_code == 401)
        check("the admin API wants the operator's token", c.post(
            "/v1/admin/tenants", headers=h, json={"tenant_id": "x1", "name": "x"}).status_code == 401)

        r = c.post("/v1/conversations", headers=h, json={"user_id": "U404"})
        check("a customer the tenant does not have: 404", r.status_code == 404, r.text)
        r = c.post("/v1/conversations", headers=h, json={"user_id": "U001", "model": "gpt-9"})
        check("a model this deployment does not offer: 400", r.status_code == 400
              and "not offered" in r.json()["detail"], r.text)
        keyless = make_app(engine, url, models=("mock", "deepseek-flash"))
        saved = {k: os.environ.pop(k) for k in ("DEEPSEEK_API_KEY",) if k in os.environ}
        try:
            with client(keyless) as c2:
                r = c2.post("/v1/conversations", headers=h,
                            json={"user_id": "U001", "model": "deepseek-flash"})
        finally:
            os.environ.update(saved)
        check("an offered model whose provider has no key on the server: 400, before anything "
              "is sent", r.status_code == 400 and "DEEPSEEK_API_KEY" in r.json()["detail"], r.text)

        r = c.post("/v1/conversations", headers={**h, "Idempotency-Key": "open-1"},
                   json={"user_id": "U001"})
        cid = r.json()["conversation_id"]
        again = c.post("/v1/conversations", headers={**h, "Idempotency-Key": "open-1"},
                       json={"user_id": "U001"})
        check("opening twice with one Idempotency-Key opens one conversation",
              r.status_code == 201 and again.status_code == 201
              and again.json()["conversation_id"] == cid
              and again.headers.get("idempotent-replayed") == "true")
        check("…the same key with another body: 422", c.post(
            "/v1/conversations", headers={**h, "Idempotency-Key": "open-1"},
            json={"user_id": "U002"}).status_code == 422)
        check("another tenant cannot see the conversation: 404, not 403",
              c.get(f"/v1/conversations/{cid}", headers=h2).status_code == 404)
        check("every response carries X-Request-ID, and a sane one is echoed back",
              c.get("/healthz", headers={"X-Request-ID": "abc-123"}).headers.get(
                  "x-request-id") == "abc-123" and len(r.headers.get("x-request-id", "")) == 32)

        log_lines: list[str] = []
        from pasarbench.service.observability import JsonFormatter

        class Grab(logging.Handler):
            def emit(self, record):
                log_lines.append(JsonFormatter().format(record))
        grab = Grab()
        lg = logging.getLogger("pasar")
        kept, lg.handlers = lg.handlers, [grab]
        lg.setLevel(logging.INFO)
        secret = "my new address is 42 Zebra Lane"
        msg = {"content": f"Hi, cancel order O1001 please. {secret}. My digits are 6789"}
        r = c.post(f"/v1/conversations/{cid}/messages", headers={**h, "Idempotency-Key": "m-1"},
                   json=msg)
        body = r.json()
        dup = c.post(f"/v1/conversations/{cid}/messages", headers={**h, "Idempotency-Key": "m-1"},
                     json=msg)
        lg.handlers = kept
        lg.setLevel(logging.WARNING)
        tools = [x["tool"] for x in body["tool_calls"]]
        check("a turn: identity first, then the order, then the cancellation the policy refuses "
              "for a delivered order", tools == ["verify_identity", "get_order", "cancel_order"]
              and body["stop_reason"] == "replied" and "P2.2" in body["reply"], body)
        transcript = c.get(f"/v1/conversations/{cid}/messages", headers=h).json()
        check("a retried message with its Idempotency-Key gets the first one's bytes, and "
              "the customer's message is stored once", dup.content == r.content
              and dup.headers.get("idempotent-replayed") == "true"
              and sum(1 for m in transcript if m["role"] == "user") == 1)
        check("the transcript is what the model saw: no system prompt unless asked",
              [m["role"] for m in transcript][:3] == ["user", "assistant", "tool"]
              and c.get(f"/v1/conversations/{cid}/messages?include_system=true",
                        headers=h).json()[0]["role"] == "system")
        check("the logs say what happened, never what was said",
              any('"msg": "turn"' in x and cid in x for x in log_lines)
              and not any("Zebra" in x for x in log_lines), log_lines[:2])

        audit = c.get(f"/v1/conversations/{cid}/tool-calls", headers=h).json()
        check("the audit log has every call, with the state of the world each read",
              [a["tool"] for a in audit] == tools and all(a["world_seq"] == 0 for a in audit))
        conv = c.get(f"/v1/conversations/{cid}", headers=h).json()
        check("the conversation keeps its provenance: prompt, policy and tools by digest",
              all(len(conv["provenance"][k]) == 16 for k in
                  ("prompt_digest", "policy_digest", "tools_digest"))
              and conv["provenance"]["world_clock"].startswith("2026-11-11T10:00")
              and conv["identity_verified"] is True, conv["provenance"])

        with c.stream("POST", f"/v1/conversations/{cid}/messages?stream=true", headers=h,
                      json={"content": "I want to talk to a human about O1001"}) as s:
            ctype = s.headers["content-type"]
            events = sse("".join(s.iter_text()))
        kinds = [e.get("event") for e in events]
        ids = [int(e["id"]) for e in events if "id" in e]
        done = json.loads(events[-1]["data"])
        check("?stream=true: Server-Sent Events as the turn runs, in order, each with its id",
              ctype.startswith("text/event-stream") and kinds[0] == "turn.started"
              and kinds[-1] == "turn.completed" and "tool.result" in kinds
              and ids == sorted(ids) and len(set(ids)) == len(ids), kinds)
        check("…the last event is the turn's whole result",
              done["stop_reason"] == "replied" and done["turn"] == 2
              and [x["tool"] for x in done["tool_calls"]] == ["get_order", "escalate_to_human"],
              done)
        kept = c.get(f"/v1/conversations/{cid}/events?after={ids[0] - 1}", headers=h).json()
        check("a dropped stream loses nothing: the events are kept, by id",
              [e["id"] for e in kept] == ids)

        sk = {**h, "Idempotency-Key": "s-1"}
        msg2 = {"content": "track O1001 please"}
        with c.stream("POST", f"/v1/conversations/{cid}/messages?stream=true", headers=sk,
                      json=msg2) as s1:
            last = sse("".join(s1.iter_text()))[-1]
        with c.stream("POST", f"/v1/conversations/{cid}/messages?stream=true", headers=sk,
                      json=msg2) as s2:
            replay = sse("".join(s2.iter_text()))
            replay_type = s2.headers["content-type"]
        check("a retried stream comes back as a stream: the turn's last event, not run again",
              replay_type.startswith("text/event-stream") and len(replay) == 1
              and replay[0]["event"] == "turn.completed"
              and json.loads(replay[0]["data"])["turn"] == json.loads(last["data"])["turn"],
              (replay_type, replay))
        with c.stream("POST", f"/v1/conversations/{cid}/messages?stream=true", headers=h,
                      json={"content": "where is O1001?"}):
            pass                                    # gone before reading a byte
        done = until(lambda: not c.get(f"/v1/conversations/{cid}", headers=h).json()[
            "turn_in_progress"] and app.state.runtime.in_flight == 0)
        check("a client gone before its stream starts: the turn still runs to its end and "
              "gives back its lease and its slot", done and c.get(
                  f"/v1/conversations/{cid}", headers=h).json()["last_stop"] == "replied")

        out = c.get("/v1/admin/outbox", headers=OP).json()
        check("the escalation's ticket is in the outbox, written with it",
              len(out) == 1 and out[0]["topic"] == "escalation.created", out)
        acked = c.post("/v1/admin/outbox/ack", headers=OP,
                       json={"event_ids": [out[0]["event_id"]]}).json()
        check("…and gone once acknowledged", acked == {"acknowledged": 1}
              and c.get("/v1/admin/outbox", headers=OP).json() == [])

        usage = {g: c.get(f"/v1/usage?group_by={g}", headers=h).json()
                 for g in ("model", "day", "conversation")}
        conv = c.get(f"/v1/conversations/{cid}", headers=h).json()
        check("the ledger adds up the same by model, by day and by conversation, and to the "
              "conversation's own cost", len({sum(r["cost_picousd"] for r in v)
                                               for v in usage.values()}) == 1
              and usage["model"][0]["cost_picousd"] == conv["cost_picousd"] > 0, usage)

        rep = c.get("/v1/admin/tenants/acme/replay", headers=OP).json()
        check("the operator can replay a tenant: every call the same",
              rep["ok"] and rep["calls"] == len(c.get(f"/v1/conversations/{cid}/tool-calls",
                                                         headers=h).json()), rep)
        from sqlalchemy import update
        from pasarbench.service import schema as S
        from pasarbench.service.store import transaction
        with transaction(engine, write=True) as conn:
            conn.execute(update(S.escalations).where(S.escalations.c.tenant_id == "acme")
                         .values(category="other"))
        rep = c.get("/v1/admin/tenants/acme/replay", headers=OP).json()
        check("…and a row edited behind the service's back is found",
              not rep["ok"] and any("escalations" in d for d in rep["world_differences"]), rep)

        newer = c.post("/v1/conversations", headers=h, json={"user_id": "U002"}).json()[
            "conversation_id"]
        page = c.get("/v1/conversations?limit=1", headers=h).json()
        page2 = c.get(f"/v1/conversations?limit=1&before={page['next']}", headers=h).json()
        check("conversations come a page at a time, newest first",
              [x["conversation_id"] for x in page["data"]] == [newer] and page["next"] == newer
              and [x["conversation_id"] for x in page2["data"]] == [cid]
              and page2["next"] is None, (page, page2))
        r = c.post(f"/v1/conversations/{cid}/close", headers=h)
        r2 = c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": "hello?"})
        check("a closed conversation takes no more messages: 409 conversation-closed",
              r.json()["status"] == "closed" and r2.status_code == 409
              and r2.json()["type"] == "/problems/conversation-closed")
        r = c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": ""})
        check("an invalid body: 422, as problem details", r.status_code == 422
              and r.json()["type"] == "/problems/invalid-request")
        m = c.get("/metrics").text
        check("metrics: by route template, never by id (no conversation id in a label)",
              'route="/v1/conversations/{cid}/messages"' in m and "cv_" not in m
              and "pasar_turns_total" in m and "pasar_model_cost_picousd_total" in m)
        check("ready: the database answers and has the schema",
              c.get("/readyz").json()["status"] == "ready")
    engine.dispose()


def test_turn_control():
    print("turn control")
    engine, url = database("sqlite", "control")
    app = make_app(engine, url, mock_latency_ms=300, max_concurrent_turns=2,
                   admission_timeout=0.05)
    with client(app) as c:
        h = tenant(c, "acme")
        cids = [c.post("/v1/conversations", headers=h, json={"user_id": u}).json()[
            "conversation_id"] for u in ("U001", "U002", "U003")]
        got: dict = {}

        def post(i, cid, key=None):
            got[i] = c.post(f"/v1/conversations/{cid}/messages",
                            headers={**h, **({"Idempotency-Key": key} if key else {})},
                            json={"content": "where is O1001?"})

        def running(cid):
            return lambda: c.get(f"/v1/conversations/{cid}", headers=h).json()[
                "turn_in_progress"]

        t = threading.Thread(target=post, args=(0, cids[0]))
        t.start()
        until(running(cids[0]))
        post(1, cids[0])
        t.join()
        check("one turn at a time in a conversation: the second gets 409 turn-in-progress, "
              "with Retry-After", got[0].status_code == 200 and got[1].status_code == 409
              and got[1].json()["type"] == "/problems/turn-in-progress"
              and "retry-after" in got[1].headers)

        t = threading.Thread(target=post, args=(2, cids[1], "k-9"))
        t.start()
        until(running(cids[1]))
        post(3, cids[1], "k-9")
        t.join()
        post(4, cids[1], "k-9")
        check("a retry while the first is still running: 409 idempotency-key-in-progress; "
              "after it, the first's answer", got[3].status_code == 409
              and got[3].json()["type"] == "/problems/idempotency-key-in-progress"
              and got[4].status_code == 200 and got[4].json() == got[2].json())

        ts = [threading.Thread(target=post, args=(5 + i, cids[i])) for i in range(2)]
        for x in ts:
            x.start()
        until(lambda: app.state.runtime.in_flight == 2)
        post(7, cids[2])
        for x in ts:
            x.join()
        check("past PASAR_MAX_CONCURRENT_TURNS a turn is turned away at once: 503, "
              "Retry-After", got[7].status_code == 503 and "retry-after" in got[7].headers
              and got[5].status_code == got[6].status_code == 200)

        from sqlalchemy import insert, update
        from pasarbench.service import schema as S
        from pasarbench.service.runtime import utcnow
        from pasarbench.service.store import transaction
        from datetime import timedelta
        cid = cids[2]
        with transaction(engine, write=True) as conn:
            conn.execute(insert(S.messages).values(
                conversation_id=cid, seq=1, turn=1, role="user", content="cancel O1003"))
            conn.execute(insert(S.messages).values(
                conversation_id=cid, seq=2, turn=1, role="assistant", content="",
                tool_calls=[{"id": "call_dead", "name": "get_order",
                             "arguments": {"order_id": "O1003"}}]))
            conn.execute(update(S.conversations).where(S.conversations.c.conversation_id == cid)
                         .values(turns=1, lease_owner="dead-process",
                                 lease_expires_at=utcnow() + timedelta(seconds=60)))
        held = c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": "hello?"})
        with transaction(engine, write=True) as conn:
            conn.execute(update(S.conversations).where(S.conversations.c.conversation_id == cid)
                         .values(lease_expires_at=utcnow() - timedelta(seconds=1)))
        r = c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": "hello?"})
        msgs = c.get(f"/v1/conversations/{cid}/messages", headers=h).json()
        repaired = next((m for m in msgs if m.get("tool_call_id") == "call_dead"), {})
        started = next(e for e in c.get(f"/v1/conversations/{cid}/events", headers=h).json()
                       if e["type"] == "turn.started" and e["turn"] == 2)
        check("a crash mid-turn: its lease holds the conversation until it expires (409), "
              "then the next turn takes it", held.status_code == 409 and r.status_code == 200)
        check("…and first answers the call the crash left unanswered, as not run",
              "interrupted" in repaired.get("content", "") and repaired.get("platform") is True
              and started["data"]["repaired"] == 1, repaired)
    engine.dispose()


def test_limits_and_budgets():
    print("limits and budgets")
    engine, url = database("sqlite", "limits")
    app = make_app(engine, url, rate=1.0, burst=3)
    with client(app) as c:
        h = tenant(c, "acme")
        codes = [c.get("/v1/conversations", headers=h) for _ in range(4)]
        check("past its rate a tenant gets 429 rate-limited, with Retry-After",
              [x.status_code for x in codes] == [200, 200, 200, 429]
              and "retry-after" in codes[3].headers, [x.status_code for x in codes])
    engine.dispose()

    engine, url = database("sqlite", "budgets")
    from pasarbench.service.config import usd_to_picousd
    app = make_app(engine, url, conversation_budget_picousd=usd_to_picousd("0.0002"))
    with client(app) as c:
        h = tenant(c, "acme")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U001"}).json()[
            "conversation_id"]
        r = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                   json={"content": "cancel O1001, digits 6789"}).json()
        r2 = c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": "hello?"})
        check("a conversation that spends its budget mid-turn stops, and the customer gets a "
              "fallback reply, not silence", r["stop_reason"] == "conversation_budget"
              and r["fallback"] and r["reply"].startswith("I'm sorry"), r)
        check("…and takes no more turns: 429 budget-exhausted", r2.status_code == 429
              and r2.json()["type"] == "/problems/budget-exhausted"
              and r2.json()["scope"] == "conversation", r2.text)
    engine.dispose()

    engine, url = database("sqlite", "daily")
    with client(make_app(engine, url)) as c:
        h2 = tenant(c, "frugal", budget="0.0001")
        cid = c.post("/v1/conversations", headers=h2, json={"user_id": "U001"}).json()[
            "conversation_id"]
        first = c.post(f"/v1/conversations/{cid}/messages", headers=h2,
                       json={"content": "where is O1001?"}).json()
        r = c.post(f"/v1/conversations/{cid}/messages", headers=h2, json={"content": "thanks"})
        check("a tenant that passes its daily limit mid-turn stops there, with a fallback "
              "reply", first["stop_reason"] == "tenant_budget" and first["fallback"], first)
        check("…and takes no more turns until midnight UTC: 429, Retry-After",
              r.status_code == 429 and r.json()["scope"] == "tenant"
              and int(r.headers["retry-after"]) > 0, r.text)
    engine.dispose()

    # A cap on a conversation's model calls in all, as the benchmark caps an
    # episode's steps (scripts/service_sweep.py sets it to them).
    engine, url = database("sqlite", "steps")
    with client(make_app(engine, url, conversation_max_steps=2)) as c:
        h = tenant(c, "acme")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U001"}).json()[
            "conversation_id"]
        r = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                   json={"content": "cancel O1001, digits 6789"}).json()
        r2 = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                    json={"content": "hello?"}).json()
        check("a conversation's cap on model calls in all holds inside a turn and across "
              "turns: two in the first, none in the next, each ended as max_steps",
              (r["stop_reason"], r["usage"]["model_calls"], r2["stop_reason"],
               r2["usage"]["model_calls"]) == ("max_steps", 2, "max_steps", 0)
              and r["fallback"], (r["stop_reason"], r["usage"], r2["stop_reason"]))
    engine.dispose()


def test_market():
    print("market")
    from pasarbench.harness.prompts import system_prompt
    engine, url = database("sqlite", "market")
    with client(make_app(engine, url)) as c:
        h = tenant(c, "acme")
        own = c.post("/v1/conversations", headers=h, json={"user_id": "U001"}).json()
        there = c.post("/v1/conversations", headers=h,
                       json={"user_id": "U001", "market": "ID"}).json()
        shown = c.get(f"/v1/conversations/{there['conversation_id']}/messages"
                      f"?include_system=true", headers=h).json()[0]["content"]
        check("a conversation is in her own country's market unless the storefront says "
              "another, and the agent is told the one it is in (WHAT_FAILED #67)",
              own["market"] == "MY" and there["market"] == "ID"
              and shown == system_prompt("U001", "ID", "preload"), (own["market"],
                                                                     there["market"]))
        bad = c.post("/v1/conversations", headers=h, json={"user_id": "U001", "market": "ZZ"})
        odd = c.post("/v1/conversations", headers=h, json={"user_id": "U001", "market": "id"})
        check("…a market no world has is refused (400), one not written as one is invalid "
              "(422)", (bad.status_code, odd.status_code) == (400, 422),
              (bad.status_code, odd.status_code))
    engine.dispose()


def test_provider_failure():
    print("provider failure")
    from pasarbench.service.resilience import CircuitBreaker, Gates
    engine, url = database("sqlite", "provider")
    app = make_app(engine, url)
    rt = app.state.runtime
    rt.gates = Gates(sleep=lambda s: None, breaker=lambda: CircuitBreaker(threshold=2,
                                                                          reset_s=60))
    calls = []

    class Down:
        name = model = "down"
        reports_usage = True

        def chat(self, messages, tools):
            calls.append(1)
            raise RuntimeError("HTTP 503 from https://provider.example: overloaded")

    real = rt.registry.backend
    rt.registry.backend = lambda spec: Down()
    with client(app) as c:
        h = tenant(c, "acme")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U001"}).json()[
            "conversation_id"]
        out = [c.post(f"/v1/conversations/{cid}/messages", headers=h,
                      json={"content": "where is O1001?"}).json() for _ in range(3)]
        check("a provider that keeps failing: each call tried three times, then the turn ends "
              "with a fallback reply", out[0]["stop_reason"] == "provider_error"
              and out[0]["fallback"] and len(calls) >= 3, out[0])
        check("…after two failed turns the breaker opens: the next fails fast, unsent",
              out[2]["stop_reason"] == "provider_unavailable" and len(calls) == 6,
              (len(calls), out[2]["stop_reason"]))
        ledger = c.get("/v1/usage", headers=h).json()
        check("…and nothing is charged for calls that never answered", ledger == [], ledger)
        rt.registry.backend = real
    engine.dispose()


def test_bad_arguments(kind: str):
    """A model's arguments must never take a turn down (the review of v75):
    an amount of 1e400, a voucher of 0, a credit past what a column holds, a
    300-character tool name -- each answered as a refusal, the turn replying."""
    print(f"bad arguments [{kind}]")
    from sqlalchemy import select
    from pasarbench.service import schema as S
    engine, url = database(kind, "badargs")
    app = make_app(engine, url)
    calls = [
        ("verify_identity", {"user_id": "U001", "phone_last4": "6789"}),
        ("issue_refund", {"order_id": "O1008", "amount_minor": float("inf"), "method": "card",
                          "reason": "x"}),
        ("issue_goodwill_voucher", {"user_id": "U001", "amount_minor": 0, "currency": "MYR",
                                    "reason": "x"}),
        ("issue_store_credit", {"user_id": "U001", "amount_minor": 10 ** 19, "currency": "MYR",
                                "reason": "x"}),
        ("t" * 300, {}),
        ("issue_store_credit", {"user_id": "U001", "amount_minor": 270_000, "currency": "SGD",
                                "reason": "x", "order_id": "O1008"}),
        ("issue_store_credit", {"user_id": "U001", "amount_minor": 100, "currency": "MYR",
                                "reason": "x", "order_id": ""}),
        # O1008 is IDR 270,000, SGD 22.14; her account is in MYR (P5.3): MYR 70 is SGD 21
        ("issue_store_credit", {"user_id": "U001", "amount_minor": 7_000, "currency": "MYR",
                                "reason": "x", "order_id": "O1008"}),
        ("issue_store_credit", {"user_id": "U001", "amount_minor": 1_000, "currency": "MYR",
                                "reason": "x", "order_id": "O1008"}),
    ]
    app.state.runtime.registry.backend = lambda spec: Plan(calls)
    with client(app) as c:
        h = tenant(c, "acme")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U001"}).json()[
            "conversation_id"]
        r = c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": "go"})
        out = r.json()
        with engine.connect() as conn:
            vouchers = conn.execute(select(S.vouchers)).all()
            credits = [(g.amount_minor, g.order_id) for g in conn.execute(
                select(S.store_credit_grants))]
        again = c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": "go"})
    errs = [x.get("error") or "" for x in out.get("tool_calls", [])]
    check(f"[{kind}] nine awkward calls in one turn: every one answered, the turn replies, "
          f"and the next turn runs", r.status_code == 200 and out["stop_reason"] == "replied"
          and len(errs) == 9 and again.status_code == 200, (r.status_code, r.text[:300]))
    check(f"[{kind}] an amount of 1e400 is the tool's to refuse, as no integer",
          r.status_code == 200 and errs[1] and "refused by the platform" not in errs[1], errs)
    check(f"[{kind}] a voucher of 0 and a credit of 10**19: refused by the platform, nothing "
          f"written", "refused by the platform" in errs[2] and "refused by the platform"
          in errs[3] and vouchers == [], errs)
    check(f"[{kind}] a 300-character tool name: an unknown tool, recorded whole",
          "unknown tool" in errs[4], errs[4][:80])
    check(f"[{kind}] store credit in a currency not the account's is refused (P5.3); an "
          f"empty order id is no order, as the tools read it",
          "account's currency, MYR" in errs[5] and errs[6] == "", errs[5:7])
    check(f"[{kind}] credit in her currency against an order in another is counted against "
          f"what was paid, in SGD: MYR 70 of an IDR 270,000 order passes, MYR 10 more does not",
          errs[7] == "" and "past what was paid" in errs[8]
          and credits == [(100, ""), (7_000, "O1008")], (errs[7:], credits))
    engine.dispose()


def test_nul(kind: str):
    """PostgreSQL's text holds no NUL character, and JSON can carry one
    ("\u0000"): from the customer it is invalid input (422), from the model it
    is stored as U+FFFD, and nothing ends in a 500 (WHAT_FAILED #65)."""
    print(f"NUL [{kind}]")
    from sqlalchemy import select
    from pasarbench.harness.types import ModelResponse, ToolCall, Usage
    from pasarbench.service import schema as S
    engine, url = database(kind, "nul")
    app = make_app(engine, url)
    mode = {"m": None}

    class Agent:
        name = model = "agent"
        reports_usage = True

        def chat(self, messages, tools):
            if mode["m"] == "args" and messages[-1].role == "user":
                return ModelResponse(tool_calls=[ToolCall(
                    name="get_order", arguments={"order_id": "O10\x0001"},
                    id=f"c{len(messages)}")], usage=Usage(9, 9))
            if mode["m"] == "content":
                return ModelResponse(content="Your order\x00 is on its way", usage=Usage(9, 9))
            return ModelResponse(content="ok", usage=Usage(9, 9))

    app.state.runtime.registry.backend = lambda spec: Agent()
    with client(app) as c:
        h = tenant(c, "acme")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U001"}).json()[
            "conversation_id"]

        def say(text):
            return c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": text})
        refused = [say("hi\u0000there").status_code,
                   c.get("/v1/conversations/cv_%00", headers=h).status_code,
                   c.post("/v1/conversations", headers=h,
                          json={"user_id": "U0\u000001"}).status_code,
                   c.get("/v1/conversations?user_id=U%0001", headers=h).status_code]
        mode["m"] = "content"
        text = say("hi")
        mode["m"] = "args"
        args = say("hi")
        mode["m"] = None
        after = say("hi")
        with engine.connect() as conn:
            charged = len(conn.execute(select(S.usage_ledger)).all())
    check(f"[{kind}] a NUL from the customer -- in her message, an id, a query -- is "
          f"invalid input: 422", refused == [422, 422, 422, 422], refused)
    check(f"[{kind}] a NUL from the model, in its reply or its arguments, is stored as "
          f"U+FFFD: the turns reply, the next runs, and every call is charged",
          text.status_code == 200 and "\ufffd" in text.json()["reply"]
          and args.status_code == 200 and args.json()["tool_calls"][0]["error"]
          == "no such order" and after.status_code == 200 and charged == 4,
          (text.status_code, args.status_code, after.status_code, charged))
    engine.dispose()


def test_order(kind: str):
    """A customer's rows come back in the benchmark's order on both databases:
    PostgreSQL moves an updated row and returns rows in no order unless asked."""
    print(f"row order [{kind}]")
    from pasarbench.db import Database
    from pasarbench.service.store import run_tool, seed_tenant
    from pasarbench.tools import call as tool_call
    engine, _ = database(kind, "order")
    seed_tenant(engine, "o1", "o1", "synthetic:20")
    user, order = "U002-C0001", "O1002-C0001"
    for k in range(3):
        run_tool(engine, tenant_id="o1", user_id=user, name="modify_shipping_address",
                 args={"order_id": order, "new_address": f"{k} Jalan Baru, 50450 KL"},
                 verified=[user], prior_actions=[])
    got = run_tool(engine, tenant_id="o1", user_id=user, name="list_user_orders",
                   args={"user_id": user}, verified=[user], prior_actions=[]).result
    from pasarbench.service.store import world_tables
    db = Database(tables=world_tables("synthetic:20"))
    db.verified = {user}
    for k in range(3):
        tool_call(db, "modify_shipping_address",
                  {"order_id": order, "new_address": f"{k} Jalan Baru, 50450 KL"})
    want = tool_call(db, "list_user_orders", {"user_id": user})
    check(f"[{kind}] after three updates to one of her orders, her orders list in the "
          f"benchmark's order", [o["order_id"] for o in got["orders"]]
          == [o["order_id"] for o in want["orders"]], (got["orders"], want["orders"]))
    engine.dispose()


def test_failures_keep_their_key():
    """A turn that began and then failed keeps its answer under its key: it may
    have changed things, and a retry must not run it again (the review of v75:
    a lost lease freed the key, and the retry escalated twice)."""
    print("failures keep their key")
    from datetime import timedelta
    from sqlalchemy import insert, select, update
    from pasarbench.service import schema as S
    from pasarbench.service.runtime import utcnow
    from pasarbench.service.store import transaction
    engine, url = database("sqlite", "failkey")
    app = make_app(engine, url)
    box = {}

    def steal(n):
        if n == 2:      # the second call: after the first tool call ran
            with transaction(engine, write=True) as conn:
                conn.execute(update(S.conversations).where(
                    S.conversations.c.conversation_id == box["cid"]).values(
                    lease_owner="another-process",
                    lease_expires_at=utcnow() + timedelta(seconds=60)))

    calls = [("verify_identity", {"user_id": "U001", "phone_last4": "6789"}),
             ("escalate_to_human", {"order_id": "O1001", "category": "other", "reason": "x"})]
    app.state.runtime.registry.backend = lambda spec: Plan(calls, before=steal)
    with client(app) as c:
        h = tenant(c, "acme")
        box["cid"] = cid = c.post("/v1/conversations", headers=h,
                                  json={"user_id": "U001"}).json()["conversation_id"]
        hk = {**h, "Idempotency-Key": "k-lost"}
        first = c.post(f"/v1/conversations/{cid}/messages", headers=hk, json={"content": "go"})
        retry = c.post(f"/v1/conversations/{cid}/messages", headers=hk, json={"content": "go"})
        with engine.connect() as conn:
            users = conn.execute(select(S.messages).where(
                S.messages.c.conversation_id == cid, S.messages.c.role == "user")).all()
            esc = conn.execute(select(S.escalations)).all()
            after = [e.type for e in conn.execute(select(S.events).where(
                S.events.c.conversation_id == cid).order_by(S.events.c.event_id))]
            ledger = conn.execute(select(S.usage_ledger)).all()
        check("a turn that loses its lease mid-way answers 409, and its key keeps that answer: "
              "the retry gets it back and runs nothing", first.status_code == 409
              and retry.status_code == 409 and retry.headers.get("idempotent-replayed")
              == "true" and retry.content == first.content and len(users) == 1
              and esc == [], (first.status_code, retry.status_code, len(users), len(esc)))
        check("…and once it lost the lease it wrote nothing more -- no error event, no "
              "fallback reply -- but the model call it made is charged",
              after[-1] == "tool.result" and "model.error" not in after
              and len(ledger) == 2, (after, len(ledger)))
        with transaction(engine, write=True) as conn:
            conn.execute(update(S.conversations).where(
                S.conversations.c.conversation_id == cid).values(
                lease_owner=None, lease_expires_at=None))
            conn.execute(insert(S.idempotency_keys).values(
                tenant_id="acme", key="k-dead", request_hash="x" * 64, state="in_progress",
                created_at=utcnow(), expires_at=utcnow() - timedelta(seconds=1)))
        app.state.runtime.registry.backend = lambda spec: Plan([])
        r = c.post(f"/v1/conversations/{cid}/messages", headers={**h, "Idempotency-Key":
                                                                  "k-dead"},
                   json={"content": "hello"})
        check("a key whose request died with its process frees itself when the turn's lease "
              "would have", r.status_code == 200, r.text[:200])
    engine.dispose()


def test_long_turns():
    """A turn longer than its lease (one step can be: a gate's wait and three
    tries at a slow provider) keeps the lease and its key alive, so a retry
    neither takes the conversation over nor runs the turn again (the second
    review of v75); and a stream works on a server that ran no startup."""
    print("long turns")
    from sqlalchemy import select
    from pasarbench.service import schema as S
    engine, url = database("sqlite", "long")
    app = make_app(engine, url, lease_seconds=1.0)
    calls = [("verify_identity", {"user_id": "U001", "phone_last4": "6789"}),
             ("get_order", {"order_id": "O1001"}),
             ("escalate_to_human", {"order_id": "O1001", "category": "other", "reason": "x"})]
    # The second model call takes 2 s, twice the lease: no write renews the
    # lease while it runs -- only the heartbeat can.
    slow = lambda n: time.sleep(2.0 if n == 2 else 0.2)       # noqa: E731
    app.state.runtime.registry.backend = lambda spec: Plan(calls, before=slow)
    with client(app) as c:
        h = tenant(c, "acme")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U001"}).json()[
            "conversation_id"]
        hk = {**h, "Idempotency-Key": "long-1"}
        got = {}

        def first():
            got["first"] = c.post(f"/v1/conversations/{cid}/messages", headers=hk,
                                  json={"content": "go"})

        t = threading.Thread(target=first)
        t.start()
        time.sleep(1.6)                     # inside the long call, past the lease
        retry = c.post(f"/v1/conversations/{cid}/messages", headers=hk, json={"content": "go"})
        other = c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": "hi"})
        t.join()
        again = c.post(f"/v1/conversations/{cid}/messages", headers=hk, json={"content": "go"})
        with engine.connect() as conn:
            esc = conn.execute(select(S.escalations)).all()
            said = conn.execute(select(S.messages).where(S.messages.c.role == "user")).all()
    check("a turn longer than its lease keeps it: a retry mid-turn is told the key is in "
          "progress, another message that a turn is running, and neither runs",
          retry.status_code == 409 and retry.json()["type"] == "/problems/idempotency-key-in-"
          "progress" and other.status_code == 409 and got["first"].status_code == 200,
          (retry.status_code, other.status_code, got["first"].status_code))
    check("…and after it, the retry gets its answer: one escalation, her message stored once",
          again.status_code == 200 and again.content == got["first"].content
          and len(esc) == 1 and len(said) == 1, (again.status_code, len(esc), len(said)))
    engine.dispose()

    engine, url = database("sqlite", "nolifespan")
    from fastapi.testclient import TestClient
    bare = TestClient(make_app(engine, url))                  # no `with`: no startup ran
    h = tenant(bare, "acme")
    cid = bare.post("/v1/conversations", headers=h, json={"user_id": "U001"}).json()[
        "conversation_id"]
    r = bare.post(f"/v1/conversations/{cid}/messages?stream=true", headers=h,
                  json={"content": "where is O1001?"})
    rt = bare.app.state.runtime
    done = until(lambda: rt.in_flight == 0)
    check("a stream on a server that ran no startup still streams, and gives its slot back",
          r.status_code == 200 and "turn.completed" in r.text and done, (r.status_code,
                                                                          r.text[:200]))
    engine.dispose()


def test_zero_usage_is_estimated():
    print("usage")
    from sqlalchemy import select
    from pasarbench.service import schema as S
    engine, url = database("sqlite", "usage0")
    app = make_app(engine, url)
    app.state.runtime.registry.backend = lambda spec: Plan([], usage=(0, 0))
    with client(app) as c:
        h = tenant(c, "acme")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U001"}).json()[
            "conversation_id"]
        c.post(f"/v1/conversations/{cid}/messages", headers=h, json={"content": "hi"})
        with engine.connect() as conn:
            row = conn.execute(select(S.usage_ledger)).first()
    check("a provider that reports no usage is estimated, marked so, and charged -- never "
          "free", row.estimated and row.prompt_tokens > 0 and row.cost_picousd > 0, row)
    engine.dispose()


def test_logs_hold_no_message():
    print("logs")
    import sys
    from pasarbench.service.observability import JsonFormatter
    said = "".join(("42 Ze", "bra Lane"))          # not in this file's source, which a traceback shows
    try:
        raise ValueError(f"Failing row contains (acme, {said}, ...)")
    except ValueError:
        rec = logging.LogRecord("pasar.runtime", logging.ERROR, __file__, 1, "tool call failed",
                                None, sys.exc_info())
    line = JsonFormatter().format(rec)
    check("an exception is logged as its type and frames, never its message (a database's "
          "message can quote the row it refused)", "Zebra" not in line
          and '"exc_type": "ValueError"' in line and "traceback" in line, line[:200])


def test_postgres_api():
    """What only PostgreSQL could get wrong: answers replayed byte for byte
    (JSONB reorders keys), and days counted in UTC whatever the server's zone."""
    print("API [postgres]")
    from datetime import datetime, timezone
    from sqlalchemy import create_engine, insert, text
    from pasarbench.service import schema as S
    from pasarbench.service.store import transaction
    # The test database's own zone made Singapore's, where 20:00 UTC is already
    # tomorrow: a session left in it would put the call on the wrong day.
    raw = create_engine(PG, isolation_level="AUTOCOMMIT")
    with raw.connect() as conn:
        name = conn.execute(text("SELECT current_database()")).scalar()
        conn.execute(text(f"ALTER DATABASE \"{name}\" SET timezone TO 'Asia/Singapore'"))
    raw.dispose()
    raw = create_engine(PG)
    with raw.connect() as conn:
        server_zone = conn.execute(text("SHOW TIME ZONE")).scalar()
    raw.dispose()
    engine, url = database("postgres", "")
    app = make_app(engine, url)
    with client(app) as c:
        h = tenant(c, "acme")
        hk = {**h, "Idempotency-Key": "pg-1"}
        cid = c.post("/v1/conversations", headers=hk, json={"user_id": "U002"}).json()[
            "conversation_id"]
        again = c.post("/v1/conversations", headers=hk, json={"user_id": "U002"})
        first = c.post(f"/v1/conversations/{cid}/messages", headers={**h, "Idempotency-Key":
                                                                     "pg-2"},
                       json={"content": "Please cancel O1002, last digits 4567"})
        dup = c.post(f"/v1/conversations/{cid}/messages", headers={**h, "Idempotency-Key":
                                                                   "pg-2"},
                     json={"content": "Please cancel O1002, last digits 4567"})
        check("[postgres] a retried request gets the first one's bytes, not just its JSON",
              dup.content == first.content and again.headers.get("idempotent-replayed")
              == "true", (first.content[:80], dup.content[:80]))
        with engine.connect() as conn:
            zone = conn.execute(text("SHOW TIME ZONE")).scalar()
        with transaction(engine, write=True) as conn:
            conn.execute(insert(S.usage_ledger).values(
                tenant_id="acme", conversation_id=cid, turn=9, step=1, provider="local",
                model="mock", price_id=None, prompt_tokens=1, cached_tokens=0,
                completion_tokens=1, estimated=True, cost_picousd=1, latency_ms=1,
                created_at=datetime(2020, 2, 29, 20, 0, tzinfo=timezone.utc)))
        days = [r["key"] for r in c.get("/v1/usage?group_by=day&since=2020-02-01"
                                        "&until=2020-04-01", headers=h).json()]
        check(f"[postgres] usage by day counts UTC days: the service's sessions are in UTC, "
              f"and the database's own zone ({server_zone}) decides nothing",
              days == ["2020-02-29"] and zone == "UTC" and server_zone == "Asia/Singapore",
              (days, zone, server_zone))
    engine.dispose()


# ---------------------------------------------------------------------------

def main() -> int:
    if MISSING:
        skip("the agent service", f"{MISSING} is not installed: pip install -r "
             f"requirements-service.txt")
        print(f"\n0 passed, 0 failed, {len(SKIP)} skipped")
        return 0
    logging.getLogger("pasar").setLevel(logging.WARNING)
    try:
        test_costs()
        test_resilience()
        test_migrations("sqlite")
        test_isolation()
        test_concurrency("sqlite")
        test_api()
        test_turn_control()
        test_limits_and_budgets()
        test_market()
        test_provider_failure()
        test_bad_arguments("sqlite")
        test_nul("sqlite")
        test_order("sqlite")
        test_failures_keep_their_key()
        test_long_turns()
        test_zero_usage_is_estimated()
        test_logs_hold_no_message()
        test_parity("sqlite")
        test_episodes("sqlite")
        if PG:
            test_migrations("postgres")
            test_concurrency("postgres")
            test_bad_arguments("postgres")
            test_nul("postgres")
            test_order("postgres")
            test_postgres_api()
            test_parity("postgres")
            test_episodes("postgres")
        else:
            skip("PostgreSQL: migrations, concurrency, parity",
                 "set PASAR_TEST_POSTGRES_URL to a database these tests may wipe")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    tail = f", {len(SKIP)} skipped" if SKIP else ""
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed{tail}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
