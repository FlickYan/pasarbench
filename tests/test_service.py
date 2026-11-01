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

    # The gate from a coroutine (v81): one limit, threads and coroutines alike.
    import asyncio
    gate = ProviderGate("p", concurrency=2, acquire_timeout=0.2, sleep=slept.append)
    held, release = threading.Event(), threading.Event()
    holders = [threading.Thread(target=lambda: gate.call(lambda: (held.set(), release.wait(5))))
               for _ in range(2)]
    for t in holders:
        t.start()
    until(lambda: gate.in_flight == 2)

    async def turned_away() -> bool:
        try:
            await gate.acall(lambda: asyncio.sleep(0))
            return False
        except Overloaded:
            return True

    async def let_in() -> str:
        async def answer() -> str:
            return "in"
        waiting = asyncio.ensure_future(gate.acall(answer))
        await asyncio.sleep(0.05)
        await asyncio.get_running_loop().run_in_executor(None, release.set)
        return await waiting

    away = asyncio.run(turned_away())
    gate.acquire_timeout = 5.0
    got = asyncio.run(let_in())
    for t in holders:
        t.join()
    check("from a coroutine, the gate's slots are the threads' slots: two held by threads "
          "turn a third caller away, and a slot freed by a thread lets a waiting coroutine in",
          away and got == "in" and gate.in_flight == 0, (away, got, gate.in_flight))

    tries: list = []

    async def flaky_async() -> str:
        tries.append(1)
        if len(tries) < 3:
            raise RuntimeError("HTTP 503 from https://provider: overloaded")
        return "answer"

    before = gate.retries
    out = asyncio.run(gate.acall(flaky_async))
    check("…its retries are the same: a transient failure tried again, counted, and the "
          "waits the gate was given", out == "answer" and len(tries) == 3
          and gate.retries - before == 2, (out, len(tries), gate.retries - before))

    from pasarbench.service.resilience import Slots
    slots = Slots(1)
    slots.acquire(1)

    async def cancelled_then_freed() -> tuple:
        waiter = asyncio.ensure_future(slots.aacquire(5))
        await asyncio.sleep(0.02)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        slots.release()
        second = await slots.aacquire(0.5)
        return waiter.cancelled(), second, slots.free

    gone, second, free = asyncio.run(cancelled_then_freed())
    check("a coroutine cancelled while it waits takes no slot and owes none: the slot freed "
          "after it goes to the next", gone and second and free == 0, (gone, second, free))


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
    if kind == "postgres":
        # 0002 on a database that already has tenants (v79): each gets its
        # sequence, starting after its counter; down again, the counter takes
        # the sequence's last value back.
        from sqlalchemy import text
        from pasarbench.service.store import id_sequence
        seq = {"s": id_sequence("old-tenant")}
        command.downgrade(cfg, "0001")
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO tenants (tenant_id, name, world, world_digest) "
                              "VALUES ('old-tenant', 'Old', 'demo', 'x')"))
            conn.execute(text("INSERT INTO id_counters (tenant_id, value) "
                              "VALUES ('old-tenant', 7)"))
        command.upgrade(cfg, "head")
        with engine.begin() as conn:
            first = conn.execute(text("SELECT nextval(CAST(:s AS regclass))"), seq).scalar()
        command.downgrade(cfg, "0001")
        with engine.begin() as conn:
            kept = conn.execute(text("SELECT value FROM id_counters "
                                     "WHERE tenant_id = 'old-tenant'")).scalar()
            gone = conn.execute(text("SELECT to_regclass(:s)"), seq).scalar()
            conn.execute(text("DELETE FROM tenants WHERE tenant_id = 'old-tenant'"))
        command.upgrade(cfg, "head")
        check("[postgres] migration 0002: a tenant there already gets its sequence after its "
              "counter (7, then 8), and down again its counter takes the last value back",
              (first, kept, gone) == (8, 8, None), (first, kept, gone))
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


def test_kit_messages():
    """scripts/service_sweep.py: what it says before it starts anything, and a
    rate as its report prints it."""
    print("the real-model kit's messages")
    import contextlib
    import io
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "service_sweep", Path(__file__).resolve().parent.parent / "scripts" / "service_sweep.py")
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    real = importlib.util.find_spec
    importlib.util.find_spec = (lambda name, *a, **kw:
                                None if name == "sqlalchemy" else real(name, *a, **kw))
    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            code = kit.main(["--self-test"])
    finally:
        importlib.util.find_spec = real
    check("the real-model kit without the service's packages says which are missing and how "
          "to install them, before it starts anything (exit 2)",
          code == 2 and "sqlalchemy" in said.getvalue()
          and "pip install -r requirements-service.txt" in said.getvalue(), said.getvalue()[:300])
    usd = "$0.001525"
    r = {"model": "deepseek-flash", "backend": "openai-compat:deepseek-flash", "simulator": "c",
         "policy_mode": "preload", "episodes": 117, "passed": 116, "tasks_all_passed": 38,
         "n_tasks": 39, "k": 3, "pass^1": 0.9915, "pass^k": 0.9744, "unfinished": 0,
         "stops": {"done": 117}, "per_trap": {},
         "cost": {"total": usd, "per_conversation_mean": usd, "per_conversation_p50": usd,
                  "per_conversation_p95": usd, "per_passed_conversation": usd},
         "tokens_per_conversation": {"prompt": 33786.0, "cached": 31624.0, "completion": 571.0,
                                     "cache_share": 0.936},
         "per_conversation": {"model_calls": 7.1, "tool_calls": 4.8, "customer_turns": 3.3},
         "turn_ms": {"p50": 2251.0, "p95": 5507.0},
         "time_share": {"model": 0.992, "tools": 0.001, "service": 0.007},
         "service_ms_per_turn": {"p50": 18.0, "p95": 36.0},
         "customer_tokens_per_conversation": {"prompt": 2608.9, "completion": 288.5},
         "checks": {"prompt_and_tools_as_the_benchmark": 117,
                    "rescored_to_the_same_verdict": 117, "not_rescorable": {}, "replay": None},
         "served_models": ["deepseek-flash"]}
    md = kit.markdown(r)
    check("…and its report prints a rate from the counts: 116 of 117 is 0.991, not the 0.992 "
          "that rounding pass^1 twice gave", "| 0.991 (116/117) | 0.974 (38/39) |" in md, md[:400])


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


def test_ids(kind: str):
    """A new row's id, from the tenant's sequence (PostgreSQL, v79): taken with
    no lock held to commit, so no write waits for another customer's."""
    print(f"ids [{kind}]")
    from pasarbench.db import Database
    from pasarbench.service.store import run_tool, seed_tenant

    engine, _ = database(kind, "ids")
    seed_tenant(engine, "ids", "ids", "synthetic:3")
    t = Database.fresh().tables
    owners = [(u, next(o for o, x in t["orders"].items() if x["user_id"] == u))
              for u in t["users"] if any(x["user_id"] == u for x in t["orders"].values())]
    people = [(f"{u}-C{i:04d}", f"{o}-C{i:04d}") for i in (1, 2, 3) for u, o in owners]

    def escalate(user, order, hooks=None):
        return run_tool(engine, tenant_id="ids", user_id=user, name="escalate_to_human",
                        args={"order_id": order, "category": "customer_request",
                              "reason": "test"},
                        verified=[user], prior_actions=[], hooks=hooks)

    took, go, box = threading.Event(), threading.Event(), {}
    held = threading.Thread(target=lambda: box.update(a=escalate(
        *people[0], hooks={"before_commit": lambda: (took.set(), go.wait(10))})))
    held.start()
    took.wait(5)
    t0 = time.monotonic()
    other = escalate(*people[1])
    waited = time.monotonic() - t0
    go.set()
    held.join()
    a = box["a"]
    check(f"[{kind}] a write that makes a row does not wait for another customer's write "
          f"still open in the tenant ({waited:.2f}s; the counter row held it to the other's "
          f"commit, 10s here)", waited < 2.0 and a.result["ok"] and other.result["ok"]
          and a.allocated != other.allocated, (waited, a.allocated, other.allocated))

    n = len(people) - 2
    gate = threading.Barrier(n)
    outs: list = [None] * n

    def at_once(i):
        gate.wait()
        outs[i] = escalate(*people[2 + i])
    threads = [threading.Thread(target=at_once, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    ids = sorted(x for o in outs for x in o.allocated)
    check(f"[{kind}] {n} customers' writes at once: each its own id, none lost or "
          f"repeated -- ESC0003 to ESC{n + 2:04d}, after the two above",
          all(o.result["ok"] for o in outs)
          and ids == [f"ESC{i:04d}" for i in range(3, n + 3)], ids)
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


def test_slices(kind: str):
    """A customer's slice, read once per write to her world (store.Slices, v80):
    a call that finds her write_seq where a slice was read uses it -- the
    same answer, and never one the world has moved past."""
    print(f"slices [{kind}]")
    import copy as _copy
    from pasarbench.service.store import (copy_world, load_slice, run_tool, seed_tenant, slices,
                                          transaction)

    engine, _ = database(kind, "slices")
    seed_tenant(engine, "sl", "sl", "demo")
    kept = slices(engine)
    _paid, _pay, proc, _users = demo_orders()
    user, order = proc["user_id"], proc["order_id"]

    def call(name, args):
        return run_tool(engine, tenant_id="sl", user_id=user, name=name, args=args,
                        verified=[user], prior_actions=[]).result

    h0, m0 = kept.hits, kept.misses
    first = call("get_order", {"order_id": order})
    again = call("get_order", {"order_id": order})
    check(f"[{kind}] the second read of a customer whose world has not changed uses the "
          f"slice the first read: the same answer, one read of her tables",
          again == first and (kept.hits - h0, kept.misses - m0) == (1, 1),
          (kept.hits - h0, kept.misses - m0))
    done = call("cancel_order", {"order_id": order, "reason": "changed my mind"})
    after = call("get_order", {"order_id": order})
    check(f"[{kind}] a write moves her write_seq, and the next read reads her world again: "
          f"the order cancelled, not the slice from before",
          done.get("ok") and after["order"]["status"] == "cancelled"
          and first["order"]["status"] == "processing" and kept.misses - m0 == 2,
          (done, after.get("order", {}).get("status")))

    with transaction(engine, write=False) as conn:
        tables, _versions = load_slice(conn, "sl", user)
    keys = {type(k) for rows in tables.values() for r in rows.values() for k in r}
    check(f"[{kind}] a slice's rows are keyed by plain str (SQLAlchemy's own names are a "
          f"subclass, which deepcopy rebuilds one by one)", keys == {str}, keys)
    nested = _copy.deepcopy(tables)
    nested["orders"][order]["tags"] = ["a", {"b": [1]}]
    copied = copy_world(nested)
    copied["orders"][order]["tags"][1]["b"].append(2)
    copied["orders"][order]["status"] = "x"
    check(f"[{kind}] copy_world is a deep copy: equal to deepcopy's, and nothing the copy "
          f"changes reaches the original (a nested value included)",
          copy_world(tables) == _copy.deepcopy(tables)
          and nested["orders"][order]["tags"] == ["a", {"b": [1]}]
          and nested["orders"][order]["status"] == "cancelled")

    kept.size, h0 = 0, kept.hits
    kept.forget("sl")
    call("get_order", {"order_id": order})
    call("get_order", {"order_id": order})
    check(f"[{kind}] PASAR_SLICE_CACHE=0 keeps none: every call reads the database",
          kept.hits == h0)
    kept.size = 2000
    engine.dispose()


def test_liveness():
    """A pooled PostgreSQL connection is checked before it is lent: one whose
    session the server ended is replaced before anyone sees it -- by its
    socket, with no round trip (store._check_on_checkout, v80)."""
    print("liveness [postgres]")
    from sqlalchemy import text
    from pasarbench.service.store import make_engine, transaction
    engine = make_engine(PG, pool_size=1, max_overflow=0)
    with transaction(engine, write=False) as conn:
        pid = conn.execute(text("SELECT pg_backend_pid()")).scalar()
    other = make_engine(PG, pool_size=1, max_overflow=0)
    with transaction(other, write=False) as conn:
        ended = conn.execute(text("SELECT pg_terminate_backend(:p, 5000)"), {"p": pid}).scalar()
    try:
        with transaction(engine, write=False) as conn:
            now = conn.execute(text("SELECT pg_backend_pid()")).scalar()
        ok, detail = ended and now != pid, (ended, pid, now)
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    check("a pooled connection whose session the server ended is replaced before it is "
          "lent: the next transaction runs, on a new session", ok, detail)

    import pasarbench.service.store as store
    pings: list = []
    real = engine.dialect.do_ping
    engine.dialect.do_ping = lambda c: (pings.append(1), real(c))[1]
    with transaction(engine, write=False) as conn:
        conn.execute(text("SELECT 1"))
    lent_quietly = len(pings)
    store.PING_AFTER, kept = 0.0, store.PING_AFTER
    try:
        with transaction(engine, write=False) as conn:
            pid = conn.execute(text("SELECT pg_backend_pid()")).scalar()
        with transaction(other, write=False) as conn:
            conn.execute(text("SELECT pg_terminate_backend(:p, 5000)"), {"p": pid})
        with transaction(engine, write=False) as conn:
            now = conn.execute(text("SELECT pg_backend_pid()")).scalar()
        ok, detail = now != pid and len(pings) >= 2, (pid, now, len(pings))
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    finally:
        store.PING_AFTER = kept
        engine.dialect.do_ping = real
    check("a connection back within seconds is lent with no round trip; one idle longer is "
          "pinged, and a session the server ended is replaced that way too",
          lent_quietly == 0 and ok, (lent_quietly, detail))
    engine.dispose()
    other.dispose()


def test_database_errors():
    """What counts as the database not answering, which the service answers
    503 for (v83) -- and what does not, on SQLite as well: a lock waited out
    is an outage to wait out, a missing table a defect to fix."""
    print("a database that does not answer [sqlite]")
    import sqlite3
    from sqlalchemy import event, text
    from sqlalchemy.exc import IntegrityError, OperationalError
    from sqlalchemy.exc import TimeoutError as PoolTimeout
    from pasarbench.service.store import (DatabaseSilent, Silence, SilencePool, make_engine,
                                          no_answer, unavailable)

    class Said(Exception):
        def __init__(self, sqlstate):
            super().__init__(sqlstate)
            self.sqlstate = sqlstate

    path = f"{TMP}/locked.db"
    first, second = sqlite3.connect(path), sqlite3.connect(path, timeout=0)
    first.execute("BEGIN IMMEDIATE")
    try:
        second.execute("BEGIN IMMEDIATE")
        locked = None
    except sqlite3.OperationalError as e:
        locked = e
    first.rollback()
    first.close()
    try:
        second.execute("SELECT * FROM no_such_table")
        missing = None
    except sqlite3.OperationalError as e:
        missing = e
    second.close()
    def op(code):
        return unavailable(OperationalError("x", {}, Said(code)))

    said = {
        "a connection lost": unavailable(OperationalError("x", {}, Exception("closed"))),
        "a statement the server ended (57014)": op("57014"),
        "a server shutting down (57P01)": op("57P01"),
        "a connection failure (08006)": op("08006"),
        "too many connections (53300)": op("53300"),
        "an I/O error (58030)": op("58030"),
        "no connection free in the pool": unavailable(PoolTimeout("x")),
        "a checkout refused while it is silent": unavailable(DatabaseSilent("x")),
        "SQLite's lock waited out": locked is not None and unavailable(
            OperationalError("x", {}, locked)),
        "a serialization failure (40001)": op("40001"),
        "a deadlock (40P01)": op("40P01"),
        "a statement too complex (54001)": op("54001"),
        "a lock not available (55P03)": op("55P03"),
        "a constraint broken": unavailable(IntegrityError("x", {}, Exception())),
        "SQLite's missing table": missing is not None and unavailable(
            OperationalError("x", {}, missing)),
    }
    want = dict.fromkeys(list(said)[:9], True) | dict.fromkeys(list(said)[9:], False)
    check("the database not answering (503) is a connection lost or refused, a statement it "
          "ended, a server going down, resources it ran out of, no connection free in time, "
          "SQLite's lock waited out; not a serialization failure or a deadlock, which are "
          "tried again, nor a limit a statement ran into, nor a constraint broken, nor a "
          "table SQLite does not have -- which SQLite calls operational too",
          said == want, {k: v for k, v in said.items() if v != want[k]})

    import psycopg
    told = {
        "a FATAL it sent (too many connections)": no_answer(psycopg.OperationalError(
            'connection to server at "127.0.0.1", port 5432 failed: FATAL:  too many '
            'connections for role "app"')),
        "timed out": no_answer(psycopg.errors.ConnectionTimeout("connection timeout expired")),
        "refused": no_answer(psycopg.OperationalError(
            'connection to server at "127.0.0.1", port 5432 failed: Connection refused')),
    }
    far = make_engine("postgresql+psycopg://x@127.0.0.1:9/x", timeout=60)
    check("a connection the server refused with a FATAL is the server answering, and does "
          "not make it silent; one that timed out or was refused does; and a trial lasts "
          "three timeouts at most before another may try",
          told == {"a FATAL it sent (too many connections)": False, "timed out": True,
                   "refused": True} and far.pasar_silence.trial_seconds == 180,
          (told, far.pasar_silence.trial_seconds))
    far.dispose()

    # Silence: refused at once while the database is silent, but for one
    # checkout at a time, every `retry` seconds; the first answer ends it.
    g = Silence(retry=0.2)

    def refused(into=None, key=None):
        try:
            g.admit()
            out = False
        except DatabaseSilent:
            out = True
        if into is not None:
            into[key] = out
        return out

    def elsewhere(key):
        t = threading.Thread(target=refused, args=(seen, key))
        t.start()
        t.join()

    seen: dict = {}
    seen["answering"] = refused()
    g.silent("a test")
    seen["at once"] = refused()
    time.sleep(0.25)
    seen["the trial"] = refused()
    elsewhere("another, meanwhile")
    seen["the trial again"] = refused()
    g.trial_over()
    elsewhere("another, at once")
    time.sleep(0.25)
    elsewhere("another, later")
    seen["the first, now"] = refused()
    g.answered()
    seen["answered"] = refused()
    spells = g.spells
    stale = Silence(retry=0.05, trial_seconds=0.2)
    stale.silent("a test")
    time.sleep(0.06)
    g = stale
    seen["a trial"] = refused()
    elsewhere("another, during it")
    time.sleep(0.25)
    elsewhere("another, the trial lost")
    want = {"answering": False, "at once": True, "the trial": False,
            "another, meanwhile": True, "the trial again": False, "another, at once": True,
            "another, later": False, "the first, now": True, "answered": False,
            "a trial": False, "another, during it": True, "another, the trial lost": False}
    check("a database seen silent is not waited on: a checkout is refused at once, but for "
          "one at a time every SILENT_RETRY seconds -- the next after a trial fails, or after "
          "one that is never heard from -- and the first answer ends it",
          seen == want and spells == 1 and unavailable(DatabaseSilent("x")),
          {k: v for k, v in seen.items() if v != want.get(k)})

    # Said once a minute at most: a database that comes and goes logs one
    # line as it stops answering and one as it answers, not one each time.
    lines = []

    class Keep(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    keep = Keep()
    logging.getLogger("pasar.db").addHandler(keep)
    try:
        g = Silence(retry=0.01)
        for _ in range(4):
            g.silent("a test")
            g.answered()
    finally:
        logging.getLogger("pasar.db").removeHandler(keep)
    check("a database that stops and starts answering four times in a minute is said to, "
          "once each way",
          len(lines) == 2 and "stopped answering" in lines[0] and "answers again" in lines[1]
          and g.spells == 4, lines)

    # A trial is a whole checkout: one whose connection has to be made again
    # and cannot be -- a dead one found in the pool -- ends the trial too,
    # and another tries a retry later, not when the trial would have expired.
    import sqlite3 as _sq
    down = {"now": False}

    def dial():
        if down["now"]:
            raise _sq.OperationalError("unable to open database file")
        return _sq.connect(":memory:", check_same_thread=False)

    pool = SilencePool(dial, pool_size=1, max_overflow=0, timeout=1)
    pool.pasar_silence = g = Silence(retry=0.1)
    dead = pool.connect()
    dead.invalidate()                           # back in the pool with no connection
    dead.close()
    g.silent("a test")
    time.sleep(0.12)
    down["now"] = True
    try:
        pool.connect()
        failed_trial = None
    except _sq.OperationalError as e:
        failed_trial = e
    ended = g._trial is None
    down["now"] = False
    time.sleep(0.12)
    seen = {}
    elsewhere("another, a retry later")
    pool.dispose()
    check("a trial whose checkout fails as it makes its connection again ends there: another "
          "tries SILENT_RETRY later",
          failed_trial is not None and ended and seen == {"another, a retry later": False},
          (failed_trial, ended, seen))

    # A write behind another process's write lock, past SQLite's wait for
    # it (30 s; 0.3 here): 503, and nothing left held.
    db, url = database("sqlite", "busy")
    db.dispose()

    @event.listens_for(db, "connect")
    def _short(dbapi_conn, _record):            # noqa: ANN001
        dbapi_conn.execute("PRAGMA busy_timeout=300")

    app = make_app(db, url)
    with client(app) as c:
        h = tenant(c, "acme")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U002"}).json()[
            "conversation_id"]
        blocker = sqlite3.connect(url.split("sqlite:///", 1)[1])
        blocker.execute("BEGIN IMMEDIATE")
        t0 = time.monotonic()
        r = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                   json={"content": "where is my order O1002?"})
        took = time.monotonic() - t0
        keyed = {**h, "Idempotency-Key": "busy-1"}
        rk = c.post(f"/v1/conversations/{cid}/messages", headers=keyed,
                    json={"content": "and O1003?"})
        blocker.rollback()
        blocker.close()
        nxt = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                     json={"content": "where is my order O1002?"})
        t0 = time.monotonic()
        again = until(lambda: c.post(f"/v1/conversations/{cid}/messages", headers=keyed,
                                     json={"content": "and O1003?"}).status_code == 200, 6)
        again_took = time.monotonic() - t0
    check("[sqlite] a write that waits out SQLite's lock is answered 503 database-unavailable, "
          "Retry-After, saying which wait it was; the conversation's next turn runs",
          r.status_code == 503 and r.json().get("type") == "/problems/database-unavailable"
          and r.headers.get("retry-after") == "5" and "SQLite" in r.json().get("detail", "")
          and took < 2 and nxt.status_code == 200,
          (r.status_code, r.json().get("type"), r.json().get("detail"), round(took, 2),
           nxt.status_code))
    check("[sqlite] …and under an Idempotency-Key, the key is forgotten once the database "
          "answers: the request sent again under it runs, within RELEASE_EVERY, not 409 "
          "until the key would expire",
          rk.status_code == 503 and again and again_took < 4,
          (rk.status_code, again, round(again_took, 2)))
    db.dispose()

    # A lease the database may have taken -- the lease's transaction failed
    # as the database stopped answering, its COMMIT perhaps heard and its
    # answer lost -- is given back once it answers, in case.
    import pasarbench.service.runtime as R
    db, url = database("sqlite", "lost")
    app = make_app(db, url)
    rt = app.state.runtime
    real = R.costs.spent_today
    calls = {"n": 0}

    def flaky(conn, tenant_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OperationalError("SELECT", {}, Exception("server closed the connection"))
        return real(conn, tenant_id)

    with client(app) as c:
        h = tenant(c, "acme", budget="5.00")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U002"}).json()[
            "conversation_id"]
        R.costs.spent_today = flaky
        try:
            lost = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                          json={"content": "where is my order O1002?"})
            owed = rt.unreleased()
        finally:
            R.costs.spent_today = real
        after = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                       json={"content": "where is my order O1002?"})
        left = rt.unreleased()
    check("[sqlite] a lease whose transaction the database did not answer is owed back all "
          "the same -- it may have been taken -- and the conversation's next turn takes it "
          "over and runs",
          lost.status_code == 503 and owed == 1 and after.status_code == 200 and left == 0,
          (lost.status_code, owed, after.status_code, left))
    db.dispose()

    bare = make_engine(sqlite_url("bare"))
    with bare.connect() as conn:
        conn.execute(text("SELECT 1"))
    app = make_app(bare, sqlite_url("bare"))
    with client(app) as c:
        asked = c.get("/v1/conversations", headers={"Authorization": "Bearer psk_x"})
        ready = c.get("/readyz")
    check("[sqlite] a database without the schema is not an outage: a request is answered "
          "500, not 503 to come back to; /readyz says what is missing",
          asked.status_code == 500 and ready.status_code == 503
          and "schema" in ready.json().get("detail", ""),
          (asked.status_code, ready.status_code, ready.json().get("detail")))
    bare.dispose()


def test_silent_database():
    """A database that does not answer (v83): every wait on it is bounded --
    a statement by the server, a lent connection by the watchdog, a new one
    by connect_timeout -- and a request is answered 503 within
    PASAR_DB_TIMEOUT; an operator's long work is not bounded; a lease the
    database did not hear given back is given back once it answers."""
    print("a silent database [postgres]")
    import socket as _socket
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError
    from pasarbench.service.store import (SILENT_RETRY, DatabaseSilent, deadline, make_engine,
                                          transaction, unavailable)

    engine = make_engine(PG, timeout=1.0)
    with transaction(engine, write=False) as conn:
        shown = [conn.execute(text(f"SHOW {n}")).scalar() for n in
                 ("statement_timeout", "idle_in_transaction_session_timeout")]
    tcp = PG.replace("@/pasar_test?host=/var/run/postgresql&port=5433",
                     "@127.0.0.1:5433/pasar_test")
    over_tcp = make_engine(tcp, timeout=1.0)
    fairy = over_tcp.raw_connection()
    try:
        sock = _socket.socket(fileno=__import__("os").dup(fairy.driver_connection.pgconn.socket))
        try:
            user_timeout = sock.getsockopt(_socket.IPPROTO_TCP, 18)     # TCP_USER_TIMEOUT
            keepalive = sock.getsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE)
        finally:
            sock.close()
    finally:
        fairy.close()
    check("[postgres] a connection says how long it may be waited on: statements ended by "
          "the server at 80% of PASAR_DB_TIMEOUT, a session idle in a transaction at three "
          "times it; over TCP, keepalives on and a user timeout of PASAR_DB_TIMEOUT",
          shown == ["800ms", "3s"] and user_timeout == 1000 and keepalive == 1,
          (shown, user_timeout, keepalive))

    t0 = time.monotonic()
    try:
        with transaction(engine, write=False) as conn:
            conn.execute(text("SELECT pg_sleep(5)"))
        ended = None
    except OperationalError as e:
        ended = e
    took = time.monotonic() - t0
    with transaction(engine, write=False) as conn:
        after = conn.execute(text("SELECT 1")).scalar()
    check("[postgres] a statement past it is ended by the server (0.8 s here) -- a 503, "
          "not a defect -- and the connection serves the next transaction",
          ended is not None and unavailable(ended) and 0.7 < took < 1.5 and after == 1,
          (type(ended).__name__, round(took, 2)))

    # A server that does not answer at all: a session stopped mid-statement
    # (pg_sleep under SIGSTOP would stop the whole test database; a lock
    # held by another session is a statement the server cannot end sooner
    # than its timeout, so the timeout is lifted for it and the watchdog
    # has to act).
    holder = make_engine(PG)
    hc = holder.connect()
    hc.execute(text("CREATE TABLE IF NOT EXISTS silent_probe (id int PRIMARY KEY)"))
    hc.execute(text("INSERT INTO silent_probe VALUES (1) ON CONFLICT DO NOTHING"))
    hc.commit()
    hc.execute(text("SELECT id FROM silent_probe WHERE id = 1 FOR UPDATE"))
    watchdog = engine.pasar_watchdog
    shut_before = watchdog.aborted
    t0 = time.monotonic()
    try:
        with deadline(1.0), engine.connect() as conn:
            conn.execute(text("SET statement_timeout = 0"))
            conn.execute(text("SELECT id FROM silent_probe WHERE id = 1 FOR UPDATE"))
        cut = None
    except OperationalError as e:
        cut = e
    took = time.monotonic() - t0
    hc.rollback()
    silence = engine.pasar_silence
    silent_then = silence.since is not None
    t0 = time.monotonic()
    try:
        with transaction(engine, write=False) as conn:
            conn.execute(text("SELECT 1"))
        turned = None
    except DatabaseSilent as e:
        turned = e
    turned_took = time.monotonic() - t0
    time.sleep(SILENT_RETRY + 0.1)
    with transaction(engine, write=False) as conn:
        again = conn.execute(text("SELECT 1")).scalar()
    ten = make_engine(PG)                       # 10 s, and a connection lent for 1 of them
    hc.execute(text("SELECT id FROM silent_probe WHERE id = 1 FOR UPDATE"))
    try:
        with deadline(1.0), ten.connect() as conn:
            conn.execute(text("SET statement_timeout = 0"))
            conn.execute(text("SELECT id FROM silent_probe WHERE id = 1 FOR UPDATE"))
    except OperationalError:
        pass
    hc.rollback()
    hc.close()
    short_cut, short_silent = ten.pasar_watchdog.aborted, ten.pasar_silence.since is not None
    with transaction(ten, write=False) as conn:
        short_after = conn.execute(text("SELECT 1")).scalar()
    ten.dispose()
    check("[postgres] a lent connection not back within its deadline is shut down by the "
          "watchdog: the statement waiting on it raises at once (1 s here), counted",
          cut is not None and unavailable(cut) and 0.9 < took < 1.6
          and watchdog.aborted == shut_before + 1,
          (type(cut).__name__, round(took, 2), watchdog.aborted - shut_before))
    check("[postgres] …and the database is taken to have stopped answering: the next "
          "checkout is refused at once (503), not waited on; one tries it a second later, "
          "its answer ends the silence, and the pool's next connection serves",
          silent_then and turned is not None and unavailable(turned) and turned_took < 0.1
          and again == 1 and silence.since is None,
          (silent_then, type(turned).__name__, round(turned_took, 3), again, silence.since))
    check("[postgres] …but one lent for less than the timeout -- a readiness probe's 2 s, a "
          "rate limit's -- shut down says only that the database is slower than that: "
          "nothing is refused",
          short_cut == 1 and not short_silent and short_after == 1,
          (short_cut, short_silent, short_after))

    one = make_engine(PG, timeout=1.0)
    hc = holder.connect()
    hc.execute(text("SELECT id FROM silent_probe WHERE id = 1 FOR UPDATE"))
    t0 = time.monotonic()
    try:
        with deadline(2.0), one.connect() as conn:
            conn.execute(text("SET statement_timeout = 0"))
            conn.execute(text("SELECT id FROM silent_probe WHERE id = 1 FOR UPDATE"))
    except OperationalError:
        pass
    held_to = time.monotonic() - t0
    hc.rollback()
    hc.close()
    one.dispose()
    check("[postgres] a deadline longer than PASAR_DB_TIMEOUT is held to it: a connection "
          "lent for 2 s on an engine of 1 is shut down at 1",
          0.9 < held_to < 1.6, round(held_to, 2))

    # A restart: the pool's sessions ended and the database taken to be
    # silent, as a connection refused while it was down leaves it. The first
    # trial after finds a dead connection in the pool, makes it again within
    # its own checkout, and answers: the silence over at once.
    named = PG + ("&" if "?" in PG else "?") + "application_name=pasar_restart"
    restart = make_engine(named, pool_size=3, max_overflow=0)
    warm = [restart.connect() for _ in range(3)]
    for w in warm:
        w.execute(text("SELECT 1"))
    for w in warm:
        w.close()
    with holder.connect() as conn:
        ended = conn.execute(text("SELECT count(pg_terminate_backend(pid)) FROM "
                                  "pg_stat_activity WHERE application_name = "
                                  "'pasar_restart'")).scalar()
    time.sleep(0.2)
    restart.pasar_silence.silent("a test")
    time.sleep(SILENT_RETRY + 0.1)
    t0 = time.monotonic()
    try:
        with transaction(restart, write=False) as conn:
            back_answer = conn.execute(text("SELECT 1")).scalar()
    except Exception as e:  # noqa: BLE001
        back_answer = type(e).__name__
    back_took = time.monotonic() - t0
    over = restart.pasar_silence.since is None
    restart.dispose()
    check("[postgres] after a restart -- the pool's sessions ended, the database taken to be "
          "silent -- the first trial finds a dead connection, makes it again within its own "
          "checkout and answers: the silence is over at once, not a dead connection a second",
          ended == 3 and back_answer == 1 and over and back_took < 1,
          (ended, back_answer, over, round(back_took, 2)))

    t0 = time.monotonic()
    with deadline(None), transaction(engine, write=True) as conn:
        conn.execute(text("SELECT pg_sleep(1.5)"))
        lifted = conn.execute(text("SHOW statement_timeout")).scalar()
    took = time.monotonic() - t0
    check("[postgres] an operator's long work (deadline(None): seeding, replay) is not "
          "bounded: 1.5 s of it on an engine of 1, its transaction's statement timeout lifted",
          took >= 1.5 and lifted == "0", (round(took, 2), lifted))

    # A database that answers nothing: a listener that never speaks.
    hole = _socket.socket()
    hole.bind(("127.0.0.1", 0))
    hole.listen(64)
    port = hole.getsockname()[1]
    silent = make_engine(f"postgresql+psycopg://postgres@127.0.0.1:{port}/x", timeout=1.0)
    t0 = time.monotonic()
    try:
        with silent.connect() as conn:
            conn.execute(text("SELECT 1"))
        refused = None
    except OperationalError as e:
        refused = e
    took = time.monotonic() - t0
    from pasarbench.service.app import create_app
    from pasarbench.service.config import Settings
    from fastapi.testclient import TestClient
    app = create_app(Settings(database_url=f"postgresql+psycopg://postgres@127.0.0.1:{port}/x",
                              admin_token="op-token", log_level="WARNING", db_timeout=1.0),
                     silent)
    c = TestClient(app)
    t0 = time.monotonic()
    ready = c.get("/readyz")
    ready_took = time.monotonic() - t0
    t0 = time.monotonic()
    asked = c.get("/v1/conversations", headers={"Authorization": "Bearer psk_x"})
    asked_took = time.monotonic() - t0
    check("[postgres] a database that answers nothing: a new connection gives up within "
          "connect_timeout (2 s here); after it, /readyz and a request are answered 503 "
          "database-unavailable at once, Retry-After, not left waiting",
          refused is not None and 1.5 < took < 3 and ready.status_code == 503
          and ready_took < 0.5 and asked.status_code == 503
          and asked.json().get("type") == "/problems/database-unavailable"
          and "has not answered" in asked.json().get("detail", "")
          and asked.headers.get("retry-after") == "5" and asked_took < 0.5,
          (round(took, 2), ready.status_code, round(ready_took, 2), asked.status_code,
           asked.json().get("type"), asked.json().get("detail"), round(asked_took, 2)))
    silent.dispose()
    slow = make_engine(f"postgresql+psycopg://postgres@127.0.0.1:{port}/x")    # 10 s: 5 to connect
    app = create_app(Settings(database_url=f"postgresql+psycopg://postgres@127.0.0.1:{port}/x",
                              admin_token="op-token", log_level="WARNING"), slow)
    c = TestClient(app)
    t0 = time.monotonic()
    ready = c.get("/readyz")
    ready_took = time.monotonic() - t0
    check("[postgres] /readyz answers within 2 s whatever the database does -- here one "
          "that never answers, where the service's own connections are given 5 s: 503 "
          "(its own connection is given 2 s, and a probe that takes longer is answered "
          "without it)",
          ready.status_code == 503 and 1.9 < ready_took < 2.6,
          (ready.status_code, round(ready_took, 2), ready.json().get("detail")))
    time.sleep(3.5)                     # the probe's thread gives up meanwhile
    slow.dispose()
    hole.close()

    # A turn whose write waits past the timeout (on a lock another session
    # holds on the messages a turn appends to, reads still allowed): 503,
    # and its lease -- the turn holds it by then -- given back once the
    # database answers, not held for its 60 s: by the releaser, or by the
    # conversation's next turn here, whichever comes first.
    schema, url = database("postgres", "silent")
    schema.dispose()
    db = make_engine(url, timeout=1.0)
    app = make_app(db, url, db_timeout=1.0)
    rt = app.state.runtime

    def stuck_turn(c, h, cid):
        lock = make_engine(PG).connect()
        lock.execute(text("LOCK TABLE messages IN SHARE MODE"))
        t0 = time.monotonic()
        r = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                   json={"content": "where is my order O1002?"})
        took = time.monotonic() - t0
        owed = rt.unreleased()
        lock.rollback()
        lock.close()
        return r, took, owed

    with client(app) as c:
        h = tenant(c, "acme")
        cids = [c.post("/v1/conversations", headers=h, json={"user_id": "U002"}).json()[
            "conversation_id"] for _ in range(3)]
        r, took, owed = stuck_turn(c, h, cids[0])
        t0 = time.monotonic()
        loop = until(lambda: rt.unreleased() == 0, 6)
        loop_took = time.monotonic() - t0
        with transaction(db, write=False) as conn:
            holder_left = conn.execute(text("SELECT lease_owner FROM conversations WHERE "
                                            "conversation_id = :c"), {"c": cids[0]}).scalar()
        r2, _, owed2 = stuck_turn(c, h, cids[1])
        t0 = time.monotonic()
        nxt = c.post(f"/v1/conversations/{cids[1]}/messages", headers=h,
                     json={"content": "hello?"})
        next_took = time.monotonic() - t0
        left = rt.unreleased()
        keyed = {**h, "Idempotency-Key": "silent-1"}
        rk, _, _ = stuck_turn(c, keyed, cids[2])
        t0 = time.monotonic()
        rerun = until(lambda: c.post(f"/v1/conversations/{cids[2]}/messages", headers=keyed,
                                     json={"content": "where is my order O1002?"}
                                     ).status_code == 200, 6)
        rerun_took = time.monotonic() - t0
    check("[postgres] a turn whose write waits past the timeout is answered 503 within it, and "
          "its lease, which it does not ask the database for again, is given back once the "
          "database answers -- by the releaser within RELEASE_EVERY (2 s), not in its 60 s",
          r.status_code == 503 and r.json().get("type") == "/problems/database-unavailable"
          and took < 3 and owed == 1 and loop and loop_took < 4 and holder_left is None,
          (r.status_code, r.json().get("type"), round(took, 2), owed, loop,
           round(loop_took, 2), holder_left))
    check("[postgres] …or by the conversation's next turn on this process, at once, in the "
          "transaction that takes its own lease: answered 200, not 409",
          r2.status_code == 503 and owed2 == 1 and nxt.status_code == 200 and next_took < 1.5
          and left == 0, (r2.status_code, owed2, nxt.status_code, round(next_took, 2), left))
    check("[postgres] …and its Idempotency-Key is forgotten once the database answers: the "
          "request sent again under it runs within RELEASE_EVERY, not 409 for a lease",
          rk.status_code == 503 and rerun and rerun_took < 4, (rk.status_code, rerun,
                                                              round(rerun_took, 2)))
    db.dispose()

    busy = make_engine(url, pool_size=1, max_overflow=0)
    app = make_app(busy, url)
    with client(app) as c:
        lent = busy.connect()
        lent.execute(text("SELECT 1"))
        t0 = time.monotonic()
        ready = c.get("/readyz")
        ready_took = time.monotonic() - t0
        lent.close()
    busy.dispose()
    check("[postgres] /readyz on a server whose pool is busy -- every connection lent -- asks "
          "the database on a connection of its own: 200 at once, not 503 for the queue",
          ready.status_code == 200 and ready_took < 1, (ready.status_code, round(ready_took, 2)))
    holder.dispose()
    engine.dispose()
    over_tcp.dispose()


def test_profiler():
    """The sampling profiler (profiling.py, v80): every thread's CPU charged to
    calls that ran throughout, waiting charged nothing -- against threads
    whose CPU is known -- and the operator's endpoint."""
    print("profiler")
    from pasarbench.service.__main__ import main as cli
    from pasarbench.service.profiling import Sampler, thread_clocks

    def spin(seconds):
        t0 = time.thread_time()
        while time.thread_time() - t0 < seconds:
            pass

    def long_spin(box):
        t0 = time.thread_time()
        spin(0.4)
        box["long"] = time.thread_time() - t0

    def bursts(box):
        t0 = time.thread_time()
        for _ in range(300):
            spin(0.0005)
            time.sleep(0.0005)
        box["bursts"] = time.thread_time() - t0

    stop = threading.Event()

    def waits(box):
        stop.wait(10)

    box: dict = {}
    s = Sampler(0.005).start()
    ts = [threading.Thread(target=f, args=(box,)) for f in (long_spin, bursts, waits)]
    for t in ts:
        t.start()
    for t in ts[:2]:
        t.join()
    stop.set()
    ts[2].join()
    rep = s.stop()
    inclusive: dict = {}
    for key, us in rep["stacks_us"].items():
        for f in set(key.split(";")):
            inclusive[f] = inclusive.get(f, 0) + us / 1e6

    def got(name):
        return sum(v for k, v in inclusive.items() if k.endswith(".<locals>." + name))

    total = sum(rep["stacks_us"].values()) / 1e6
    if not thread_clocks():
        skip("the profiler's CPU accounting", "no per-thread CPU clocks (Linux has them)")
    else:
        check(f"a thread's CPU in one long call is charged to that call "
              f"({got('long_spin'):.3f} s of {box['long']:.3f})",
              0.85 * box["long"] <= got("long_spin") <= 1.05 * box["long"])
        check(f"a thread's CPU in short bursts between waits is charged to the call that ran "
              f"throughout ({got('bursts'):.3f} s of {box['bursts']:.3f}), and the waits to "
              f"nothing", 0.85 * box["bursts"] <= got("bursts") <= 1.05 * box["bursts"])
        check(f"a thread waiting the whole time is charged nothing "
              f"({got('waits') * 1000:.1f} ms of {total * 1000:.0f})",
              got("waits") < 0.01 * total)
        check(f"what the stacks were charged is the process's CPU, the sampler's own aside "
              f"({rep['sampled_seconds']:.2f} + {rep['sampler_cpu_seconds']:.2f} of "
              f"{rep['process_cpu_seconds']:.2f} s)",
              rep["sampled_seconds"] + rep["sampler_cpu_seconds"]
              >= 0.9 * rep["process_cpu_seconds"])
    check("a profile names code, never values: every frame a module and a function",
          all(":" in f for k in rep["stacks_us"] for f in k.split(";")))

    engine, url = database("sqlite", "profile")
    with client(make_app(engine, url)) as c:
        check("the profile is the operator's", c.get("/v1/admin/profile?seconds=0.1")
              .status_code == 401)
        r = c.get("/v1/admin/profile?seconds=0.3&interval_ms=2", headers=OP)
        body = r.json()
        check("GET /v1/admin/profile samples the process for the seconds asked and answers "
              "the profile", r.status_code == 200 and 0.25 <= body["wall_seconds"] < 2
              and body["mode"] in ("cpu", "wall") and isinstance(body["stacks_us"], dict),
              r.text[:300])
        long: dict = {}
        th = threading.Thread(target=lambda: long.update(
            r=c.get("/v1/admin/profile?seconds=60", headers=OP)))
        t0 = time.monotonic()
        th.start()
        from pasarbench.service.profiling import running
        until(lambda: running() is not None)
        busy = c.get("/v1/admin/profile?seconds=1", headers=OP).status_code
        ended = c.delete("/v1/admin/profile", headers=OP).status_code
        th.join(10)
        check("one profile at a time (409), and DELETE ends it: its GET answers at once",
              busy == 409 and ended == 204 and long["r"].status_code == 200
              and time.monotonic() - t0 < 10, (busy, ended))
        check("…and with none running, DELETE is 404",
              c.delete("/v1/admin/profile", headers=OP).status_code == 404)
    engine.dispose()
    path, folded = os.path.join(TMP, "p.json"), os.path.join(TMP, "p.folded")
    with open(path, "w") as f:
        json.dump(rep, f)
    import contextlib
    import io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli(["profile", path, "--turns", "10", "--folded", folded])
    lines = open(folded).read().splitlines()
    check("`python -m pasarbench.service profile` sums it up by layer and by function, "
          "per turn, and writes the stacks a flame graph reads",
          code == 0 and "by layer" in out.getvalue() and "ms/turn" in out.getvalue()
          and lines and all(ln.rsplit(" ", 1)[1].isdigit() for ln in lines), out.getvalue()[:300])


def test_shared_limits(kind: str):
    """What a deployment's processes share through the database (limits.py,
    v82): who is serving and each one's share of a provider's limit, a
    tenant's bucket, and a breaker's opening -- each checked against what
    the same processes did with every limit their own."""
    print(f"shared limits [{kind}]")
    import asyncio
    from collections import Counter
    from types import SimpleNamespace
    from sqlalchemy import text
    from pasarbench.service.limits import Deployment, SharedRate, processes, split
    from pasarbench.service.resilience import (CircuitBreaker, CircuitOpen, Gates, ProviderGate,
                                               Slots, TenantLimits)

    if kind == "sqlite":
        shares = {n: [split(16, n, r) for r in range(n)] for n in range(1, 20)}
        check("a provider's 16 calls at once, shared out: 6, 5 and 5 over three processes, as "
              "even as whole numbers go, adding up to 16 -- and past 16 processes one each, "
              "never none", shares[3] == [6, 5, 5]
              and all(sum(v) == 16 for n, v in shares.items() if n <= 16)
              and shares[19] == [1] * 19 and split(0, 2, 1) == 0, shares[3])

        # A share that moves (Slots.resize): waiters let in as it grows, a
        # holder never robbed as it shrinks.
        s = Slots(1)
        s.acquire(0)
        waited = []
        t = threading.Thread(target=lambda: waited.append(s.acquire(2.0)))
        t.start()
        time.sleep(0.05)
        s.resize(2)
        t.join(3)

        async def coroutine_waits() -> bool:
            got = asyncio.ensure_future(s.aacquire(2.0))
            await asyncio.sleep(0.05)
            await asyncio.get_running_loop().run_in_executor(None, s.resize, 3)
            return await got

        let_in = asyncio.run(coroutine_waits())
        s.resize(1)                              # three held, one allowed
        refused = s.acquire(0.05)
        s.release()
        s.release()
        refused_still = s.acquire(0.05)
        s.release()
        then = s.acquire(0.05)
        check("a limit resized: growing lets a waiting thread and a waiting coroutine in; "
              "shrinking below the slots in use takes none away, and no one gets one until "
              "enough are released", waited == [True] and let_in and not refused
              and not refused_still and then, (waited, let_in, refused, refused_still, then))

        now = [0.0]
        told = []
        b = CircuitBreaker(threshold=2, reset_s=30, clock=lambda: now[0],
                           on_open=told.append)
        b.record(False)
        b.record(False)
        b.record(False)                          # a call in flight failing after: no news
        now[0] = 31
        trial = b.allow()
        b.record(False)                          # the trial failed: opened again
        check("a breaker says when it opens -- after its failures, and again when its trial "
              "fails -- and not for each call failing while it is open",
              told == [30, 30] and trial is None, told)
        h = CircuitBreaker(reset_s=30, clock=lambda: now[0])
        held = h.hold(12)
        blocked = h.blocked()
        now[0] += 12.5
        after = (h.state, h.allow())
        own = CircuitBreaker(threshold=1, reset_s=30, clock=lambda: now[0])
        own.record(False)
        kept = not own.hold(5) and abs(own.blocked() - 30) < 1e-9
        check("held open for another process's breaker: no call for 12 s, then one trial as "
              "after an opening of its own; an opening of its own that lasts longer stays",
              held and abs(blocked - 12) < 1e-9 and after == ("half_open", None) and kept,
              (held, blocked, after, kept))

        tries = []
        g = ProviderGate("p", concurrency=1, attempts=3,
                         sleep=lambda _s: g.breaker.hold(30))

        def down():
            tries.append(1)
            raise RuntimeError("HTTP 503 from https://provider: down")

        try:
            g.call(down)
            stopped = None
        except CircuitOpen as e:
            stopped = e
        check("a call waiting to try again stops if its provider's breaker opens meanwhile "
              "-- here or in another process: one attempt, not three",
              stopped is not None and tries == [1], (stopped, tries))

        g2 = ProviderGate("p", concurrency=1, acquire_timeout=5)
        holding, free = threading.Event(), threading.Event()

        def holder():                            # its own call fails too: a success would close it
            holding.set()
            free.wait(5)
            raise RuntimeError("HTTP 400 from https://provider: no")

        def hold_slot():
            try:
                g2.call(holder)
            except RuntimeError:
                pass

        t = threading.Thread(target=hold_slot)
        t.start()
        holding.wait(5)
        sent, outcome = [], []

        def waiter():
            try:
                g2.call(lambda: sent.append(1))
                outcome.append("sent")
            except CircuitOpen:
                outcome.append("open")

        w = threading.Thread(target=waiter)
        w.start()
        time.sleep(0.05)
        g2.breaker.hold(30)                      # opened while it waited for the slot
        free.set()
        t.join(5)
        w.join(5)
        check("…and so does a call that waited for its slot while the breaker opened: not "
              "sent at all, and not counted a failure of the provider's (one: the holder's)",
              outcome == ["open"] and not sent and g2.breaker.failures == 1,
              (outcome, sent, g2.breaker.failures))

    # --- the database's part ------------------------------------------------
    engine, url = database(kind, "shared-limits")
    a, b2 = SharedRate(engine, 2.0, 3), SharedRate(engine, 2.0, 3)
    got = [(a if i % 2 == 0 else b2).take("acme") for i in range(4)]
    other = a.take("globex")
    asked = []
    for x in (a, b2):
        inner = x._sqlite_take if kind == "sqlite" else x._pg_take
        setattr(x, "_sqlite_take" if kind == "sqlite" else "_pg_take",
                lambda bucket, f=inner: (asked.append(bucket), f(bucket))[1])
    again = [b2.take("acme") for _ in range(20)]
    time.sleep(0.55)
    later = b2.take("acme")
    own = [TenantLimits(2.0, 3), TenantLimits(2.0, 3)]
    alone = [own[i % 2].take("acme") for i in range(6)]
    check(f"[{kind}] one bucket a tenant for two processes: of 4 requests in turn to each at "
          f"2 a second, bursts of 3, the 4th is refused, told to wait half a second -- where "
          f"each process's own bucket admitted 6; another tenant's is its own",
          got[:3] == [None] * 3 and got[3] is not None and 0.3 < got[3] <= 0.51
          and other is None and later is None and alone == [None] * 6, (got, other, later))
    check(f"[{kind}] …and a refusal is remembered until the time it gave: 20 more requests "
          f"refused with no statement, the next after it with one",
          all(w is not None and w <= got[3] for w in again) and asked == ["acme"],
          (again[:3], asked))
    from pasarbench.service.limits import _clock
    with engine.begin() as conn:                 # 40 just spent, at 0.1 a second
        conn.execute(text(f"INSERT INTO rate_limits (bucket, level, at) VALUES "
                          f"('spent', 40, {_clock(engine)})"))
    fresh = SharedRate(engine, 10.0, 40)
    first_wait = fresh.take("spent")
    time.sleep(0.12)
    then_in = fresh.take("spent")
    check(f"[{kind}] a bucket spent under other settings -- 40 requests at 0.1 a second -- "
          f"drains at these: at 10 a second refused a tenth of a second, then admitted, "
          f"where a bucket kept in seconds would have refused for 400",
          first_wait is not None and first_wait <= 0.11 and then_in is None,
          (first_wait, then_in))
    with engine.begin() as conn:                 # written by a clock 30 s ahead
        conn.execute(text(f"INSERT INTO rate_limits (bucket, level, at) VALUES "
                          f"('ahead', 40, {_clock(engine)} + 30)"))
    told = SharedRate(engine, 10.0, 40).take("ahead")
    check(f"[{kind}] a full bucket whose time is 30 s ahead says so: wait 30.1 s, not 0.1 "
          f"and a statement every tenth of a second",
          told is not None and 29.5 <= told <= 30.2, told)
    # Requests that wait on each other's lock (PostgreSQL's now() is a
    # statement's start): the bucket's time never moves back, so none drains twice.
    crowd = [SharedRate(engine, 3000.0, 40) for _ in range(4)]
    let_in = Counter()
    crowd_until = time.monotonic() + 1.5
    crowd_from = time.monotonic()

    def crowded(r, i):
        while time.monotonic() < crowd_until:
            if r.take("crowd") is None:
                let_in[i] += 1

    ts = [threading.Thread(target=crowded, args=(crowd[i % 4], i)) for i in range(24)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    allowed = 40 + 3000.0 * (time.monotonic() - crowd_from)
    check(f"[{kind}] 24 clients on 4 processes at 3,000 a second admit no more than the "
          f"limit -- the bucket's time never moves back, so no interval is drained twice "
          f"(with it moving back, 1.06-1.29 times the limit on PostgreSQL)",
          sum(let_in.values()) <= 1.03 * allowed,
          (sum(let_in.values()), round(allowed)))
    # A rolling change of PASAR_RATE: two processes, two settings, one bucket.
    old, new = SharedRate(engine, 10.0, 40), SharedRate(engine, 1.0, 40)
    admitted = Counter()
    stop_at = time.monotonic() + 1.0

    def hammer(r, name):
        while time.monotonic() < stop_at:
            if r.take("rolling") is None:
                admitted[name] += 1

    ts = [threading.Thread(target=hammer, args=(r, n)) for r, n in ((old, "old"), (new, "new"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    total = sum(admitted.values())
    check(f"[{kind}] two processes with different settings on one bucket -- a rolling "
          f"change from 10 a second to 1 -- admit no more in a second than the looser allows "
          f"(40 at once and 10 more), each admission counted by whoever made it",
          40 <= total <= 52, dict(admitted))

    if kind == "sqlite":
        from pasarbench.service.config import Settings
        from pasarbench.service.store import make_engine
        bare = make_engine(sqlite_url("not-migrated"))
        broken = SharedRate(bare, 1.0, 2)
        alone = [broken.take("acme") for _ in range(3)]
        try:
            Settings(limits="cluster").validate()
            refused = False
        except ValueError:
            refused = True
        check("a bucket that cannot be read -- a database not migrated to 0003 -- is this "
              "process's own, as before v82: 2 admitted at bursts of 2, the 3rd refused, the "
              "failure counted once (the database asked again 5 s on); PASAR_LIMITS is auto, "
              "deployment or process",
              alone[:2] == [None, None] and alone[2] is not None and broken.errors["rate"] == 1
              and refused and Settings().limits == "auto", (alone, dict(broken.errors)))
        bare.dispose()

    beat = 0.2
    cfg = SimpleNamespace(heartbeat_seconds=beat, provider_concurrency=16, rate=10.0, burst=40)

    def node(name: str) -> Deployment:
        return Deployment(engine, cfg, Gates(16, 0.5), node=name)

    d = [node("n1"), node("n2"), node("n3")]
    for x in d:
        x.start()
    for x in d:
        x.sync()
    first = [x.share for x in d]
    resized = all(x.gates.get("p").concurrency == x.share for x in d)
    d[2].stop()
    moved = until(lambda: [x.share for x in d[:2]] == [8, 8], 3)
    d[1]._stop.set()                             # dies: no row deleted, no word
    d[1]._wake.set()
    d[1]._thread.join(3)
    t0 = time.monotonic()
    alone_again = until(lambda: d[0].share == 16 and d[0].nodes == ["n1"], 6)
    gone_after = time.monotonic() - t0
    check(f"[{kind}] three processes take 6, 5 and 5 of a provider's 16 calls at once; one "
          f"that leaves gives its share back at once, and one that dies within three "
          f"heartbeats (0.6 s here)",
          first == [6, 5, 5] and resized and moved and alone_again and gone_after < 2.5,
          (first, resized, moved, [x.share for x in d], round(gone_after, 2)))
    view = d[0].view()
    check(f"[{kind}] GET /v1/admin/nodes: the processes as the database has them, the one "
          f"no longer counted marked so, and this one's share",
          view["processes"] == 1 and view["provider_concurrency"] == {"deployment": 16,
                                                                     "this_process": 16}
          and {n["node"]: n["counted"] for n in view["nodes"]} == {"n1": True, "n2": False},
          view)
    d[0].stop()
    hung = node("hung")
    hung.start()
    hung._beat.acquire()                         # a heartbeat the database never answers
    t0 = time.monotonic()
    hung.stop(timeout=0.5)
    took = time.monotonic() - t0
    hung._beat.release()
    gone = until(lambda: "hung" not in [r["node"] for r in processes(engine)[0]], 3)
    hung.sync()
    with engine.connect() as conn:
        after = conn.execute(text("SELECT count(*) FROM service_nodes WHERE node = 'hung'")
                             ).scalar()
    check(f"[{kind}] a process stopping waits for the database at most its timeout (0.5 s "
          f"here: the process goes, and its row expires on its own); the heartbeat it was "
          f"waiting for ends before the row is deleted, and none after it writes the row back",
          took < 0.9 and gone and after == 0, (round(took, 2), gone, after))

    # On PostgreSQL the word comes at once: heartbeats 30 s apart, here.
    slow = SimpleNamespace(**{**vars(cfg), "heartbeat_seconds": 30.0 if kind == "postgres"
                              else beat})
    p1, p2 = (Deployment(engine, slow, Gates(16, 0.5), node=n) for n in ("p1", "p2"))
    p1.start()
    p2.start()
    joined = until(lambda: p1.share == p2.share == 8, 1.5)
    for _ in range(5):
        p1.gates.get("prov").breaker.record(False)
    heard = until(lambda: p2.gates.get("prov").breaker.state == "open", 1.5)
    calls = []
    try:
        p2.gates.get("prov").call(lambda: calls.append(1))
        fast = False
    except CircuitOpen:
        fast = True
    p2.gates.get("prov").breaker.record(True)    # a call of its own got through after all
    p2.sync()
    p2.sync()
    once = p2.gates.get("prov").breaker.state == "closed"
    check(f"[{kind}] a breaker that opens in one process opens in the other"
          f"{' at once (LISTEN/NOTIFY)' if kind == 'postgres' else ', by its next heartbeat'}: "
          f"its calls fail fast, unsent -- and the same opening, read again, does not open it "
          f"again once a call there has closed it",
          heard and fast and not calls and once, (heard, fast, calls, once))
    listening = p1.view()["hears_at_once"]
    p2.stop()
    left = until(lambda: p1.share == 16, 1.5)
    if kind == "postgres":
        check("[postgres] …and a process joining or leaving is heard of at once, not at the "
              "next heartbeat 30 s on: each of two counts the other as it joins, and the one "
              "left takes all 16 as the other leaves", joined and left and listening,
              (joined, left, listening))
    p1.stop()
    if kind == "sqlite":
        import contextlib
        import io
        from pasarbench.service.__main__ import main as cli
        os.environ["PASAR_WORKERS"] = "4x"
        try:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                listed = cli(["--database-url", url, "nodes"])
                served = cli(["--database-url", url, "serve"])
        finally:
            del os.environ["PASAR_WORKERS"]
        check("a malformed PASAR_WORKERS stops `serve` with one line, and no other command: "
              "`nodes` lists the processes and the breakers open",
              listed == 0 and served == 2 and "PASAR_WORKERS" in err.getvalue()
              and json.loads(out.getvalue())["breakers_open"] is not None,
              (listed, served, err.getvalue()[:200]))
        from pasarbench.service.__main__ import shared_on_sqlite
        check("`serve --workers 2` on one SQLite file with PASAR_LIMITS=auto shares the "
              "limits; one process, PostgreSQL, or a PASAR_LIMITS said outright, as set",
              shared_on_sqlite("auto", "sqlite:///x.db", 2)
              and not shared_on_sqlite("auto", "sqlite:///x.db", 1)
              and not shared_on_sqlite("process", "sqlite:///x.db", 2)
              and not shared_on_sqlite("auto", "postgresql+psycopg://h/db", 2))
    if kind == "postgres":
        import logging as _logging
        caught = []

        class Catch(_logging.Handler):
            def emit(self, record):
                caught.append(record)

        pool_log = _logging.getLogger("sqlalchemy.pool")
        catch = Catch()
        pool_log.addHandler(catch)
        q = Deployment(engine, cfg, Gates(16, 0.5), node="q1")
        q.start()
        first_pid = q.listener_pid
        with engine.begin() as conn:
            conn.execute(text("SELECT pg_terminate_backend(:p)"), {"p": first_pid})
        back = until(lambda: q.listening and q.listener_pid not in (None, first_pid), 5)
        q.stop()
        pool_log.removeHandler(catch)
        check("[postgres] a listener whose session the server ends listens again on a new "
              "one, a heartbeat on, with nothing logged outside the service's own lines",
              back and not caught, (back, [r.getMessage()[:80] for r in caught]))
        with engine.connect() as conn:
            kinds = dict(conn.execute(text(
                "SELECT relname, relpersistence FROM pg_class WHERE relname IN "
                "('service_nodes', 'rate_limits', 'provider_breakers')")).all())
        check("[postgres] the shared tables are UNLOGGED: no disk write a request",
              kinds == {"service_nodes": "u", "rate_limits": "u", "provider_breakers": "u"},
              kinds)

    # Two apps -- two processes -- on one database.
    h = None
    auto = make_app(engine, url).state.deployment
    check(f"[{kind}] PASAR_LIMITS=auto shares the limits on PostgreSQL and keeps them each "
          f"process's own on SQLite",
          (auto is not None) == (kind == "postgres"), auto)
    apps = [make_app(engine, url, rate=1.0, burst=2, provider_concurrency=16,
                     heartbeat_seconds=beat, limits="deployment") for _ in range(2)]
    with client(apps[0]) as c1, client(apps[1]) as c2:
        h = tenant(c1, "shared")
        codes = [(c1 if i % 2 == 0 else c2).get("/v1/conversations", headers=h)
                 for i in range(3)]
        until(lambda: all(c.get("/v1/admin/nodes", headers=OP).json()["processes"] == 2
                          for c in (c1, c2)), 3)
        nodes = [c.get("/v1/admin/nodes", headers=OP).json() for c in (c1, c2)]
    apart = [make_app(engine, url, rate=1.0, burst=2, limits="process") for _ in range(2)]
    with client(apart[0]) as c1, client(apart[1]) as c2:
        h2 = tenant(c1, "apart")
        own_codes = [(c1 if i % 2 == 0 else c2).get("/v1/conversations", headers=h2).status_code
                     for i in range(5)]
        apart_view = c1.get("/v1/admin/nodes", headers=OP).json()
    check(f"[{kind}] two servers on one database: a tenant at 1 a second, bursts of 2, is "
          f"refused its 3rd request whichever server it reaches, told when to come back; "
          f"each counts two processes, 8 calls at once each of 16 -- with PASAR_LIMITS=process "
          f"each server admitted 2 of its own",
          [r.status_code for r in codes] == [200, 200, 429]
          and codes[2].headers.get("retry-after") == "1"
          and [n["processes"] for n in nodes] == [2, 2]
          and [n["provider_concurrency"]["this_process"] for n in nodes] == [8, 8]
          and own_codes == [200, 200, 200, 200, 429] and apart_view["limits"] == "process",
          ([r.status_code for r in codes], [n.get("processes") for n in nodes], own_codes))
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
    ("GET", "/v1/admin/profile"), ("DELETE", "/v1/admin/profile"), ("GET", "/v1/admin/nodes"),
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


def test_waiting_turns():
    """A turn waiting on its model holds no thread (v81): two dozen turns at
    once, on four worker threads, with a model that takes half a second a
    call, finish together -- not four at a time."""
    print("waiting turns")
    engine, url = database("sqlite", "waiting")
    n, threads, latency = 24, 4, 0.5
    app = make_app(engine, url, threads=threads, max_concurrent_turns=n,
                   provider_concurrency=n, mock_latency_ms=int(latency * 1000))
    with client(app) as c:
        h = tenant(c, "acme", world="synthetic:4")
        from pasarbench.db import Database
        seed = Database.fresh().tables
        people = [(u, next(o for o, x in seed["orders"].items() if x["user_id"] == u))
                  for u in seed["users"] if any(x["user_id"] == u for x in seed["orders"].values())]
        mine = [(f"{u}-C{i:04d}", f"{o}-C{i:04d}") for i in range(1, 5) for u, o in people][:n]
        cids = [(c.post("/v1/conversations", headers=h, json={"user_id": u}).json()[
            "conversation_id"], o) for u, o in mine]
        out: list = [None] * len(cids)
        gate = threading.Barrier(len(cids))
        peak = [0]

        def one(i):
            cid, order = cids[i]
            gate.wait()
            out[i] = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                            json={"content": f"where is my order {order}?"})

        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(cids))]
        before = threading.active_count()
        t0 = time.monotonic()
        for t in ts:
            t.start()
        while any(t.is_alive() for t in ts):
            peak[0] = max(peak[0], threading.active_count())
            time.sleep(0.02)
        took = time.monotonic() - t0
    ok = [r.status_code == 200 and r.json()["stop_reason"] == "replied" for r in out]
    calls = [r.json()["usage"]["model_calls"] for r in out if r.status_code == 200]
    serial = len(cids) / threads * max(calls or [3]) * latency
    check(f"{len(cids)} turns at once on {threads} worker threads, {max(calls or [0])} model "
          f"calls of {latency:.1f} s each: all done in {took:.1f} s -- they waited together; "
          f"held a thread each, four at a time, they would take {serial:.0f} s",
          all(ok) and took < serial / 2, (sum(ok), took))
    workers = peak[0] - before - len(cids)
    check(f"…and the process held at most {workers} threads beyond its clients' while they "
          f"waited: its worker pools' {threads} each, and the test client's own",
          workers <= 2 * threads + 4, (peak[0], before))
    engine.dispose()

    # A running turn's transactions have threads of their own: every other
    # handler's thread taken -- here a profile, holding the one there is for
    # its three seconds -- a turn already admitted still finishes in its own
    # time, not after the burst.
    engine, url = database("sqlite", "bulkhead")
    app = make_app(engine, url, threads=1, mock_latency_ms=200)
    with client(app) as c:
        h = tenant(c, "acme")
        cid = c.post("/v1/conversations", headers=h, json={"user_id": "U002"}).json()[
            "conversation_id"]
        box: dict = {}

        def turn():
            t0 = time.monotonic()
            box["r"] = c.post(f"/v1/conversations/{cid}/messages", headers=h,
                              json={"content": "where is my order O1002?"})
            box["took"] = time.monotonic() - t0

        def busy():
            box["profile"] = c.get("/v1/admin/profile?seconds=3", headers=OP).status_code

        rt = app.state.runtime
        tt = threading.Thread(target=turn)
        tt.start()
        until(lambda: rt.in_flight == 1)
        bt = threading.Thread(target=busy)
        bt.start()
        from pasarbench.service.profiling import running
        until(lambda: running() is not None)
        tt.join(10)
        bt.join(10)
    check(f"a turn already running finishes in its own time ({box.get('took', 0):.1f} s, three "
          f"model calls of 0.2 s) while every other handler's thread is taken for three "
          f"seconds: its transactions have threads of their own",
          box.get("r") is not None and box["r"].status_code == 200
          and box["took"] < 2.5 and box.get("profile") == 200,
          (box.get("took"), box.get("profile")))
    engine.dispose()

    import asyncio
    from pasarbench.service.config import Settings
    from pasarbench.service.runtime import Runtime
    engine, url = database("sqlite", "cancelled")
    rt = Runtime(engine, Settings(database_url=url, threads=1, max_concurrent_turns=2,
                                  log_level="WARNING"))

    async def cancelled_waiting() -> int:
        busy = asyncio.ensure_future(rt.in_thread(time.sleep, 0.5))
        await asyncio.sleep(0.05)
        waiting = asyncio.ensure_future(rt.abegin_turn("acme", "cv_" + "0" * 24))
        await asyncio.sleep(0.05)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await busy
        return rt.admission.free
    free = asyncio.run(cancelled_waiting())
    check("…and a turn cancelled while it waits for one of them gives its slot back",
          free == 2, free)
    engine.dispose()

    from pasarbench.harness.types import Message
    from pasarbench.service.models import MockAgent
    steady, spread = MockAgent(latency_ms=100), MockAgent(latency_ms=100, jitter=0.3)
    waits = [spread._latency() for _ in range(2000)]
    said = [Message(role="system", content="user_id **U001** in the SG market"),
            Message(role="user", content="where is my order O1001?")]
    check("the mock's latency strays as a provider's does, by up to its jitter "
          f"({min(waits):.3f}–{max(waits):.3f} s around 0.100), and what it says does not",
          0.07 <= min(waits) < 0.08 and 0.12 < max(waits) <= 0.13
          and steady._latency() == 0.1 and spread.respond(said, []) == steady.respond(said, []),
          (min(waits), max(waits)))


def test_transport():
    """What the service sends a provider is what the benchmark's _post sends
    (v81): the same body, the backend's headers, urllib's own agent; and what
    goes wrong is the same error, retried or not as before -- a redirect
    followed nowhere, a key sent nowhere else. Over one connection kept alive,
    where urllib opens one a call."""
    print("transport")
    import asyncio
    import http.server
    from pasarbench.harness import backends as B
    from pasarbench.harness.types import Message
    from pasarbench.service.models import Transport, _Unhalted
    from pasarbench.service.resilience import is_transient

    seen: list = []
    ports: set = set()
    answer = {"choices": [{"message": {"content": "hi", "tool_calls": [
        {"id": "c1", "function": {"name": "get_order", "arguments": "{\"order_id\": \"O1\"}"}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                                  "prompt_cache_hit_tokens": 4}, "model": "m"}

    class Provider(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.path, dict(self.headers.items()), body))
            ports.add(self.client_address[1])
            code, out, extra = 200, json.dumps(answer).encode(), {}
            if self.path.startswith("/s503"):
                code, out = 503, b"overloaded"
            elif self.path.startswith("/s400"):
                code, out = 400, b"bad request"
            elif self.path.startswith("/moved"):
                code, out, extra = 302, b"", {"Location": f"http://127.0.0.1:{other}/steal"}
            elif self.path.startswith("/slow"):
                time.sleep(1.5)
            try:
                self.send_response(code)
                for k, v in {"Content-Type": "application/json", **extra}.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            except (BrokenPipeError, ConnectionResetError):
                pass                            # a client that timed out has gone

    servers = [http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider) for _ in range(2)]
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    port, other = servers[0].server_address[1], servers[1].server_address[1]
    base = f"http://127.0.0.1:{port}"
    msgs = [Message(role="system", content="you help"), Message(role="user", content="där?")]
    tools = [{"type": "function", "function": {"name": "get_order", "description": "d",
                                               "parameters": {"type": "object"}}}]
    backend = B.OpenAICompatBackend("m", base, api_key="sk-test", extra_body={"seed": 1})
    sync = backend.chat(msgs, tools)
    by_urllib = seen[-1]
    t = Transport()
    wrapped = _Unhalted(backend, B, t)

    async def calls(n):
        outs = [await wrapped.achat(msgs, tools) for _ in range(n)]
        await t.aclose()
        return outs

    ports.clear()
    outs = asyncio.run(calls(3))
    by_httpx = seen[-1]
    same_headers = all(by_httpx[1].get(k) == by_urllib[1].get(k) for k in
                       ("Authorization", "Content-Type", "User-Agent", "Accept-Encoding",
                        "Content-Length"))
    check("the same request: path, body byte for byte, the key, content type, urllib's user "
          "agent and encoding",
          by_httpx[0] == by_urllib[0] and by_httpx[2] == by_urllib[2] and same_headers,
          (by_urllib[1], by_httpx[1]))
    check("…the same answer read from it, and three calls over one connection kept alive",
          all(o == sync for o in outs) and len(ports) == 1, (outs[0], sync, len(ports)))

    def both(path, timeout=5.0):
        url = f"{base}{path}"
        got = []
        try:
            B._post(url, {"a": 1}, {"Authorization": "Bearer sk-test"}, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            got.append(e)

        async def go():
            try:
                await t.post(url, {"a": 1}, {"Authorization": "Bearer sk-test"}, timeout=timeout)
            except Exception as e:  # noqa: BLE001
                got.append(e)
            finally:
                await t.aclose()
        asyncio.run(go())
        return got

    errs = {path: both(path) for path in ("/s503", "/s400", "/moved")}
    errs["/slow"] = both("/slow", timeout=0.5)
    servers[1].shutdown()
    servers[1].server_close()
    dead = f"http://127.0.0.1:{other}"
    closed = []
    try:
        B._post(dead, {}, {})
    except Exception as e:  # noqa: BLE001
        closed.append(e)

    async def refused():
        try:
            await t.post(dead, {}, {})
        except Exception as e:  # noqa: BLE001
            closed.append(e)
        finally:
            await t.aclose()
    asyncio.run(refused())
    errs["refused"] = closed
    same = {k: (len(v) == 2 and is_transient(v[0]) == is_transient(v[1])
                and (str(v[0]) == str(v[1]) if k in ("/s503", "/s400", "/moved") else True))
            for k, v in errs.items()}
    check("the same failures: a 503 and a 400 the same error word for word, retried and not; "
          "a timeout and a refused connection both retried", all(same.values())
          and is_transient(errs["/s503"][1]) and not is_transient(errs["/s400"][1])
          and is_transient(errs["/slow"][1]) and is_transient(errs["refused"][1]),
          {k: [f"{type(e).__name__}: {e}"[:80] for e in v] for k, v in errs.items()})
    check("…and a redirect is an error, followed nowhere: the key never reaches the host it "
          "names", "HTTP 302" in str(errs["/moved"][1])
          and not any(path == "/steal" for path, _h, _b in seen))
    servers[0].shutdown()
    servers[0].server_close()


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
        test_database_errors()
        test_bad_arguments("sqlite")
        test_nul("sqlite")
        test_order("sqlite")
        test_failures_keep_their_key()
        test_long_turns()
        test_waiting_turns()
        test_transport()
        test_zero_usage_is_estimated()
        test_logs_hold_no_message()
        test_slices("sqlite")
        test_profiler()
        test_shared_limits("sqlite")
        test_parity("sqlite")
        test_episodes("sqlite")
        test_kit_messages()
        if PG:
            test_migrations("postgres")
            test_concurrency("postgres")
            test_ids("postgres")
            test_slices("postgres")
            test_liveness()
            test_silent_database()
            test_shared_limits("postgres")
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
