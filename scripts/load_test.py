"""
Load test for the agent service: a real server (uvicorn, its own process),
real HTTP, concurrent customers, the mock model -- so what is measured is the
service itself, then the service under a model's latency.

    python scripts/load_test.py --database-url sqlite:///data/load.db
    python scripts/load_test.py --database-url postgresql+psycopg://user:pw@host/db \\
        --turns 2000 --concurrency 32 --customers 200
    python scripts/load_test.py ... --hot            every worker on one customer
    python scripts/load_test.py ... --mock-latency-ms 400
    python scripts/load_test.py ... --concurrency 1000 --mock-latency-ms 10000 \
        --mock-jitter 0.3 --ramp-s 20     a thousand customers on a slow model, each
                                          call 7-13 s, arriving over 20 s (v81)

The database must be one the test may fill (it migrates it, and makes a tenant
of its own each run). Each worker opens a conversation for its customer and
sends turns: half ask where an order is (reads), half ask for a voucher or a
human (writes: an insert under the customer's lock, and an outbox ticket),
each with the customer's digits. A conversation is replaced every 10 turns.

It reports turns a second and the latency a client sees, p50/p95/p99, beside
the server's own time of each turn; every status that was not 200, and every
request that failed in transport (both fail the run); the time the tool calls
spent in their transactions (from the audit log); the CPU a turn cost -- the
server's, this client's, and the whole machine's, the database's among the
rest -- and the server's threads and memory at their peak (Linux: /proc); the
event loop's lag (pasar_event_loop_lag_seconds); and then replays the tenant:
every tool call the run made, run again against the rebuilt world, must answer
and change the same.

    python scripts/load_test.py ... --profile p.json

profiles the server while it serves the turns (GET /v1/admin/profile; one
server process, --workers 1, for all of it) and prints where its CPU went,
per turn (profiling.py).

    python scripts/load_test.py ... --statements      PostgreSQL: what a turn sent
    python scripts/load_test.py ... --db-latency-ms 1 a database 1 ms further off
    python scripts/load_test.py ... --root ../v83     another source tree's server

--statements counts what the database ran for each turn, from its own
statistics (pg_stat_statements, which the database must have: installed with
CREATE EXTENSION, loaded by shared_preload_libraries, with track_utility on
so that BEGIN and COMMIT count), read as the turns start and as they end --
nothing is reset: the statistics are the server's, every database's -- and
the difference is what the turns ran: every statement, the transactions'
BEGIN and COMMIT among them, each a round trip; or why it could not count.
--db-latency-ms puts a relay between the server and its database (a TCP URL),
a process of its own, that holds what it is sent for half that each way, as
a database in another rack or zone is further off; the round trip it adds is
measured idle and sampled through the run, and its CPU is reported apart
from the database's. (v84)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROFILE_SECONDS = 600           # the longest a profile runs; the test ends it when it is done


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * len(xs) + 0.5)) - 1))
    return xs[k]


def _ticks(fields: list[str]) -> float:
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def process_cpu(pid: int) -> float | None:
    """The CPU seconds a process and its children (a server's workers) have
    used so far, every thread of each; None where there is no /proc."""
    try:
        total = 0.0
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue                       # gone since the listing
            if int(d.name) == pid or int(fields[1]) == pid:
                total += _ticks(fields)
        return total
    except (OSError, ValueError, IndexError):
        return None


def server_threads(pid: int) -> tuple[int, float] | None:
    """The threads a process has now and its resident memory in MB (Linux:
    /proc), its children's included."""
    try:
        threads, rss = 0, 0.0
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                status = (d / "status").read_text()
            except OSError:
                continue
            fields = dict(line.split(":\t", 1) for line in status.splitlines() if ":\t" in line)
            if int(d.name) == pid or fields.get("PPid", "").strip() == str(pid):
                threads += int(fields.get("Threads", "0"))
                rss += int(fields.get("VmRSS", "0 kB").split()[0]) / 1024
        return threads, rss
    except (OSError, ValueError):
        return None


def loop_lag(base: str) -> dict[str, float] | None:
    """The server's event-loop lag histogram (pasar_event_loop_lag_seconds,
    v81), as {upper bound: count so far}; None from a server without it."""
    import httpx
    try:
        text = httpx.get(f"{base}/metrics", timeout=30).text
    except httpx.HTTPError:
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("pasar_event_loop_lag_seconds_bucket{"):
            le = line.split('le="', 1)[1].split('"', 1)[0]
            out[le] = float(line.rsplit(" ", 1)[1])
    return out or None


def lag_between(a: dict[str, float] | None, b: dict[str, float] | None) -> dict | None:
    """The lag while the turns ran: its median and p99 (a bucket's upper
    bound, in ms) and the share of timers more than 100 ms late."""
    if not a or not b:
        return None
    bounds = sorted(b, key=lambda le: float("inf") if le == "+Inf" else float(le))
    cum = [(le, b[le] - a.get(le, 0.0)) for le in bounds]
    n = cum[-1][1]
    if n <= 0:
        return None

    def q(p: float) -> str:
        le = next(le for le, c in cum if c >= p * n)
        return ">10000" if le == "+Inf" else f"{float(le) * 1000:g}"
    late = n - dict(cum).get("0.1", n)
    return {"timers": int(n), "p50_ms": q(0.5), "p99_ms": q(0.99),
            "over_100ms": round(late / n, 4)}


def machine_cpu() -> float | None:
    """The CPU seconds every process on the machine has used (/proc/stat),
    idle and I/O wait not counted."""
    try:
        v = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
        return (sum(v[:8]) - v[3] - v[4]) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def own_cpu() -> float:
    t = os.times()
    return t.user + t.system


def relay(host: str, port: int, latency_ms: float) -> None:
    """A TCP relay between the server and its database, holding what passes
    for half the latency it is given in each direction (--db-latency-ms):
    every round trip the server makes costs that much more, as it would to a
    database further off. A process of its own, its CPU counted apart (the
    report's cpu_ms_per_turn.relay), on one thread: select() -- whose timeout is kept to
    the microsecond, where an event loop's are whole milliseconds -- reads
    what arrives, and each chunk is sent on when it is due. It prints the
    port it listens on, and runs until it is killed."""
    import heapq
    import select as _select
    delay = latency_ms / 2000
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(256)
    print(listener.getsockname()[1], flush=True)
    peer: dict[socket.socket, socket.socket] = {}
    due: list[tuple[float, int, socket.socket, bytes]] = []     # (when, n, to, data)
    n = 0
    while True:
        wait = None if not due else max(0.0, due[0][0] - time.monotonic())
        readable, _, _ = _select.select([listener, *peer], [], [], wait)
        now = time.monotonic()
        for r in readable:
            if r is listener:
                client, _ = listener.accept()
                try:
                    upstream = socket.create_connection((host, port))
                except OSError:
                    client.close()
                    continue
                for s_ in (client, upstream):
                    s_.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                peer[client], peer[upstream] = upstream, client
                continue
            try:
                data = r.recv(262144)
            except OSError:
                data = b""
            n += 1
            heapq.heappush(due, (now + delay, n, peer[r], data))
            if not data:                        # its end, after what came before it
                peer.pop(r, None)
        now = time.monotonic()
        while due and due[0][0] <= now:
            _when, _n, to, data = heapq.heappop(due)
            try:
                if data:
                    to.sendall(data)
                else:
                    to.shutdown(socket.SHUT_WR)
            except OSError:
                pass


def round_trip_ms(url: str, n: int = 200) -> float:
    """The median time of a SELECT 1 on one connection to `url`, in ms."""
    from sqlalchemy import create_engine, text
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            times = []
            for _ in range(n):
                t0 = time.perf_counter()
                conn.execute(text("SELECT 1"))
                times.append(time.perf_counter() - t0)
        return round(statistics.median(times) * 1000, 3)
    finally:
        engine.dispose()


class sample_round_trips:
    """A SELECT 1 every 0.2 s while the turns run, on a connection of its own,
    straight to the database and through the relay: the round trip the relay
    adds under the run's load, not only when it is idle."""

    def __init__(self, direct: str, relayed: str):
        self.times: dict[str, list[float]] = {"direct": [], "relayed": []}
        self.done = threading.Event()
        self.threads = [threading.Thread(target=self._run, args=(k, u), daemon=True)
                        for k, u in (("direct", direct), ("relayed", relayed))]
        for t in self.threads:
            t.start()

    def _run(self, kind: str, url: str) -> None:
        from sqlalchemy import create_engine, text
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                while not self.done.wait(0.2):
                    t0 = time.perf_counter()
                    conn.execute(text("SELECT 1"))
                    self.times[kind].append((time.perf_counter() - t0) * 1000)
        finally:
            engine.dispose()

    def stop(self) -> dict[str, float] | None:
        self.done.set()
        for t in self.threads:
            t.join(10)
        d, r = self.times["direct"], self.times["relayed"]
        if not d or not r:
            return None
        added = statistics.median(r) - statistics.median(d)
        return {"direct_p50_ms": round(statistics.median(d), 3),
                "relayed_p50_ms": round(statistics.median(r), 3),
                "relayed_p95_ms": round(pct(r, 95), 3), "added_p50_ms": round(added, 3),
                "samples": len(r)}


CONTROL = ("BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE")


def statements(url: str) -> tuple[dict[int, tuple[str, int]] | None, str | None]:
    """What PostgreSQL has run on this database so far, by query
    (pg_stat_statements: {queryid: (query, calls)}), read and not reset --
    the statistics are the server's, every database's and every user's --
    so that a run's are what it adds; or None and why not."""
    from sqlalchemy import create_engine, text
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            utility = conn.execute(text(
                "SELECT setting FROM pg_settings "
                "WHERE name = 'pg_stat_statements.track_utility'")).scalar()
            if utility is None:
                return None, ("pg_stat_statements is not loaded: the server needs it in "
                              "shared_preload_libraries")
            if utility != "on":
                return None, ("pg_stat_statements.track_utility is off: BEGIN and COMMIT "
                              "would not be counted")
            if conn.execute(text("SELECT to_regclass('pg_stat_statements')")).scalar() is None:
                return None, ("pg_stat_statements is not installed in this database: "
                              "CREATE EXTENSION pg_stat_statements")
            rows = conn.execute(text(
                "SELECT queryid, query, calls FROM pg_stat_statements WHERE dbid = "
                "(SELECT oid FROM pg_database WHERE datname = current_database())"))
            return {int(q): (t, int(c)) for q, t, c in rows}, None
    except Exception as e:  # noqa: BLE001 -- not allowed, not reachable
        return None, f"pg_stat_statements could not be read ({type(e).__name__})"
    finally:
        engine.dispose()


def ran_between(a: dict[int, tuple[str, int]], b: dict[int, tuple[str, int]]
                ) -> list[tuple[str, int]]:
    """What ran between two readings of statements()."""
    return [(t, c - a.get(q, ("", 0))[1]) for q, (t, c) in b.items()
            if c - a.get(q, ("", 0))[1] > 0]


def per_turn(rows: list[tuple[str, int]], turns: int) -> dict[str, Any]:
    """A turn's statements, its transactions' control among them, and the
    queries it ran most."""
    import re
    total = sum(c for _q, c in rows)
    control = sum(c for q, c in rows if q.lstrip().upper().startswith(CONTROL))
    top = sorted(rows, key=lambda r: -r[1])[:12]
    return {"statements": round(total / turns, 1), "control": round(control / turns, 1),
            "top": [[round(c / turns, 2), re.sub(r"\s+", " ", q)[:100]] for q, c in top]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--turns", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--customers", type=int, default=50, help="synthetic customers (x6)")
    ap.add_argument("--hot", action="store_true", help="every worker on one customer")
    ap.add_argument("--ramp-s", type=float, default=0.0,
                    help="each client starts at a random moment in the first this many seconds, "
                         "rather than all at once")
    ap.add_argument("--mock-latency-ms", type=int, default=0)
    ap.add_argument("--mock-jitter", type=float, default=0.0,
                    help="each mock call's latency strays by up to this fraction of it")
    ap.add_argument("--threads", type=int, default=64,
                    help="the server's worker threads (PASAR_THREADS); every client's turn "
                         "is admitted (PASAR_MAX_CONCURRENT_TURNS), waiting on its model "
                         "without one (v81)")
    ap.add_argument("--workers", type=int, default=1, help="server processes (uvicorn)")
    ap.add_argument("--tenants", type=int, default=1,
                    help="spread the workers over this many tenants (each its own id counter)")
    ap.add_argument("--provider-concurrency", type=int, default=16,
                    help="model calls in flight per provider, per server process")
    ap.add_argument("--json", help="write the result here too")
    ap.add_argument("--profile", help="profile the server while it serves the turns: the "
                                      "profile's JSON here, its summary on stdout")
    ap.add_argument("--profile-interval-ms", type=float, default=10)
    ap.add_argument("--statements", action="store_true",
                    help="PostgreSQL: count what the database ran for each turn "
                         "(pg_stat_statements)")
    ap.add_argument("--db-latency-ms", type=float, default=0.0,
                    help="add this much to each of the server's round trips to its database "
                         "(a relay, a process of its own; a TCP URL)")
    ap.add_argument("--root", help="the source tree whose server to run (default: this one)")
    ap.add_argument("--relay", nargs=3, help=argparse.SUPPRESS)      # host port ms: the relay
    if "--relay" in sys.argv:
        host, port, ms = sys.argv[sys.argv.index("--relay") + 1:][:3]
        relay(host, int(port), float(ms))
        return 0
    args = ap.parse_args()

    import httpx
    from sqlalchemy import select

    from pasarbench.db import Database
    from pasarbench.service import schema as S
    from pasarbench.service.store import make_engine

    port = free_port()
    token = secrets.token_urlsafe(16)
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}  # no key goes anywhere
    env.update(PASAR_DATABASE_URL=args.database_url, PASAR_ADMIN_TOKEN=token,
               PASAR_MOCK_LATENCY_MS=str(args.mock_latency_ms),
               PASAR_MOCK_JITTER=str(args.mock_jitter), PASAR_RATE="100000",
               PASAR_BURST="100000", PASAR_THREADS=str(args.threads),
               PASAR_MAX_CONCURRENT_TURNS=str(max(args.concurrency, 1)),
               PASAR_PROVIDER_CONCURRENCY=str(args.provider_concurrency),
               PASAR_LOG_LEVEL="WARNING", PYTHONDONTWRITEBYTECODE="1")
    py = sys.executable
    root = Path(args.root).resolve() if args.root else ROOT
    env["PYTHONPATH"] = os.pathsep.join([str(root), *filter(None, [env.get("PYTHONPATH")])])
    subprocess.run([py, "-m", "pasarbench.service", "migrate"], env=env, cwd=root, check=True,
                   stdout=subprocess.DEVNULL)
    relayer, rtt, relayed = None, None, None
    if args.db_latency_ms > 0:
        from sqlalchemy.engine import make_url
        u = make_url(args.database_url)
        if not u.host or u.host.startswith("/"):
            raise SystemExit("--db-latency-ms needs a TCP --database-url (host and port)")
        relayer = subprocess.Popen([py, __file__, "--relay", u.host, str(u.port or 5432),
                                    str(args.db_latency_ms)], stdout=subprocess.PIPE,
                                   text=True)
        try:
            relayed = u.set(host="127.0.0.1", port=int(relayer.stdout.readline())
                            ).render_as_string(hide_password=False)
            rtt = {"direct_ms": round_trip_ms(args.database_url),
                   "relayed_ms": round_trip_ms(relayed)}
        except BaseException:
            relayer.kill()
            raise
        env["PASAR_DATABASE_URL"] = relayed
    import tempfile
    errlog = tempfile.NamedTemporaryFile(prefix="pasar-load-", suffix=".log", delete=False)
    server = subprocess.Popen([py, "-m", "pasarbench.service", "serve", "--port", str(port),
                               "--workers", str(args.workers)],
                              env=env, cwd=root, stdout=subprocess.DEVNULL, stderr=errlog)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{base}/readyz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise SystemExit("the server did not come up: " +
                             Path(errlog.name).read_text(errors="replace")[-4000:])
        run_id = secrets.token_hex(3)
        op = {"Authorization": f"Bearer {token}"}
        tenants = []
        for n in range(args.tenants):
            r = httpx.post(f"{base}/v1/admin/tenants", headers=op, timeout=120,
                           json={"tenant_id": f"load-{run_id}-{n}", "name": "load test",
                                 "world": f"synthetic:{args.customers}"})
            r.raise_for_status()
            tenants.append((f"load-{run_id}-{n}",
                            {"Authorization": f"Bearer {r.json()['api_key']}"}))

        seed = Database.fresh().tables
        people = [(u, d["verified_phone_last4"],
                   [o for o, x in seed["orders"].items() if x["user_id"] == u])
                  for u, d in seed["users"].items()]
        people = [p for p in people if p[2]]
        customers = [(f"{u}-C{i:04d}", digits, [f"{o}-C{i:04d}" for o in orders])
                     for i in range(1, args.customers + 1) for u, digits, orders in people]

        latencies: list[float] = []
        served: list[float] = []                # the turn's own time, as the server timed it
        statuses: Counter = Counter()
        stops: Counter = Counter()
        lock = threading.Lock()
        next_turn = [0]

        def worker(w: int) -> None:
            user, digits, orders = customers[0 if args.hot else w % len(customers)]
            h = tenants[w % len(tenants)][1]
            if args.ramp_s:
                time.sleep(random.uniform(0, args.ramp_s))
            with httpx.Client(base_url=base, headers=h, timeout=300) as c:
                cid, used = None, 0
                while True:
                    with lock:
                        n = next_turn[0]
                        if n >= args.turns:
                            return
                        next_turn[0] += 1
                    try:
                        if cid is None or used >= 10:
                            cid = c.post("/v1/conversations", json={"user_id": user}).json()[
                                "conversation_id"]
                            used = 0
                    except (httpx.HTTPError, KeyError, ValueError) as e:
                        # Counted, not lost: a turn the client could not send is a failure.
                        with lock:
                            statuses[f"no conversation: {type(e).__name__}"] += 1
                        cid = None
                        continue
                    order = orders[n % len(orders)]
                    text = [f"where is my order {order}?",
                            f"can I get a voucher for the delay on {order}? digits {digits}",
                            f"track {order} please",
                            f"I want a human agent for {order}, my digits are {digits}"][n % 4]
                    t0 = time.perf_counter()
                    try:
                        resp = c.post(f"/v1/conversations/{cid}/messages",
                                      json={"content": text})
                    except httpx.HTTPError as e:
                        with lock:
                            statuses[f"transport: {type(e).__name__}"] += 1
                        cid = None                  # the turn may have run: a new conversation
                        continue
                    dt = time.perf_counter() - t0
                    used += 1
                    with lock:
                        statuses[resp.status_code] += 1
                        if resp.status_code == 200:
                            latencies.append(dt)
                            body = resp.json()
                            stops[body["stop_reason"]] += 1
                            served.append(body["latency_ms"] / 1000)

        profiled: dict[str, Any] = {}

        def profile() -> None:
            profiled["response"] = httpx.get(
                f"{base}/v1/admin/profile", headers=op, timeout=PROFILE_SECONDS + 120,
                params={"seconds": PROFILE_SECONDS, "interval_ms": args.profile_interval_ms})

        profiler = threading.Thread(target=profile) if args.profile else None
        if profiler is not None:
            profiler.start()
        lag0 = loop_lag(base)
        cpu0 = (process_cpu(server.pid), own_cpu(), machine_cpu())
        peak = [0, 0.0]
        watching = threading.Event()

        def watch() -> None:
            while not watching.wait(0.5):
                now = server_threads(server.pid)
                if now:
                    peak[0], peak[1] = max(peak[0], now[0]), max(peak[1], now[1])

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        before, why_not = statements(args.database_url) if args.statements else (None, None)
        sampler = sample_round_trips(args.database_url, relayed) if relayer else None
        relay0 = process_cpu(relayer.pid) if relayer else None
        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        cpu1 = (process_cpu(server.pid), own_cpu(), machine_cpu())
        relay1 = process_cpu(relayer.pid) if relayer else None
        after, why_after = statements(args.database_url) if before is not None else (None, None)
        ran = ran_between(before, after) if before is not None and after is not None else None
        sampled = sampler.stop() if sampler else None
        lag = lag_between(lag0, loop_lag(base))
        watching.set()
        watcher.join()
        if profiler is not None:
            for _ in range(200):               # the profile may not have begun yet
                if httpx.delete(f"{base}/v1/admin/profile", headers=op).status_code == 204 \
                        or not profiler.is_alive():
                    break
                time.sleep(0.05)
            profiler.join()
            profiled["response"].raise_for_status()
            Path(args.profile).write_text(profiled["response"].text)
        ok = statuses.get(200, 0)
        spent = [None if a is None or b is None else (b - a) for a, b in zip(cpu0, cpu1)]
        cpu_per_turn = None
        if ok and spent[0] is not None and spent[2] is not None:
            server_ms, client_ms, machine_ms = (x * 1000 / ok for x in spent)
            relay_ms = ((relay1 - relay0) * 1000 / ok
                        if relay0 is not None and relay1 is not None else 0.0)
            cpu_per_turn = {"server": round(server_ms, 1), "load_client": round(client_ms, 1),
                            "machine": round(machine_ms, 1),
                            **({"relay": round(relay_ms, 1)} if relayer else {}),
                            "rest": round(machine_ms - server_ms - client_ms - relay_ms, 1)}

        engine = make_engine(args.database_url)
        with engine.connect() as conn:
            rows = conn.execute(select(S.tool_calls.c.latency_ms, S.tool_calls.c.is_write,
                                       S.tool_calls.c.attempts)
                                .where(S.tool_calls.c.tenant_id.in_([t for t, _ in tenants]),
                                       S.tool_calls.c.dispatched.is_(True))).all()
        engine.dispose()
        reads = [r.latency_ms for r in rows if not r.is_write]
        writes = [r.latency_ms for r in rows if r.is_write]
        reports = [httpx.get(f"{base}/v1/admin/tenants/{t}/replay", headers=op,
                             timeout=1800).json() for t, _ in tenants]
        replay = {"calls": sum(r["calls"] for r in reports),
                  "identical": sum(r["identical"] for r in reports),
                  "world_identical": all(r["world_identical"] for r in reports),
                  "ok": all(r["ok"] for r in reports),
                  "seconds": round(sum(r["seconds"] for r in reports), 3)}
        db = args.database_url.split(":", 1)[0].split("+")[0]
        out = {
            "database": db, "turns": sum(statuses.values()), "ok": statuses.get(200, 0),
            "concurrency": args.concurrency, "customers": 1 if args.hot else
            min(len(customers), args.concurrency), "hot": args.hot,
            "mock_latency_ms": args.mock_latency_ms, "mock_jitter": args.mock_jitter,
            "ramp_s": args.ramp_s,
            "server_processes": args.workers,
            "tenants": args.tenants, "provider_concurrency": args.provider_concurrency,
            "cpus": os.cpu_count(), "seconds": round(wall, 2),
            "turns_per_s": round(statuses.get(200, 0) / wall, 1),
            "turn_ms": {f"p{p}": round(pct(latencies, p) * 1000, 1) for p in (50, 95, 99)},
            "turn_ms_mean": round(statistics.mean(latencies) * 1000, 1) if latencies else None,
            "server_turn_ms": {f"p{p}": round(pct(served, p) * 1000, 1) for p in (50, 95, 99)},
            "cpu_ms_per_turn": cpu_per_turn,
            "event_loop_lag": lag,
            "server_threads_peak": peak[0] or None,
            "server_rss_mb_peak": round(peak[1], 1) or None,
            "statuses": dict(statuses), "stops": dict(stops),
            "tool_calls": len(rows), "writes": len(writes),
            "read_ms": {f"p{p}": pct(reads, p) for p in (50, 99)},
            "write_ms": {f"p{p}": pct(writes, p) for p in (50, 99)},
            "conflict_retries": sum(r.attempts - 1 for r in rows),
            "replay": replay,
            "statements_per_turn": (per_turn(ran, ok) if ran and ok else
                                    ({"why_not": why_not or why_after} if args.statements
                                     else None)),
            "db_round_trip": {**rtt, "during_run": sampled} if rtt else None,
            "root": str(root),
        }
        print(json.dumps(out, indent=2))
        if args.profile:
            from pasarbench.service.profiling import merge, summarize
            print("\n" + summarize(merge([profiled["response"].json()]), turns=ok))
        who = "hot: 1 customer" if args.hot else f"{out['customers']} customers"
        print(f"\n| {db} | {args.workers} | {args.tenants} | {who} "
              f"| {args.concurrency} | {args.mock_latency_ms} ms | {out['turns_per_s']} "
              f"| {out['turn_ms']['p50']} / {out['turn_ms']['p95']} / {out['turn_ms']['p99']} "
              f"| {out['write_ms']['p50']} / {out['write_ms']['p99']} "
              f"| {'identical' if replay.get('ok') else 'MISMATCH'} ({replay.get('calls')}) |")
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=2))
        return 0 if replay.get("ok") and statuses.get(200, 0) == sum(statuses.values()) else 1
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
        if relayer is not None:
            relayer.kill()
            relayer.wait()
        errlog.close()
        os.unlink(errlog.name)


if __name__ == "__main__":
    raise SystemExit(main())
