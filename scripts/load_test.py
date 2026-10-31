"""
Load test for the agent service: a real server (uvicorn, its own process),
real HTTP, concurrent customers, the mock model -- so what is measured is the
service itself, then the service under a model's latency.

    python scripts/load_test.py --database-url sqlite:///data/load.db
    python scripts/load_test.py --database-url postgresql+psycopg://user:pw@host/db \\
        --turns 2000 --concurrency 32 --customers 200
    python scripts/load_test.py ... --hot            every worker on one customer
    python scripts/load_test.py ... --mock-latency-ms 400

The database must be one the test may fill (it migrates it, and makes a tenant
of its own each run). Each worker opens a conversation for its customer and
sends turns: half ask where an order is (reads), half ask for a voucher or a
human (writes: an insert under the customer's lock, and an outbox ticket),
each with the customer's digits. A conversation is replaced every 10 turns.

It reports turns a second and the latency a client sees, p50/p95/p99, every
status that was not 200, the time the tool calls spent in their transactions
(from the audit log), the CPU a turn cost -- the server's, this client's, and
the whole machine's, the database's among the rest (Linux: /proc) -- and then
replays the tenant: every tool call the run made, run again against the
rebuilt world, must answer and change the same.

    python scripts/load_test.py ... --profile p.json

profiles the server while it serves the turns (GET /v1/admin/profile; one
server process, --workers 1, for all of it) and prints where its CPU went,
per turn (profiling.py).
"""

from __future__ import annotations

import argparse
import json
import os
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--turns", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--customers", type=int, default=50, help="synthetic customers (x6)")
    ap.add_argument("--hot", action="store_true", help="every worker on one customer")
    ap.add_argument("--mock-latency-ms", type=int, default=0)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--workers", type=int, default=1, help="server processes (uvicorn)")
    ap.add_argument("--tenants", type=int, default=1,
                    help="spread the workers over this many tenants (each its own id counter)")
    ap.add_argument("--provider-concurrency", type=int, default=16,
                    help="model calls in flight per provider, per server process")
    ap.add_argument("--json", help="write the result here too")
    ap.add_argument("--profile", help="profile the server while it serves the turns: the "
                                      "profile's JSON here, its summary on stdout")
    ap.add_argument("--profile-interval-ms", type=float, default=10)
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
               PASAR_MOCK_LATENCY_MS=str(args.mock_latency_ms), PASAR_RATE="100000",
               PASAR_BURST="100000", PASAR_THREADS=str(args.threads),
               PASAR_MAX_CONCURRENT_TURNS=str(min(args.threads, max(args.concurrency, 1))),
               PASAR_PROVIDER_CONCURRENCY=str(args.provider_concurrency),
               PASAR_LOG_LEVEL="WARNING", PYTHONDONTWRITEBYTECODE="1")
    py = sys.executable
    subprocess.run([py, "-m", "pasarbench.service", "migrate"], env=env, cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    import tempfile
    errlog = tempfile.NamedTemporaryFile(prefix="pasar-load-", suffix=".log", delete=False)
    server = subprocess.Popen([py, "-m", "pasarbench.service", "serve", "--port", str(port),
                               "--workers", str(args.workers)],
                              env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=errlog)
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
        statuses: Counter = Counter()
        stops: Counter = Counter()
        lock = threading.Lock()
        next_turn = [0]

        def worker(w: int) -> None:
            user, digits, orders = customers[0 if args.hot else w % len(customers)]
            h = tenants[w % len(tenants)][1]
            with httpx.Client(base_url=base, headers=h, timeout=300) as c:
                cid, used = None, 0
                while True:
                    with lock:
                        n = next_turn[0]
                        if n >= args.turns:
                            return
                        next_turn[0] += 1
                    if cid is None or used >= 10:
                        cid = c.post("/v1/conversations", json={"user_id": user}).json()[
                            "conversation_id"]
                        used = 0
                    order = orders[n % len(orders)]
                    text = [f"where is my order {order}?",
                            f"can I get a voucher for the delay on {order}? digits {digits}",
                            f"track {order} please",
                            f"I want a human agent for {order}, my digits are {digits}"][n % 4]
                    t0 = time.perf_counter()
                    resp = c.post(f"/v1/conversations/{cid}/messages", json={"content": text})
                    dt = time.perf_counter() - t0
                    used += 1
                    with lock:
                        statuses[resp.status_code] += 1
                        if resp.status_code == 200:
                            latencies.append(dt)
                            stops[resp.json()["stop_reason"]] += 1

        profiled: dict[str, Any] = {}

        def profile() -> None:
            profiled["response"] = httpx.get(
                f"{base}/v1/admin/profile", headers=op, timeout=PROFILE_SECONDS + 120,
                params={"seconds": PROFILE_SECONDS, "interval_ms": args.profile_interval_ms})

        profiler = threading.Thread(target=profile) if args.profile else None
        if profiler is not None:
            profiler.start()
        cpu0 = (process_cpu(server.pid), own_cpu(), machine_cpu())
        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        cpu1 = (process_cpu(server.pid), own_cpu(), machine_cpu())
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
            cpu_per_turn = {"server": round(server_ms, 1), "load_client": round(client_ms, 1),
                            "machine": round(machine_ms, 1),
                            "rest": round(machine_ms - server_ms - client_ms, 1)}

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
            "mock_latency_ms": args.mock_latency_ms, "server_processes": args.workers,
            "tenants": args.tenants, "provider_concurrency": args.provider_concurrency,
            "cpus": os.cpu_count(), "seconds": round(wall, 2),
            "turns_per_s": round(statuses.get(200, 0) / wall, 1),
            "turn_ms": {f"p{p}": round(pct(latencies, p) * 1000, 1) for p in (50, 95, 99)},
            "turn_ms_mean": round(statistics.mean(latencies) * 1000, 1) if latencies else None,
            "cpu_ms_per_turn": cpu_per_turn,
            "statuses": dict(statuses), "stops": dict(stops),
            "tool_calls": len(rows), "writes": len(writes),
            "read_ms": {f"p{p}": pct(reads, p) for p in (50, 99)},
            "write_ms": {f"p{p}": pct(writes, p) for p in (50, 99)},
            "conflict_retries": sum(r.attempts - 1 for r in rows),
            "replay": replay,
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
        errlog.close()
        os.unlink(errlog.name)


if __name__ == "__main__":
    raise SystemExit(main())
