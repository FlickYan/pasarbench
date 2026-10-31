"""
Do the service's limits hold for a deployment, not just a process? (v82)

Two servers on one PostgreSQL database -- two processes, as two replicas
behind a load balancer are, each client talking to one of them -- in front
of a stand-in provider: an OpenAI-compatible server on this machine that
answers as the mock agent does, after a model's latency, and counts what it
is sent. Each part runs twice: with PASAR_LIMITS=process, every limit a
process's own as before v82, and with deployment.

    provider    32 clients, 16 a server; the provider takes 8 calls at once
                and refuses (429) past that, as a provider's concurrency limit
                does, and each server is told 8 (PASAR_PROVIDER_CONCURRENCY):
                the most the two servers had in flight together (sampled every
                0.1 s), what the provider refused, the turns that got through
    tenant      a tenant allowed 5 requests a second, bursts of 5, asking as
                fast as 8 clients can for 10 s: what the deployment admitted
    outage      the provider fails every call (503) for 45 s, 14 clients on
                one server and 1 on the other, 16 calls at once allowed: the
                calls it was sent from each, and how soon the light server's
                turns failed fast
    membership  (deployment) the second server killed (SIGKILL), started
                again, stopped (SIGTERM): how long the first takes to count it

    python scripts/limits_check.py --database-url postgresql+psycopg://... \\
        [--parts provider,tenant,outage,membership] [--json out.json]

The database must be one the check may fill: it migrates it and makes
tenants of its own. The stand-in is local, so the servers send it "EMPTY" for
a key (models.py), and they are started with no variable that holds one.

(No `from __future__ import annotations`: the stand-in's handlers are
FastAPI's, which reads their annotations.)
"""

import argparse
import asyncio
import json
import os
import random
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SECRET = ("_API_KEY", "_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


# ---------------------------------------------------------------------------
# The stand-in provider
# ---------------------------------------------------------------------------

def standin(port: int, latency_ms: float, jitter: float, max_concurrency: int) -> None:
    """An OpenAI-compatible chat endpoint at /<lane>/v1 -- one lane a server,
    so that each one's calls are counted apart -- answering as the mock agent
    would, after `latency_ms` (strayed by `jitter`), refusing with 429 past
    `max_concurrency` calls at once, or failing every call with 503 when told
    to (POST /control {"mode": "down"})."""
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from pasarbench.harness.types import Message, ToolCall
    from pasarbench.service.models import MockAgent

    agent = MockAgent()
    st: dict[str, Any] = {}

    def reset(mode: str = "up") -> None:
        st.update(mode=mode, in_flight=0, peak=0, calls=Counter(), refused=Counter(),
                  failed=Counter(), answered=Counter(), first=None, last=None)

    reset()
    app = FastAPI()

    def wire_to_messages(wire: list[dict]) -> list[Message]:
        out = []
        for m in wire:
            if m.get("role") == "tool":
                out.append(Message(role="tool", content=m.get("content") or "",
                                   tool_call_id=m.get("tool_call_id"), name=m.get("name")))
                continue
            calls = [ToolCall(name=tc["function"]["name"],
                              arguments=json.loads(tc["function"].get("arguments") or "{}"),
                              id=tc["id"]) for tc in m.get("tool_calls") or []]
            out.append(Message(role=m["role"], content=m.get("content") or "",
                               tool_calls=calls))
        return out

    @app.post("/{lane}/v1/chat/completions")
    async def chat(lane: str, request: Request) -> Any:
        body = await request.json()
        now = time.monotonic()
        st["calls"][lane] += 1
        st["first"] = st["first"] or now
        st["last"] = now
        if st["mode"] == "down":
            st["failed"][lane] += 1
            return JSONResponse({"error": {"message": "the stand-in is down",
                                           "type": "server_error"}}, status_code=503)
        if st["in_flight"] >= max_concurrency:
            st["refused"][lane] += 1
            return JSONResponse({"error": {"message": "too many requests at once",
                                           "type": "rate_limit"}}, status_code=429,
                                headers={"Retry-After": "1"})
        st["in_flight"] += 1
        st["peak"] = max(st["peak"], st["in_flight"])
        try:
            await asyncio.sleep(latency_ms / 1000 * random.uniform(1 - jitter, 1 + jitter))
            resp = agent.respond(wire_to_messages(body.get("messages") or []),
                                 body.get("tools") or [])
        finally:
            st["in_flight"] -= 1
        st["answered"][lane] += 1
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content or None}
        if resp.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.name,
                                               "arguments": json.dumps(tc.arguments)}}
                                 for tc in resp.tool_calls]
        return {"id": f"standin-{secrets.token_hex(4)}", "object": "chat.completion",
                "model": "standin-1",
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": "tool_calls" if resp.tool_calls else "stop"}],
                "usage": {"prompt_tokens": resp.usage.prompt_tokens,
                          "completion_tokens": resp.usage.completion_tokens}}

    @app.post("/control")
    async def control(request: Request) -> Any:
        reset((await request.json()).get("mode", "up"))
        return {"mode": st["mode"]}

    @app.get("/stats")
    async def stats() -> Any:
        return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in st.items()}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Servers
# ---------------------------------------------------------------------------

def clean_env() -> dict[str, str]:
    """This process's environment without anything that holds a key."""
    return {k: v for k, v in os.environ.items()
            if not k.upper().endswith(SECRET) and not k.upper().startswith("HF_")}


class Server:
    def __init__(self, lane: str, env: dict[str, str], workdir: Path, standin_port: int):
        self.lane, self.port = lane, free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        models = workdir / f"models-{lane}.json"
        models.write_text(json.dumps([{
            "name": "standin", "provider": "standin", "model": "standin-1", "api": "chat",
            "base_url": f"http://127.0.0.1:{standin_port}/{lane}/v1", "temperature": 0.0}]))
        self.env = {**env, "PASAR_MODELS_FILE": str(models), "PASAR_MODELS": "standin",
                    "PASAR_DEFAULT_MODEL": "standin"}
        self.log = workdir / f"server-{lane}.log"
        self.proc: subprocess.Popen | None = None

    def start(self) -> "Server":
        import httpx
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "pasarbench.service", "serve", "--port", str(self.port)],
            env=self.env, cwd=ROOT, stdout=subprocess.DEVNULL,
            stderr=open(self.log, "a"))
        for _ in range(150):
            try:
                if httpx.get(f"{self.base}/readyz", timeout=1).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        raise SystemExit(f"server {self.lane} did not come up: "
                         + self.log.read_text(errors="replace")[-3000:])

    def stop(self, sig: int = signal.SIGTERM) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.send_signal(sig)
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(10)


def forget_limits(url: str) -> None:
    """Each part from nothing shared: no bucket and no breaker left open by the
    part before (a part's servers stop gracefully, so their rows go with them)."""
    from sqlalchemy import text

    from pasarbench.service.store import make_engine, transaction
    engine = make_engine(url)
    with transaction(engine, write=True) as conn:
        conn.execute(text("DELETE FROM rate_limits"))
        conn.execute(text("DELETE FROM provider_breakers"))
    engine.dispose()


class Rig:
    """The stand-in and a pair of servers, started for one part, in one mode."""

    def __init__(self, args: argparse.Namespace, workdir: Path, standin_port: int):
        self.args, self.workdir, self.standin_port = args, workdir, standin_port
        self.token = secrets.token_urlsafe(16)
        self.op = {"Authorization": f"Bearer {self.token}"}

    def env(self, mode: str, **extra: str) -> dict[str, str]:
        env = clean_env()
        env.update(PASAR_DATABASE_URL=self.args.database_url, PASAR_ADMIN_TOKEN=self.token,
                   PASAR_LIMITS=mode, PASAR_PROVIDER_CONCURRENCY="8",
                   PASAR_PROVIDER_ACQUIRE_TIMEOUT="30", PASAR_RATE="100000",
                   PASAR_BURST="100000", PASAR_MAX_CONCURRENT_TURNS="256",
                   PASAR_LOG_LEVEL="WARNING", PYTHONDONTWRITEBYTECODE="1")
        env.update(extra)
        return env

    def servers(self, mode: str, **extra: str) -> list[Server]:
        forget_limits(self.args.database_url)
        env = self.env(mode, **extra)
        return [Server(lane, env, self.workdir, self.standin_port).start() for lane in "ab"]

    def provider(self, mode: str) -> None:
        import httpx
        httpx.post(f"http://127.0.0.1:{self.standin_port}/control", json={"mode": mode},
                   timeout=5).raise_for_status()

    def provider_stats(self) -> dict[str, Any]:
        import httpx
        return httpx.get(f"http://127.0.0.1:{self.standin_port}/stats", timeout=5).json()

    def tenant(self, server: Server, customers: int) -> tuple[dict[str, str], list]:
        import httpx

        from pasarbench.db import Database
        tid = f"limits-{secrets.token_hex(3)}"
        r = httpx.post(f"{server.base}/v1/admin/tenants", headers=self.op, timeout=300,
                       json={"tenant_id": tid, "name": "limits check",
                             "world": f"synthetic:{customers}"})
        r.raise_for_status()
        seed = Database.fresh().tables
        people = [(u, d["verified_phone_last4"],
                   [o for o, x in seed["orders"].items() if x["user_id"] == u])
                  for u, d in seed["users"].items()]
        people = [p for p in people if p[2]]
        who = [(f"{u}-C{i:04d}", digits, [f"{o}-C{i:04d}" for o in orders])
               for i in range(1, customers + 1) for u, digits, orders in people]
        return {"Authorization": f"Bearer {r.json()['api_key']}"}, who


def turns(server: Server, headers: dict, customer: tuple, until: float,
          out: list, lock: threading.Lock) -> None:
    """One client's turns against one server until `until`: (lane, start,
    seconds, status, stop reason) each."""
    import httpx
    user, digits, orders = customer
    n = 0
    with httpx.Client(base_url=server.base, headers=headers, timeout=120) as c:
        cid = None
        while time.monotonic() < until:
            t0 = time.monotonic()
            try:
                if cid is None or n % 10 == 0:
                    cid = c.post("/v1/conversations", json={"user_id": user}).json()[
                        "conversation_id"]
                order = orders[n % len(orders)]
                text = [f"where is my order {order}?",
                        f"can I get a voucher for the delay on {order}? digits {digits}",
                        f"track {order} please",
                        f"I want a human agent for {order}, my digits are {digits}"][n % 4]
                r = c.post(f"/v1/conversations/{cid}/messages", json={"content": text})
                stop = r.json().get("stop_reason") if r.status_code == 200 else None
                status: Any = r.status_code
            except (httpx.HTTPError, KeyError, ValueError) as e:
                status, stop, cid = f"transport: {type(e).__name__}", None, None
            n += 1
            with lock:
                out.append((server.lane, t0, time.monotonic() - t0, status, stop))


def in_flight_sampler(servers: list[Server], op: dict, stop: threading.Event,
                      out: list) -> None:
    """The two servers' calls in flight to the provider, together, every 0.1 s."""
    import httpx
    with httpx.Client(timeout=5) as c:
        while not stop.is_set():
            total = 0
            for s in servers:
                try:
                    g = c.get(f"{s.base}/v1/admin/gates", headers=op).json()["providers"]
                    total += sum(p["in_flight"] for p in g.values())
                except (httpx.HTTPError, KeyError, ValueError):
                    pass
            out.append(total)
            stop.wait(0.1)


# ---------------------------------------------------------------------------
# The parts
# ---------------------------------------------------------------------------

def part_provider(rig: Rig, mode: str, seconds: float) -> dict[str, Any]:
    servers = rig.servers(mode)
    try:
        h, who = rig.tenant(servers[0], 10)
        marks = [s.log.stat().st_size for s in servers]
        rig.provider("up")
        out: list = []
        lock = threading.Lock()
        until = time.monotonic() + seconds
        clients = [threading.Thread(target=turns, args=(servers[i % 2], h, who[i], until,
                                                        out, lock)) for i in range(32)]
        stop, sampled = threading.Event(), []
        sampler = threading.Thread(target=in_flight_sampler, args=(servers, rig.op, stop,
                                                                   sampled))
        sampler.start()
        for t in clients:
            t.start()
        for t in clients:
            t.join()
        stop.set()
        sampler.join()
        prov = rig.provider_stats()
        ok = [x for x in out if x[3] == 200 and x[4] == "replied"]
        stops = Counter(x[4] or str(x[3]) for x in out)
        sent = sum(prov["calls"].values())
        refused = sum(prov["refused"].values())
        opened = sum(len(logged(s, m, "breaker opened here")) for s, m in zip(servers, marks))
        return {"mode": mode, "seconds": seconds, "clients": 32,
                "breaker_openings": opened,
                "in_flight_peak": max(sampled or [0]),
                "in_flight_p95": pct(sampled, 95),
                "provider_calls": sent, "provider_refused": refused,
                "refused_share": round(refused / sent, 3) if sent else None,
                "provider_peak": prov["peak"],
                "turns": len(out), "turns_answered": len(ok),
                "answered_per_s": round(len(ok) / seconds, 2),
                "turn_p50_s": pct([x[2] for x in ok], 50),
                "turn_p95_s": pct([x[2] for x in ok], 95),
                "stops": dict(stops)}
    finally:
        for s in servers:
            s.stop()


def part_tenant(rig: Rig, mode: str, seconds: float) -> dict[str, Any]:
    import httpx
    servers = rig.servers(mode, PASAR_RATE="5", PASAR_BURST="5")
    try:
        h, _who = rig.tenant(servers[0], 1)
        codes: Counter = Counter()
        lock = threading.Lock()
        until = time.monotonic() + seconds

        def hammer(server: Server) -> None:
            with httpx.Client(base_url=server.base, headers=h, timeout=30) as c:
                while time.monotonic() < until:
                    r = c.get("/v1/conversations", params={"limit": 1})
                    with lock:
                        codes[(server.lane, r.status_code)] += 1

        ts = [threading.Thread(target=hammer, args=(servers[i % 2],)) for i in range(8)]
        t0 = time.monotonic()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        took = time.monotonic() - t0
        admitted = sum(n for (_l, code), n in codes.items() if code == 200)
        return {"mode": mode, "seconds": round(took, 2), "rate": 5, "burst": 5,
                "requests": sum(codes.values()), "admitted": admitted,
                "admitted_per_s": round(admitted / took, 2),
                "allowed": round(5 + 5 * took, 1),
                "by_server": {lane: {str(code): n for (l2, code), n in codes.items()
                                     if l2 == lane} for lane in "ab"}}
    finally:
        for s in servers:
            s.stop()


def logged(server: Server, since: int, needle: str) -> list[float]:
    """When (epoch seconds) the server logged a line holding `needle`, since
    byte `since` of its log."""
    from datetime import datetime, timezone
    out = []
    with open(server.log, "rb") as f:
        f.seek(since)
        for line in f.read().decode(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if needle in rec.get("msg", ""):
                out.append(datetime.strptime(rec["ts"], "%Y-%m-%dT%H:%M:%S.%fZ")
                           .replace(tzinfo=timezone.utc).timestamp())
    return out


def part_outage(rig: Rig, mode: str, seconds: float) -> dict[str, Any]:
    servers = rig.servers(mode, PASAR_PROVIDER_CONCURRENCY="16")
    try:
        h, who = rig.tenant(servers[0], 4)
        marks = [s.log.stat().st_size for s in servers]
        rig.provider("down")
        wall = time.time()
        out: list = []
        lock = threading.Lock()
        start = time.monotonic()
        until = start + seconds
        lanes = [servers[0]] * 14 + [servers[1]]
        clients = [threading.Thread(target=turns, args=(s, h, who[i], until, out, lock))
                   for i, s in enumerate(lanes)]
        for t in clients:
            t.start()
        for t in clients:
            t.join()
        prov = rig.provider_stats()
        light = sorted((x for x in out if x[0] == "b"), key=lambda x: x[1])
        fast_from = next((round(x[1] - start, 2) for x in light
                          if x[4] == "provider_unavailable" and x[2] < 0.2), None)
        first10 = [x[2] for x in light if x[1] - start < 10]
        own = [logged(s, m, "breaker opened here") for s, m in zip(servers, marks)]
        held = [logged(s, m, "its breaker opened in") for s, m in zip(servers, marks)]
        first_open = min((t for ts in own for t in ts), default=None)
        heard = [round(ts[0] - first_open, 3) if ts and first_open else None for ts in held]
        return {"mode": mode, "seconds": seconds,
                "breaker_openings": {s.lane: {"own": [round(t - wall, 2) for t in o],
                                              "held_for_the_other": [round(t - wall, 2)
                                                                     for t in hd]}
                                     for s, o, hd in zip(servers, own, held)},
                "light_server_heard_after_s": heard[1],
                "provider_calls": prov["calls"], "provider_calls_total":
                    sum(prov["calls"].values()),
                "light_server_turns": len(light),
                "light_server_fails_fast_from_s": fast_from,
                "light_server_first_10s_turn_s": (round(statistics.mean(first10), 2)
                                                  if first10 else None),
                "stops": {lane: dict(Counter(x[4] or str(x[3]) for x in out if x[0] == lane))
                          for lane in "ab"}}
    finally:
        rig.provider("up")
        for s in servers:
            s.stop()


def part_membership(rig: Rig) -> dict[str, Any]:
    import httpx
    servers = rig.servers("deployment")
    a, b = servers

    def seen() -> tuple[int | None, int | None]:
        try:
            v = httpx.get(f"{a.base}/v1/admin/nodes", headers=rig.op, timeout=5).json()
            return v["processes"], v["provider_concurrency"]["this_process"]
        except (httpx.HTTPError, KeyError, ValueError):
            return None, None

    def wait_for(processes: int, limit: float = 40.0) -> float | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit:
            if seen()[0] == processes:
                return round(time.monotonic() - t0, 2)
            time.sleep(0.1)
        return None

    try:
        out: dict[str, Any] = {"heartbeat_seconds": 5, "both": seen()}
        b.stop(signal.SIGKILL)
        out["killed_counted_out_s"] = wait_for(1)
        out["alone"] = seen()
        t0 = time.monotonic()
        b.start()
        out["started_counted_s"] = round(time.monotonic() - t0, 2)
        out["started_counted_in_s_after_ready"] = wait_for(2)
        out["together_again"] = seen()
        b.stop(signal.SIGTERM)
        out["stopped_counted_out_s"] = wait_for(1)
        return out
    finally:
        for s in servers:
            s.stop()


def priced() -> None:
    """The stand-in's row of the price book -- free: a conversation is never
    run on a model the service cannot charge for."""
    from sqlalchemy import select

    from pasarbench.service import schema as S
    from pasarbench.service.store import make_engine, transaction
    engine = make_engine(ARGS.database_url)
    p = S.price_book
    with transaction(engine, write=True) as conn:
        if conn.execute(select(p.c.price_id).where(p.c.provider == "standin")).first() is None:
            conn.execute(p.insert().values(
                price_id="standin:standin-1:2026-01-01", provider="standin", model="standin-1",
                input_microusd_per_mtok=0, cached_microusd_per_mtok=0,
                output_microusd_per_mtok=0, effective_from="2026-01-01",
                source="scripts/limits_check.py: the stand-in provider, free"))
    engine.dispose()


ARGS = argparse.Namespace()


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--database-url", required=False)
    ap.add_argument("--parts", default="provider,tenant,outage,membership")
    ap.add_argument("--modes", default="process,deployment")
    ap.add_argument("--provider-seconds", type=float, default=30)
    ap.add_argument("--outage-seconds", type=float, default=45)
    ap.add_argument("--latency-ms", type=float, default=400)
    ap.add_argument("--json")
    ap.add_argument("--standin", type=int, help=argparse.SUPPRESS)   # the stand-in's port
    args = ARGS = ap.parse_args()
    if args.standin:
        standin(args.standin, args.latency_ms, 0.3, 8)
        return 0
    if not (args.database_url or "").startswith("postgresql"):
        raise SystemExit("--database-url: a PostgreSQL database the check may fill")
    env = clean_env()
    env.update(PASAR_DATABASE_URL=args.database_url, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-m", "pasarbench.service", "migrate"], env=env, cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    priced()
    workdir = Path(tempfile.mkdtemp(prefix="pasar-limits-"))
    port = free_port()
    prov = subprocess.Popen([sys.executable, __file__, "--standin", str(port), "--latency-ms",
                             str(args.latency_ms)], env=env, cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=open(workdir / "standin.log", "a"))
    import httpx
    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/stats", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.1)
    rig = Rig(args, workdir, port)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    result: dict[str, Any] = {"parts": {}}
    try:
        for part in parts:
            if part == "membership":
                result["parts"][part] = part_membership(rig)
                print(json.dumps({part: result["parts"][part]}), flush=True)
                continue
            for mode in modes:
                fn = {"provider": lambda m: part_provider(rig, m, args.provider_seconds),
                      "tenant": lambda m: part_tenant(rig, m, 10.0),
                      "outage": lambda m: part_outage(rig, m, args.outage_seconds)}[part]
                got = fn(mode)
                result["parts"].setdefault(part, {})[mode] = got
                print(json.dumps({part: {mode: got}}), flush=True)
    finally:
        prov.terminate()
        prov.wait(10)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
