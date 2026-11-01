"""
Limits a deployment shares (v82).

Until v82 every limit was a process's own (resilience.py), and a deployment
of N processes -- `serve --workers N`, or N replicas behind a load balancer
-- allowed N times each: a tenant N times its rate, a provider N times the
calls it takes at once, and N breakers that each had to see the provider
fail five times before it stopped calling. With PASAR_LIMITS=deployment --
the default on PostgreSQL -- the processes that share a database share their
limits through it:

    who is serving    each process writes its row of service_nodes as it
                      starts and every PASAR_HEARTBEAT_SECONDS (5); the rows
                      written in the last three heartbeats are the
                      deployment. A process that shuts down deletes its row;
                      one that dies stops being counted 15 seconds on.
    provider calls    each process takes its share of
                      PASAR_PROVIDER_CONCURRENCY -- 16 over three processes
                      is 6, 5 and 5 -- and no more, with no statement a call:
                      the shares move when the processes do.
    tenant requests   one bucket a tenant, in the database (rate_limits),
                      counted in requests: a request is one statement, which
                      admits it or says how long until it would be.
    breaker openings  a process whose breaker opens writes until when
                      (provider_breakers); every other opens its own until
                      then, and a call stops retrying a provider whose
                      breaker is open (resilience.py).

On PostgreSQL a process hears of a join, a leave or an opening at once
(LISTEN/NOTIFY); elsewhere, or behind a pooler that drops notifications, at
its next heartbeat. The three tables are UNLOGGED there: none of it is worth
a disk write a request, and after a crash it starts again empty -- every
bucket full, every process back at its next heartbeat, every breaker closed.

What stays a process's own is what one process can hold: the turns it runs
at once (PASAR_MAX_CONCURRENT_TURNS) and its threads. PASAR_LIMITS=process
makes every limit a process's own again, as before v82 -- the default on
SQLite, one machine's file, where a request's check would be a write
transaction waiting on the database's one write lock behind any long write
(WHAT_FAILED #71); `serve --workers N` there shares them all the same, and
PASAR_LIMITS=deployment does for other processes on that one machine.

A database that cannot be asked -- not migrated to 0003, or failing --
leaves each limit this process's own, as before v82, and says so (the log,
pasar_limits_errors_total); a process stopping never waits more than 5 s for
it to answer.

What sharing does not do (docs/SERVICE.md, v82): a process's share is fixed
by the count, not by its load -- a process with more turns waiting than its
share does not borrow another's idle slots -- and while the count changes the
deployment can exceed its limit by a process's old share, for as long as the
calls begun under it take.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from collections import Counter
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import __version__
from .resilience import Gates, TenantLimits
from .store import deadline, is_sqlite, transaction

log = logging.getLogger("pasar.limits")

CHANNEL = "pasar_limits"        # LISTEN/NOTIFY: a process joined, left, or opened a breaker
EXPIRY = 3                      # heartbeats a process may miss before it is not counted
FORGET = 3600.0                 # seconds after which a dead process's row is deleted
LOCAL_SECONDS = 5.0             # after a failed statement, counted here alone this long
RATE_SECONDS = 2.0              # the longest a request's check waits on the database (v83)
STOP_SECONDS = 5.0              # the longest a process stopping waits for the database


def split(total: int, n: int, rank: int) -> int:
    """The share of `total` that the `rank`-th of `n` processes takes: as
    even as whole numbers allow (16 over three is 6, 5 and 5), and never less
    than one -- a process with none could never call. Past `total` processes
    each still takes one: the deployment then allows more than `total`. (A
    total of 0 is 0: no calls, as an operator set it.)"""
    if total <= 0:
        return 0
    if n <= 1:
        return total
    base, extra = divmod(total, n)
    return max(1, base + (1 if rank < extra else 0))


def _clock(engine: Engine) -> str:
    """The database's clock, in seconds since the epoch: one clock for every
    process, wherever it runs. PostgreSQL's now() is the transaction's start --
    in autocommit, the statement's."""
    if is_sqlite(engine):
        return "((julianday('now') - 2440587.5) * 86400.0)"
    return "CAST(extract(epoch FROM now()) AS double precision)"


def processes(engine: Engine) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The processes serving on this database, as their rows say -- which
    are counted (not past the time each gave), how long each has been up,
    when it was last heard of -- and the breakers open, on the database's
    clock: what GET /v1/admin/nodes and `python -m pasarbench.service nodes`
    show."""
    now_sql = _clock(engine)
    with transaction(engine, write=False) as conn:
        now = float(conn.execute(text(f"SELECT {now_sql}")).scalar())
        rows = conn.execute(text(
            "SELECT node, host, pid, version, started_at, seen_at, expires_at "
            "FROM service_nodes ORDER BY node")).all()
        opened = conn.execute(text(
            "SELECT provider, open_until - :now, opened_by FROM provider_breakers "
            "WHERE open_until > :now ORDER BY provider"), {"now": now}).all()
    return ([{"node": r[0], "host": r[1], "pid": r[2], "version": r[3],
              "up_s": round(now - r[4], 1), "seen_s_ago": round(now - r[5], 1),
              "counted": r[6] > now} for r in rows],
            [{"provider": p, "for_s": round(float(left), 1), "opened_by": by}
             for p, left, by in opened])


def _hint(e: Exception) -> str:
    """What a failed statement may mean, when it says so: one of 0003's
    tables missing (and not, say, PgBouncer's prepared statement)."""
    said = str(e)
    if any(t in said for t in ("rate_limits", "service_nodes", "provider_breakers")) and (
            "does not exist" in said or "no such table" in said):
        return " -- is the database migrated to 0003 (python -m pasarbench.service migrate)?"
    return ""


class SharedRate:
    """A tenant's requests, counted for the whole deployment: `rate` a second
    on average and `burst` at once, as resilience.TokenBucket counts them for
    one process.

    The bucket holds requests: a row keeps how many (`level`) and when that
    was written (`at`), and it drains at `rate` a second. A request is
    admitted while the bucket, drained to now, holds at most burst - 1, and
    adds one -- GCRA, in requests. One statement admits or refuses, on the
    database's clock; a refusal reads what the bucket holds, which says how
    long until a request would be admitted. In requests, not in seconds, so
    that processes with other settings -- during a rolling change of
    PASAR_RATE or PASAR_BURST -- agree on what a bucket holds: each admission
    adds one whoever makes it, and the bucket drains at the rate of whichever
    process writes it next, so the deployment admits no more than the larger
    burst and the faster rate allow together -- the looser setting's limit
    when one of them changes; and a new setting takes effect at once on a
    bucket filled under the old one. On PostgreSQL the statement goes on a pooled
    connection in autocommit, through psycopg itself: through SQLAlchemy a
    statement of this shape cost twice the CPU (208 µs against 104, measured
    on this machine).

    A refusal is remembered until the time it gives, and the requests until
    then refused here with no statement: the bucket only drains with time, so
    a process with these settings could admit nothing sooner -- and a client
    sending as fast as it can past its limit costs the database a statement
    for each request that arrives as a refusal runs out, not one a request.

    A bucket that cannot be read -- a database not migrated to 0003, a
    statement that fails, or one not answered within RATE_SECONDS (v83: the
    connection's watchdog shuts it down) -- is this process's own for the next
    LOCAL_SECONDS (resilience.TenantLimits, as before v82), then asked again;
    it is counted (pasar_limits_errors_total{part="rate"}) and logged once a
    minute."""

    def __init__(self, engine: Engine, rate: float, burst: int, metrics: Any = None,
                 errors: Counter | None = None):
        self.engine, self.rate, self.burst, self.metrics = engine, rate, burst, metrics
        self.errors = errors if errors is not None else Counter()
        self.room = max(0, burst - 1)           # what it may hold for one more to go in
        self._pg = not is_sqlite(engine)
        now = _clock(engine)
        # `at` never moves back: a statement that waited on the row's lock
        # behind one that started later (PostgreSQL's now() is a statement's
        # start) would write an earlier time, and the next would drain the
        # bucket for that interval twice -- 1.4 times the limit at 3,000 a
        # second (WHAT_FAILED #71). A refusal says how far ahead `at` is too:
        # the bucket drains only once its time has come (a clock stepped back).
        if self._pg:
            held = f"GREATEST(r.level - GREATEST({now} - r.at, 0) * %(rate)s, 0)"
            self._take = (
                f"INSERT INTO rate_limits AS r (bucket, level, at) "
                f"VALUES (%(bucket)s, 1, {now}) "
                f"ON CONFLICT (bucket) DO UPDATE SET level = {held} + 1, "
                f"at = GREATEST(r.at, {now}) "
                f"WHERE {held} <= %(room)s RETURNING r.level")
            self._held = (f"SELECT GREATEST(level - GREATEST({now} - at, 0) * %(rate)s, 0), "
                          f"GREATEST(at - {now}, 0) FROM rate_limits WHERE bucket = %(bucket)s")
        else:
            held = f"max(level - max({now} - at, 0) * :rate, 0)"
            self._take = text(
                f"INSERT INTO rate_limits (bucket, level, at) VALUES (:bucket, 1, {now}) "
                f"ON CONFLICT (bucket) DO UPDATE SET level = {held} + 1, at = max(at, {now}) "
                f"WHERE {held} <= :room RETURNING level")
            self._held = text(f"SELECT max(level - max({now} - at, 0) * :rate, 0), "
                              f"max(at - {now}, 0) FROM rate_limits WHERE bucket = :bucket")
        self._params = {"rate": max(0.0, float(rate)), "room": float(self.room)}
        self._logged = 0.0
        self._lock = threading.Lock()           # the two below, shared by the request threads
        self._refused: dict[str, float] = {}   # bucket: when (time.monotonic) one may be admitted
        self._local_until = 0.0                 # until when this process counts alone
        self._local = TenantLimits(rate, burst)

    def take(self, bucket: str) -> float | None:
        """None if the request is admitted; else seconds until one would be."""
        if self.burst < 1:
            return 1.0 / self.rate if self.rate > 0 else 60.0
        now = time.monotonic()
        with self._lock:
            alone = now < self._local_until
            until = None if alone else self._refused.get(bucket)
            if until is not None and now >= until:
                del self._refused[bucket]
                until = None
        if alone:
            return self._local.take(bucket)
        if until is not None:
            return until - now
        try:
            held = self._pg_take(bucket) if self._pg else self._sqlite_take(bucket)
        except Exception as e:  # noqa: BLE001 -- see the class: counted here alone a while
            self._failed(e)
            with self._lock:
                self._local_until = time.monotonic() + LOCAL_SECONDS
            return self._local.take(bucket)
        if held is None:
            return None
        level, ahead = held
        wait = 60.0 if self.rate <= 0 else max(0.001, (level - self.room) / self.rate + ahead)
        with self._lock:
            if len(self._refused) > 10_000:     # tenants that stopped asking
                self._refused = {b: u for b, u in self._refused.items() if u > now}
            self._refused[bucket] = now + wait
        return wait

    def _pg_take(self, bucket: str) -> tuple[float, float] | None:
        """None if admitted; else what the bucket holds now, and how far
        ahead of now its time is."""
        params = {"bucket": bucket, **self._params}
        with deadline(RATE_SECONDS):            # a database this slow: counted here instead
            fairy = self.engine.raw_connection()
        conn = fairy.driver_connection
        try:
            conn.autocommit = True
            try:
                if conn.execute(self._take, params).fetchone() is not None:
                    return None
                row = conn.execute(self._held, params).fetchone()
                return (float(row[0]), float(row[1])) if row else (self.room + 1.0, 0.0)
            finally:
                if not conn.closed:
                    conn.autocommit = False
        except Exception:
            if conn.closed or getattr(conn, "broken", False):
                fairy.invalidate()
            raise
        finally:
            fairy.close()

    def _sqlite_take(self, bucket: str) -> tuple[float, float] | None:
        params = {"bucket": bucket, **self._params}
        with transaction(self.engine, write=True) as conn:
            if conn.execute(self._take, params).first() is not None:
                return None
            row = conn.execute(self._held, params).first()
            return (float(row[0]), float(row[1])) if row else (self.room + 1.0, 0.0)

    def _failed(self, e: Exception) -> None:
        self.errors["rate"] += 1
        if self.metrics is not None:
            self.metrics.limits_errors.labels("rate").inc()
        if time.monotonic() - self._logged > 60:
            self._logged = time.monotonic()
            log.warning(f"the shared rate limit could not be read ({type(e).__name__}): "
                        f"this process counts each tenant's requests alone for "
                        f"{LOCAL_SECONDS:.0f}s, as before v82{_hint(e)}")


class Deployment:
    """This process, as one of the deployment's: its row of service_nodes,
    its share of each provider's limit, the tenants' shared buckets, and the
    breakers the others opened. `start` before serving, `stop` after."""

    def __init__(self, engine: Engine, settings: Any, gates: Gates, metrics: Any = None,
                 node: str | None = None):
        self.engine, self.settings, self.gates, self.metrics = engine, settings, gates, metrics
        # Replicas of one deployment share their hostnames' first characters
        # and often a PID (1, in a container): 48 random bits tell them apart.
        self.node = node or (f"{socket.gethostname()[:24]}:{os.getpid()}:"
                             f"{uuid.uuid4().hex[:12]}")
        self.period = float(settings.heartbeat_seconds)
        self.total = int(settings.provider_concurrency)
        self.errors: Counter = Counter()
        self.rate = SharedRate(engine, settings.rate, settings.burst, metrics, self.errors)
        self.share = self.total
        self.nodes: list[str] = [self.node]
        self.synced_at: float | None = None             # time.monotonic() of the last sync
        self.listening = False
        self._pg = not is_sqlite(engine)
        # LISTEN needs psycopg's own connection: another driver hears at heartbeats.
        self._can_listen = self._pg and engine.dialect.driver == "psycopg"
        self._adopted: dict[str, tuple[str, float]] = {}  # provider: the opening held last
        self._lock = threading.Lock()
        self._beat = threading.Lock()           # one heartbeat at a time, and none past leaving
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._listener: Any = None
        self._listener_lock = threading.Lock()
        self.listener_pid: int | None = None    # its session on the server, for an operator
        self._listen_after = 0.0
        self._logged: dict[str, float] = {}
        self._syncs = 0
        now = _clock(engine)
        self._q_now = text(f"SELECT {now}")
        # A process says until when to count it -- three of its own heartbeats
        # on -- so that processes with heartbeats of their own agree on the count.
        self._q_beat = text(
            "INSERT INTO service_nodes (node, host, pid, version, started_at, seen_at, "
            "expires_at) VALUES (:node, :host, :pid, :version, :now, :now, :until) "
            "ON CONFLICT (node) DO UPDATE SET seen_at = excluded.seen_at, "
            "expires_at = excluded.expires_at")
        self._q_live = text("SELECT node FROM service_nodes WHERE expires_at > :now ORDER BY node")
        self._q_open = text("SELECT provider, open_until - :now, opened_by, opened_at "
                            "FROM provider_breakers WHERE open_until > :now")
        self._q_forget = text("DELETE FROM service_nodes WHERE expires_at < :before")
        self._q_leave = text("DELETE FROM service_nodes WHERE node = :node")
        self._q_opened = text(
            "INSERT INTO provider_breakers (provider, open_until, opened_at, opened_by) "
            "VALUES (:provider, :until, :now, :node) "
            "ON CONFLICT (provider) DO UPDATE SET open_until = excluded.open_until, "
            "opened_at = excluded.opened_at, opened_by = excluded.opened_by "
            "WHERE excluded.open_until > provider_breakers.open_until")
        self._q_notify = text("SELECT pg_notify(:channel, :what)")
        gates.on_open = self.opened

    # --- joining and leaving ------------------------------------------------
    def start(self) -> None:
        """Join: listen, write the row, tell the others, count -- before the
        process serves anything -- then keep it current from a thread."""
        if self._can_listen:
            self._listen()
        try:
            self.sync(say="join")
        except Exception as e:  # noqa: BLE001 -- not migrated, or not reachable: the thread retries
            self._failed("heartbeat", e)
        self._thread = threading.Thread(target=self._run, name="pasar-limits", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = STOP_SECONDS) -> None:
        """Leave: the row deleted and the others told, so that they take this
        process's share up at once rather than 15 seconds on -- after any
        heartbeat in flight, so that none writes the row back. Never longer
        than `timeout`: a database that does not answer (a partition, a
        restart) is left to forget the row itself, and the process goes."""
        deadline = time.monotonic() + timeout
        self._stop.set()
        self._wake.set()
        left = threading.Event()

        def leave() -> None:
            try:
                with self._beat:                # the heartbeat in flight, if any, first
                    with transaction(self.engine, write=True) as conn:
                        conn.execute(self._q_leave, {"node": self.node})
                        if self._pg:
                            conn.execute(self._q_notify, {"channel": CHANNEL, "what": "leave"})
            except Exception as e:  # noqa: BLE001 -- the others stop counting it 15 s on
                self._failed("heartbeat", e)
            finally:
                left.set()

        threading.Thread(target=leave, name="pasar-limits-leave", daemon=True).start()
        left.wait(max(0.0, deadline - time.monotonic()))
        if self._thread is not None:
            self._thread.join(max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():         # in a call the database never answered
                return
        self._close_listener()

    # --- keeping count ------------------------------------------------------
    def sync(self, say: str | None = None) -> None:
        """One heartbeat: this process's row, then who is serving and which
        breakers are open, all on the database's clock; `say` tells the
        others (PostgreSQL) once it commits. None once it is leaving."""
        with self._beat:
            if self._stop.is_set():
                return
            with transaction(self.engine, write=True) as conn:
                now = float(conn.execute(self._q_now).scalar())
                conn.execute(self._q_beat, {"node": self.node,
                                            "host": socket.gethostname()[:255],
                                            "pid": os.getpid(), "version": __version__,
                                            "now": now, "until": now + EXPIRY * self.period})
                live = [r[0] for r in conn.execute(self._q_live, {"now": now})]
                openings = conn.execute(self._q_open, {"now": now}).all()
                self._syncs += 1
                if self._syncs % 60 == 1:
                    conn.execute(self._q_forget, {"before": now - FORGET})
                if say and self._pg:
                    conn.execute(self._q_notify, {"channel": CHANNEL, "what": say})
        self._counted(live)
        for provider, left, by, at in openings:
            self._hold(provider, float(left), by, float(at))

    def _counted(self, live: list[str]) -> None:
        if self.node not in live:               # its row is written in the same transaction
            live = sorted([*live, self.node])
        share = split(self.total, len(live), live.index(self.node))
        with self._lock:                        # one count at a time, its gates resized with it
            before, self.share, self.nodes = self.share, share, live
            self.synced_at = time.monotonic()
            if share != before:
                self.gates.resize(share)
        if share != before:
            log.info(f"{len(live)} processes share the limits: this one's share of each "
                     f"provider's {self.total} calls at once is {share}")
        if self.metrics is not None:
            self.metrics.service_nodes.set(len(live))
            self.metrics.provider_share.set(share)

    def _hold(self, provider: str, left: float, by: str, at: float) -> None:
        """Another process's opening, held here too -- once: a call here that
        succeeds after it closes this one's breaker, and the same opening read
        again at the next heartbeat does not open it again."""
        if by == self.node or left <= 0:
            return
        with self._lock:
            if self._adopted.get(provider) == (by, at):
                return
            self._adopted[provider] = (by, at)
        if self.gates.get(provider).breaker.hold(left):
            if self.metrics is not None:
                self.metrics.breaker_openings.labels(provider, "another process").inc()
            log.warning(f"model provider {provider!r}: its breaker opened in {by}; "
                        f"not called from here for {left:.0f}s")

    def opened(self, provider: str, seconds: float) -> None:
        """This process's breaker for `provider` opened (Gates.on_open): the
        others are told from a thread of its own -- a breaker opens inside a
        model call, which may be on the event loop."""
        if self.metrics is not None:
            self.metrics.breaker_openings.labels(provider, "this process").inc()
        threading.Thread(target=self._publish, args=(provider, seconds), daemon=True,
                         name="pasar-limits-publish").start()

    def _publish(self, provider: str, seconds: float) -> None:
        try:
            with transaction(self.engine, write=True) as conn:
                now = float(conn.execute(self._q_now).scalar())
                conn.execute(self._q_opened, {"provider": provider, "until": now + seconds,
                                              "now": now, "node": self.node})
                if self._pg:
                    conn.execute(self._q_notify, {"channel": CHANNEL, "what": "breaker"})
        except Exception as e:  # noqa: BLE001 -- the others find out from their own calls
            self._failed("publish", e)

    # --- the thread -----------------------------------------------------------
    def _run(self) -> None:
        next_beat = time.monotonic() + self.period
        while not self._stop.is_set():
            heard = self._wait(min(1.0, max(0.0, next_beat - time.monotonic())))
            if self._stop.is_set():
                break
            if heard or time.monotonic() >= next_beat:
                try:
                    self.sync()
                except Exception as e:  # noqa: BLE001 -- the next heartbeat tries again
                    self._failed("heartbeat", e)
                next_beat = time.monotonic() + self.period
        self._close_listener()

    def _wait(self, timeout: float) -> bool:
        """Up to `timeout` seconds for word from another process: True if any came."""
        if self._can_listen and self._listener is None and \
                time.monotonic() >= self._listen_after:
            if self._listen():
                return True                     # it may have missed something meanwhile
        if self._listener is None:
            woke = self._wake.wait(timeout)
            self._wake.clear()
            return woke and not self._stop.is_set()
        try:
            # Whatever has come, read at once (several joining together are one sync).
            return bool(list(self._listener.notifies(timeout=timeout, stop_after=1)))
        except Exception as e:  # noqa: BLE001 -- the session ended: listen again, a heartbeat on
            self._failed("listen", e)
            self._close_listener()
            self._listen_after = time.monotonic() + self.period
            return False

    def _listen(self) -> bool:
        """A connection of its own, not the pool's -- psycopg's, made with the
        engine's own arguments -- listening on CHANNEL. Not the pool's: a
        pooled connection the server ended is reset as it is closed, and the
        pool logs that failure as a traceback of its own."""
        import psycopg
        conn = None
        try:
            cargs, cparams = self.engine.dialect.create_connect_args(self.engine.url)
            conn = psycopg.connect(*cargs, **{**cparams, "connect_timeout": 5,
                                              "autocommit": True})
            conn.execute(f"LISTEN {CHANNEL}")
        except Exception as e:  # noqa: BLE001 -- heartbeats alone carry the news
            self._failed("listen", e)
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._listen_after = time.monotonic() + self.period
            return False
        with self._listener_lock:
            self._listener, self.listening = conn, True
            self.listener_pid = conn.info.backend_pid
        return True

    def _close_listener(self) -> None:
        with self._listener_lock:
            conn, self._listener, self.listening = self._listener, None, False
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _failed(self, part: str, e: Exception) -> None:
        self.errors[part] += 1
        if self.metrics is not None:
            self.metrics.limits_errors.labels(part).inc()
        if time.monotonic() - self._logged.get(part, -1e9) > 60:
            self._logged[part] = time.monotonic()
            then = {"heartbeat": "this process keeps its last count",
                    "listen": "the others are heard of at each heartbeat instead",
                    "publish": "the others find out from their own calls"}.get(part, "")
            log.warning(f"shared limits, {part}: {type(e).__name__}: {str(e)[:200]} -- "
                        f"{then}{_hint(e)}")

    # --- what the operator sees --------------------------------------------
    def view(self) -> dict[str, Any]:
        """GET /v1/admin/nodes: the processes, as the database has them now,
        and what this one counts and holds."""
        rows: list[dict[str, Any]] = []
        openings: list[dict[str, Any]] = []
        try:
            rows, openings = processes(self.engine)
        except Exception as e:  # noqa: BLE001
            self._failed("heartbeat", e)
        with self._lock:
            share, nodes, synced = self.share, list(self.nodes), self.synced_at
        return {
            "limits": "deployment", "setting": getattr(self.settings, "limits", "deployment"),
            "node": self.node,
            "processes": len(nodes),
            "nodes": rows,
            "provider_concurrency": {"deployment": self.total, "this_process": share},
            "tenant_rate": {"per_second": self.settings.rate, "burst": self.settings.burst,
                            "shared": True},
            "breakers_open": openings,
            "heartbeat_seconds": self.period,
            "synced_s_ago": None if synced is None else round(time.monotonic() - synced, 1),
            "hears_at_once": self.listening,
            "errors": dict(self.errors),
        }
