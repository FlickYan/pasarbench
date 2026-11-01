"""
The commerce world in SQL, and how a tool call reads and changes it.

The benchmark's tools (pasarbench/tools.py) are pure functions over an
in-memory world, and they are what the benchmark scores. The service runs
those same functions: each tool call takes one customer's slice of the world
into a `pasarbench.db.Database` -- read from the database, or the one kept
from her last call when her world has not changed since (Slices) -- runs the
tool on a copy of it, and writes the rows it changed back -- in one short
transaction, never across a model call -- with the conversation's record of
the call (the tool's message, the audit rows) in the same transaction, so a
call is either done and recorded, or neither.

    read tool   one snapshot of the slice (REPEATABLE READ on PostgreSQL), no lock
    write tool  the customer's row locked first (SELECT ... FOR UPDATE on
                PostgreSQL; on SQLite every writing transaction begins
                IMMEDIATE and holds the database), the slice as of the write
                sequence the lock found, the tool run, each changed row updated
                only if its version is still the one read (else the call runs
                again, on what is there now), each new row inserted, the
                customer's write sequence advanced

Two conversations of one customer therefore write one at a time, and two
customers never wait for each other. A new row's id comes from the tenant's
sequence on PostgreSQL, taken without a lock held to commit (v79; before it, a
counter row that every creating write of the tenant waited on -- docs/SERVICE.md
measures both); on SQLite, which has one writer anyway, from a counter row.

The lock and the version check guard different things: the version check
catches a read-modify-write of one row gone stale; only the lock stops two
inserts that each read the same balance (two refunds, each under the order's
total) -- tests/test_service.py races both on PostgreSQL.

A slice is the customer's own rows: another customer's order does not exist
for the tools, whatever the model asks -- authorization in the data layer, not
in the prompt. The catalog -- products, sellers, livestream claims -- is public
and loaded as asked for, through a per-process cache (no tool writes it).

Two invariants the platform keeps whatever the model decides: what goes back
for an order (refunds and store credit in its currency) never passes what was
paid for it (`strict_money`), and a new row names only the customer's own
orders. A call that would break either is refused, logged as a refused
attempt as the tools log their own refusals, and nothing is written.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import socket
import sqlite3
import threading
import time
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from sqlalchemy import (Connection, Engine, bindparam, create_engine, event, insert, select,
                        text, update)
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.exc import TimeoutError as PoolTimeout
from sqlalchemy.pool import QueuePool

from ..db import CURRENCY_EXPONENT, SGD_RATE, Action, Database, fmt_money, seed
from ..tools import CHECKS, TOOLS, WRITE_TOOLS
from ..tools import call as tool_call
from . import schema as S

# The currencies the tools take, from their own schema.
CURRENCIES = tuple(TOOLS["issue_store_credit"]["schema"]["function"]["parameters"]
                   ["properties"]["currency"]["enum"])

# The tables a customer's slice holds; the catalog is loaded as the tools ask.
CATALOG = ("products", "sellers", "livestream_claims")
VERSIONED = {"users", "orders", "order_items", "payments"}


class StoreError(Exception):
    """A tool call the store cannot carry out as asked."""


class Conflict(StoreError):
    """A row changed between this call's read and its write: nothing was
    written, and the call is run again on what is there now."""


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def storable(x: Any) -> Any:
    """x as a database keeps it, on either: a float that is not a number (inf,
    -inf, nan) as its name -- JSON has none, though 1e400 parses as one, and
    PostgreSQL's JSONB refuses it -- and a NUL character as U+FFFD, which
    PostgreSQL's text cannot hold at all (SQLite can, so no SQLite test saw it).
    What the model sends is made storable before anything reads it
    (runtime.py), so a call and its replay see the same arguments."""
    if isinstance(x, str):
        return x.replace("\x00", "\ufffd") if "\x00" in x else x
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)
    if isinstance(x, dict):
        return {storable(k): storable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [storable(v) for v in x]
    return x


def _json(obj: Any) -> str:
    return json.dumps(storable(obj))


def is_sqlite(engine: Engine) -> bool:
    return engine.dialect.name == "sqlite"


def id_sequence(tenant_id: str) -> str:
    """The PostgreSQL sequence a tenant's new ids come from. Named by a digest
    of the tenant's id: a sequence's name is an identifier (63 bytes, no
    hyphen unquoted), and a tenant's id may be 63 characters with hyphens.
    Migration 0002 names them the same way; never change one without the other."""
    return "ids_" + hashlib.sha256(tenant_id.encode()).hexdigest()[:24]


DB_TIMEOUT = 10.0       # seconds the service waits on the database (PASAR_DB_TIMEOUT, v83)


SQLITE_BUSY_SECONDS = 30.0      # a write waits for SQLite's one write lock this long


def make_engine(url: str, *, pool_size: int = 10, max_overflow: int = 20,
                echo: bool = False, timeout: float = DB_TIMEOUT) -> Engine:
    """An engine for SQLite or PostgreSQL.

    SQLite: one file, WAL so readers do not block the writer, foreign keys on,
    a 30-second busy wait, and transactions that say whether they write: a
    write begins IMMEDIATE, taking the write lock before it reads, so two
    writers cannot both read a balance and both add to it."""
    if url.startswith("sqlite"):
        engine = create_engine(url, echo=echo, pool_size=pool_size, max_overflow=max_overflow,
                               hide_parameters=True, json_serializer=_json,
                               connect_args={"check_same_thread": False,
                                             "timeout": SQLITE_BUSY_SECONDS})

        @event.listens_for(engine, "connect")
        def _connect(dbapi_conn, _record):            # noqa: ANN001
            dbapi_conn.isolation_level = None         # BEGIN is ours (below)
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute(f"PRAGMA busy_timeout={int(SQLITE_BUSY_SECONDS * 1000)}")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

        @event.listens_for(engine, "begin")
        def _begin(conn):                             # noqa: ANN001
            mode = "IMMEDIATE" if conn.get_execution_options().get("pasar_write") else "DEFERRED"
            conn.exec_driver_sql(f"BEGIN {mode}")

        return engine
    # hide_parameters: a failed statement's error never carries the values it
    # was given (a customer's address, what she said). timezone=UTC: the
    # server's own zone never decides what day a timestamp falls on.
    #
    # PostgreSQL that does not answer (v83) -- stopped, partitioned, a pooler
    # hung -- is waited on for `timeout` seconds at most, never as long as
    # the network holds a connection: a statement the server runs is ended by
    # it after 80% of that (statement_timeout, the connection kept); a
    # connection lent from the pool comes back within it or its socket is
    # shut down (Watchdog); a new one is made within 5 s or not at all
    # (connect_timeout); a free one is waited for as long (pool_timeout); a
    # session left inside a transaction is ended by the server after three
    # times it (idle_in_transaction_session_timeout); and the kernel gives up
    # on a peer that has gone silent (keepalives, tcp_user_timeout).
    #
    # And once it has been seen not to answer, it is not waited on by every
    # request in turn: until a connection answers again, a checkout is
    # refused at once, but for one at a time that tries it (Silence).
    ms = int(timeout * 1000)
    engine = create_engine(
        url, echo=echo, pool_size=pool_size, max_overflow=max_overflow, pool_recycle=1800,
        pool_timeout=timeout, poolclass=SilencePool, hide_parameters=True,
        json_serializer=_json,
        connect_args={"options": f"-c timezone=UTC -c statement_timeout={int(ms * 0.8)} "
                                 f"-c idle_in_transaction_session_timeout={ms * 3}",
                      "connect_timeout": max(2, int(min(timeout, 5))),
                      "keepalives": 1, "keepalives_idle": 10, "keepalives_interval": 5,
                      "keepalives_count": 3, "tcp_user_timeout": ms})
    silence = Silence(trial_seconds=max(30.0, 3 * timeout))
    engine.pool.pasar_silence = silence         # type: ignore[attr-defined]
    engine.pasar_silence = silence              # type: ignore[attr-defined]

    @event.listens_for(engine, "do_connect")
    def _dial(dialect, _record, cargs, cparams):      # noqa: ANN001
        if not in_checkout():                   # (a checkout was admitted whole)
            silence.admit()
        try:
            return dialect.connect(*cargs, **cparams)
        except dialect.loaded_dbapi.OperationalError as e:
            if no_answer(e):
                silence.silent("a connection could not be made")
            raise

    _check_on_checkout(engine, Watchdog(timeout, silence), silence)
    return engine


PING_AFTER = 10.0       # seconds idle in the pool past which a connection is pinged


class _Unset:
    pass


_DEADLINE = threading.local()


@contextmanager
def deadline(seconds: float | None) -> Iterator[None]:
    """The longest a connection taken in this block, in this thread, may be
    lent: `seconds`, or None for work an operator asked for that takes as
    long as it takes -- seeding a world, replaying a tenant, forgetting
    expired keys: no watchdog, and on PostgreSQL no statement or idle
    timeout in its transactions (v83)."""
    before = getattr(_DEADLINE, "seconds", _Unset)
    _DEADLINE.seconds = seconds
    try:
        yield
    finally:
        if before is _Unset:
            del _DEADLINE.seconds
        else:
            _DEADLINE.seconds = before


def unbounded() -> bool:
    """Whether this thread is in a deadline(None) block."""
    return getattr(_DEADLINE, "seconds", _Unset) is None


class Watchdog:
    """Every connection lent from the pool comes back within its deadline,
    or its socket is shut down: a statement waiting on a database that does
    not answer -- stopped, partitioned, a pooler hung -- then raises at once
    (the server closed the connection, as psycopg says), the pool replaces
    the connection, and the request is answered 503 (v83). The server's own
    statement_timeout ends what it is running first, keeping the connection;
    this is for when the server cannot answer at all.

    Lending and taking back cost a transaction one dictionary operation each,
    no lock and no wakeup: the watchdog looks every `every` seconds (a
    quarter of a second at most), so a deadline is kept to within that. It
    shuts down only the socket it was lent with -- the same file, by its
    inode, noted when the connection was made -- never one that took its
    descriptor's number since."""

    def __init__(self, seconds: float, silence: "Silence | None" = None):
        self.seconds = seconds
        self.silence = silence
        self.aborted = 0
        self.every = min(0.25, seconds / 4)
        self._lent: dict[int, tuple[float, int, tuple[int, int], bool]] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def lend(self, key: int, sock: tuple[int, tuple[int, int]], seconds: float) -> None:
        """Watch `key`'s socket (its descriptor and identity) for `seconds`,
        the timeout at most. One lent for the whole timeout that has to be
        shut down says the database has stopped answering (Silence); one lent
        for less -- a readiness probe's 2 s, a rate limit's -- says only that
        it is slower than that."""
        seconds = min(seconds, self.seconds)
        self._lent[key] = (time.monotonic() + seconds, sock[0], sock[1],
                           seconds >= self.seconds)
        if self._thread is None:
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, daemon=True,
                                                    name="pasar-db-watchdog")
                    self._thread.start()

    def back(self, key: int) -> bool:
        """Whether `key` was lent and is back in time."""
        return self._lent.pop(key, None) is not None

    def _run(self) -> None:
        while True:
            time.sleep(self.every)
            try:
                self._look()
            except Exception:  # noqa: BLE001 -- the watchdog itself must not stop
                log.exception("the database watchdog failed a round")

    def _look(self) -> None:
        now = time.monotonic()
        for key, (due, fd, ident, whole) in list(self._lent.items()):
            if due > now or self._lent.pop(key, None) is None:
                continue                        # not due, or back meanwhile
            if _shut(fd, ident):
                self.aborted += 1
                if whole and self.silence is not None:
                    self.silence.silent(f"a connection lent did not answer in "
                                        f"{self.seconds:g}s")


SILENT_RETRY = 1.0      # seconds between tries of a database that has stopped answering
TOLD_EVERY = 60.0       # a silence is logged once a minute at most
log = logging.getLogger("pasar.db")
_CHECKOUT = threading.local()


def in_checkout() -> bool:
    """Whether this thread is inside a pool checkout admitted by Silence."""
    return getattr(_CHECKOUT, "depth", 0) > 0


class DatabaseSilent(PoolTimeout):
    """A checkout refused at once: the database has stopped answering
    (Silence). A pool timeout's kind, so that it is answered as one: 503."""


class Silence:
    """Whether this process's database has stopped answering, as it last
    saw: from a connection lent for the whole timeout that the watchdog had
    to shut down, or one that could not be made, until a connection answers
    again: one lent comes back in time.

    While it is silent a checkout is refused at once (DatabaseSilent), not
    waited on: a request that came during an outage would otherwise wait
    out its own timeout, holding a thread and a connection, and so would
    every one after it, until the pool and the threads ran out -- each
    bounded, and together without bound. One checkout at a time, every
    SILENT_RETRY seconds, is let through to try it for real: a request, a
    heartbeat, a lease given back; its answer, or any other connection's,
    ends the silence (v83)."""

    def __init__(self, retry: float = SILENT_RETRY, trial_seconds: float = 30.0):
        self.retry = retry
        self.trial_seconds = trial_seconds      # a trial not heard from in this long is lost
        self.since: float | None = None         # monotonic; None while it answers
        self.refused = 0
        self.spells = 0
        self._refused_before = 0
        self._until = 0.0
        self._trial: tuple[int, float] | None = None    # (thread, since)
        self._lock = threading.Lock()
        self._told = -TOLD_EVERY                # when a silence was last logged
        self._telling = False                   # this one was: its end is logged too

    def admit(self) -> None:
        """Return to take a connection; raise DatabaseSilent to be refused."""
        if self.since is None:
            return
        me = threading.get_ident()
        with self._lock:
            since = self.since
            if since is None:
                return
            now = time.monotonic()
            trial = self._trial
            if trial is not None and trial[0] == me:
                return
            if trial is not None and now - trial[1] > self.trial_seconds:
                trial = self._trial = None
            # An operator's unbounded work is no trial: nothing would end it.
            if trial is None and now >= self._until and not unbounded():
                self._trial = (me, now)
                return
            self.refused += 1
        raise DatabaseSilent(f"the database has not answered for {now - since:.0f}s")

    def silent(self, why: str) -> None:
        """A connection did not answer, or could not be made."""
        with self._lock:
            now = time.monotonic()
            if self.since is not None:
                return                          # (the trials' own ends set the next)
            self._until = now + self.retry
            self.since = now
            self.spells += 1
            self._refused_before = self.refused
            self._telling = now - self._told >= TOLD_EVERY
            if not self._telling:
                return
            self._told = now
        log.warning(f"the database has stopped answering ({why}): until it does, a request "
                    f"is answered 503 at once, and one at a time tries it every "
                    f"{self.retry:g}s (said once a minute at most)")

    def trial_over(self) -> None:
        """This thread's checkout failed: if it was the trial, another may
        try SILENT_RETRY seconds on -- or sooner after one that failed at
        once, a connection refused (a database restarting), which costs
        nothing to try again: twice as long as it took, a tenth of
        SILENT_RETRY at least."""
        if self._trial is None:
            return
        with self._lock:
            trial = self._trial
            if trial is not None and trial[0] == threading.get_ident():
                now = time.monotonic()
                self._trial = None
                self._until = now + min(self.retry, max(self.retry / 10, 2 * (now - trial[1])))

    def answered(self) -> None:
        if self.since is None:
            return
        with self._lock:
            if self.since is None:
                return
            took = time.monotonic() - self.since
            self.since, self._trial = None, None
            n = self.refused - self._refused_before
            if not self._telling:
                return
        log.warning(f"the database answers again, after {took:.1f}s; {n} "
                    f"{'checkout' if n == 1 else 'checkouts'} refused at once meanwhile")


class SilencePool(QueuePool):
    """QueuePool, refusing a checkout at once while its database is silent:
    before it waits for a connection or makes one. A checkout is admitted
    once, whole -- its wait, a connection made, a dead one found and
    replaced within it -- and a trial's that fails, wherever it fails, ends
    the trial."""

    pasar_silence: Silence | None = None

    def connect(self):                          # noqa: ANN201
        silence = self.pasar_silence
        if silence is None:
            return super().connect()
        silence.admit()
        depth = getattr(_CHECKOUT, "depth", 0)
        _CHECKOUT.depth = depth + 1
        try:
            return super().connect()
        except BaseException:
            silence.trial_over()
            raise
        finally:
            _CHECKOUT.depth = depth

    def recreate(self):                         # noqa: ANN202 -- engine.dispose()
        new = super().recreate()
        new.pasar_silence = self.pasar_silence
        return new


def _shut(fd: int, ident: tuple[int, int]) -> bool:
    """Shut down the socket on `fd` if it is still the one `ident` names --
    checked on a duplicate of it, so that the descriptor checked is the one
    shut down even if libpq closes `fd` and its number is taken meanwhile."""
    try:
        dup = os.dup(fd)
    except OSError:
        return False
    try:
        st = os.fstat(dup)
        if (st.st_dev, st.st_ino) != ident:
            return False
        sock = socket.socket(fileno=dup)
        dup = -1                                # the socket closes it
        try:
            sock.shutdown(socket.SHUT_RDWR)
        finally:
            sock.close()
        return True
    except OSError:
        return False
    finally:
        if dup >= 0:
            os.close(dup)


def _check_on_checkout(engine: Engine, watchdog: Watchdog | None = None,
                       silence: "Silence | None" = None) -> None:
    """A pooled connection is checked before it is lent: one used within
    PING_AFTER seconds by looking at its socket -- a server that ended the
    session (a restart, an administrator, a crash) leaves it readable or
    closed, and an idle session receives nothing -- and one idle longer by a
    round trip, as pool_pre_ping would on every checkout. A connection found
    dead is replaced before anyone sees it. A turn checks out a connection
    for each of its transactions: a ping each was an eighth of their round
    trips (docs/SERVICE.md, v80). The watchdog has it from before the check
    -- a ping to a server that does not answer is a wait like any other --
    until it is back. One lent and back in time says the database answers
    (Silence) -- not one made: a server can take connections and answer no
    statement."""
    engine.pasar_watchdog = watchdog            # type: ignore[attr-defined]

    @event.listens_for(engine, "connect")
    def _connected(dbapi_conn, record):           # noqa: ANN001
        record.info["pasar_used"] = time.monotonic()
        pgconn = getattr(dbapi_conn, "pgconn", None)
        if watchdog is not None and pgconn is not None:
            try:                                # the socket, as the watchdog will know it
                fd = pgconn.socket
                st = os.fstat(fd)
                record.info["pasar_sock"] = (fd, (st.st_dev, st.st_ino))
            except Exception:  # noqa: BLE001 -- no socket to watch: the check finds it
                pass

    @event.listens_for(engine, "checkin")
    def _returned(dbapi_conn, record):            # noqa: ANN001
        record.info["pasar_used"] = time.monotonic()
        if watchdog is not None and watchdog.back(id(record)) and dbapi_conn is not None \
                and silence is not None:
            silence.answered()

    if watchdog is not None:
        @event.listens_for(engine, "invalidate")
        def _dropped(_dbapi_conn, record, _exc):  # noqa: ANN001
            watchdog.back(id(record))
            if silence is not None and not in_checkout():
                silence.trial_over()            # (within one, the checkout's end says)

    @event.listens_for(engine, "checkout")
    def _lent(dbapi_conn, record, _proxy):        # noqa: ANN001
        from sqlalchemy.exc import DisconnectionError
        if watchdog is not None:
            seconds = getattr(_DEADLINE, "seconds", watchdog.seconds)
            sock = record.info.get("pasar_sock")
            if seconds is not None and sock is not None:
                watchdog.lend(id(record), sock, seconds)
        try:
            if time.monotonic() - record.info.get("pasar_used", 0.0) <= PING_AFTER:
                quiet = _quiet(dbapi_conn)
                if quiet:
                    return
                if quiet is False:
                    raise DisconnectionError("the server ended the session")
            try:                                  # idle long, or no way to tell: ping it
                engine.dialect.do_ping(dbapi_conn)
            except Exception as e:  # noqa: BLE001 -- any failure: a new connection
                raise DisconnectionError(f"ping failed: {type(e).__name__}") from None
        except DisconnectionError:
            if watchdog is not None:
                watchdog.back(id(record))
            raise


def _quiet(dbapi_conn: Any) -> bool | None:
    """True: an idle session the server has said nothing on -- open on this
    side, its socket with nothing to read. False: one it ended (an ended
    session sends its reason and closes, which reads as readable). None: no
    way to tell without a round trip (a driver other than psycopg 3, or the
    check failing), and the caller pings."""
    import select as _select
    pgconn = getattr(dbapi_conn, "pgconn", None)
    if pgconn is None:
        return None
    try:
        if dbapi_conn.closed or getattr(dbapi_conn, "broken", False):
            return False
        fd = pgconn.socket
        if hasattr(_select, "poll"):            # no FD_SETSIZE limit, unlike select()
            p = _select.poll()
            p.register(fd, _select.POLLIN | _select.POLLERR | _select.POLLHUP)
            return not p.poll(0)
        return not _select.select([fd], [], [], 0)[0]
    except Exception:  # noqa: BLE001 -- the check itself failed: ask the server
        return None


@contextmanager
def transaction(engine: Engine, *, write: bool, snapshot: bool = False) -> Iterator[Connection]:
    """One transaction. `write`: it will write, and on SQLite takes the write
    lock at its start. `snapshot`: on PostgreSQL every statement reads the
    same snapshot (REPEATABLE READ) -- SQLite's transactions always do."""
    opts: dict[str, Any] = {"pasar_write": write}
    if snapshot and not is_sqlite(engine):
        opts["isolation_level"] = "REPEATABLE READ"
    with engine.connect().execution_options(**opts) as conn:
        with conn.begin():
            if unbounded() and not is_sqlite(engine):   # an operator's long work (deadline)
                conn.exec_driver_sql("SET LOCAL statement_timeout = 0")
                conn.exec_driver_sql("SET LOCAL idle_in_transaction_session_timeout = 0")
            yield conn


SQLITE_UNAVAILABLE = {"BUSY", "LOCKED", "IOERR", "CANTOPEN", "FULL", "NOMEM", "PROTOCOL"}
# SQLSTATE classes that say the database could not answer: a connection
# exception, resources it ran out of, an operator's intervention (a statement
# ended at statement_timeout, a shutdown), a system error.
UNAVAILABLE_SQLSTATE = {"08", "53", "57", "58"}


def unavailable(e: BaseException) -> bool:
    """Whether `e` says the database did not answer, or could not -- a
    connection lost, refused, timed out or shut down by the watchdog, a
    statement it ended at statement_timeout, no free connection in the pool
    in time, a checkout refused while it is silent, SQLite's lock waited out
    -- and not that a statement was wrong, nor a conflict tried again (a
    serialization failure or a deadlock, SQLSTATE class 40, which psycopg
    also calls operational), nor a limit a statement ran into (class 54):
    what the service answers 503 for (v83)."""
    from sqlalchemy.exc import DBAPIError, OperationalError
    if isinstance(e, PoolTimeout):              # DatabaseSilent among them
        return True
    if not isinstance(e, DBAPIError):
        return False
    if e.connection_invalidated:                # SQLAlchemy saw the connection go
        return True
    orig = getattr(e, "orig", None)
    if isinstance(orig, sqlite3.Error):
        if not isinstance(e, OperationalError):
            return False
        # SQLite says "operational" of a missing table and a syntax error
        # too: only a lock waited out, or the file not there to be had, is
        # the database not answering.
        name = getattr(orig, "sqlite_errorname", None)        # Python 3.11+
        if name:
            return name.split("_")[1] in SQLITE_UNAVAILABLE
        return any(m in str(orig) for m in ("database is locked", "database table is locked",
                                            "disk I/O error", "unable to open database file",
                                            "database or disk is full"))
    code = getattr(orig, "sqlstate", None) or ""
    if not code:                                # the client's own: lost, refused, timed out
        return isinstance(e, OperationalError)
    return code[:2] in UNAVAILABLE_SQLSTATE


def no_answer(e: BaseException) -> bool:
    """Whether a connection that could not be made got no answer from the
    server -- timed out, refused, unreachable, closed as it started -- not
    one: a FATAL the server sent (too many connections, a password refused,
    no such database) is the server answering, and does not make it silent."""
    return "FATAL" not in str(e)


def create_all(engine: Engine) -> None:
    """The tables, for tests and a first local run; deployments migrate
    (alembic upgrade head)."""
    S.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------

def world_tables(world: str) -> dict[str, dict[str, dict[str, Any]]]:
    """A starting world, by name -- a pure function of the name, so replay can
    rebuild it: "demo" is the benchmark's seed; "task:<id>" a task's world
    (the seed with its patch, as Database.fresh builds it); "synthetic:<n>"
    the seed's customers and their orders n times over, renamed, for load."""
    if world == "demo":
        return Database.fresh().tables
    if world.startswith("task:"):
        from ..rescore import all_tasks
        task = all_tasks().get(world[5:])
        if task is None:
            raise ValueError(f"no task {world[5:]!r}")
        return Database.fresh(task.db_patch).tables
    if world.startswith("synthetic:"):
        return _synthetic(int(world.split(":", 1)[1]))
    raise ValueError(f"unknown world {world!r}: demo, task:<id> or synthetic:<n>")


def _synthetic(n: int) -> dict[str, dict[str, dict[str, Any]]]:
    """The seed's customers, orders and everything that hangs off them, n
    times, each copy's ids suffixed -C0001...; catalog shared."""
    base = seed()
    out = {t: dict(rows) if t in CATALOG else {} for t, rows in base.items()}
    keys = {"users": "user_id", "orders": "order_id", "order_items": "order_item_id",
            "shipments": "shipment_id", "payments": "payment_id"}
    for i in range(1, n + 1):
        sfx = f"-C{i:04d}"
        for table, key in keys.items():
            for row in base[table].values():
                r = copy.deepcopy(row)
                r[key] = r[key] + sfx
                for ref in ("user_id", "order_id"):
                    if ref in r and ref != key:
                        r[ref] = r[ref] + sfx
                out[table][r[key]] = r
    return out


def world_digest(tables: dict[str, dict[str, dict[str, Any]]]) -> str:
    blob = json.dumps({t: tables[t] for t in S.WORLD_ORDER if t in tables}, sort_keys=True,
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _row_values(table: str, row: dict[str, Any]) -> dict[str, Any]:
    t, _ = S.WORLD_TABLES[table]
    cols = set(t.c.keys())
    extra = set(row) - cols
    if extra:
        raise StoreError(f"{table} row has fields the schema does not: {sorted(extra)}")
    return dict(row)


def seed_tenant(engine: Engine, tenant_id: str, name: str, world: str,
                daily_budget_picousd: int | None = None) -> str:
    """A tenant and its starting world, in one transaction. Returns the
    world's digest."""
    tables = world_tables(world)
    digest = world_digest(tables)
    slices(engine).forget(tenant_id)
    # A world of thousands of customers is seconds of inserts in one
    # transaction: an operator's long work, with no deadline (v83).
    with deadline(None), transaction(engine, write=True) as conn:
        conn.execute(insert(S.tenants).values(tenant_id=tenant_id, name=name, world=world,
                                              world_digest=digest,
                                              daily_budget_picousd=daily_budget_picousd))
        conn.execute(insert(S.id_counters).values(tenant_id=tenant_id, value=0))
        if conn.dialect.name == "postgresql":
            # Its new ids: a sequence (see _run_once), made with the tenant and
            # gone with it if the seeding fails -- PostgreSQL's DDL is transactional.
            conn.execute(text(f"CREATE SEQUENCE {id_sequence(tenant_id)} MINVALUE 1 START 1"))
        for table in S.WORLD_ORDER:
            rows = [{"tenant_id": tenant_id, "pos": i, **_row_values(table, r)}
                    for i, r in enumerate(tables.get(table, {}).values())]
            if rows:
                conn.execute(insert(S.WORLD_TABLES[table][0]), rows)
    return digest


# A new row's place: after every seeded row, in the order its customer's writes
# committed, and within one write in the order the tool made them -- the order
# the benchmark's dict would hold them in.
NEW_ROWS = 10 ** 12


def new_pos(customer_seq: int, i: int) -> int:
    return NEW_ROWS + customer_seq * 4096 + i


# ---------------------------------------------------------------------------
# A customer's slice, as the tools see it
# ---------------------------------------------------------------------------
#
# The statements a tool call runs are built once, with named parameters.
# Built for each call, a statement cost more CPU in SQLAlchemy than its round
# trip did -- every value coerced into a parameter, the statement's cache key
# computed anew -- and a tool call runs a dozen (docs/SERVICE.md, v80).

def _columns(t: Any) -> tuple[tuple[int, str], ...]:
    """(position, name) of the columns a tool sees, in the table's order. The
    names as plain str: SQLAlchemy's are a subclass, which copy.deepcopy
    rebuilds one by one."""
    return tuple((i, str(c.key)) for i, c in enumerate(t.c) if c.key not in S.SERVICE_COLUMNS)


_COLS = {table: _columns(t) for table, (t, _key) in S.WORLD_TABLES.items()}
_AT = {table: {str(c.key): i for i, c in enumerate(t.c)}
       for table, (t, _key) in S.WORLD_TABLES.items()}


def _plain(table: str, row: Any) -> dict[str, Any]:
    """A world row as the tools see it, from a `select(table)` row."""
    return {k: row[i] for i, k in _COLS[table]}


def _by(t: Any, column: str, many: bool, ordered: bool = True) -> Any:
    q = select(t).where(t.c.tenant_id == bindparam("tenant"),
                        t.c[column].in_(bindparam("ids", expanding=True)) if many
                        else t.c[column] == bindparam("id"))
    return q.order_by(t.c.pos) if ordered else q


_Q_USER = {table: _by(S.WORLD_TABLES[table][0], "user_id", False, table != "users")
           for table in ("users", "orders", "store_credit_grants", "vouchers")}
_Q_ORDERS = {table: _by(S.WORLD_TABLES[table][0], "order_id", True)
             for table in ("order_items", "shipments", "payments", "refunds", "returns",
                           "escalations", "seller_messages")}
_Q_KEYS = {table: _by(S.WORLD_TABLES[table][0], S.WORLD_TABLES[table][1], True, False)
           for table in ("products", "sellers")}
_Q_CLAIMS = _by(S.livestream_claims, "livestream_id", False)
_c = S.customers
_Q_SEQ = select(_c.c.write_seq).where(_c.c.tenant_id == bindparam("tenant"),
                                      _c.c.user_id == bindparam("user"))
_Q_SEQ_LOCKED = _Q_SEQ.with_for_update()
_Q_BUMP = (update(_c).where(_c.c.tenant_id == bindparam("tenant"),
                            _c.c.user_id == bindparam("user"))
           .values(write_seq=_c.c.write_seq + 1).returning(_c.c.write_seq))
_k = S.id_counters
_Q_COUNTER = (update(_k).where(_k.c.tenant_id == bindparam("tenant"))
              .values(value=_k.c.value + 1).returning(_k.c.value))
_Q_NEXTVAL = text("SELECT nextval(CAST(:seq AS regclass))")
_Q_OUTBOX = insert(S.outbox)
_a = S.actions
_Q_ACTIONS = select(_a.c.tool, _a.c.args, _a.c.ok, _a.c.error, _a.c.denied).where(
    _a.c.conversation_id == bindparam("cid")).order_by(_a.c.seq)
_Q_ACTIONS_BEFORE = select(_a.c.tool, _a.c.args, _a.c.ok, _a.c.error, _a.c.denied).where(
    _a.c.conversation_id == bindparam("cid"),
    _a.c.call_seq < bindparam("before")).order_by(_a.c.seq)


class Catalog:
    """Catalog rows this process has read -- products, sellers, a livestream's
    claims. No tool writes them (diff() refuses), so a row once read stays
    true, and a tool call reads them here instead of from the database: three
    fewer queries a call. One per engine (two databases may hold one tenant
    id), least recently used out past `size` entries."""

    def __init__(self, size: int = 50_000):
        self.size = size
        self._rows: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Any:
        with self._lock:
            v = self._rows.get(key)
            if v is not None:
                self._rows.move_to_end(key)
            return v

    def put(self, key: tuple, value: Any) -> None:
        with self._lock:
            self._rows[key] = value
            self._rows.move_to_end(key)
            while len(self._rows) > self.size:
                self._rows.popitem(last=False)


_CATALOGS: "weakref.WeakKeyDictionary[Engine, Catalog]" = weakref.WeakKeyDictionary()
_CATALOGS_LOCK = threading.Lock()


def catalog(engine: Engine) -> Catalog:
    with _CATALOGS_LOCK:
        c = _CATALOGS.get(engine)
        if c is None:
            c = _CATALOGS[engine] = Catalog()
        return c


class Slices:
    """Customers' slices as a tool call last read them, each with the write
    it was read at: the customer's write_seq, which every change to her
    world advances in the transaction that makes it (_run_once). A call that
    finds her write_seq where a slice was read uses that slice instead of
    reading a dozen tables again -- the same rows, and the same versions, by
    construction. One per engine (two databases may hold one tenant id),
    least recently used out past `size` customers; 0 keeps none.

    What it relies on: that nothing else changes a customer's world rows.
    Nothing in the service does -- replay reports a row changed behind its
    back as a difference -- and a deployment where another system writes them
    runs with PASAR_SLICE_CACHE=0. A slice is never changed in place: a call
    gets a copy to run its tool on (copy_world)."""

    def __init__(self, size: int = 2000):
        self.size = size
        self._rows: OrderedDict[tuple[str, str], tuple[int, Any, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, tenant_id: str, user_id: str, seq: int) -> tuple[Any, Any] | None:
        with self._lock:
            got = self._rows.get((tenant_id, user_id))
            if got is None or got[0] != seq:
                self.misses += 1
                return None
            self._rows.move_to_end((tenant_id, user_id))
            self.hits += 1
            return got[1], got[2]

    def put(self, tenant_id: str, user_id: str, seq: int, tables: Any, versions: Any) -> None:
        if self.size <= 0:
            return
        with self._lock:
            got = self._rows.get((tenant_id, user_id))
            if got is not None and got[0] > seq:
                return                  # a later write's slice is already here
            self._rows[(tenant_id, user_id)] = (seq, tables, versions)
            self._rows.move_to_end((tenant_id, user_id))
            while len(self._rows) > self.size:
                self._rows.popitem(last=False)

    def forget(self, tenant_id: str, user_id: str | None = None) -> None:
        """A customer's slice, or every slice of a tenant (seeded anew: tests)."""
        with self._lock:
            for k in [k for k in self._rows if k[0] == tenant_id
                      and (user_id is None or k[1] == user_id)]:
                del self._rows[k]


_SLICES: "weakref.WeakKeyDictionary[Engine, Slices]" = weakref.WeakKeyDictionary()


def slices(engine: Engine) -> Slices:
    with _CATALOGS_LOCK:
        c = _SLICES.get(engine)
        if c is None:
            c = _SLICES[engine] = Slices()
        return c


_PLAIN = frozenset({str, int, float, bool, type(None)})


World = dict[str, dict[str, dict[str, Any]]]


def copy_world(tables: World) -> World:
    """A deep copy of world tables. Their rows hold plain values -- str, int,
    float, bool, None, as the columns do -- and a row of them is copied as a
    dict; any other row through copy.deepcopy. The same copy, in a twentieth
    of deepcopy's time."""
    return {name: {rid: (row.copy() if all(type(v) in _PLAIN for v in row.values())
                         else copy.deepcopy(row)) for rid, row in rows.items()}
            for name, rows in tables.items()}


def _catalog_rows(conn: Connection, cache: Catalog | None, tenant_id: str, table: str,
                  ids: list[str]) -> list[dict[str, Any]]:
    """Catalog rows by id, in their table's order: from the cache, the rest
    from the database."""
    key = S.WORLD_TABLES[table][1]
    got: list[tuple[int, dict[str, Any]]] = []
    missing = []
    for i in ids:
        hit = cache.get((tenant_id, table, i)) if cache else None
        (got.append(hit) if hit is not None else missing.append(i))
    if missing:
        pos = _AT[table]["pos"]
        for r in conn.execute(_Q_KEYS[table], {"tenant": tenant_id, "ids": missing}):
            entry = (int(r[pos]), _plain(table, r))
            if cache:
                cache.put((tenant_id, table, entry[1][key]), entry)
            got.append(entry)
    return [dict(row) for _pos, row in sorted(got, key=lambda e: e[0])]


def _claims(conn: Connection, cache: Catalog | None, tenant_id: str,
            livestream_id: str) -> list[dict[str, Any]]:
    hit = cache.get((tenant_id, "claims_of", livestream_id)) if cache else None
    if hit is None:
        hit = [_plain("livestream_claims", r) for r in conn.execute(
            _Q_CLAIMS, {"tenant": tenant_id, "id": livestream_id})]
        if cache:
            cache.put((tenant_id, "claims_of", livestream_id), hit)
    return [dict(r) for r in hit]


class StoreDatabase(Database):
    """The benchmark's Database over one customer's slice. Catalog rows the
    slice lacks are read on demand; new ids come from `allocate` -- the
    tenant's counter, or in a replay the ids the call was given."""

    def __init__(self, conn: Connection, tenant_id: str, tables,
                 allocate: Callable[[str], str], cache: Catalog | None = None):
        super().__init__(tables=tables)
        self._conn = conn
        self._tenant = tenant_id
        self._allocate = allocate
        self._cache = cache
        self.allocated: list[str] = []

    def next_id(self, prefix: str) -> str:
        new = self._allocate(prefix)
        self.allocated.append(new)
        return new

    def row(self, table: str, row_id: str):
        r = self.tables[table].get(row_id)
        if r is None and table in ("products", "sellers") and isinstance(row_id, str):
            got = _catalog_rows(self._conn, self._cache, self._tenant, table, [row_id])
            if got:
                r = self.tables[table][row_id] = got[0]
        return r

    def where(self, table: str, **kw: Any):
        if (table == "livestream_claims" and isinstance(kw.get("livestream_id"), str)
                and not any(c["livestream_id"] == kw["livestream_id"]
                            for c in self.tables[table].values())):
            for d in _claims(self._conn, self._cache, self._tenant, kw["livestream_id"]):
                self.tables[table][d["claim_id"]] = d
        return super().where(table, **kw)


def load_slice(conn: Connection, tenant_id: str, user_id: str,
               cache: Catalog | None = None) -> tuple[
        dict[str, dict[str, dict[str, Any]]], dict[tuple[str, str], int]]:
    """The customer's rows, every table the tools read, and the version of
    each versioned row as read; the catalog rows they name, through `cache`."""
    tables: dict[str, dict[str, dict[str, Any]]] = {t: {} for t in S.WORLD_TABLES}
    tables["notes"] = {}
    versions: dict[tuple[str, str], int] = {}

    def put(table: str, rows) -> list[dict[str, Any]]:
        key = S.WORLD_TABLES[table][1]
        cols, at, into = _COLS[table], _AT[table][key], tables[table]
        version = _AT[table]["version"] if table in VERSIONED else None
        out = []
        for r in rows:
            if version is not None:
                versions[(table, r[at])] = r[version]
            d = {k: r[i] for i, k in cols}
            into[d[key]] = d
            out.append(d)
        return out

    who = {"tenant": tenant_id, "id": user_id}
    put("users", conn.execute(_Q_USER["users"], who))
    orders = put("orders", conn.execute(_Q_USER["orders"], who))
    order_ids = [r["order_id"] for r in orders]
    if order_ids:
        theirs = {"tenant": tenant_id, "ids": order_ids}
        for table, q in _Q_ORDERS.items():
            put(table, conn.execute(q, theirs))
    for table in ("store_credit_grants", "vouchers"):
        put(table, conn.execute(_Q_USER[table], who))
    product_ids = sorted({r["product_id"] for r in tables["order_items"].values()})
    for d in _catalog_rows(conn, cache, tenant_id, "products", product_ids):
        tables["products"][d["product_id"]] = d
    seller_ids = sorted({r["seller_id"] for r in orders}
                        | {r["seller_id"] for r in tables["products"].values()})
    for d in _catalog_rows(conn, cache, tenant_id, "sellers", seller_ids):
        tables["sellers"][d["seller_id"]] = d
    for live in sorted({r["livestream_id"] for r in orders if r.get("livestream_id")}):
        for d in _claims(conn, cache, tenant_id, live):
            tables["livestream_claims"][d["claim_id"]] = d
    return tables, versions


# ---------------------------------------------------------------------------
# Changes
# ---------------------------------------------------------------------------

def diff(before: dict[str, dict[str, dict[str, Any]]],
         after: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """What a tool changed in the world tables: rows it inserted, and for each
    row it changed, each field's old and new value. A row it deleted is an
    error: no tool deletes."""
    inserts: dict[str, dict[str, Any]] = {}
    updates: dict[str, dict[str, dict[str, list[Any]]]] = {}
    for table in S.WORLD_TABLES:
        b, a = before.get(table, {}), after.get(table, {})
        gone = set(b) - set(a)
        if gone:
            raise StoreError(f"a tool deleted {table} rows {sorted(gone)}")
        for rid, row in a.items():
            if rid not in b:
                # A catalog row new since the call began was read on demand
                # (StoreDatabase.row): no tool creates catalog rows.
                if table not in CATALOG:
                    inserts.setdefault(table, {})[rid] = copy.deepcopy(row)
            elif row != b[rid]:
                if table in CATALOG:
                    raise StoreError(f"a tool changed catalog row {table} {rid}")
                fields = {k: [b[rid].get(k), v] for k, v in row.items() if b[rid].get(k) != v}
                updates.setdefault(table, {})[rid] = fields
    return {"insert": inserts, "update": updates}


def apply_changes(conn: Connection, tenant_id: str, changes: dict[str, Any],
                  versions: dict[tuple[str, str], int], customer_seq: int) -> None:
    """Write a tool's changes: each update only if the row is still the
    version read (Conflict if not), each insert in parent-first order, placed
    after everything before it (new_pos)."""
    for table, rows in changes["update"].items():
        t, key = S.WORLD_TABLES[table]
        for rid, fields in rows.items():
            values = {k: new for k, (_old, new) in fields.items()}
            stmt = update(t).where(t.c.tenant_id == tenant_id, t.c[key] == rid)
            if table in VERSIONED:
                stmt = stmt.where(t.c.version == versions[(table, rid)])
                values["version"] = versions[(table, rid)] + 1
            if conn.execute(stmt.values(**values)).rowcount != 1:
                raise Conflict(f"{table} {rid} changed under this write")
    for table in S.WORLD_ORDER:
        rows = changes["insert"].get(table)
        if rows:
            conn.execute(insert(S.WORLD_TABLES[table][0]),
                         [{"tenant_id": tenant_id, "pos": new_pos(customer_seq, i),
                           **_row_values(table, r)} for i, r in enumerate(rows.values())])


# ---------------------------------------------------------------------------
# A tool call
# ---------------------------------------------------------------------------

def result_digest(result: Any) -> str:
    """A tool result's fingerprint: what replay compares, key order aside."""
    blob = json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class ToolOutcome:
    result: dict[str, Any]              # what the model is shown
    is_write: bool
    actions: list[Action]               # what the call added to the action log
    world_seq: int                      # writes committed to the customer before it read
    changes: dict[str, Any] | None = None
    customer_seq: int | None = None     # the write it committed as, if it changed anything
    allocated: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    attempts: int = 1
    seconds: float = 0.0

    @property
    def result_sha(self) -> str:
        return result_digest(self.result)


def _logged(name: str, args: Any) -> dict[str, Any]:
    """A call's arguments as the tools log a refused write: every parameter of
    the tool, None where not given."""
    if not isinstance(args, dict):
        return {}
    props = TOOLS[name]["schema"]["function"]["parameters"]["properties"] if name in TOOLS else args
    return {k: args.get(k) for k in props}


def _sgd(amount_minor: int, currency: str) -> Fraction:
    """An amount's value in SGD at the platform's rates (db.SGD_RATE), exactly:
    the money guard compares sums across currencies, and floats would let a
    sum that equals the order's total read as a hair over it, or under."""
    return (Fraction(int(amount_minor), 10 ** CURRENCY_EXPONENT[currency])
            * Fraction(str(SGD_RATE[currency])))


def _whole(amount: Any) -> bool:
    """An amount the tools' schema takes: a positive whole number. Anything
    else is the tool's to refuse, in its words."""
    return (not isinstance(amount, bool) and isinstance(amount, (int, float))
            and (not isinstance(amount, float) or amount.is_integer()) and amount > 0)


def money_guard(db: Database, name: str, args: Any) -> str | None:
    """The platform's own rules about money, beside the policy the model
    follows, checked once the tool's own answer would not come first (a
    customer the tools would refuse as unverified, an amount they would
    refuse as no integer):

    - store credit is given in the customer's account currency (P5.3): credit
      in another would be added to her balance as the wrong minor units;
    - what has gone back for an order -- its refunds, and store credit naming
      it, in any currency, valued in SGD at the platform's rates -- never
      passes what was paid for it.

    Credit that names no order is the policy's to bound, not the platform's:
    there is nothing paid to measure it against."""
    if name not in ("issue_refund", "issue_store_credit") or not isinstance(args, dict):
        return None
    amount = args.get("amount_minor")
    if not _whole(amount):
        return None
    currency = None
    if name == "issue_store_credit":
        uid = args.get("user_id")
        u = db.row("users", uid) if isinstance(uid, str) else None
        if u is None or uid not in db.verified or args.get("currency") not in CURRENCIES:
            return None
        currency = args["currency"]
        if currency != u["store_credit_currency"]:
            return (f"refused by the platform: store credit is given in the account's "
                    f"currency, {u['store_credit_currency']} (P5.3). Nothing was paid out.")
    order_id = args.get("order_id")
    if not isinstance(order_id, str) or not order_id:
        return None
    o = db.row("orders", order_id)
    if o is None or o["user_id"] not in db.verified:
        return None                 # the tool's own answer comes first
    currency = currency or o["currency"]          # a refund is in the order's (P5.2)
    back = (sum(_sgd(r["amount_minor"], r["currency"]) for r in db.where("refunds",
                                                                         order_id=order_id))
            + sum(_sgd(g["amount_minor"], g["currency"])
                  for g in db.where("store_credit_grants", order_id=order_id)))
    total = _sgd(o["total_minor"], o["currency"])
    if back + _sgd(int(amount), currency) > total:
        rate = _sgd(1, o["currency"])
        gone = fmt_money(int(back / rate), o["currency"])
        return (f"refused by the platform: {gone} of {fmt_money(o['total_minor'], o['currency'])} "
                f"has already gone back for order {order_id}, and this would take it past what "
                f"was paid. Nothing was paid out.")
    return None


def isolation_guard(changes: dict[str, Any], own_orders: set[str]) -> str | None:
    """A new row that names an order names one of this customer's. The tools
    cannot see another customer's orders (the slice holds none), but a store
    credit takes any order id it is given and writes it down; on the platform
    it may not."""
    for rows in changes["insert"].values():
        for row in rows.values():
            oid = row.get("order_id")
            if oid not in (None, "") and oid not in own_orders:
                return (f"refused by the platform: {oid} is not one of this customer's "
                        f"orders. Nothing was written.")
    return None


INT64 = 2 ** 63


def value_guard(changes: dict[str, Any]) -> str | None:
    """Amounts the store can hold: a new row's amount is a positive whole
    number of minor units, and no number leaves a 64-bit column's range. The
    tools grant a voucher of 0, or a credit of 10**19; a store refuses them."""
    for table, rows in changes["insert"].items():
        for row in rows.values():
            amount = row.get("amount_minor")
            if amount is not None and (not isinstance(amount, int) or isinstance(amount, bool)
                                       or not 0 < amount < INT64):
                return (f"refused by the platform: an amount is a positive whole number of "
                        f"minor units below {INT64:,}, not {amount!r}. Nothing was written.")
    for rows in (*changes["insert"].values(), *(
            {rid: {k: v[1] for k, v in f.items()} for rid, f in t.items()}
            for t in changes["update"].values())):
        for row in rows.values():
            for k, v in row.items():
                if isinstance(v, int) and not isinstance(v, bool) and not -INT64 <= v < INT64:
                    return (f"refused by the platform: {k} would be {v}, past what the store "
                            f"holds. Nothing was written.")
    return None


def _rule(e: Exception) -> str:
    """What a refused write broke, in the same words on either database --
    replay runs on SQLite whatever the call ran on -- and never the values it
    carried."""
    return "a value out of range" if isinstance(e, DataError) else "a rule of the store"


def _retryable(exc: BaseException) -> bool:
    """A conflict this call can be run again after: a row changed under it, or
    PostgreSQL gave up on the transaction (serialization failure, deadlock)."""
    if isinstance(exc, Conflict):
        return True
    return getattr(getattr(exc, "orig", None), "sqlstate", None) in ("40001", "40P01")


Record = Callable[[Connection, "ToolOutcome"], None]


def run_tool(engine: Engine, *, tenant_id: str, user_id: str, name: str, args: Any,
             verified: list[str], prior_actions: list[Action], strict_money: bool = True,
             record: Record | None = None, replay_ids: list[str] | None = None,
             lock: bool = True, attempts: int = 3,
             hooks: dict[str, Callable[[], None]] | None = None) -> ToolOutcome:
    """One tool call against the customer's slice, in its own transaction.

    `verified` and `prior_actions` are the conversation's: who it has verified
    and what it has called so far (initiate_return reads the latter).
    `record(conn, outcome)` writes the conversation's side -- the tool message,
    the audit rows -- in the same transaction as the world's change, so a call
    is either done and recorded or neither. A conflict runs the call again, up
    to `attempts` times, on what is there now.

    `replay_ids`: the ids the call was given when it ran (replay.py). `lock`
    and `hooks` are for tests: the customer's lock taken or not, and
    callbacks at two points of the transaction -- "after_read" (the slice read
    and the tool run, nothing written) and "before_commit" -- to hold one
    transaction open while another runs."""
    t0 = time.monotonic()
    for attempt in range(1, attempts + 1):
        try:
            return _run_once(engine, tenant_id, user_id, name, args, verified, prior_actions,
                             strict_money, record, replay_ids, lock, hooks or {}, attempt, t0)
        except Exception as e:  # noqa: BLE001 -- re-raised unless it is a conflict
            if attempt == attempts or not _retryable(e):
                raise
            slices(engine).forget(tenant_id, user_id)       # run again on what is there now
            time.sleep(random.random() * 0.005 * (2 ** attempt))
    raise AssertionError("unreachable")


def _run_once(engine: Engine, tenant_id: str, user_id: str, name: str, args: Any,
              verified: list[str], prior_actions: list[Action], strict_money: bool,
              record: Record | None, replay_ids: list[str] | None, lock: bool,
              hooks: dict[str, Callable[[], None]], attempt: int, t0: float) -> ToolOutcome:
    is_write = name in WRITE_TOOLS
    # SQLite: a transaction that writes anything -- the world, or only the
    # audit rows -- takes the write lock at BEGIN. PostgreSQL: a read reads one
    # snapshot (REPEATABLE READ), so the slice it loads is one state of the
    # world; a write reads under the customer's lock, the latest state.
    with transaction(engine, write=is_write or record is not None,
                     snapshot=not is_write) as conn:
        who = {"tenant": tenant_id, "user": user_id}
        # The customer's lock: every write to her world waits here.
        world_seq = conn.execute(_Q_SEQ_LOCKED if is_write and lock and not is_sqlite(engine)
                                 else _Q_SEQ, who).scalar_one_or_none()
        if world_seq is None:
            raise StoreError(f"no customer {user_id!r} in tenant {tenant_id!r}")
        world_seq = int(world_seq)
        cache, kept = catalog(engine), slices(engine)
        # Her slice as of this write_seq: read before, or read now and kept.
        # `before` is never changed (a cached slice is shared); the tool runs
        # on a copy of it.
        hit = kept.get(tenant_id, user_id, world_seq)
        if hit is None:
            before, versions = load_slice(conn, tenant_id, user_id, cache)
            kept.put(tenant_id, user_id, world_seq, before, versions)
        else:
            before, versions = hit
        tables = copy_world(before)
        given = None if replay_ids is None else list(replay_ids)
        spare = [0]

        def allocate(prefix: str) -> str:
            if given is not None:
                if not given:
                    # A call that kept no id (refused after it asked for one):
                    # what it is given is thrown away with its rows -- or, if
                    # this time it writes them, replay reports the difference.
                    spare[0] += 1
                    return f"{prefix}~{spare[0]}"
                if not given[0].startswith(prefix):
                    raise StoreError(f"replay: the call asked for a {prefix} id where its "
                                     f"log holds {given[0]}")
                return given.pop(0)
            if is_sqlite(engine):
                # One writer at a time: the counter's row costs nothing to hold.
                n = conn.execute(_Q_COUNTER, {"tenant": tenant_id}).scalar_one()
            else:
                # The tenant's sequence: its next value at once, with no lock held
                # to commit. The counter's row was held from a write's first new id
                # to its commit, and every write of the tenant that made a row
                # waited behind it (docs/SERVICE.md, Measured). What a sequence
                # gives is never taken back: a call rolled back -- a conflict run
                # again -- leaves a gap, which replay does not mind (it gives each
                # call its recorded ids), and a fresh tenant's one conversation
                # still names what it makes as the benchmark's episode does.
                n = conn.execute(_Q_NEXTVAL, {"seq": id_sequence(tenant_id)}).scalar_one()
            return f"{prefix}{int(n):04d}"

        db = StoreDatabase(conn, tenant_id, tables, allocate, cache)
        db.verified = set(verified)
        db.action_log = list(prior_actions)
        n_before = len(db.action_log)
        refusal = money_guard(db, name, args) if strict_money else None
        if refusal is not None:
            db.log(name, _logged(name, args), ok=False, error=refusal)
            result: dict[str, Any] = {"ok": False, "error": refusal}
        else:
            result = tool_call(db, name, args, checks=CHECKS)
        changes = diff(before, db.tables)
        if changes["insert"] or changes["update"]:
            refusal = isolation_guard(changes, set(before["orders"]))
            if refusal is not None:
                db.action_log[n_before:] = [Action(tool=name, args=_logged(name, args),
                                                   ok=False, error=refusal)]
                result = {"ok": False, "error": refusal}
                changes = {"insert": {}, "update": {}}
        if "after_read" in hooks:
            hooks["after_read"]()
        seq = None
        if changes["insert"] or changes["update"]:
            if not is_write:
                raise StoreError(f"read tool {name} changed the world")
            refusal = value_guard(changes)
            if refusal is None:
                try:
                    # A savepoint: a change the database refuses (a rule no guard
                    # foresaw) is undone alone, and the call is still recorded,
                    # as refused, in this transaction.
                    with conn.begin_nested():
                        seq = int(conn.execute(_Q_BUMP, who).scalar_one())
                        apply_changes(conn, tenant_id, changes, versions, seq)
                        # The transactional outbox: the ticket a human picks up is
                        # written with the escalation, or not at all.
                        for row in changes["insert"].get("escalations", {}).values():
                            conn.execute(_Q_OUTBOX, {"tenant_id": tenant_id,
                                                     "topic": "escalation.created",
                                                     "payload": {"user_id": user_id, **row}})
                except (IntegrityError, DataError) as e:
                    seq = None
                    refusal = (f"refused by the platform: the store does not accept this "
                               f"change ({_rule(e)}). Nothing was written.")
            if refusal is not None:
                db.action_log[n_before:] = [Action(tool=name, args=_logged(name, args),
                                                   ok=False, error=refusal)]
                result = {"ok": False, "error": refusal}
                changes = {"insert": {}, "update": {}}
        outcome = ToolOutcome(result=result, is_write=is_write,
                              actions=db.action_log[n_before:], world_seq=int(world_seq),
                              changes=changes if seq is not None else None, customer_seq=seq,
                              allocated=list(db.allocated) if seq is not None else [],
                              verified=sorted(db.verified), attempts=attempt,
                              seconds=time.monotonic() - t0)
        if "before_commit" in hooks:
            hooks["before_commit"]()
        if record is not None:
            record(conn, outcome)
    return outcome


# ---------------------------------------------------------------------------
# The whole tenant, for the benchmark's verifier
# ---------------------------------------------------------------------------

def tenant_world(conn: Connection, tenant_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Every world row the tenant holds, keyed as the benchmark's tables are."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for table, (t, key) in S.WORLD_TABLES.items():
        out[table] = {}
        for r in conn.execute(select(t).where(t.c.tenant_id == tenant_id)
                              .order_by(t.c.pos, t.c[key])):
            d = _plain(table, r)
            out[table][d[key]] = d
    out.setdefault("notes", {})
    return out


def conversation_actions(conn: Connection, conversation_id: str,
                         before_call: int | None = None) -> list[Action]:
    """The conversation's action log, as the benchmark's Database keeps it;
    with `before_call`, as it stood when that call ran."""
    rows = (conn.execute(_Q_ACTIONS, {"cid": conversation_id}) if before_call is None else
            conn.execute(_Q_ACTIONS_BEFORE, {"cid": conversation_id, "before": before_call}))
    return [Action(tool=r.tool, args=dict(r.args or {}), ok=bool(r.ok), error=r.error,
                   denied=bool(r.denied)) for r in rows]


_KEY = re.compile(r"[^A-Za-z0-9_.:-]")


def safe_id(text: str, limit: int = 64) -> str:
    return _KEY.sub("-", text)[:limit]
