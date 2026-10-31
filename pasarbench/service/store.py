"""
The commerce world in SQL, and how a tool call reads and changes it.

The benchmark's tools (pasarbench/tools.py) are pure functions over an
in-memory world, and they are what the benchmark scores. The service runs
those same functions: each tool call loads one customer's slice of the world
into a `pasarbench.db.Database`, runs the tool on it, and writes the rows it
changed back -- in one short transaction, never across a model call -- with
the conversation's record of the call (the tool's message, the audit rows) in
the same transaction, so a call is either done and recorded, or neither.

    read tool   one snapshot of the slice (REPEATABLE READ on PostgreSQL), no lock
    write tool  the customer's row locked first (SELECT ... FOR UPDATE on
                PostgreSQL; on SQLite every writing transaction begins
                IMMEDIATE and holds the database), the slice loaded under the
                lock, the tool run, each changed row updated only if its
                version is still the one read (else the call runs again, on
                what is there now), each new row inserted, the customer's write
                sequence advanced

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
import math
import random
import re
import threading
import time
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from sqlalchemy import Connection, Engine, create_engine, event, insert, select, text, update
from sqlalchemy.exc import DataError, IntegrityError

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


def make_engine(url: str, *, pool_size: int = 10, max_overflow: int = 20,
                echo: bool = False) -> Engine:
    """An engine for SQLite or PostgreSQL.

    SQLite: one file, WAL so readers do not block the writer, foreign keys on,
    a 30-second busy wait, and transactions that say whether they write: a
    write begins IMMEDIATE, taking the write lock before it reads, so two
    writers cannot both read a balance and both add to it."""
    if url.startswith("sqlite"):
        engine = create_engine(url, echo=echo, pool_size=pool_size, max_overflow=max_overflow,
                               hide_parameters=True, json_serializer=_json,
                               connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _connect(dbapi_conn, _record):            # noqa: ANN001
            dbapi_conn.isolation_level = None         # BEGIN is ours (below)
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
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
    return create_engine(url, echo=echo, pool_size=pool_size, max_overflow=max_overflow,
                         pool_pre_ping=True, pool_recycle=1800, hide_parameters=True,
                         json_serializer=_json, connect_args={"options": "-c timezone=UTC"})


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
            yield conn


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
    with transaction(engine, write=True) as conn:
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

def _plain(table: str, row: Any) -> dict[str, Any]:
    d = dict(row._mapping)
    for c in S.SERVICE_COLUMNS:
        d.pop(c, None)
    return d


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


def _catalog_rows(conn: Connection, cache: Catalog | None, tenant_id: str, table: str,
                  ids: list[str]) -> list[dict[str, Any]]:
    """Catalog rows by id, in their table's order: from the cache, the rest
    from the database."""
    t, key = S.WORLD_TABLES[table]
    got: list[tuple[int, dict[str, Any]]] = []
    missing = []
    for i in ids:
        hit = cache.get((tenant_id, table, i)) if cache else None
        (got.append(hit) if hit is not None else missing.append(i))
    if missing:
        for r in conn.execute(select(t).where(t.c.tenant_id == tenant_id, t.c[key].in_(missing))):
            entry = (int(r.pos), _plain(table, r))
            if cache:
                cache.put((tenant_id, table, entry[1][key]), entry)
            got.append(entry)
    return [dict(row) for _pos, row in sorted(got, key=lambda e: e[0])]


def _claims(conn: Connection, cache: Catalog | None, tenant_id: str,
            livestream_id: str) -> list[dict[str, Any]]:
    hit = cache.get((tenant_id, "claims_of", livestream_id)) if cache else None
    if hit is None:
        lc = S.livestream_claims
        hit = [_plain("livestream_claims", r) for r in conn.execute(select(lc).where(
            lc.c.tenant_id == tenant_id, lc.c.livestream_id == livestream_id)
            .order_by(lc.c.pos))]
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
        t, key = S.WORLD_TABLES[table]
        out = []
        for r in rows:
            m = r._mapping
            if table in VERSIONED:
                versions[(table, m[key])] = m["version"]
            d = _plain(table, r)
            tables[table][d[key]] = d
            out.append(d)
        return out

    c, o = S.customers, S.orders
    put("users", conn.execute(select(c).where(c.c.tenant_id == tenant_id,
                                              c.c.user_id == user_id)))
    orders = put("orders", conn.execute(select(o).where(o.c.tenant_id == tenant_id,
                                                        o.c.user_id == user_id)
                                        .order_by(o.c.pos)))
    order_ids = [r["order_id"] for r in orders]
    if order_ids:
        for table in ("order_items", "shipments", "payments", "refunds", "returns",
                      "escalations", "seller_messages"):
            t, _ = S.WORLD_TABLES[table]
            put(table, conn.execute(select(t).where(t.c.tenant_id == tenant_id,
                                                    t.c.order_id.in_(order_ids))
                                    .order_by(t.c.pos)))
    for table in ("store_credit_grants", "vouchers"):
        t, _ = S.WORLD_TABLES[table]
        put(table, conn.execute(select(t).where(t.c.tenant_id == tenant_id,
                                                t.c.user_id == user_id).order_by(t.c.pos)))
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
        c = S.customers
        q = select(c.c.write_seq).where(c.c.tenant_id == tenant_id, c.c.user_id == user_id)
        if is_write and lock and not is_sqlite(engine):
            q = q.with_for_update()     # the customer's lock: every write to her world waits here
        world_seq = conn.execute(q).scalar_one_or_none()
        if world_seq is None:
            raise StoreError(f"no customer {user_id!r} in tenant {tenant_id!r}")
        cache = catalog(engine)
        tables, versions = load_slice(conn, tenant_id, user_id, cache)
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
                k = S.id_counters
                n = conn.execute(update(k).where(k.c.tenant_id == tenant_id)
                                 .values(value=k.c.value + 1).returning(k.c.value)).scalar_one()
            else:
                # The tenant's sequence: its next value at once, with no lock held
                # to commit. The counter's row was held from a write's first new id
                # to its commit, and every write of the tenant that made a row
                # waited behind it (docs/SERVICE.md, Measured). What a sequence
                # gives is never taken back: a call rolled back -- a conflict run
                # again -- leaves a gap, which replay does not mind (it gives each
                # call its recorded ids), and a fresh tenant's one conversation
                # still names what it makes as the benchmark's episode does.
                n = conn.execute(text("SELECT nextval(CAST(:seq AS regclass))"),
                                 {"seq": id_sequence(tenant_id)}).scalar_one()
            return f"{prefix}{int(n):04d}"

        db = StoreDatabase(conn, tenant_id, tables, allocate, cache)
        db.verified = set(verified)
        db.action_log = list(prior_actions)
        n_before = len(db.action_log)
        before = copy.deepcopy(db.tables)
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
                        seq = int(conn.execute(update(c).where(c.c.tenant_id == tenant_id,
                                                               c.c.user_id == user_id)
                                               .values(write_seq=c.c.write_seq + 1)
                                               .returning(c.c.write_seq)).scalar_one())
                        apply_changes(conn, tenant_id, changes, versions, seq)
                        # The transactional outbox: the ticket a human picks up is
                        # written with the escalation, or not at all.
                        for row in changes["insert"].get("escalations", {}).values():
                            conn.execute(insert(S.outbox).values(
                                tenant_id=tenant_id, topic="escalation.created",
                                payload={"user_id": user_id, **row}))
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
    a = S.actions
    q = select(a).where(a.c.conversation_id == conversation_id)
    if before_call is not None:
        q = q.where(a.c.call_seq < before_call)
    return [Action(tool=r.tool, args=dict(r.args or {}), ok=bool(r.ok), error=r.error,
                   denied=bool(r.denied)) for r in conn.execute(q.order_by(a.c.seq))]


_KEY = re.compile(r"[^A-Za-z0-9_.:-]")


def safe_id(text: str, limit: int = 64) -> str:
    return _KEY.sub("-", text)[:limit]
