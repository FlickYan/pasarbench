"""
The service's tables (SQLAlchemy Core), for SQLite and PostgreSQL alike.

Two groups. The commerce world -- customers, orders, payments, refunds and the
rest -- holds exactly the rows and fields the benchmark's world holds
(pasarbench/db.py), one table each, every row keyed by its tenant first: two
tenants may use the same order id and never see each other's. The service's
own tables hold what a live deployment needs and the benchmark never did:
tenants and their keys, conversations and their messages, the audit log of
every tool call, idempotency keys, the usage ledger, the price book, the id
counter and the outbox.

Money is integer minor units in its currency, as in the benchmark (IDR and VND
have no minor unit); model cost is integer picodollars (costs.py). Times of the
world are kept as the benchmark writes them, "YYYY-MM-DD HH:MM" in Singapore
time, because the tools read them that way; the service's own times are
timestamps with a zone.
"""

from __future__ import annotations

from sqlalchemy import (JSON, BigInteger, Boolean, CheckConstraint, Column, DateTime, Float,
                        ForeignKeyConstraint, Index, Integer, MetaData, String, Table, Text,
                        func)
from sqlalchemy.dialects.postgresql import JSONB

# Stable constraint names, so a migration names what it creates the same way on
# every database (Alembic's recommendation).
metadata = MetaData(naming_convention={
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
})

# JSONB on PostgreSQL (indexable, compact), JSON text elsewhere.
Json = JSON().with_variant(JSONB(), "postgresql")


def _now():
    return Column("created_at", DateTime(timezone=True), nullable=False,
                  server_default=func.now())


def _tenant():
    return Column("tenant_id", String(64), primary_key=True)


def _pos():
    """Where the row stands in its table's order: the order the benchmark's
    dicts keep -- the seed's, then each new row in the order it was made -- so
    a tool that lists rows lists them as the benchmark's would, on either
    database (PostgreSQL returns rows in no order unless asked, and moves a
    row when it is updated)."""
    return Column("pos", BigInteger, nullable=False)


def _tenant_fk(table: str):
    return ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE",
                                name=f"fk_{table}_tenant_id_tenants")


ORDER_STATUSES = ("pending", "processing", "shipped", "delivered", "cancelled")
ITEM_STATUSES = ("pending", "processing", "shipped", "delivered", "cancelled",
                 "return_requested")
PAYMENT_STATUSES = ("pending", "authorized", "captured", "collected", "refunded", "released")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


# ---------------------------------------------------------------------------
# Tenants and their keys
# ---------------------------------------------------------------------------

tenants = Table(
    "tenants", metadata,
    Column("tenant_id", String(64), primary_key=True),
    Column("name", String(200), nullable=False),
    # What the world was built from ("demo", "task:<id>", "synthetic:<n>") and
    # its digest: replay rebuilds the starting world from this, not from a copy.
    Column("world", String(100), nullable=False),
    Column("world_digest", String(16), nullable=False),
    # A spend cap a day, in picodollars; none when null (costs.py).
    Column("daily_budget_picousd", BigInteger),
    _now(),
)

api_keys = Table(
    "api_keys", metadata,
    Column("key_id", String(32), primary_key=True),
    Column("tenant_id", String(64), nullable=False),
    # SHA-256 of the key: the key itself is shown once and never stored.
    Column("key_hash", String(64), nullable=False, unique=True),
    _now(),
    Column("revoked_at", DateTime(timezone=True)),
    _tenant_fk("api_keys"),
)

# A tenant's ids for new rows (RET0001, REF0002, ...), one count for every
# prefix as the benchmark's Database has, so a conversation on a fresh tenant
# names what it creates exactly as the benchmark does. On SQLite this row is the
# count. On PostgreSQL the tenant's sequence is (store.id_sequence, migration
# 0002): it gives a value without a lock held to commit; this row stays at 0
# there, and takes the sequence's value back if 0002 is undone.
id_counters = Table(
    "id_counters", metadata,
    _tenant(),
    Column("value", BigInteger, nullable=False, server_default="0"),
    _tenant_fk("id_counters"),
)

# ---------------------------------------------------------------------------
# The commerce world (pasarbench/db.py), one row per benchmark row
# ---------------------------------------------------------------------------

customers = Table(
    "customers", metadata,
    _tenant(),
    Column("user_id", String(64), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("email", String(320), nullable=False),
    Column("phone", String(32), nullable=False),
    Column("country", String(2), nullable=False),
    Column("language", String(16), nullable=False),
    Column("tier", String(16), nullable=False),
    Column("joined", String(10), nullable=False),
    Column("store_credit_minor", BigInteger, nullable=False),
    Column("store_credit_currency", String(3), nullable=False),
    Column("verified_phone_last4", String(4), nullable=False),
    # Optimistic version of the row, and the sequence of writes made to this
    # customer's world: the lock every write takes is on this row.
    Column("version", Integer, nullable=False, server_default="1"),
    Column("write_seq", BigInteger, nullable=False, server_default="0"),
    _pos(),
    CheckConstraint("store_credit_minor >= 0", name="credit_nonnegative"),
    _tenant_fk("customers"),
)

sellers = Table(
    "sellers", metadata,
    _tenant(),
    Column("seller_id", String(64), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("country", String(2), nullable=False),
    Column("rating", Float, nullable=False),
    Column("cod_enabled", Boolean, nullable=False),
    Column("official_store", Boolean, nullable=False),
    _pos(),
    _tenant_fk("sellers"),
)

products = Table(
    "products", metadata,
    _tenant(),
    Column("product_id", String(64), primary_key=True),
    Column("seller_id", String(64), nullable=False),
    Column("title", String(300), nullable=False),
    Column("category", String(64), nullable=False),
    Column("price_minor", BigInteger, nullable=False),
    Column("currency", String(3), nullable=False),
    Column("stock", Integer, nullable=False),
    Column("hazmat", Boolean, nullable=False),
    Column("perishable", Boolean, nullable=False),
    _pos(),
    ForeignKeyConstraint(["tenant_id", "seller_id"], ["sellers.tenant_id", "sellers.seller_id"],
                         name="fk_products_seller_id_sellers"),
    _tenant_fk("products"),
)

orders = Table(
    "orders", metadata,
    _tenant(),
    Column("order_id", String(64), primary_key=True),
    Column("user_id", String(64), nullable=False),
    Column("seller_id", String(64), nullable=False),
    Column("status", String(16), nullable=False),
    Column("payment_method", String(16), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("subtotal_minor", BigInteger, nullable=False),
    Column("shipping_minor", BigInteger, nullable=False),
    Column("discount_minor", BigInteger, nullable=False),
    Column("total_minor", BigInteger, nullable=False),
    Column("created", String(16), nullable=False),
    Column("source", String(16), nullable=False),
    Column("livestream_id", String(64)),
    Column("shipping_address", Text, nullable=False),
    Column("version", Integer, nullable=False, server_default="1"),
    _pos(),
    CheckConstraint(_in("status", ORDER_STATUSES), name="status"),
    CheckConstraint("total_minor >= 0", name="total_nonnegative"),
    ForeignKeyConstraint(["tenant_id", "user_id"], ["customers.tenant_id", "customers.user_id"],
                         name="fk_orders_user_id_customers"),
    ForeignKeyConstraint(["tenant_id", "seller_id"], ["sellers.tenant_id", "sellers.seller_id"],
                         name="fk_orders_seller_id_sellers"),
    _tenant_fk("orders"),
)
Index("ix_orders_tenant_user", orders.c.tenant_id, orders.c.user_id)


def _order_child(name: str, id_column: str, *columns, versioned: bool = False,
                 checks: tuple = ()) -> Table:
    """A table of rows that belong to one order: its id, the order's, and an
    index to find them by the order, which is how every tool reads them."""
    t = Table(
        name, metadata,
        _tenant(),
        Column(id_column, String(64), primary_key=True),
        Column("order_id", String(64), nullable=False),
        *columns,
        *([Column("version", Integer, nullable=False, server_default="1")] if versioned else []),
        _pos(),
        *checks,
        ForeignKeyConstraint(["tenant_id", "order_id"], ["orders.tenant_id", "orders.order_id"],
                             name=f"fk_{name}_order_id_orders"),
        _tenant_fk(name),
    )
    Index(f"ix_{name}_tenant_order", t.c.tenant_id, t.c.order_id)
    return t


order_items = _order_child(
    "order_items", "order_item_id",
    Column("product_id", String(64), nullable=False),
    Column("qty", Integer, nullable=False),
    Column("unit_price_minor", BigInteger, nullable=False),
    Column("item_status", String(20), nullable=False),
    versioned=True,
    checks=(CheckConstraint("qty > 0", name="qty_positive"),
            CheckConstraint(_in("item_status", ITEM_STATUSES), name="item_status")),
)

shipments = _order_child(
    "shipments", "shipment_id",
    Column("courier", String(100), nullable=False),
    Column("tracking_no", String(100), nullable=False),
    Column("status", String(20), nullable=False),
    Column("shipped", String(16)),
    Column("delivered", String(16)),
    Column("customs_status", String(16), nullable=False),
    Column("last_scan", String(16)),
)

payments = _order_child(
    "payments", "payment_id",
    Column("method", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("amount_minor", BigInteger, nullable=False),
    Column("currency", String(3), nullable=False),
    Column("paid", String(16)),
    Column("instrument_last4", String(4)),
    versioned=True,
    checks=(CheckConstraint(_in("status", PAYMENT_STATUSES), name="status"),
            CheckConstraint("amount_minor >= 0", name="amount_nonnegative")),
)

refunds = _order_child(
    "refunds", "refund_id",
    Column("amount_minor", BigInteger, nullable=False),
    Column("currency", String(3), nullable=False),
    Column("method", String(16), nullable=False),
    Column("reason", Text, nullable=False),
    Column("status", String(16), nullable=False),
    Column("created", String(16), nullable=False),
    checks=(CheckConstraint("amount_minor > 0", name="amount_positive"),),
)

returns = _order_child(
    "returns", "return_id",
    Column("order_item_id", String(64), nullable=False),
    Column("reason", Text, nullable=False),
    Column("photo_evidence", Boolean, nullable=False),
    Column("status", String(16), nullable=False),
    Column("created", String(16), nullable=False),
)

escalations = _order_child(
    "escalations", "escalation_id",
    Column("category", String(32), nullable=False),
    Column("reason", Text, nullable=False),
    Column("status", String(16), nullable=False),
    Column("created", String(16), nullable=False),
)

seller_messages = _order_child(
    "seller_messages", "message_id",
    Column("seller_id", String(64), nullable=False),
    Column("message", Text, nullable=False),
    Column("created", String(16), nullable=False),
)

livestream_claims = Table(
    "livestream_claims", metadata,
    _tenant(),
    Column("claim_id", String(64), primary_key=True),
    Column("livestream_id", String(64), nullable=False),
    Column("seller_id", String(64), nullable=False),
    Column("product_id", String(64), nullable=False),
    Column("timestamp", String(16), nullable=False),
    Column("claim_text", Text, nullable=False),
    Column("verified_fulfilled", Boolean, nullable=False),
    _pos(),
    _tenant_fk("livestream_claims"),
)
Index("ix_livestream_claims_tenant_live", livestream_claims.c.tenant_id,
      livestream_claims.c.livestream_id)


def _customer_child(name: str, id_column: str, *columns, checks: tuple = ()) -> Table:
    t = Table(
        name, metadata,
        _tenant(),
        Column(id_column, String(64), primary_key=True),
        Column("user_id", String(64), nullable=False),
        *columns,
        _pos(),
        *checks,
        ForeignKeyConstraint(["tenant_id", "user_id"],
                             ["customers.tenant_id", "customers.user_id"],
                             name=f"fk_{name}_user_id_customers"),
        _tenant_fk(name),
    )
    Index(f"ix_{name}_tenant_user", t.c.tenant_id, t.c.user_id)
    return t


store_credit_grants = _customer_child(
    "store_credit_grants", "grant_id",
    Column("amount_minor", BigInteger, nullable=False),
    Column("currency", String(3), nullable=False),
    Column("reason", Text, nullable=False),
    # The order it was for, if any -- not a foreign key: the tool accepts an id
    # it cannot find, and the row says what the agent wrote.
    Column("order_id", String(64)),
    Column("created", String(16), nullable=False),
    checks=(CheckConstraint("amount_minor > 0", name="amount_positive"),),
)

vouchers = _customer_child(
    "vouchers", "voucher_id",
    Column("amount_minor", BigInteger, nullable=False),
    Column("currency", String(3), nullable=False),
    Column("reason", Text, nullable=False),
    Column("created", String(16), nullable=False),
    checks=(CheckConstraint("amount_minor > 0", name="amount_positive"),),
)

# The benchmark's table name -> the service's table and its key column.
WORLD_TABLES: dict[str, tuple[Table, str]] = {
    "users": (customers, "user_id"),
    "sellers": (sellers, "seller_id"),
    "products": (products, "product_id"),
    "orders": (orders, "order_id"),
    "order_items": (order_items, "order_item_id"),
    "shipments": (shipments, "shipment_id"),
    "payments": (payments, "payment_id"),
    "livestream_claims": (livestream_claims, "claim_id"),
    "refunds": (refunds, "refund_id"),
    "returns": (returns, "return_id"),
    "escalations": (escalations, "escalation_id"),
    "seller_messages": (seller_messages, "message_id"),
    "store_credit_grants": (store_credit_grants, "grant_id"),
    "vouchers": (vouchers, "voucher_id"),
}
# Insert order, parents first, for the foreign keys.
WORLD_ORDER = ("users", "sellers", "products", "orders", "order_items", "shipments",
               "payments", "livestream_claims", "refunds", "returns", "escalations",
               "seller_messages", "store_credit_grants", "vouchers")
# Columns the service adds to a world row, never shown to the tools.
SERVICE_COLUMNS = frozenset({"tenant_id", "version", "write_seq", "pos"})

# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

conversations = Table(
    "conversations", metadata,
    Column("conversation_id", String(32), primary_key=True),
    Column("tenant_id", String(64), nullable=False),
    Column("user_id", String(64), nullable=False),
    Column("market", String(2), nullable=False),
    Column("status", String(16), nullable=False, server_default="open"),
    # What produced it, fixed when it opens: the model and its settings, and
    # digests of the prompt, the policy and the tool schemas it was shown;
    # what the tools and the platform's guards checked (replay runs the calls
    # under the same); the world's clock, which the tools read.
    Column("model", String(200), nullable=False),
    Column("model_settings", Json, nullable=False),
    Column("policy_mode", String(16), nullable=False),
    Column("prompt_digest", String(16), nullable=False),
    Column("policy_digest", String(16), nullable=False),
    Column("tools_digest", String(16), nullable=False),
    Column("tool_checks", Integer, nullable=False),
    Column("strict_money", Boolean, nullable=False),
    Column("world_clock", String(32), nullable=False),
    # Customers verified in it (identity is verified per conversation).
    Column("verified", Json, nullable=False),
    Column("turns", Integer, nullable=False, server_default="0"),
    Column("steps", Integer, nullable=False, server_default="0"),
    Column("tool_calls", Integer, nullable=False, server_default="0"),
    Column("n_messages", Integer, nullable=False, server_default="0"),
    Column("n_actions", Integer, nullable=False, server_default="0"),
    Column("prompt_tokens", BigInteger, nullable=False, server_default="0"),
    Column("cached_tokens", BigInteger, nullable=False, server_default="0"),
    Column("completion_tokens", BigInteger, nullable=False, server_default="0"),
    Column("cost_picousd", BigInteger, nullable=False, server_default="0"),
    # The turn lease: one turn at a time, across processes (app.py).
    Column("lease_owner", String(64)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("last_stop", String(32)),
    _now(),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("closed_at", DateTime(timezone=True)),
    CheckConstraint("status IN ('open', 'closed')", name="status"),
    ForeignKeyConstraint(["tenant_id", "user_id"], ["customers.tenant_id", "customers.user_id"],
                         name="fk_conversations_user_id_customers"),
    _tenant_fk("conversations"),
)
Index("ix_conversations_tenant_created", conversations.c.tenant_id, conversations.c.created_at)

# What the model sees, in order: the system prompt, her turns, its replies and
# tool calls, the tool results.
messages = Table(
    "messages", metadata,
    Column("conversation_id", String(32), primary_key=True),
    Column("seq", Integer, primary_key=True),
    Column("turn", Integer, nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("tool_calls", Json),
    # A provider's call id and the tool name a model sent: any length they come in.
    Column("tool_call_id", Text),
    Column("name", Text),
    # Written by the platform, not the model or the customer: the system
    # prompt, a fallback reply when a turn could not finish, the answer to a
    # tool call a crash left without one.
    Column("platform", Boolean, nullable=False, server_default="0"),
    _now(),
    ForeignKeyConstraint(["conversation_id"], ["conversations.conversation_id"],
                         ondelete="CASCADE", name="fk_messages_conversation_id_conversations"),
)

# Every tool call the model made, read or write, run or refused: what it was
# asked, what it answered, and enough to run it again exactly (replay.py) --
# the state of the customer's world it read (world_seq: how many writes had
# committed to it), who the conversation had verified, the ids it was given,
# and for a call that changed the world, the rows it changed and the write
# sequence it committed as.
tool_calls = Table(
    "tool_calls", metadata,
    Column("conversation_id", String(32), primary_key=True),
    Column("seq", Integer, primary_key=True),
    Column("tenant_id", String(64), nullable=False),
    Column("user_id", String(64), nullable=False),
    Column("turn", Integer, nullable=False),
    Column("step", Integer, nullable=False),
    Column("call_id", Text, nullable=False),
    Column("tool", Text, nullable=False),
    Column("args", Json, nullable=False),
    Column("ok", Boolean, nullable=False),
    Column("error", Text),
    Column("result_sha", String(16), nullable=False),
    Column("is_write", Boolean, nullable=False),
    # False for a call answered without reaching the tools: arguments that
    # were not JSON, a tool the agent is not shown, the turn's budget spent,
    # or a crash before it ran. Replay runs only the ones that reached them.
    Column("dispatched", Boolean, nullable=False),
    Column("world_seq", BigInteger),
    Column("verified_before", Json, nullable=False),
    Column("customer_seq", BigInteger),
    Column("changes", Json),
    Column("allocated_ids", Json),
    Column("attempts", Integer, nullable=False, server_default="1"),
    Column("latency_ms", Integer, nullable=False),
    _now(),
    ForeignKeyConstraint(["conversation_id"], ["conversations.conversation_id"],
                         ondelete="CASCADE", name="fk_tool_calls_conversation_id_conversations"),
)
Index("ix_tool_calls_customer", tool_calls.c.tenant_id, tool_calls.c.user_id,
      tool_calls.c.world_seq)

# The benchmark's action log, line for line (pasarbench/db.py Action): what the
# verifier reads. A call adds none (a read refused for its arguments), one, or
# two (a cancellation and the refund it settled).
actions = Table(
    "actions", metadata,
    Column("conversation_id", String(32), primary_key=True),
    Column("seq", Integer, primary_key=True),
    Column("call_seq", Integer, nullable=False),
    Column("tool", String(64), nullable=False),
    Column("args", Json, nullable=False),
    Column("ok", Boolean, nullable=False),
    Column("error", Text),
    Column("denied", Boolean, nullable=False),
    _now(),
    ForeignKeyConstraint(["conversation_id", "call_seq"],
                         ["tool_calls.conversation_id", "tool_calls.seq"],
                         ondelete="CASCADE", name="fk_actions_conversation_id_tool_calls"),
)

# Events a turn emits, in order (runtime.py): what an SSE stream sends, kept.
events = Table(
    "events", metadata,
    Column("event_id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True,
           autoincrement=True),
    Column("conversation_id", String(32), nullable=False),
    Column("turn", Integer, nullable=False),
    Column("type", String(32), nullable=False),
    Column("payload", Json, nullable=False),
    _now(),
    ForeignKeyConstraint(["conversation_id"], ["conversations.conversation_id"],
                         ondelete="CASCADE", name="fk_events_conversation_id_conversations"),
)
Index("ix_events_conversation", events.c.conversation_id, events.c.event_id)

# A request's Idempotency-Key, what it was for, and the answer it got. While
# the request runs, the key expires with the turn's lease; once answered, after
# PASAR_IDEMPOTENCY_HOURS.
idempotency_keys = Table(
    "idempotency_keys", metadata,
    Column("tenant_id", String(64), primary_key=True),
    Column("key", String(200), primary_key=True),
    Column("request_hash", String(64), nullable=False),
    Column("state", String(16), nullable=False),
    Column("status_code", Integer),
    # The body exactly as it was sent: a retry gets the same bytes back (a
    # JSON column would not keep them -- PostgreSQL's JSONB reorders keys).
    Column("response", Text),
    _now(),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("state IN ('in_progress', 'completed')", name="state"),
    _tenant_fk("idempotency_keys"),
)

# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

price_book = Table(
    "price_book", metadata,
    Column("price_id", String(200), primary_key=True),
    Column("provider", String(64), nullable=False),
    Column("model", String(200), nullable=False),
    # Microdollars a million tokens: exact for every price the plans list.
    Column("input_microusd_per_mtok", BigInteger, nullable=False),
    Column("cached_microusd_per_mtok", BigInteger, nullable=False),
    Column("output_microusd_per_mtok", BigInteger, nullable=False),
    Column("effective_from", String(10), nullable=False),
    Column("source", Text, nullable=False),
    Index("ix_price_book_model", "provider", "model", "effective_from", unique=True),
)

usage_ledger = Table(
    "usage_ledger", metadata,
    Column("entry_id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True,
           autoincrement=True),
    Column("tenant_id", String(64), nullable=False),
    Column("conversation_id", String(32), nullable=False),
    Column("turn", Integer, nullable=False),
    Column("step", Integer, nullable=False),
    Column("provider", String(64), nullable=False),
    Column("model", String(200), nullable=False),
    Column("price_id", String(200)),
    Column("prompt_tokens", BigInteger, nullable=False),
    Column("cached_tokens", BigInteger, nullable=False),
    Column("completion_tokens", BigInteger, nullable=False),
    # Counted by the service, not reported by a provider: the mock and the
    # scripted model, or a provider that sent no usage.
    Column("estimated", Boolean, nullable=False),
    Column("cost_picousd", BigInteger, nullable=False),
    Column("latency_ms", Integer, nullable=False),
    # What the provider says it ran, when it says: an alias can be re-pointed.
    Column("served_model", String(200)),
    _now(),
    ForeignKeyConstraint(["conversation_id"], ["conversations.conversation_id"],
                         ondelete="CASCADE", name="fk_usage_ledger_conversation_id_conversations"),
    _tenant_fk("usage_ledger"),
)
Index("ix_usage_ledger_tenant_created", usage_ledger.c.tenant_id, usage_ledger.c.created_at)
Index("ix_usage_ledger_conversation", usage_ledger.c.conversation_id)

# ---------------------------------------------------------------------------
# Outbox: what the world's changes tell the outside, written in the same
# transaction as the change (an escalation is a ticket a human picks up).
# ---------------------------------------------------------------------------

outbox = Table(
    "outbox", metadata,
    Column("event_id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True,
           autoincrement=True),
    Column("tenant_id", String(64), nullable=False),
    Column("topic", String(64), nullable=False),
    Column("payload", Json, nullable=False),
    _now(),
    Column("delivered_at", DateTime(timezone=True)),
    Column("attempts", Integer, nullable=False, server_default="0"),
    _tenant_fk("outbox"),
)
Index("ix_outbox_pending", outbox.c.delivered_at, outbox.c.event_id)
