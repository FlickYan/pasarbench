"""Ids from a sequence per tenant, on PostgreSQL (v79).

A tenant's new ids came from its row of id_counters, updated in the tool
call's own transaction and so locked until that call committed: every write
of the tenant that made a row waited for every other one (docs/SERVICE.md,
Measured). On PostgreSQL they now come from a sequence per tenant
(store.id_sequence), which hands out its next value at once and holds
nothing. This makes one for each tenant there is, starting after its counter;
store.seed_tenant makes one with each new tenant. SQLite keeps the counter: it
has one writer at a time anyway.

Down: each counter takes back its sequence's last value, and the sequences go.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-31
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0002'
down_revision: str | Sequence[str] | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _sequence(tenant_id: str) -> str:
    """store.id_sequence as it is at this revision: a migration keeps its own
    copy, so that a later change to the app cannot change what it did."""
    return "ids_" + hashlib.sha256(tenant_id.encode()).hexdigest()[:24]


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for tenant_id, value in bind.execute(sa.text(
            "SELECT tenant_id, value FROM id_counters")).all():
        op.execute(f"CREATE SEQUENCE IF NOT EXISTS {_sequence(tenant_id)} "
                   f"MINVALUE 1 START {int(value) + 1}")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for (tenant_id,) in bind.execute(sa.text("SELECT tenant_id FROM id_counters")).all():
        name = _sequence(tenant_id)
        if bind.execute(sa.text("SELECT to_regclass(:n)"), {"n": name}).scalar() is None:
            continue
        last, called = bind.execute(sa.text(f"SELECT last_value, is_called FROM {name}")).one()
        bind.execute(sa.text("UPDATE id_counters SET value = :v WHERE tenant_id = :t"),
                     {"v": int(last) if called else int(last) - 1, "t": tenant_id})
        op.execute(f"DROP SEQUENCE {name}")
