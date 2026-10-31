"""What the processes of a deployment share (v82): who is serving
(service_nodes), a request bucket a tenant (rate_limits) and the providers'
breakers that are open (provider_breakers) -- schema.py, limits.py.

UNLOGGED on PostgreSQL, as schema.py makes them: none of it is worth a disk
write a request, and after a crash it starts again empty. A process of v81's
on a database migrated to 0003 does not read them; one of v82's on a database
not yet migrated keeps each limit its own, says so in its log, and counts it
(pasar_limits_errors_total) -- migrate before, or as, v82 rolls out.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-01
"""

from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0003'
down_revision: str | Sequence[str] | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("service_nodes", "rate_limits", "provider_breakers")


def upgrade() -> None:
    op.create_table('service_nodes',
    sa.Column('node', sa.String(length=64), nullable=False),
    sa.Column('host', sa.String(length=255), nullable=False),
    sa.Column('pid', sa.Integer(), nullable=False),
    sa.Column('version', sa.String(length=32), nullable=False),
    sa.Column('started_at', sa.Float(), nullable=False),
    sa.Column('seen_at', sa.Float(), nullable=False),
    sa.Column('expires_at', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('node', name=op.f('pk_service_nodes'))
    )
    op.create_table('rate_limits',
    sa.Column('bucket', sa.String(length=128), nullable=False),
    sa.Column('level', sa.Float(), nullable=False),
    sa.Column('at', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('bucket', name=op.f('pk_rate_limits'))
    )
    op.create_table('provider_breakers',
    sa.Column('provider', sa.String(length=64), nullable=False),
    sa.Column('open_until', sa.Float(), nullable=False),
    sa.Column('opened_at', sa.Float(), nullable=False),
    sa.Column('opened_by', sa.String(length=64), nullable=False),
    sa.PrimaryKeyConstraint('provider', name=op.f('pk_provider_breakers'))
    )
    if op.get_bind().dialect.name == "postgresql":
        for name in TABLES:
            op.execute(f"ALTER TABLE {name} SET UNLOGGED")


def downgrade() -> None:
    for name in reversed(TABLES):
        op.drop_table(name)
