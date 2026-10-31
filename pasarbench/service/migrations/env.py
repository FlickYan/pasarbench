"""
Alembic's environment: the service's tables (schema.py) and the database URL
it is given -- by `python -m pasarbench.service migrate`, or PASAR_DATABASE_URL
for the alembic command itself (alembic.ini at the repo root):

    PASAR_DATABASE_URL=... alembic revision --autogenerate -m "what changed"
    PASAR_DATABASE_URL=... alembic upgrade head

SQLite alters a table by copying it (render_as_batch); PostgreSQL in place.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from pasarbench.service.schema import metadata

config = context.config
URL = config.get_main_option("sqlalchemy.url") or os.environ.get("PASAR_DATABASE_URL", "")
if not URL:
    raise SystemExit("no database URL: set PASAR_DATABASE_URL")


def offline() -> None:
    context.configure(url=URL, target_metadata=metadata, literal_binds=True,
                      render_as_batch=URL.startswith("sqlite"), compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def online() -> None:
    engine = engine_from_config({"sqlalchemy.url": URL}, prefix="sqlalchemy.",
                                poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=metadata,
                          render_as_batch=connection.dialect.name == "sqlite",
                          compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    offline()
else:
    online()
