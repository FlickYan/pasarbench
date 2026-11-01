"""
    python -m pasarbench.service migrate                 schema to the latest revision, prices
    python -m pasarbench.service create-tenant acme --name "Acme Mart" [--world demo]
                                 [--daily-budget-usd 25]  a tenant, its world, its first key
    python -m pasarbench.service create-key acme         another key (shown once)
    python -m pasarbench.service revoke-key key_...
    python -m pasarbench.service serve [--host 0.0.0.0] [--port 8000] [--workers N]
                                 N: PASAR_WORKERS, else 1 -- processes sharing the limits
    python -m pasarbench.service replay acme             re-run every tool call; exit 1 on a mismatch
    python -m pasarbench.service usage acme [--group-by model|day|conversation]
    python -m pasarbench.service gc                      forget expired Idempotency-Keys
    python -m pasarbench.service nodes                   the processes serving on this database,
                                 which are counted, and the breakers open (v82)
    python -m pasarbench.service profile p.json [--turns N] [--folded p.folded]
                                 what a profile (GET /v1/admin/profile) adds up to

The database is PASAR_DATABASE_URL (config.py); `--database-url` overrides it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MIGRATIONS = Path(__file__).parent / "migrations"


def alembic_config(url: str):  # noqa: ANN201 -- alembic is imported here, not at the top
    from alembic.config import Config
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def _sqlite_dir(url: str) -> None:
    """sqlite:///data/pasar.db: make data/ first."""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)


def shared_on_sqlite(limits: str, url: str, workers: int) -> bool:
    """`serve --workers N` on one SQLite file with PASAR_LIMITS=auto: N
    processes are a deployment, and their limits are shared -- `auto` keeps
    them a process's own on SQLite for the one process it usually is, never
    to hand N of them N times each (WHAT_FAILED #71)."""
    return workers > 1 and url.startswith("sqlite") and limits == "auto"


def main(argv: list[str] | None = None) -> int:
    from .config import Settings, usd_to_picousd

    ap = argparse.ArgumentParser(prog="python -m pasarbench.service",
                                 description="The PasarBench agent service.")
    ap.add_argument("--database-url", help="default: PASAR_DATABASE_URL")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="upgrade the schema to the latest revision")
    t = sub.add_parser("create-tenant", help="a tenant, its starting world and its first key")
    t.add_argument("tenant_id")
    t.add_argument("--name", required=True)
    t.add_argument("--world", default="demo", help="demo | task:<task_id> | synthetic:<n>")
    t.add_argument("--daily-budget-usd")
    k = sub.add_parser("create-key")
    k.add_argument("tenant_id")
    r = sub.add_parser("revoke-key")
    r.add_argument("key_id")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    # Processes: each its own event loop and threads, every limit shared with
    # the others through the database (PASAR_LIMITS, v82).
    s.add_argument("--workers", type=int, help="default: PASAR_WORKERS, else 1")
    p = sub.add_parser("replay", help="re-run every tool call of a tenant and compare")
    p.add_argument("tenant_id")
    u = sub.add_parser("usage")
    u.add_argument("tenant_id")
    u.add_argument("--group-by", default="model", choices=("model", "day", "conversation"))
    sub.add_parser("gc", help="forget expired Idempotency-Keys")
    sub.add_parser("nodes", help="the processes serving on this database, which are counted, "
                                 "and the breakers open (v82)")
    pr = sub.add_parser("profile", help="what a profile (GET /v1/admin/profile) adds up to")
    pr.add_argument("files", nargs="+", help="profiles, as the endpoint answers them (JSON)")
    pr.add_argument("--turns", type=int, help="the turns served while it ran: CPU per turn")
    pr.add_argument("--folded", help="write its stacks here, for a flame graph")
    pr.add_argument("--top", type=int, default=25)
    args = ap.parse_args(argv)

    if args.cmd == "profile":
        from .profiling import folded, merge, summarize
        profile = merge([json.loads(Path(f).read_text()) for f in args.files])
        print(summarize(profile, args.turns, args.top))
        if args.folded:
            Path(args.folded).write_text(folded(profile))
        return 0

    settings = Settings.from_env()
    url = args.database_url or settings.database_url
    _sqlite_dir(url)

    if args.cmd == "serve":
        import uvicorn

        from .observability import uvicorn_log_config
        workers = args.workers
        if workers is None:
            raw = os.environ.get("PASAR_WORKERS", "").strip() or "1"
            if not raw.isdigit() or int(raw) < 1:
                print(f"error: PASAR_WORKERS={raw!r}: a number of processes, 1 or more",
                      file=sys.stderr)
                return 2
            workers = int(raw)
        if shared_on_sqlite(settings.limits, url, workers):
            os.environ["PASAR_LIMITS"] = "deployment"
            print(f"note: {workers} processes on one SQLite file share their limits "
                  f"(PASAR_LIMITS=deployment): each request's check is a write transaction",
                  file=sys.stderr)
        os.environ["PASAR_DATABASE_URL"] = url
        uvicorn.run("pasarbench.service.app:create_app", factory=True, host=args.host,
                    port=args.port, workers=workers, log_level="warning",
                    log_config=uvicorn_log_config(), proxy_headers=True,
                    timeout_keep_alive=settings.keep_alive_seconds)
        return 0

    from .store import make_engine, seed_tenant, transaction
    engine = make_engine(url, timeout=settings.db_timeout)
    try:
        if args.cmd == "migrate":
            from alembic import command

            from .costs import seed_prices
            command.upgrade(alembic_config(url), "head")
            with transaction(engine, write=True) as conn:
                added = seed_prices(conn)
            print(f"schema at head; price book: {added} rows added")
        elif args.cmd == "create-tenant":
            from .auth import create_key
            budget = usd_to_picousd(args.daily_budget_usd) if args.daily_budget_usd else None
            digest = seed_tenant(engine, args.tenant_id, args.name, args.world, budget)
            key_id, secret = create_key(engine, args.tenant_id)
            print(f"tenant {args.tenant_id}: world {args.world} ({digest})")
            print(f"API key {key_id} -- shown once, keep it:\n{secret}")
        elif args.cmd == "create-key":
            from .auth import create_key
            key_id, secret = create_key(engine, args.tenant_id)
            print(f"API key {key_id} -- shown once, keep it:\n{secret}")
        elif args.cmd == "revoke-key":
            from .auth import revoke_key
            print("revoked" if revoke_key(engine, args.key_id) else "no live key by that id")
        elif args.cmd == "replay":
            from .replay import replay_tenant
            report = replay_tenant(engine, args.tenant_id)
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        elif args.cmd == "usage":
            from .costs import usage_report
            from .store import deadline
            with deadline(None), transaction(engine, write=False) as conn:   # an operator's
                rows = usage_report(conn, args.tenant_id, args.group_by)
            print(f"{'key':<34} {'calls':>7} {'prompt':>11} {'cached':>11} {'out':>9} "
                  f"{'hit':>6} {'cost':>12}")
            for row in rows:
                print(f"{row['key']:<34} {row['calls']:>7} {row['prompt_tokens']:>11} "
                      f"{row['cached_tokens']:>11} {row['completion_tokens']:>9} "
                      f"{row['cache_hit_rate']:>6.1%} {row['cost_usd']:>12}")
        elif args.cmd == "gc":
            from .auth import Idempotency
            print(f"{Idempotency(engine, settings.idempotency_hours).purge()} expired keys removed")
        elif args.cmd == "nodes":
            from .limits import processes
            rows, opened = processes(engine)
            print(json.dumps({"processes": rows, "breakers_open": opened}, indent=2))
    except Exception as e:  # noqa: BLE001 -- a CLI says what went wrong, in one line
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
