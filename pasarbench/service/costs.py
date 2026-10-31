"""
What a model call costs, in exact integers.

Prices are kept as microdollars per million tokens -- every price the plans
list ($0.003625 a million for MiMo's cached input included) is a whole number
of those -- and a call's cost as picodollars: a microdollar per million tokens
is a picodollar per token, so

    cost = (prompt - cached) * input + cached * cached_price + completion * output

is integer arithmetic with no rounding anywhere, and a month of a tenant's
calls adds up to the same total in any order. A float would not.

The price book is a table with effective dates, never a constant: a provider
changes a price, a row is added, and every ledger entry keeps the price it was
charged at (usage_ledger.price_id).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import Connection, func, insert, select

from . import schema as S

PICO = 10 ** 12


def micro(dollars_per_mtok: str | float) -> int:
    """$ per million tokens -> integer microdollars per million tokens, exactly
    (from the decimal string, never through a float's binary expansion)."""
    v = Decimal(str(dollars_per_mtok)) * 1_000_000
    if v != v.to_integral_value():
        raise ValueError(f"{dollars_per_mtok} $/Mtok is not a whole number of microdollars")
    return int(v)


@dataclass(frozen=True)
class Price:
    price_id: str
    provider: str
    model: str
    input: int          # microdollars per million tokens = picodollars per token
    cached: int
    output: int


def cost_picousd(prompt: int, cached: int, completion: int, price: Price | None) -> int:
    if price is None:
        return 0
    cached = min(max(cached, 0), max(prompt, 0))
    return (prompt - cached) * price.input + cached * price.cached + completion * price.output


def usd(picousd: int) -> str:
    """$1.234567 -- six places, which is what a conversation's cost needs."""
    return f"${Decimal(picousd) / PICO:.6f}"


# The prices the runs have been charged at (RUNBOOK 1s: the five labs' own
# pages, read 2026-10-13; OpenCode Go's table, read 2026-10-16). "mock" is the
# service's own test model, priced as DeepSeek-V4.1-Flash so that load tests
# exercise the ledger with realistic numbers -- marked as such.
PRICE_BOOK: list[tuple[str, str, str, str, str, str, str]] = [
    # provider, model, input, cached, output ($/Mtok), effective from, source
    ("deepseek", "deepseek-v4-pro", "1.32", "0.044", "3.96", "2026-10-13", "DeepSeek pricing page, peak"),
    ("deepseek", "deepseek-flash", "0.30", "0.006", "1.20", "2026-10-13", "DeepSeek pricing page"),
    ("xiaomi", "mimo-v2.6-flash", "0.14", "0.003", "0.28", "2026-10-13", "Xiaomi MiMo pricing page"),
    ("xai", "grok-4.3", "1.25", "0.20", "2.50", "2026-10-13", "xAI pricing page"),
    ("zai", "glm-5.3-flash", "0.15", "0.03", "0.50", "2026-10-13", "Z.ai pricing page"),
    ("opencode-go", "claude-haiku-5-5", "0.10", "0.01", "0.50", "2026-10-16", "OpenCode Go table"),
    ("opencode-go", "gpt-6-luna", "0.10", "0.01", "0.50", "2026-10-16", "OpenCode Go table"),
    ("opencode-go", "minimax-m3", "0.30", "0.06", "1.20", "2026-10-16", "OpenCode Go table"),
    ("opencode-go", "kimi-k2.6", "0.95", "0.16", "4.00", "2026-10-16", "OpenCode Go table"),
    ("opencode-go", "longcat-2.0", "0.30", "0.006", "1.20", "2026-10-16", "OpenCode Go table"),
    ("opencode-go", "hy3", "0.14", "0.035", "0.58", "2026-10-16", "OpenCode Go table"),
    ("opencode-go", "mimo-v2.6-pro", "0.435", "0.003625", "0.87", "2026-10-16", "OpenCode Go table"),
    ("opencode-go", "glm-5.3", "1.40", "0.26", "4.40", "2026-10-16", "OpenCode Go table"),
    ("opencode-go", "kimi-k3", "3.00", "0.30", "15.00", "2026-10-16", "OpenCode Go table"),
    ("local", "mock", "0.30", "0.006", "1.20", "2026-10-13", "simulated: priced as deepseek-flash"),
    ("local", "scripted", "0", "0", "0", "2026-10-13", "replayed reference solution: free"),
]


def seed_prices(conn: Connection) -> int:
    """Add the book's rows that are not in the table yet. Returns how many."""
    have = {r.price_id for r in conn.execute(select(S.price_book.c.price_id))}
    rows = []
    for provider, model, i, c, o, since, source in PRICE_BOOK:
        pid = f"{provider}:{model}:{since}"
        if pid not in have:
            rows.append(dict(price_id=pid, provider=provider, model=model,
                             input_microusd_per_mtok=micro(i),
                             cached_microusd_per_mtok=micro(c),
                             output_microusd_per_mtok=micro(o),
                             effective_from=since, source=source))
    if rows:
        conn.execute(insert(S.price_book), rows)
    return len(rows)


def price_for(conn: Connection, provider: str, model: str, on: str | None = None) -> Price | None:
    """The price in effect on `on` (YYYY-MM-DD; today when None): the latest
    row on or before it."""
    on = on or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p = S.price_book
    r = conn.execute(select(p).where(p.c.provider == provider, p.c.model == model,
                                     p.c.effective_from <= on)
                     .order_by(p.c.effective_from.desc()).limit(1)).first()
    if r is None:
        return None
    return Price(r.price_id, r.provider, r.model, r.input_microusd_per_mtok,
                 r.cached_microusd_per_mtok, r.output_microusd_per_mtok)


def spent_today(conn: Connection, tenant_id: str) -> int:
    """What the tenant's calls have cost since midnight UTC, in picodollars."""
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    u = S.usage_ledger
    return int(conn.execute(select(func.coalesce(func.sum(u.c.cost_picousd), 0))
                            .where(u.c.tenant_id == tenant_id, u.c.created_at >= start))
               .scalar_one())


def usage_report(conn: Connection, tenant_id: str, group_by: str,
                 since: datetime | None = None, until: datetime | None = None) -> list[dict[str, Any]]:
    """The tenant's ledger, summed by model, by day or by conversation."""
    u = S.usage_ledger
    if group_by == "model":
        key = u.c.model
    elif group_by == "conversation":
        key = u.c.conversation_id
    elif group_by == "day":
        key = func.date(u.c.created_at)          # UTC days: every timestamp here is UTC
    else:
        raise ValueError("group_by is model, day or conversation")
    q = (select(key.label("key"), func.count().label("calls"),
                func.sum(u.c.prompt_tokens).label("prompt_tokens"),
                func.sum(u.c.cached_tokens).label("cached_tokens"),
                func.sum(u.c.completion_tokens).label("completion_tokens"),
                func.sum(u.c.cost_picousd).label("cost_picousd"))
         .where(u.c.tenant_id == tenant_id).group_by(key).order_by(key))
    if since is not None:
        q = q.where(u.c.created_at >= since)
    if until is not None:
        q = q.where(u.c.created_at < until)
    out = []
    for r in conn.execute(q):
        cost = int(r.cost_picousd or 0)
        prompt = int(r.prompt_tokens or 0)
        out.append({"key": str(r.key), "calls": int(r.calls), "prompt_tokens": prompt,
                    "cached_tokens": int(r.cached_tokens or 0),
                    "completion_tokens": int(r.completion_tokens or 0),
                    "cache_hit_rate": round(int(r.cached_tokens or 0) / prompt, 4) if prompt else 0.0,
                    "cost_picousd": cost, "cost_usd": usd(cost)})
    return out
