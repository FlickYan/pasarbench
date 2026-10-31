"""
The service's settings, from PASAR_* environment variables -- never from a
file the service finds by itself. Provider keys are read by the model
registry (models.py), each from its own provider's variables only.

    PASAR_DATABASE_URL            sqlite:///data/pasar.db (local) or
                                  postgresql+psycopg://user:pass@host:5432/db
    PASAR_ADMIN_TOKEN             the operator's token for /v1/admin/*; unset:
                                  those routes answer 404 (use the CLI)
    PASAR_DEFAULT_MODEL           mock
    PASAR_MODELS                  the models tenants may ask for, comma-separated;
                                  "scripted:*" allows every reference solution
    PASAR_MODELS_FILE             a JSON file of more model presets (models.py)
    PASAR_POLICY_MODE             preload
    PASAR_STRICT_MONEY            1: the platform refuses a refund past what was paid
    PASAR_MAX_STEPS_PER_TURN      16 model calls a turn
    PASAR_MAX_TOOL_CALLS_PER_TURN 24
    PASAR_MAX_TURN_SECONDS        120
    PASAR_MAX_TURNS               40 customer turns a conversation
    PASAR_CONVERSATION_BUDGET_USD 0.50 a conversation, all its model calls
    PASAR_CONVERSATION_MAX_TOKENS 2000000
    PASAR_CONVERSATION_MAX_STEPS  0: model calls a conversation may make in all (0: no cap;
                                  scripts/service_sweep.py sets the benchmark's 30)
    PASAR_PROVIDER_CONCURRENCY    16 calls in flight to one provider: the deployment's,
                                  shared out between its processes (PASAR_LIMITS)
    PASAR_PROVIDER_ACQUIRE_TIMEOUT 30 seconds to wait for one of them
    PASAR_MAX_CONCURRENT_TURNS    256 turns at once, per process; past it, 503. A turn
                                  waiting on its model holds no thread (v81): this
                                  bounds memory and the CPU a burst can ask for
    PASAR_ADMISSION_TIMEOUT       2 seconds a turn waits to be admitted
    PASAR_THREADS                 64 worker threads: a turn's transactions, and every
                                  handler but a turn's
    PASAR_RATE / PASAR_BURST      10 requests a second a tenant, bursts of 40: the
                                  deployment's (PASAR_LIMITS)
    PASAR_LIMITS                  auto: deployment on PostgreSQL, and on SQLite for
                                  `serve --workers N`; else process.
                                  deployment: the processes that share the database share
                                  the provider limits, the tenants' buckets and the
                                  breakers' openings (limits.py, v82); process: each its
                                  own, as before v82 -- N processes allow N times each.
                                  On SQLite a shared bucket is a write transaction a
                                  request, behind any long write (WHAT_FAILED #71)
    PASAR_HEARTBEAT_SECONDS       5: how often a process says it is serving; one silent for
                                  three heartbeats is not counted
    PASAR_LEASE_SECONDS           60: a turn's hold on its conversation, renewed every
                                  third of it while the turn runs; a dead process's
                                  turn frees its conversation within one
    PASAR_IDEMPOTENCY_HOURS       24
    PASAR_POOL_SIZE / PASAR_MAX_OVERFLOW  10 / 20 database connections
    PASAR_SLICE_CACHE             2000 customers' slices kept, per process, by the write
                                  each was read at (store.Slices); 0 where another
                                  system writes the world tables
    PASAR_MOCK_LATENCY_MS         0: the mock model's simulated latency
    PASAR_MOCK_JITTER             0: how far each call's latency strays from it, a
                                  fraction (0.3: anywhere from 70% to 130%), as a
                                  provider's does
    PASAR_KEEP_ALIVE_SECONDS      75: how long `serve` keeps an idle connection open --
                                  longer than a load balancer's usual 60, so that the
                                  balancer closes first (uvicorn's own default, 5,
                                  races a client sending its next request)
    PASAR_LOG_LEVEL               INFO
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from decimal import Decimal


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()


def usd_to_picousd(usd: str) -> int:
    return int(Decimal(usd) * 10 ** 12)


@dataclass
class Settings:
    database_url: str = "sqlite:///data/pasar.db"
    admin_token: str = ""
    default_model: str = "mock"
    models: tuple[str, ...] = ("mock", "scripted:*")
    models_file: str = ""
    policy_mode: str = "preload"
    strict_money: bool = True
    max_steps_per_turn: int = 16
    max_tool_calls_per_turn: int = 24
    max_turn_seconds: float = 120.0
    max_turns: int = 40
    conversation_budget_picousd: int = usd_to_picousd("0.50")
    conversation_max_tokens: int = 2_000_000
    conversation_max_steps: int = 0
    provider_concurrency: int = 16
    provider_acquire_timeout: float = 30.0
    max_concurrent_turns: int = 256
    admission_timeout: float = 2.0
    threads: int = 64
    rate: float = 10.0
    burst: int = 40
    lease_seconds: float = 60.0
    idempotency_hours: float = 24.0
    pool_size: int = 10
    max_overflow: int = 20
    slice_cache: int = 2000
    limits: str = "auto"
    heartbeat_seconds: float = 5.0
    mock_latency_ms: int = 0
    mock_jitter: float = 0.0
    keep_alive_seconds: float = 75.0
    log_level: str = "INFO"
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            database_url=_env("PASAR_DATABASE_URL", cls.database_url),
            admin_token=_env("PASAR_ADMIN_TOKEN", ""),
            default_model=_env("PASAR_DEFAULT_MODEL", cls.default_model),
            models=tuple(m.strip() for m in _env("PASAR_MODELS", "mock,scripted:*").split(",")
                         if m.strip()),
            models_file=_env("PASAR_MODELS_FILE", ""),
            policy_mode=_env("PASAR_POLICY_MODE", cls.policy_mode),
            strict_money=_env("PASAR_STRICT_MONEY", "1") not in ("0", "false", "no"),
            max_steps_per_turn=int(_env("PASAR_MAX_STEPS_PER_TURN", "16")),
            max_tool_calls_per_turn=int(_env("PASAR_MAX_TOOL_CALLS_PER_TURN", "24")),
            max_turn_seconds=float(_env("PASAR_MAX_TURN_SECONDS", "120")),
            max_turns=int(_env("PASAR_MAX_TURNS", "40")),
            conversation_budget_picousd=usd_to_picousd(_env("PASAR_CONVERSATION_BUDGET_USD",
                                                            "0.50")),
            conversation_max_tokens=int(_env("PASAR_CONVERSATION_MAX_TOKENS", "2000000")),
            conversation_max_steps=int(_env("PASAR_CONVERSATION_MAX_STEPS", "0")),
            provider_concurrency=int(_env("PASAR_PROVIDER_CONCURRENCY", "16")),
            provider_acquire_timeout=float(_env("PASAR_PROVIDER_ACQUIRE_TIMEOUT", "30")),
            max_concurrent_turns=int(_env("PASAR_MAX_CONCURRENT_TURNS", "256")),
            admission_timeout=float(_env("PASAR_ADMISSION_TIMEOUT", "2")),
            threads=int(_env("PASAR_THREADS", "64")),
            rate=float(_env("PASAR_RATE", "10")),
            burst=int(_env("PASAR_BURST", "40")),
            lease_seconds=float(_env("PASAR_LEASE_SECONDS", "60")),
            idempotency_hours=float(_env("PASAR_IDEMPOTENCY_HOURS", "24")),
            pool_size=int(_env("PASAR_POOL_SIZE", "10")),
            max_overflow=int(_env("PASAR_MAX_OVERFLOW", "20")),
            slice_cache=int(_env("PASAR_SLICE_CACHE", "2000")),
            limits=_env("PASAR_LIMITS", "auto").lower(),
            heartbeat_seconds=float(_env("PASAR_HEARTBEAT_SECONDS", "5")),
            mock_latency_ms=int(_env("PASAR_MOCK_LATENCY_MS", "0")),
            mock_jitter=float(_env("PASAR_MOCK_JITTER", "0")),
            keep_alive_seconds=float(_env("PASAR_KEEP_ALIVE_SECONDS", "75")),
            log_level=_env("PASAR_LOG_LEVEL", "INFO").upper(),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if self.policy_mode not in ("preload", "jit", "preload-named"):
            raise ValueError(f"PASAR_POLICY_MODE {self.policy_mode!r}: preload, jit or "
                             f"preload-named")
        if self.max_concurrent_turns < 1 or self.threads < 1:
            raise ValueError("PASAR_MAX_CONCURRENT_TURNS and PASAR_THREADS are at least 1")
        if self.limits not in ("auto", "deployment", "process"):
            raise ValueError(f"PASAR_LIMITS {self.limits!r}: auto, deployment or process")
        if self.heartbeat_seconds <= 0:
            raise ValueError("PASAR_HEARTBEAT_SECONDS is more than 0")
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
                raise ValueError(f"{f.name} is negative")

    def public(self) -> dict[str, object]:
        """What /v1/admin/settings may show: everything but the token."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in ("admin_token", "extra")}
