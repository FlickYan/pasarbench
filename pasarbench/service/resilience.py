"""
What stands between the service and a model provider, and between a tenant
and the service.

A model call is the slow, failing, rate-limited part of every turn: seconds
long, billed by the token, and refused with 429s or 503s when the provider is
busy (the OpenCode Go runs met both, RUNBOOK 1s). Each provider gets a gate:

    a concurrency limit   at most N calls in flight to it from this process;
                          a turn that cannot get a slot within the timeout is
                          told the service is busy (503, Retry-After) instead of
                          queueing without bound -- backpressure
    retries               a transient failure (429, 5xx, a timeout, a reset
                          connection) is tried again, up to three attempts,
                          after an exponentially growing, fully jittered wait,
                          so that many turns that failed together do not all
                          come back together
    a circuit breaker     after five failures in a row the provider is not
                          called for 30 seconds; turns fail fast instead of
                          each waiting out its own timeouts; then one trial
                          call decides whether it closes again

And each tenant gets a token bucket on its requests (app.py): a burst, then a
steady rate; past it, 429 with Retry-After.

All of it is per process. Several replicas behind a load balancer each keep
their own (docs/SERVICE.md: what moves to Redis, and when).
"""

from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# What a provider's failure says when trying again may work. The backends
# raise RuntimeError("HTTP 503 from ...") and urllib's own errors.
TRANSIENT = re.compile(
    r"HTTP (408|409|425|429|500|502|503|504|529)\b|timed? ?out|Timeout|temporarily|"
    r"Connection (reset|refused|aborted)|Remote end closed|EOF occurred|"
    r"Temporary failure in name resolution|URLError", re.I)


class Overloaded(Exception):
    """No slot to the provider within the timeout: the caller should back off."""

    def __init__(self, provider: str, retry_after: float):
        super().__init__(f"model provider {provider!r} is at its concurrency limit")
        self.provider = provider
        self.retry_after = retry_after


class CircuitOpen(Exception):
    def __init__(self, provider: str, retry_after: float):
        super().__init__(f"model provider {provider!r} is failing; not called for "
                         f"{retry_after:.0f}s")
        self.provider = provider
        self.retry_after = retry_after


def is_transient(exc: BaseException) -> bool:
    return bool(TRANSIENT.search(f"{type(exc).__name__}: {exc}"))


def backoff(attempt: int, base: float = 0.5, cap: float = 8.0,
            rand: Callable[[], float] = random.random) -> float:
    """Full jitter: a wait drawn uniformly from [0, min(cap, base * 2^n))."""
    return rand() * min(cap, base * (2 ** attempt))


def retry(fn: Callable[[], T], *, attempts: int = 3, base: float = 0.5, cap: float = 8.0,
          sleep: Callable[[float], None] = time.sleep,
          rand: Callable[[], float] = random.random,
          on_retry: Callable[[int, BaseException, float], None] | None = None) -> T:
    """fn(), tried again after a transient failure; anything else raises at once."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if attempt == attempts - 1 or not is_transient(e):
                raise
            wait = backoff(attempt, base, cap, rand)
            if on_retry is not None:
                on_retry(attempt + 1, e, wait)
            sleep(wait)
    raise AssertionError("unreachable")


@dataclass
class CircuitBreaker:
    """Closed until `threshold` failures in a row; open for `reset_s`; then
    half-open: one call is let through, and its outcome closes or re-opens it."""
    threshold: int = 5
    reset_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None
    trial: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state()

    def _state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.reset_s else "open"

    def allow(self) -> float | None:
        """None if the call may go; else seconds until it might."""
        with self._lock:
            st = self._state()
            if st == "closed":
                return None
            if st == "half_open" and not self.trial:
                self.trial = True
                return None
            return max(0.0, self.reset_s - (self.clock() - (self.opened_at or 0.0))) or 1.0

    def record(self, ok: bool) -> None:
        with self._lock:
            self.trial = False
            if ok:
                self.failures = 0
                self.opened_at = None
            else:
                self.failures += 1
                if self.failures >= self.threshold or self.opened_at is not None:
                    self.opened_at = self.clock()


class ProviderGate:
    """One provider's limit, retries and breaker, shared by every turn in the
    process that calls it."""

    def __init__(self, name: str, concurrency: int = 16, acquire_timeout: float = 30.0,
                 attempts: int = 3, breaker: CircuitBreaker | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.name = name
        self.concurrency = concurrency
        self.acquire_timeout = acquire_timeout
        self.attempts = attempts
        self.breaker = breaker or CircuitBreaker()
        self._slots = threading.BoundedSemaphore(concurrency)
        self._sleep = sleep
        self._lock = threading.Lock()
        self.in_flight = 0
        self.retries = 0

    def call(self, fn: Callable[[], T]) -> T:
        wait = self.breaker.allow()
        if wait is not None:
            raise CircuitOpen(self.name, wait)
        if not self._slots.acquire(timeout=self.acquire_timeout):
            raise Overloaded(self.name, retry_after=max(1.0, self.acquire_timeout / 4))
        with self._lock:
            self.in_flight += 1
        try:
            def counted(attempt: int, _e: BaseException, _w: float) -> None:
                with self._lock:
                    self.retries += 1
            out = retry(fn, attempts=self.attempts, sleep=self._sleep, on_retry=counted)
        except Exception:
            self.breaker.record(False)
            raise
        else:
            self.breaker.record(True)
            return out
        finally:
            with self._lock:
                self.in_flight -= 1
            self._slots.release()


class Gates:
    """The process's gates, one per provider, made on first use."""

    def __init__(self, concurrency: int = 16, acquire_timeout: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep,
                 breaker: Callable[[], CircuitBreaker] = CircuitBreaker):
        self.concurrency = concurrency
        self.acquire_timeout = acquire_timeout
        self._sleep = sleep
        self._breaker = breaker
        self._gates: dict[str, ProviderGate] = {}
        self._lock = threading.Lock()

    def get(self, provider: str) -> ProviderGate:
        with self._lock:
            g = self._gates.get(provider)
            if g is None:
                g = self._gates[provider] = ProviderGate(
                    provider, self.concurrency, self.acquire_timeout,
                    breaker=self._breaker(), sleep=self._sleep)
            return g

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {n: {"in_flight": g.in_flight, "retries": g.retries,
                        "breaker": g.breaker.state} for n, g in self._gates.items()}


class TokenBucket:
    """`rate` requests a second on average, `burst` at once."""

    def __init__(self, rate: float, burst: int, clock: Callable[[], float] = time.monotonic):
        self.rate = rate
        self.burst = burst
        self.clock = clock
        self.tokens = float(burst)
        self.at = clock()
        self._lock = threading.Lock()

    def take(self) -> float | None:
        """None if allowed; else seconds until a token is free."""
        with self._lock:
            now = self.clock()
            self.tokens = min(self.burst, self.tokens + (now - self.at) * self.rate)
            self.at = now
            if self.tokens >= 1:
                self.tokens -= 1
                return None
            return (1 - self.tokens) / self.rate if self.rate > 0 else 60.0


class TenantLimits:
    def __init__(self, rate: float, burst: int):
        self.rate, self.burst = rate, burst
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def take(self, tenant_id: str) -> float | None:
        with self._lock:
            b = self._buckets.get(tenant_id)
            if b is None:
                b = self._buckets[tenant_id] = TokenBucket(self.rate, self.burst)
        return b.take()
