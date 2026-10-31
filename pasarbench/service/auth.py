"""
Tenants' API keys, and the Idempotency-Key a request may carry.

An API key is shown once, when it is made, and stored as its SHA-256: a copy
of the database holds no key that works. A lookup is cached for 30 seconds,
so a revoked key stops working within 30 seconds on every process.

An Idempotency-Key makes a retried request safe: the first request with a key
runs and its answer is kept for PASAR_IDEMPOTENCY_HOURS; a retry with the same
key and the same body gets that answer back, byte for byte (Idempotent-Replayed:
true), without running again -- one customer message, one reply, one refund,
however many times a flaky network makes the client send it. The answer is kept
whatever it was, an error included, once the request has begun to change
anything: a retry must never run a turn twice. A request turned away before it
did anything (busy, a budget spent) frees its key. The same key with a
different body is refused (422); a retry while the first is still running is
told so (409) -- for as long as the first could be running: a key whose request
died with its process expires with the turn's lease.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from datetime import timedelta

from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.exc import IntegrityError

from . import errors as E
from . import schema as S
from .runtime import _aware, utcnow
from .store import transaction


def hash_key(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def create_key(engine: Engine, tenant_id: str) -> tuple[str, str]:
    """A new key for the tenant: (key_id, the key). The key is not kept."""
    key_id = "key_" + secrets.token_hex(6)
    secret = "psk_" + secrets.token_urlsafe(32)
    with transaction(engine, write=True) as conn:
        if conn.execute(select(S.tenants.c.tenant_id).where(
                S.tenants.c.tenant_id == tenant_id)).first() is None:
            raise E.NotFound(f"no tenant {tenant_id!r}")
        conn.execute(insert(S.api_keys).values(key_id=key_id, tenant_id=tenant_id,
                                               key_hash=hash_key(secret), created_at=utcnow()))
    return key_id, secret


def revoke_key(engine: Engine, key_id: str) -> bool:
    with transaction(engine, write=True) as conn:
        return conn.execute(update(S.api_keys).where(S.api_keys.c.key_id == key_id,
                                                     S.api_keys.c.revoked_at.is_(None))
                            .values(revoked_at=utcnow())).rowcount == 1


class KeyCache:
    def __init__(self, engine: Engine, ttl: float = 30.0):
        self.engine, self.ttl = engine, ttl
        self._cache: dict[str, tuple[float, tuple[str, str] | None]] = {}
        self._lock = threading.Lock()

    def lookup(self, secret: str) -> tuple[str, str] | None:
        """(tenant_id, key_id) for a live key; None otherwise."""
        h = hash_key(secret)
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(h)
        if hit and hit[0] > now:
            return hit[1]
        k = S.api_keys
        with self.engine.connect() as conn:
            row = conn.execute(select(k.c.tenant_id, k.c.key_id, k.c.revoked_at)
                               .where(k.c.key_hash == h)).first()
        found = (row.tenant_id, row.key_id) if row is not None and row.revoked_at is None else None
        with self._lock:
            if len(self._cache) > 10_000:
                self._cache.clear()
            self._cache[h] = (now + self.ttl, found)
        return found


def request_hash(method: str, path: str, body: bytes) -> str:
    return hashlib.sha256(method.encode() + b" " + path.encode() + b"\n" + body).hexdigest()


class Idempotency:
    def __init__(self, engine: Engine, ttl_hours: float, running_seconds: float = 300.0):
        self.engine = engine
        self.ttl = timedelta(hours=ttl_hours)
        self.running = timedelta(seconds=running_seconds)

    def begin(self, tenant_id: str, key: str, req_hash: str) -> tuple[int, str] | None:
        """None: run the request (the key is now held for it). (status, body):
        the answer the first request with this key got, as it was sent."""
        if not key or len(key) > 200:
            raise E.BadRequest("Idempotency-Key must be 1 to 200 characters")
        k = S.idempotency_keys
        now = utcnow()
        try:
            with transaction(self.engine, write=True) as conn:
                row = conn.execute(select(k).where(k.c.tenant_id == tenant_id,
                                                   k.c.key == key)).first()
                if row is not None and _aware(row.expires_at) <= now:
                    conn.execute(delete(k).where(k.c.tenant_id == tenant_id, k.c.key == key))
                    row = None
                if row is None:
                    conn.execute(insert(k).values(tenant_id=tenant_id, key=key,
                                                  request_hash=req_hash, state="in_progress",
                                                  created_at=now,
                                                  expires_at=now + self.running))
                    return None
        except IntegrityError:
            # Another request with this key inserted it first, a moment ago.
            raise E.IdempotencyInProgress(retry_after=1) from None
        if row.request_hash != req_hash:
            raise E.IdempotencyMismatch()
        if row.state == "in_progress":
            raise E.IdempotencyInProgress(retry_after=1)
        return int(row.status_code), row.response

    def extend(self, tenant_id: str, key: str) -> None:
        """Keep a running request's key from expiring: called with the turn's
        lease heartbeat, so the key lives exactly as long as the turn."""
        k = S.idempotency_keys
        with transaction(self.engine, write=True) as conn:
            conn.execute(update(k).where(k.c.tenant_id == tenant_id, k.c.key == key,
                                         k.c.state == "in_progress")
                         .values(expires_at=utcnow() + self.running))

    def complete(self, tenant_id: str, key: str, status: int, body: str,
                 req_hash: str) -> None:
        """Keep the answer. The first answer stands; and an answer is kept even
        if its key's row is gone (expired and taken by a retry that was then
        turned away): a retry must find it, not run the request again."""
        k = S.idempotency_keys
        now = utcnow()
        values = dict(state="completed", status_code=status, response=body,
                      expires_at=now + self.ttl)
        try:
            with transaction(self.engine, write=True) as conn:
                if conn.execute(update(k).where(k.c.tenant_id == tenant_id, k.c.key == key,
                                                k.c.state == "in_progress")
                                .values(**values)).rowcount == 1:
                    return
                if conn.execute(select(k.c.key).where(k.c.tenant_id == tenant_id,
                                                      k.c.key == key)).first() is None:
                    conn.execute(insert(k).values(tenant_id=tenant_id, key=key,
                                                  request_hash=req_hash, created_at=now,
                                                  **values))
        except IntegrityError:
            pass                        # another answer was kept first: it stands

    def abandon(self, tenant_id: str, key: str) -> None:
        k = S.idempotency_keys
        with transaction(self.engine, write=True) as conn:
            conn.execute(delete(k).where(k.c.tenant_id == tenant_id, k.c.key == key,
                                         k.c.state == "in_progress"))

    def purge(self) -> int:
        k = S.idempotency_keys
        with transaction(self.engine, write=True) as conn:
            return conn.execute(delete(k).where(k.c.expires_at <= utcnow())).rowcount
