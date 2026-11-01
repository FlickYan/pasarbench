"""
What the service answers when it does not do what was asked: RFC 9457
problem details (application/problem+json), each kind with its own `type`
(a URI reference relative to the service), so a client branches on the type,
never on the wording.
"""

from __future__ import annotations

from typing import Any

PROBLEMS = "/problems/"


class ServiceError(Exception):
    status = 500
    kind = "internal"
    title = "Internal error"

    def __init__(self, detail: str = "", *, retry_after: float | None = None,
                 **extra: Any):
        super().__init__(detail or self.title)
        self.detail = detail or self.title
        self.retry_after = retry_after
        self.extra = extra

    def problem(self, instance: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {"type": PROBLEMS + self.kind, "title": self.title,
                               "status": self.status, "detail": self.detail}
        if instance:
            out["instance"] = instance
        out.update(self.extra)
        return out

    def headers(self) -> dict[str, str]:
        if self.retry_after is None:
            return {}
        return {"Retry-After": str(max(1, int(round(self.retry_after))))}


class BadRequest(ServiceError):
    status, kind, title = 400, "bad-request", "Bad request"


class Unauthorized(ServiceError):
    status, kind, title = 401, "unauthorized", "Missing or invalid API key"


class NotFound(ServiceError):
    status, kind, title = 404, "not-found", "Not found"


class Conflict(ServiceError):
    status, kind, title = 409, "conflict", "Conflict"


class TurnInProgress(Conflict):
    kind, title = "turn-in-progress", "A turn is already running in this conversation"


class ConversationClosed(Conflict):
    kind, title = "conversation-closed", "The conversation is closed"


class IdempotencyMismatch(ServiceError):
    status, kind, title = 422, "idempotency-key-reused", \
        "The Idempotency-Key was used for a different request"


class IdempotencyInProgress(Conflict):
    kind, title = "idempotency-key-in-progress", \
        "A request with this Idempotency-Key is still being processed"


class RateLimited(ServiceError):
    status, kind, title = 429, "rate-limited", "Too many requests"


class BudgetExhausted(ServiceError):
    status, kind, title = 429, "budget-exhausted", "The spending limit is reached"


class Overloaded(ServiceError):
    status, kind, title = 503, "overloaded", "The service is at capacity"


class Unavailable(ServiceError):
    status, kind, title = 503, "unavailable", "A dependency is unavailable"


class DatabaseUnavailable(Unavailable):
    """The database did not answer in time, or could not (v83): nothing that
    was asked of it is known to have been done -- what a turn had committed
    before stays, as after a crash."""
    kind, title = "database-unavailable", "The database did not answer"
