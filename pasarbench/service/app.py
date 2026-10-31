"""
The HTTP API (FastAPI): what a storefront's backend calls to give its
customers the agent.

    POST /v1/conversations                     open one, for a customer (Idempotency-Key)
    POST /v1/conversations/{id}/messages       the customer's message; the agent's reply,
                                               or ?stream=true for Server-Sent Events as the
                                               turn runs (Idempotency-Key)
    GET  /v1/conversations[/{id}[/messages|/events|/tool-calls]]
    POST /v1/conversations/{id}/close
    GET  /v1/usage                             the ledger, by model, day or conversation
    /v1/admin/...                              tenants, keys, replay, the outbox (operator token)
    GET  /healthz /readyz /metrics /docs /openapi.json

Every error is application/problem+json (errors.py) with its own type; 429
and 503 carry Retry-After. Every request carries X-Request-ID back, and logs
one JSON line (observability.py). Handlers are synchronous and run in a pool
of PASAR_THREADS threads: a turn holds one while it waits on its model, and
at most PASAR_MAX_CONCURRENT_TURNS turns run at once -- past that the service
answers 503 at once rather than queueing without bound.

(No `from __future__ import annotations` here: FastAPI reads the handlers'
annotations at runtime, and the dependencies they name live in create_app.)
"""

import asyncio
import hmac
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import InvalidOperation
from typing import Annotated, Any, AsyncIterator

import anyio.from_thread
import anyio.to_thread
from fastapi import Depends, FastAPI, Header, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import AfterValidator, BaseModel, Field
from sqlalchemy import Engine, inspect, select, text, update
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from . import errors as E
from . import schema as S
from .auth import Idempotency, KeyCache, create_key, request_hash, revoke_key
from .config import Settings, usd_to_picousd
from .costs import seed_prices, usage_report
from .observability import Metrics, configure_logging
from .replay import replay_tenant
from .resilience import TenantLimits
from .runtime import LeaseLost, Runtime, Turn, iso, utcnow
from .store import make_engine, seed_tenant, transaction

log = logging.getLogger("pasar.http")
PROBLEM = "application/problem+json"
REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
TENANT_ID = r"^[a-z0-9][a-z0-9-]{1,62}$"
CONVERSATION_ID = r"^cv_[0-9a-f]{24}$"
KEY_ID = r"^key_[0-9a-f]{12}$"
NAME = r"^[^\x00-\x1f]{1,64}$"          # an id a storefront names: no control characters


def _no_nul(v: str) -> str:
    """PostgreSQL's text holds no NUL character: refused here, as invalid
    input, instead of failing deep in a transaction."""
    if "\x00" in v:
        raise ValueError("must not contain a NUL character")
    return v


Text = Annotated[str, AfterValidator(_no_nul)]
Cid = Annotated[str, Path(pattern=CONVERSATION_ID)]


# ---------------------------------------------------------------------------
# Request and response bodies (the OpenAPI document is built from these)
# ---------------------------------------------------------------------------

class Problem(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    instance: str | None = None


class OpenConversation(BaseModel):
    user_id: str = Field(pattern=NAME, description="the customer, as the storefront knows "
                         "her: her identity is still verified in the conversation, by the "
                         "agent, as the policy says")
    model: Text | None = Field(None, max_length=200,
                               description="default: PASAR_DEFAULT_MODEL")


class CustomerMessage(BaseModel):
    content: Text = Field(min_length=1, max_length=8000)


class Usage(BaseModel):
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int


class TurnUsage(Usage):
    model_calls: int


class Provenance(BaseModel):
    prompt_digest: str
    policy_digest: str
    tools_digest: str
    tool_checks: int
    strict_money: bool
    world_clock: str
    model_settings: dict[str, Any]


class Conversation(BaseModel):
    conversation_id: str
    user_id: str
    market: str
    status: str
    model: str
    policy_mode: str
    provenance: Provenance
    identity_verified: bool
    turns: int
    model_calls: int
    tool_calls: int
    usage: Usage
    cost_picousd: int
    cost_usd: str
    last_stop: str | None
    turn_in_progress: bool
    created_at: str | None
    updated_at: str | None
    closed_at: str | None


class ConversationPage(BaseModel):
    data: list[Conversation]
    next: str | None = Field(description="pass as `before` for the next page; null at the end")


class ToolCallSummary(BaseModel):
    tool: str
    args: Any
    ok: bool | None
    error: str | None


class TurnResult(BaseModel):
    conversation_id: str
    turn: int
    reply: str
    fallback: bool = Field(description="the agent could not finish, and the platform replied")
    stop_reason: str = Field(description="replied | max_steps | max_turn_seconds | "
                             "conversation_budget | tenant_budget | provider_unavailable | "
                             "provider_error")
    tool_calls: list[ToolCallSummary]
    usage: TurnUsage
    cost_picousd: int
    cost_usd: str
    latency_ms: int
    error: str | None = None


class TranscriptMessage(BaseModel):
    seq: int
    turn: int
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    platform: bool | None = None


class Event(BaseModel):
    id: int
    type: str
    turn: int
    data: dict[str, Any]
    at: str | None = None


class AuditedToolCall(BaseModel):
    seq: int
    turn: int
    step: int
    tool: str
    args: Any
    ok: bool
    error: str | None
    is_write: bool
    dispatched: bool
    world_seq: int | None
    customer_seq: int | None
    allocated_ids: list[str] | None
    attempts: int
    latency_ms: int


class UsageRow(BaseModel):
    key: str
    calls: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    cache_hit_rate: float
    cost_picousd: int
    cost_usd: str


class NewTenant(BaseModel):
    tenant_id: str = Field(pattern=TENANT_ID)
    name: Text = Field(min_length=1, max_length=200)
    world: Text = Field("demo", max_length=100,
                        description="demo | task:<task_id> | synthetic:<n>")
    daily_budget_usd: Text | None = Field(None, max_length=32,
                                          description="a decimal string, e.g. \"25.00\"")


class TenantCreated(BaseModel):
    tenant_id: str
    world: str
    world_digest: str
    key_id: str
    api_key: str = Field(description="shown once; only its SHA-256 is kept")


class KeyCreated(BaseModel):
    key_id: str
    api_key: str


class OutboxAck(BaseModel):
    event_ids: list[int] = Field(max_length=1000)


ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": Problem, "content": {PROBLEM: {}}}
    for code in (400, 401, 404, 409, 422, 429, 503)}


# ---------------------------------------------------------------------------
# Middleware: request id, access log, metrics -- plain ASGI, so a streamed
# response streams through it
# ---------------------------------------------------------------------------

class Observe:
    def __init__(self, app: Any, metrics: Metrics):
        self.app, self.metrics = app, metrics

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = next((v.decode() for k, v in scope.get("headers", [])
                    if k == b"x-request-id"), "")
        if not REQUEST_ID.match(rid):
            rid = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        status = [500]
        t0 = time.monotonic()

        async def send_with_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                status[0] = message["status"]
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + [(b"x-request-id", rid.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            seconds = time.monotonic() - t0
            route = getattr(scope.get("route"), "path", "unmatched")
            method = scope["method"]
            self.metrics.http_requests.labels(method, route, str(status[0])).inc()
            self.metrics.http_seconds.labels(method, route).observe(seconds)
            log.info("request", extra={"request_id": rid, "method": method, "route": route,
                                       "status": status[0], "ms": int(seconds * 1000),
                                       "tenant_id": scope["state"].get("tenant_id")})


def sent(model: type[BaseModel], out: dict[str, Any]) -> dict[str, Any]:
    """A response body as its model writes it -- every field, defaults
    included -- rendered once (JSONResponse), and those bytes are what an
    Idempotency-Key keeps."""
    return model.model_validate(out).model_dump(mode="json")


def problem_response(err: E.ServiceError, request: Request) -> JSONResponse:
    return JSONResponse(err.problem(instance=request.url.path), status_code=err.status,
                        media_type=PROBLEM, headers=err.headers())


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    if engine is None:
        engine = make_engine(settings.database_url, pool_size=settings.pool_size,
                             max_overflow=settings.max_overflow)
    metrics = Metrics()
    runtime = Runtime(engine, settings, metrics=metrics)
    limits = TenantLimits(settings.rate, settings.burst)
    keys = KeyCache(engine)
    idem = Idempotency(engine, settings.idempotency_hours, settings.lease_seconds)
    loop_box: dict[str, asyncio.AbstractEventLoop] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        anyio.to_thread.current_default_thread_limiter().total_tokens = settings.threads
        # The loop a streamed turn's thread hands its events to (post_message).
        loop_box["loop"] = asyncio.get_running_loop()
        try:
            with transaction(engine, write=True) as conn:
                added = seed_prices(conn)
            if added:
                log.info(f"price book: {added} rows added")
        except Exception:  # noqa: BLE001 -- not migrated yet: /readyz says so
            log.warning("price book not seeded: is the database migrated?")
        yield

    app = FastAPI(title="PasarBench Agent Service", version=__version__, lifespan=lifespan,
                  description="The customer-service agent PasarBench measures, as a backend: "
                              "conversations, streamed turns, an audited and replayable "
                              "record of every tool call, and a cost ledger.")
    app.state.runtime, app.state.settings, app.state.metrics = runtime, settings, metrics
    app.state.engine = engine
    app.add_middleware(Observe, metrics=metrics)

    @app.exception_handler(E.ServiceError)
    async def _service_error(request: Request, err: E.ServiceError) -> JSONResponse:
        return problem_response(err, request)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, err: RequestValidationError) -> JSONResponse:
        detail = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}"
                           for e in err.errors())[:1000]
        return problem_response(_Invalid(detail), request)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, err: StarletteHTTPException) -> JSONResponse:
        e = E.ServiceError(str(err.detail))
        e.status, e.kind, e.title = err.status_code, f"http-{err.status_code}", str(err.detail)
        return problem_response(e, request)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, err: Exception) -> JSONResponse:
        log.error("unhandled", exc_info=err,
                  extra={"request_id": request.scope.get("state", {}).get("request_id")})
        return problem_response(E.ServiceError("an unexpected error; it has been logged"),
                                request)

    # --- who is calling ---------------------------------------------------
    def tenant(request: Request,
               authorization: Annotated[str | None, Header()] = None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise E.Unauthorized("send Authorization: Bearer <api key>")
        found = keys.lookup(authorization[7:].strip())
        if found is None:
            raise E.Unauthorized()
        tenant_id = found[0]
        request.scope["state"]["tenant_id"] = tenant_id
        wait = limits.take(tenant_id)
        if wait is not None:
            metrics.rejections.labels("rate_limited").inc()
            raise E.RateLimited(f"more than {settings.rate:g} requests a second", retry_after=wait)
        return tenant_id

    def operator(authorization: Annotated[str | None, Header()] = None) -> None:
        token = settings.admin_token
        if not token:
            raise E.NotFound("the admin API is off: set PASAR_ADMIN_TOKEN, or use the CLI")
        given = (authorization or "")[7:].strip() if (authorization or "").startswith("Bearer ") else ""
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise E.Unauthorized("the operator's token is needed here")

    Tenant = Annotated[str, Depends(tenant)]
    IdemKey = Annotated[str | None, Header(alias="Idempotency-Key")]

    def idempotent(tenant_id: str, key: str | None, request: Request, body: BaseModel,
                   stream: bool = False) -> Response | None:
        """The stored answer, if this key has one -- the bytes the first
        request was sent; a stream asked for again gets its last event -- or
        None to run the request."""
        if key is None:
            return None
        rh = request_hash(request.method, request.url.path, body.model_dump_json().encode())
        request.state.idem_hash = rh
        got = idem.begin(tenant_id, key, rh)
        if got is None:
            return None
        status, stored = got
        replayed = {"Idempotent-Replayed": "true"}
        if stream and status == 200:
            return Response(f"event: turn.completed\ndata: {stored}\n\n",
                            media_type="text/event-stream", headers=replayed)
        return Response(stored, status_code=status, headers=replayed,
                        media_type=PROBLEM if status >= 400 else "application/json")

    def answered(request: Request, tenant_id: str, key: str | None,
                 response: JSONResponse) -> JSONResponse:
        """A response as it goes out, kept under its key if it has one."""
        if key:
            idem.complete(tenant_id, key, response.status_code, response.body.decode(),
                          request.state.idem_hash)
        return response

    def keepalive(tenant_id: str, key: str | None) -> Any:
        """A running request's key lives as long as its turn (the lease's heartbeat)."""
        return (lambda: idem.extend(tenant_id, key)) if key else None

    def failed(e: BaseException, request: Request) -> JSONResponse:
        """What a turn that began and then failed answers -- kept under the
        key like any answer: the turn may have changed things, and a retry
        must not run it again."""
        if isinstance(e, LeaseLost):
            return problem_response(E.Conflict(str(e)), request)
        if isinstance(e, E.ServiceError):
            return problem_response(e, request)
        log.error("turn failed", exc_info=e,
                  extra={"request_id": request.scope.get("state", {}).get("request_id")})
        return problem_response(E.ServiceError("the turn failed; it has been logged"), request)

    # --- health -----------------------------------------------------------
    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        """The process is up."""
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"], responses={503: ERRORS[503]})
    def readyz() -> dict[str, Any]:
        """The database answers and holds the schema: send traffic here."""
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                migrated = inspect(conn).has_table("conversations")
        except Exception as e:  # noqa: BLE001
            raise E.Unavailable(f"database: {type(e).__name__}") from None
        if not migrated:
            raise E.Unavailable("the database has no schema: run `python -m pasarbench.service "
                                "migrate`")
        return {"status": "ready", "database": engine.dialect.name,
                "turns_in_flight": runtime.in_flight}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def prometheus() -> Response:
        for provider, g in runtime.gates.snapshot().items():
            metrics.provider_in_flight.labels(provider).set(g["in_flight"])
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # --- conversations ----------------------------------------------------
    @app.post("/v1/conversations", status_code=201, response_model=Conversation,
              tags=["conversations"], responses=ERRORS)
    def open_conversation(body: OpenConversation, request: Request, tenant_id: Tenant,
                          idempotency_key: IdemKey = None) -> Any:
        """Open a conversation for one of the tenant's customers."""
        if (stored := idempotent(tenant_id, idempotency_key, request, body)) is not None:
            return stored
        try:
            out = sent(Conversation, runtime.open_conversation(tenant_id, body.user_id,
                                                               body.model))
        except BaseException:
            # One transaction, rolled back: nothing happened, and the key is free.
            if idempotency_key:
                idem.abandon(tenant_id, idempotency_key)
            raise
        return answered(request, tenant_id, idempotency_key, JSONResponse(out, status_code=201))

    @app.get("/v1/conversations", response_model=ConversationPage, tags=["conversations"],
             responses=ERRORS)
    def list_conversations(tenant_id: Tenant,
                           user_id: Annotated[str | None, Query(pattern=NAME)] = None,
                           limit: Annotated[int, Query(ge=1, le=200)] = 50,
                           before: Annotated[str | None,
                                             Query(pattern=CONVERSATION_ID)] = None) -> Any:
        """The tenant's conversations, newest first."""
        return runtime.list_conversations(tenant_id, user_id, limit, before)

    @app.get("/v1/conversations/{cid}", response_model=Conversation, tags=["conversations"],
             responses=ERRORS)
    def get_conversation(cid: Cid, tenant_id: Tenant) -> Any:
        return runtime.get_conversation(tenant_id, cid)

    @app.post("/v1/conversations/{cid}/close", response_model=Conversation,
              tags=["conversations"], responses=ERRORS)
    def close_conversation(cid: Cid, tenant_id: Tenant) -> Any:
        return runtime.close(tenant_id, cid)

    @app.get("/v1/conversations/{cid}/messages", response_model=list[TranscriptMessage],
             response_model_exclude_none=True, tags=["conversations"], responses=ERRORS)
    def transcript(cid: Cid, tenant_id: Tenant, include_system: bool = False) -> Any:
        """Every message, as the model saw them: the customer's, the agent's,
        the tools' answers."""
        return runtime.messages(tenant_id, cid, include_system)

    @app.get("/v1/conversations/{cid}/events", response_model=list[Event],
             tags=["conversations"], responses=ERRORS)
    def events(cid: Cid, tenant_id: Tenant, after: Annotated[int, Query(ge=0)] = 0,
               limit: Annotated[int, Query(ge=1, le=1000)] = 500) -> Any:
        """What the turns did, step by step: a stream's events, kept. Pass the
        last id you have as `after` to pick up where a dropped stream left off."""
        return runtime.events(tenant_id, cid, after, limit)

    @app.get("/v1/conversations/{cid}/tool-calls", response_model=list[AuditedToolCall],
             tags=["audit"], responses=ERRORS)
    def audit(cid: Cid, tenant_id: Tenant) -> Any:
        """Every tool call, as the audit log keeps it."""
        return runtime.tool_calls(tenant_id, cid)

    @app.post("/v1/conversations/{cid}/messages", response_model=TurnResult,
              tags=["conversations"],
              responses={**ERRORS, 200: {"content": {"text/event-stream": {}}}})
    def post_message(cid: Cid, body: CustomerMessage, request: Request, tenant_id: Tenant,
                     idempotency_key: IdemKey = None, stream: bool = False) -> Any:
        """The customer's message. The answer is the agent's reply when the
        turn is done -- or, with `stream=true` (or Accept: text/event-stream),
        Server-Sent Events as it runs: turn.started, model.call, tool.result
        ..., turn.completed. A client that loses the stream loses nothing: the
        turn runs to its end, and GET .../events?after= has every event."""
        stream = stream or "text/event-stream" in request.headers.get("accept", "")
        if (stored := idempotent(tenant_id, idempotency_key, request, body, stream)) is not None:
            return stored
        try:
            turn = runtime.begin_turn(tenant_id, cid)
        except BaseException:
            # Turned away before anything was written: the key is free.
            if idempotency_key:
                idem.abandon(tenant_id, idempotency_key)
            raise
        if stream:
            return streamed(turn, body.content, tenant_id, idempotency_key, request)
        try:
            response = JSONResponse(sent(TurnResult, turn.run(
                body.content, keepalive=keepalive(tenant_id, idempotency_key))))
        except BaseException as e:  # noqa: BLE001 -- answered, and the answer kept
            response = failed(e, request)
        return answered(request, tenant_id, idempotency_key, response)

    def streamed(turn: Turn, content: str, tenant_id: str, key: str | None,
                 request: Request) -> StreamingResponse:
        """The turn on its own thread, started here -- not when the stream is
        first read, which a client gone before the response starts never does:
        the turn runs to its end, gives its lease and slot back and keeps its
        answer under the key, whether or not anyone reads the stream."""
        try:
            # The loop the stream is read on: kept at startup, or asked of the
            # loop itself from this worker thread when the server ran no
            # startup (uvicorn --lifespan off).
            loop = loop_box.get("loop") or anyio.from_thread.run_sync(asyncio.get_running_loop)
        except BaseException:
            turn.release()              # the slot and the lease back, not held to expiry
            raise
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def put(item: tuple[str, Any]) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:        # the loop is gone (shutdown): the turn still finishes
                pass

        def work() -> None:
            try:
                out = sent(TurnResult, turn.run(content, emit=lambda ev: put(("event", ev)),
                                                keepalive=keepalive(tenant_id, key)))
                answered(request, tenant_id, key, JSONResponse(out))
                put(("done", out))
            except BaseException as e:  # noqa: BLE001 -- told to the stream, and kept
                response = failed(e, request)
                try:
                    answered(request, tenant_id, key, response)
                finally:
                    put(("error", response.body.decode()))

        try:
            threading.Thread(target=work, name="pasar-turn", daemon=True).start()
        except BaseException:
            turn.release()
            raise

        async def events() -> AsyncIterator[bytes]:
            while True:
                kind, item = await queue.get()
                if kind == "event":
                    data = json.dumps(item["data"], ensure_ascii=False, default=str)
                    yield f"id: {item['id']}\nevent: {item['type']}\ndata: {data}\n\n".encode()
                elif kind == "error":
                    yield f"event: error\ndata: {item}\n\n".encode()
                    return
                else:
                    return

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # --- usage --------------------------------------------------------------
    @app.get("/v1/usage", response_model=list[UsageRow], tags=["usage"], responses=ERRORS)
    def usage(tenant_id: Tenant,
              group_by: Annotated[str, Query(pattern="^(model|day|conversation)$")] = "model",
              since: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
              until: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None) -> Any:
        """What the tenant's model calls cost, from the ledger: exact to the
        picodollar, UTC days, `until` exclusive."""
        def day(d: str | None) -> datetime | None:
            return None if d is None else datetime.strptime(d, "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        with engine.connect() as conn:
            return usage_report(conn, tenant_id, group_by, day(since), day(until))

    # --- the operator -------------------------------------------------------
    Operator = Depends(operator)

    @app.post("/v1/admin/tenants", status_code=201, response_model=TenantCreated,
              tags=["admin"], dependencies=[Operator], responses=ERRORS)
    def new_tenant(body: NewTenant) -> Any:
        """A tenant, its starting world and its first API key."""
        budget = None
        if body.daily_budget_usd is not None:
            try:
                budget = usd_to_picousd(body.daily_budget_usd)
            except InvalidOperation:
                raise E.BadRequest("daily_budget_usd is a decimal string, e.g. \"25.00\"")
        with engine.connect() as conn:
            if conn.execute(select(S.tenants.c.tenant_id).where(
                    S.tenants.c.tenant_id == body.tenant_id)).first() is not None:
                raise E.Conflict(f"tenant {body.tenant_id!r} exists")
        try:
            digest = seed_tenant(engine, body.tenant_id, body.name, body.world, budget)
        except ValueError as e:
            raise E.BadRequest(str(e)) from None
        key_id, secret = create_key(engine, body.tenant_id)
        return {"tenant_id": body.tenant_id, "world": body.world, "world_digest": digest,
                "key_id": key_id, "api_key": secret}

    @app.post("/v1/admin/tenants/{tenant_id}/keys", status_code=201, response_model=KeyCreated,
              tags=["admin"], dependencies=[Operator], responses=ERRORS)
    def new_key(tenant_id: Annotated[str, Path(pattern=TENANT_ID)]) -> Any:
        key_id, secret = create_key(engine, tenant_id)
        return {"key_id": key_id, "api_key": secret}

    @app.post("/v1/admin/keys/{key_id}/revoke", status_code=204, tags=["admin"],
              dependencies=[Operator], responses=ERRORS)
    def revoke(key_id: Annotated[str, Path(pattern=KEY_ID)]) -> Response:
        """Within 30 seconds every process refuses it (auth.KeyCache)."""
        if not revoke_key(engine, key_id):
            raise E.NotFound(f"no live key {key_id!r}")
        return Response(status_code=204)

    @app.get("/v1/admin/tenants/{tenant_id}/replay", tags=["admin"], dependencies=[Operator],
             responses=ERRORS)
    def replay(tenant_id: Annotated[str, Path(pattern=TENANT_ID)]) -> dict[str, Any]:
        """Every tool call the tenant's conversations made, run again against
        its starting world, and compared (replay.py)."""
        try:
            return replay_tenant(engine, tenant_id)
        except Exception as e:
            if "no tenant" in str(e):
                raise E.NotFound(str(e)) from None
            raise

    @app.get("/v1/admin/outbox", tags=["admin"], dependencies=[Operator], responses=ERRORS)
    def outbox(limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> list[dict[str, Any]]:
        """Events not yet delivered -- an escalation is a ticket for a human:
        a relay reads them here and acknowledges each once delivered."""
        o = S.outbox
        with engine.connect() as conn:
            rows = conn.execute(select(o).where(o.c.delivered_at.is_(None))
                                .order_by(o.c.event_id).limit(limit))
            return [{"event_id": r.event_id, "tenant_id": r.tenant_id, "topic": r.topic,
                     "payload": r.payload, "created_at": iso(r.created_at)} for r in rows]

    @app.post("/v1/admin/outbox/ack", tags=["admin"], dependencies=[Operator], responses=ERRORS)
    def outbox_ack(body: OutboxAck) -> dict[str, int]:
        o = S.outbox
        with transaction(engine, write=True) as conn:
            n = conn.execute(update(o).where(o.c.event_id.in_(body.event_ids),
                                             o.c.delivered_at.is_(None))
                             .values(delivered_at=utcnow(), attempts=o.c.attempts + 1)).rowcount
        return {"acknowledged": n}

    @app.get("/v1/admin/gates", tags=["admin"], dependencies=[Operator], responses=ERRORS)
    def gates() -> dict[str, Any]:
        """Each model provider's gate: calls in flight, retries, its breaker."""
        return {"providers": runtime.gates.snapshot(), "turns_in_flight": runtime.in_flight}

    return app


class _Invalid(E.ServiceError):
    status, kind, title = 422, "invalid-request", "The request is not valid"

