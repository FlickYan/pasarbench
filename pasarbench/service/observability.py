"""
What the service tells its operator: Prometheus metrics at /metrics, and one
JSON object a line on stderr for each request and each turn.

The logs never hold what a customer or the agent said, nor an argument of a
tool call: ids, counts, timings and outcomes only. The conversation itself is
in the database, behind the tenant's key. That holds for errors too, by
construction: an exception is logged as its type, its SQLSTATE and its
traceback's frames, never its message -- a database error's message can quote
the row it refused (PostgreSQL's "Failing row contains ..."), and the engines
keep bound values out of theirs (store.make_engine, hide_parameters).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
LAG = (0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class Metrics:
    """One registry per app, so two apps in one process (the tests) do not
    share counters."""

    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.http_requests = Counter("pasar_http_requests_total", "HTTP requests",
                                     ["method", "route", "status"], registry=r)
        self.http_seconds = Histogram("pasar_http_request_seconds", "HTTP request latency",
                                      ["method", "route"], buckets=LATENCY, registry=r)
        self.turns = Counter("pasar_turns_total", "Conversation turns, by how they ended",
                             ["stop_reason"], registry=r)
        self.turn_seconds = Histogram("pasar_turn_seconds", "Turn latency", buckets=LATENCY,
                                      registry=r)
        self.turns_in_flight = Gauge("pasar_turns_in_flight", "Turns running now", registry=r)
        self.model_calls = Counter("pasar_model_calls_total", "Model calls",
                                   ["provider", "model", "outcome"], registry=r)
        self.model_seconds = Histogram("pasar_model_call_seconds", "Model call latency",
                                       ["provider"], buckets=LATENCY, registry=r)
        self.tokens = Counter("pasar_model_tokens_total", "Model tokens",
                              ["provider", "model", "kind"], registry=r)
        self.cost = Counter("pasar_model_cost_picousd_total", "Model cost, picodollars",
                            ["provider", "model"], registry=r)
        self.tool_calls = Counter("pasar_tool_calls_total", "Tool calls",
                                  ["tool", "outcome"], registry=r)
        self.tool_seconds = Histogram("pasar_tool_call_seconds",
                                      "Tool call latency, its transaction included",
                                      ["kind"], buckets=LATENCY, registry=r)
        self.conflicts = Counter("pasar_store_conflicts_total",
                                 "Tool calls run again after a write conflict", registry=r)
        self.rejections = Counter("pasar_rejections_total", "Requests turned away",
                                  ["reason"], registry=r)
        self.provider_in_flight = Gauge("pasar_provider_in_flight",
                                        "Model calls in flight, per provider", ["provider"],
                                        registry=r)
        # How late the event loop runs what is due on it (v81): every waiting
        # turn's next step -- its model's answer, a transaction's result -- is
        # a callback on the one loop, and a loop behind delays them all.
        self.loop_lag = Histogram("pasar_event_loop_lag_seconds",
                                  "How late the event loop ran a timer due on it",
                                  buckets=LAG, registry=r)
        # What the processes of a deployment share (v82, limits.py).
        self.service_nodes = Gauge("pasar_service_nodes",
                                   "Processes sharing the deployment's limits, this one "
                                   "among them", registry=r)
        self.provider_share = Gauge("pasar_provider_concurrency_share",
                                    "This process's share of each provider's calls at once",
                                    registry=r)
        self.breaker_openings = Counter("pasar_breaker_openings_total",
                                        "Provider breakers opened here, by this process's "
                                        "calls or another's", ["provider", "by"], registry=r)
        self.limits_errors = Counter("pasar_limits_errors_total",
                                     "Shared limits that could not be read or written",
                                     ["part"], registry=r)
        # A database that does not answer (v83, store.py).
        self.database_unavailable = Counter("pasar_database_unavailable_total",
                                            "Requests answered 503: the database did not "
                                            "answer in time, or could not, or had stopped "
                                            "answering (why: statement_timeout, pool, "
                                            "connection, silent)", ["why"],
                                            registry=r)
        self.db_watchdog_aborts = Gauge("pasar_db_watchdog_aborts",
                                        "Connections the watchdog shut down, lent past "
                                        "PASAR_DB_TIMEOUT, since the process started",
                                        registry=r)
        self.database_silent = Gauge("pasar_database_silent",
                                     "1 while the database has stopped answering and "
                                     "requests are refused at once", registry=r)

    # --- what the runtime reports -------------------------------------------
    def model_call(self, provider: str, model: str, seconds: float, prompt: int, cached: int,
                   completion: int, cost: int, outcome: str = "ok") -> None:
        self.model_calls.labels(provider, model, outcome).inc()
        self.model_seconds.labels(provider).observe(seconds)
        if outcome == "ok":
            self.tokens.labels(provider, model, "prompt").inc(prompt)
            self.tokens.labels(provider, model, "cached").inc(cached)
            self.tokens.labels(provider, model, "completion").inc(completion)
            self.cost.labels(provider, model).inc(cost)

    def tool_call(self, tool: str, outcome: str, is_write: bool, seconds: float,
                  attempts: int) -> None:
        self.tool_calls.labels(tool, outcome).inc()
        self.tool_seconds.labels("write" if is_write else "read").observe(seconds)
        if attempts > 1:
            self.conflicts.inc(attempts - 1)

    def turn(self, stop_reason: str, seconds: float) -> None:
        self.turns.labels(stop_reason).inc()
        self.turn_seconds.observe(seconds)


async def watch_loop(metrics: "Metrics", every: float = 0.1) -> None:
    """The event loop's lag, measured as long as it runs: a timer every
    `every` seconds, and how late it fired."""
    loop = asyncio.get_running_loop()
    while True:
        t0 = loop.time()
        await asyncio.sleep(every)
        metrics.loop_lag.observe(max(0.0, loop.time() - t0 - every))


class JsonFormatter(logging.Formatter):
    FIELDS = ("request_id", "tenant_id", "conversation_id", "method", "route", "status",
              "ms", "turn", "stop_reason", "steps", "tool_calls", "cost_picousd", "event")

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k in self.FIELDS:
            v = getattr(record, k, None)
            if v is not None:
                out[k] = v
        if record.exc_info and record.exc_info[0] is not None:
            etype, evalue, tb = record.exc_info
            out["exc_type"] = etype.__name__
            code = getattr(getattr(evalue, "orig", None), "sqlstate", None)
            if code:
                out["sqlstate"] = code
            out["traceback"] = "".join(traceback.format_tb(tb))[-4000:]
        return json.dumps(out, ensure_ascii=False)


def uvicorn_log_config(level: str = "WARNING") -> dict[str, Any]:
    """uvicorn's own loggers through the same formatter: it logs an exception
    a handler re-raised (Starlette re-raises a 500's), message and all
    otherwise. Its access log is off: the app logs every request itself."""
    return {
        "version": 1, "disable_existing_loggers": False,
        "formatters": {"json": {"()": "pasarbench.service.observability.JsonFormatter"}},
        "handlers": {"stderr": {"class": "logging.StreamHandler", "formatter": "json",
                                "stream": "ext://sys.stderr"}},
        "loggers": {"uvicorn": {"handlers": ["stderr"], "level": level, "propagate": False},
                    "uvicorn.error": {"handlers": ["stderr"], "level": level,
                                      "propagate": False},
                    "uvicorn.access": {"handlers": [], "level": "CRITICAL",
                                       "propagate": False}},
    }


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger("pasar")
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        root.setLevel(level)
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
    root.setLevel(level)
    root.propagate = False
