"""
PasarBench Agent Service: the customer-service agent the benchmark measures,
served as a backend (docs/SERVICE.md).

The benchmark core (pasarbench/*) has no dependencies and never imports this
package. The service needs FastAPI, SQLAlchemy, Alembic, psycopg and
prometheus-client (requirements-service.txt), and it reuses the benchmark's own
parts unchanged -- the tools and their checks, the policy, the system prompt,
the model backends -- so that what the benchmark scores is what the service
runs (tests/test_service.py: every reference solution, through the HTTP API,
leaves the world and the action log the benchmark's own episode leaves).

    schema.py         the tables: tenants and keys, the commerce world, conversations,
                      the audit log, idempotency keys, the price book, the ledger,
                      the outbox; migrations/ builds them (Alembic)
    store.py          a tool call: one customer's slice loaded, the tool run on it, its
                      changes written back in one short transaction under that
                      customer's lock, with the call's audit rows
    runtime.py        a turn: admission, the conversation's lease, model calls outside
                      any transaction, tool calls inside one each, budgets, events
    models.py         the models a conversation may run on: the mock agent, the
                      reference solutions, real providers
    resilience.py     between the service and a provider: a concurrency gate, retries
                      with jitter, a circuit breaker; per-tenant token buckets
    costs.py          the price book and the ledger, in exact integer picodollars
    replay.py         every tool call run again against the rebuilt starting world
    auth.py           API keys (stored hashed) and Idempotency-Keys
    app.py            the HTTP API: REST, Server-Sent Events, problem details
    observability.py  Prometheus metrics and JSON logs, with no conversation in them
    config.py         PASAR_* settings
    errors.py         what the API answers when it does not do what was asked

    python -m pasarbench.service --help
"""

__version__ = "0.1.0"
