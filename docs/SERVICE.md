# The agent service

The benchmark measures a customer-service agent; `pasarbench/service` runs it
for real customers. A storefront's backend opens a conversation for one of its
customers, posts her messages, and gets the agent's replies — whole, or as
Server-Sent Events while the turn runs. Underneath: the benchmark's own tools,
policy, system prompt and model backends, unchanged, over a SQL database
(PostgreSQL, or SQLite for one machine), with every tool call one short locked
transaction, recorded so that it can be run again and must come out the same.

What holds it together, and how each claim is checked:

- **What the benchmark scores is what the service runs.** Every one of the 313
  reference solutions, sent through the HTTP API, passes its task — and leaves
  the world and the action log exactly as the benchmark's own episode leaves
  them, on SQLite and on PostgreSQL (`tests/test_service.py`, parity).
- **Money cannot leave twice.** Eight concurrent refunds of 30% of one order:
  three paid, five refused, on both databases. Without the customer's lock, two
  raced refunds of 60% both pass — 19,500 paid back on a 16,250 order — which is
  what the lock is for (PostgreSQL; the tests race it both ways).
- **Every tool call can be run again.** The audit log holds what each call read,
  who was verified, the ids it was given and what it changed; replay rebuilds
  the starting world from its name and re-runs every call, which must answer and
  change exactly what it did. It does — 1,926 calls made by 64 concurrent
  clients, in the largest load run — and a row edited behind the service's back
  is found.
- **A turn never ends in silence, and stops when its budget is spent.** A
  failing provider, a spent budget or a cap ends the turn with a fallback reply;
  every model call is charged in the ledger, in exact integer picodollars.
- **The model is an untrusted client.** Its arguments pass the tools' checks,
  the platform's guards (money, isolation, values) and the database's own rules;
  whatever fails is answered as a refusal, and the turn goes on.
- **A real model sees what the benchmark shows it.** The system prompt and the
  tool schemas, byte for byte, on all 313 tasks; and the benchmark's own
  episodes run through the API — a real model as the agent, the benchmark's
  simulated customer — are traced in the benchmark's format, so its tools
  re-score them and its comparisons read them (`scripts/service_sweep.py`,
  [below](#with-a-real-model)).

An independent review of the service before release reported thirteen defects
— a model's bad arguments could crash a turn, a key could be freed after its
turn had written, PostgreSQL returned rows in another order — twelve reproduced
with a script, the thirteenth with two local servers. Its second pass, over the
fixes, found three more that the fixes had made, and NUL characters on
PostgreSQL. Each is fixed and pinned by a test ([WHAT_FAILED](WHAT_FAILED.md)
#59–#66).

```
pip install -r requirements-service.txt      # the benchmark itself still needs nothing
python -m tests.test_service                 # 98 checks on SQLite; PASAR_TEST_POSTGRES_URL, 27 more
python scripts/service_sweep.py --self-test  # the real-model kit on a stand-in provider: no key
```

---

## Run it

**On one machine, SQLite:**

```bash
pip install -r requirements-service.txt
export PASAR_DATABASE_URL=sqlite:///data/pasar.db
python -m pasarbench.service migrate
python -m pasarbench.service create-tenant acme --name "Acme Mart" --world demo
#   API key key_3f2a... -- shown once, keep it:
#   psk_...
python -m pasarbench.service serve --port 8000          # http://127.0.0.1:8000/docs
```

**PostgreSQL, in containers:**

```bash
docker compose -f deploy/compose.yaml up --build        # PostgreSQL 16, migrate, the API
docker compose -f deploy/compose.yaml run --rm api create-tenant acme --name "Acme Mart"
```

**The published image** (Docker Hub, once CI publishes it:
[below](#the-image-on-docker-hub)). Its entrypoint is the CLI above; with no
command, it serves. SQLite in a volume, to try it:

```bash
IMAGE=<Docker Hub user>/pasarbench-service
docker run --rm -v pasar-data:/app/data $IMAGE migrate
docker run --rm -v pasar-data:/app/data $IMAGE create-tenant acme --name "Acme Mart"
docker run -d --name pasar -v pasar-data:/app/data -p 127.0.0.1:8000:8000 $IMAGE
```

For PostgreSQL, give each `docker run` `-e PASAR_DATABASE_URL=postgresql+psycopg://...`.

The compose file lives in `deploy/` on purpose: Compose fills `${...}` from a
`.env` beside its file, and the repo root's `.env` holds the benchmark's API
keys. With the file in `deploy/`, a key reaches the service only when the shell
running compose has it set, and only the variables the file names
(`DEEPSEEK_API_KEY`, `OPENCODE_API_KEY`); the image is built without any
`.env` (`.dockerignore`). The API listens on 127.0.0.1 only.

**A conversation:**

```bash
KEY=psk_...
J=(-H "Authorization: Bearer $KEY" -H "Content-Type: application/json")
curl -s localhost:8000/v1/conversations "${J[@]}" -H "Idempotency-Key: open-7f3" \
     -d '{"user_id": "U002"}'           # "market": "ID" if she shops on another site
#  {"conversation_id": "cv_2c8a...", "model": "mock", "provenance": {"prompt_digest": ...}, ...}

curl -s localhost:8000/v1/conversations/cv_2c8a.../messages "${J[@]}" -H "Idempotency-Key: msg-1" \
     -d '{"content": "Please cancel O1002, last digits 4567"}'
#  {"reply": "Thank you, you're verified. Order O1002 is processing (total SGD 129.00).
#   I've cancelled order O1002. ...", "stop_reason": "replied",
#   "tool_calls": [{"tool": "verify_identity", ...}, {"tool": "get_order", ...},
#                  {"tool": "cancel_order", ...}], "usage": {...}, "cost_usd": "$0.00...", ...}

curl -sN "localhost:8000/v1/conversations/cv_2c8a.../messages?stream=true" "${J[@]}" \
     -d '{"content": "Where is it now?"}'
#  id: 41
#  event: turn.started
#  ...
#  event: tool.result
#  data: {"step": 1, "tool": "get_shipment", "ok": true, ...}
#  ...
#  event: turn.completed
```

`mock` is the service's own agent (`models.py`): rules, not a model — it reads
the message for an order id, four digits and what is asked, calls the tools a
careful agent would in the order the policy wants, and answers from what they
returned. It needs no key and gives the same answer to the same conversation,
which is what load tests and demos need. `scripted:<task_id>` replays a task's
reference solution. Real models — `deepseek-v4-pro` and `deepseek-flash` built
in, more in a JSON file (`deploy/models.example.json`, `PASAR_MODELS_FILE`) —
are offered only when named in `PASAR_MODELS` and only with their provider's
key on the server, read from that provider's own variables and sent nowhere
else.

---

## How a turn runs

```
POST /v1/conversations/{id}/messages
  │  Bearer key ─▶ tenant           (SHA-256 lookup, cached 30 s)        401
  │  token bucket, per tenant                                             429 Retry-After
  │  Idempotency-Key: seen? ─▶ the stored answer, byte for byte           (replayed)
  ▼
Runtime.begin_turn
  │  admission: at most PASAR_MAX_CONCURRENT_TURNS here                   503 Retry-After
  │  the lease: UPDATE ... WHERE (free or expired) AND open RETURNING *   409
  │  budgets, on the row the lease returned: the conversation's, the day  429
  ▼
Turn.run ── txn: answer what a crash left unanswered; her message ────────▶ event turn.started
  │
  ├─▶ model call ··· outside any transaction, through the provider's gate:
  │     a slot (16 in flight) · retries with full jitter · circuit breaker
  │   txn: the ledger row (always) · counters + the lease renewed (one UPDATE)
  │        · the assistant message ──────────────────────────────────────▶ event model.call
  │
  ├─▶ each tool call: store.run_tool, ONE transaction
  │     lock the customer (FOR UPDATE) · load her slice, in order · run the
  │     benchmark's tool · guards (money, isolation, values) · diff
  │     · SAVEPOINT: write_seq + 1 · UPDATE ... WHERE version = v · INSERT · outbox
  │     · the tool message and audit rows, the lease checked ───────────▶ event tool.result
  │     a version conflict: rolled back, run again on what is there now (3 tries)
  │     a rule the database enforces: undone to the savepoint, recorded as refused
  │     any other failure: answered as one the platform could not carry out
  │
  └─ until the agent replies — or a cap, a budget, the provider: a fallback reply
     txn: the lease checked and released, the fallback ─────────────────▶ event turn.completed
```

Every step commits before the next begins, and each event a stream sends is the
row that step wrote. A heartbeat renews the turn's lease every third of it
while the turn runs, however long one model call takes, and every write the turn
makes checks that it still holds it: a turn that lost its lease writes nothing
more. A process that dies mid-turn leaves a conversation whose
last committed step is true; its lease expires (`PASAR_LEASE_SECONDS`), and the
next turn first answers any tool call left without a result —
`interrupted: this call was not run` — because a model API refuses a
conversation with an unanswered call, and because the call did not run: its
world change and its tool message commit together or not at all.

---

## The seven areas

### 1. System determinism

The same inputs give the same outputs, and the service can prove it after the
fact.

- **One clock.** The world's time is frozen at 11 November 2026, 10:00
  Singapore (the 11.11 sale), as the benchmark's; the prompt says so and the
  conversation records it (`provenance.world_clock`). The service's own times
  are UTC.
- **Ids as the benchmark makes them.** One counter per tenant, so a
  conversation on a fresh tenant names what it creates exactly as the
  benchmark's episode does (`RET0001`, `REF0002`); each call's ids are kept.
- **Provenance.** A conversation records the model and its settings, digests of
  its system prompt, the policy and the tool schemas it was shown, the tools'
  checks version and the platform's money guard — fixed when it opens.
- **Replay** (`replay.py`, `python -m pasarbench.service replay <tenant>`,
  `GET /v1/admin/tenants/{id}/replay`). A tenant's starting world is a pure
  function of its name (`demo`, `task:<id>`, `synthetic:<n>`, digest checked).
  Every dispatched tool call is re-run against it, customer by customer, in the
  order of the states they read (`world_seq`: the writes committed to the
  customer before the call read); each must read the same state, give the same
  answer (`result_sha`) and make the same changes, and the rebuilt world must
  equal the database. The log and the world are read in one snapshot, so a
  tenant can be replayed while it takes traffic. A mismatch means the tools or
  the world builder changed (the digests say which), a row was edited outside
  the service, or a write ran without the lock.
- **The benchmark's order.** Every world row keeps its place (`pos`): the
  seed's order, then new rows in the order their customer's writes committed.
  A tool that lists rows lists them as the benchmark's dicts would — on
  PostgreSQL too, which moves an updated row and returns rows in no order
  unless asked.
- **Exactly-once effects.** An `Idempotency-Key` makes a retried request
  return the first one's answer, byte for byte, instead of running again (one
  customer message, one refund) — whatever that answer was, an error included,
  once the turn had begun; a tool call's change and its record commit
  together.
- **The same model calls.** Real models run at temperature 0 with reasoning
  off where the provider has a switch (RUNBOOK 1s); `mock` and `scripted` are
  deterministic, which is what lets the parity test and the load test's replay
  be exact.

*Checked:* parity — 313/313 identical to the benchmark's episodes on both
databases; replay in the tests and after every load run (`identical` in the
table below); a tampered row found.

### 2. Performance and concurrency

- **No transaction spans a model call.** A model call takes seconds; a tool
  transaction takes milliseconds. Connections are pooled and held only for the
  latter.
- **Locks where the invariants are.** A write locks its customer (one row,
  `SELECT ... FOR UPDATE`), so two conversations of one customer write one at a
  time and two customers never wait for each other. Versioned rows
  (`UPDATE ... WHERE version = v`) catch a read-modify-write gone stale; the
  lock is what stops two inserts that each read the same total (two refunds).
  The tests race both on PostgreSQL: without the lock the version check still
  turns a stale cancellation into a retry that keeps both changes, and two
  refunds both pass.
- **Reads take no lock** and see one snapshot (REPEATABLE READ).
- **Backpressure, not queues.** Per process: at most `PASAR_MAX_CONCURRENT_TURNS`
  turns (else 503 at once, Retry-After); per provider, at most
  `PASAR_PROVIDER_CONCURRENCY` calls in flight (a turn waits up to 30 s for a
  slot, then ends with a fallback reply, `provider_unavailable`); per tenant, a
  token bucket (429). A provider that fails five times in a row is not called
  for 30 s (circuit breaker); transient errors are retried with full jitter,
  three tries.
- **One turn at a time per conversation**, across processes: a lease row with
  an expiry (60 s), taken with the row it protects (`UPDATE ... RETURNING`),
  renewed by a heartbeat every 20 s while the turn runs and checked by every
  write the turn makes; a dead process frees its conversation within a minute.
- **Fewer round trips.** The catalog (products, sellers, livestream claims) is
  read-only to the tools and cached per process; the step's counters and the
  lease renewal are one statement. A turn of this workload is 68.6 statements in
  9 transactions.

*Measured* — below.

### 3. Core data and storage

- **Schema** (`schema.py`): every world table keyed by tenant first
  (`PRIMARY KEY (tenant_id, order_id)`), so two tenants can share order ids and
  never see each other's; foreign keys, CHECK constraints (statuses, amounts,
  non-negative credit), an index for every way a tool reads (`orders` by
  customer, an order's children by order); JSONB on PostgreSQL. Money is
  integer minor units, as in the benchmark.
- **Authorization in the data layer.** A tool call sees one customer's slice:
  another customer's order does not exist for it, whatever the model asks. A
  new row may name only the customer's own orders (the isolation guard). Store
  credit is given in the customer's account currency (P5.3), and what goes
  back for an order — its refunds and the credit naming it, in any currency,
  counted exactly in SGD at the platform's rates — never passes what was paid
  (the money guard); credit naming no order is the policy's to bound, not the
  platform's. A new
  row's amount is a positive whole number a column can hold (the value guard),
  and a write the database refuses anyway is undone to a savepoint and recorded
  as refused, by the rule it broke. A refused call is logged as the tools log
  their own refusals.
- **Migrations** (Alembic, `pasarbench/service/migrations`): the tests migrate
  an empty database to head and compare it with `schema.py` — no difference on
  either database — then down to nothing and up again.
- **Audit, outbox, ledger.** `tool_calls` and `actions` (the benchmark's action
  log, line for line) record every call; `events` every step; `outbox` gets an
  `escalation.created` ticket in the same transaction as the escalation (a relay
  reads `GET /v1/admin/outbox` and acknowledges); `usage_ledger` one row per
  model call.
- **SQLite** for one machine and the tests (WAL, foreign keys on, a write
  transaction takes the database's write lock at BEGIN); **PostgreSQL** for
  anything with more than one process.

### 4. Interfaces and protocols

- **REST under `/v1`**, an OpenAPI document generated from typed request and
  response models (`/docs`, `/openapi.json`); the tests pin the exact route
  list.
- **Server-Sent Events** (`?stream=true` or `Accept: text/event-stream`): each
  committed step as it happens, each with its event id. The turn's thread starts
  before the response does, so a client that drops the stream — or is gone
  before it starts — loses nothing: the turn runs to its end, and
  `GET .../events?after=<id>` returns every event since.
- **Errors are RFC 9457 problem details** (`application/problem+json`), each
  kind its own `type` — `turn-in-progress`, `budget-exhausted`,
  `idempotency-key-reused`, ... — so a client branches on the type, never the
  wording; 429 and 503 carry `Retry-After` wherever waiting helps (not a
  conversation that has spent its budget: it needs a new one).
- **Idempotency-Key** on the two POSTs that create things; replayed answers
  carry `Idempotent-Replayed: true` and the first answer's bytes (kept as text:
  JSONB would reorder their keys), a retried stream its last event. Freed only
  for a request turned away before it wrote anything; a running request's key
  lives as long as its turn (the lease's heartbeat), and one whose request died
  with its process expires with the lease. The same key with another body: 422;
  while the first is still running: 409.
- **Input with the shape it should have.** Every id in a path or query must look
  like one, and no field may carry a NUL character, which PostgreSQL's text
  cannot hold (422); a NUL the model sends is stored as U+FFFD.
- **Auth:** a tenant's `Bearer` key, stored as SHA-256, revocable (effective
  within 30 s everywhere); the operator's token for `/v1/admin`, off unless set.
  Another tenant's conversation answers 404, not 403.
- **Pagination by key**, not offset (`before=<id>`); `X-Request-ID` on every
  response.
- **Operations:** `/healthz` (the process), `/readyz` (the database answers and
  has the schema), `/metrics` (Prometheus: requests by route template, turns by
  how they stopped, model calls, tokens and cost, tool calls, conflicts,
  rejections, calls in flight per provider — never an id in a label). Logs are
  one JSON object a line, with ids, counts and timings, never what anyone said —
  by construction: an exception is logged as its type, SQLSTATE and frames, not
  its message (a database's can quote the row it refused), the engines keep
  bound values out of their errors, and uvicorn's loggers share the formatter.
- **Keys go to their provider and stop there.** Each from its provider's own
  variables, to that provider's endpoint; a redirect is refused, not followed
  (urllib would follow it, `Authorization` and all, to any host).

### 5. Testing and quality assurance

`tests/test_service.py`, in the repo's own style, run by `run_tests.sh` when
the service's dependencies are installed and by its own CI job against a real
PostgreSQL (`.github/workflows/tests.yml`), which also builds the image and
runs it:

| what | how |
|---|---|
| parity with the benchmark | all 313 reference solutions through the HTTP API; verdict, final world and action log against the benchmark's own episode, and the system prompt the model is shown, byte for byte; on SQLite and PostgreSQL |
| the benchmark's episodes through the API | the episode driver behind `scripts/service_sweep.py`: each traced step for step as the benchmark traces the same episode (calls, arguments, the answers' digests, replies, context length), re-scored to the same verdict; the whole kit on a stand-in provider with no key (`--self-test`, in CI): the model's real HTTP path, the customer's, the ledger, replay, and no key in the server's process |
| replay | a quarter of those tenants replayed; a tampered row found; every load run replayed |
| concurrency | eight refunds raced; the lock removed and the double refund shown; a stale cancellation retried by the version check (PostgreSQL) |
| isolation | another customer's order invisible, a credit naming it refused, two tenants' rows apart |
| protocol | idempotency (replay, mismatch, in progress), the lease (409), admission (503), rate limit (429), SSE order and recovery, problem details, auth and revocation, pagination |
| failure | a provider down: retries, fallback reply, breaker open, nothing charged; a crash mid-turn: the lease holds, then the unanswered call is answered; a lease lost mid-turn: nothing more written, the answer kept under the key |
| a model's arguments | 1e400, a voucher of 0, a credit of 10**19, a 300-character tool name, credit not in the account's currency, credit against an order in another currency counted in SGD, an empty order id, NUL characters from the customer and the model — each answered, the turn replying, on both databases |
| long turns | a model call twice the lease: the heartbeat keeps the lease and the key, a retry and another message are both turned away, and the retry then gets the answer (the test fails with the heartbeat off); a stream on a server that ran no startup |
| order and logs | a customer's rows in the benchmark's order after updates, on both databases; an exception logged without its message |
| money | prices exact to the microdollar, costs additive, effective dates, budgets for a conversation and a tenant's day |
| schema | migration to head equals `schema.py`; down and up again |
| the API's shape | the route list pinned; every `/v1` operation documents its problem responses |
| the image | on every push, run as a user runs it (`scripts/smoke_image.sh`): its CLI migrates and makes a tenant; its HEALTHCHECK says healthy; it runs as a user other than root and holds no `.env`; over real HTTP a turn, the turn retried under its key (the same bytes, no second turn), a stream, the audit log (`scripts/smoke_api.py`); every call replayed; stopped by SIGTERM. Published only after it passes |

`scripts/load_test.py` measures a real server (uvicorn in its own process,
real HTTP) and replays what it made. `scripts/smoke_api.py` checks any running
server, a deployed one included, on a tenant made for it.

### 6. Cost accounting

- **Exact.** Prices are kept as integer microdollars per million tokens —
  every listed price is a whole number of them, MiMo's cached $0.003625 among
  them — so a call's cost is integer picodollars:
  `(prompt − cached) × input + cached × cached_price + completion × output`.
  No float anywhere; a month of calls sums the same in any order (tested).
- **A price book with dates** (`price_book`): a provider changes a price, a row
  is added, and each ledger row keeps the `price_id` it was charged at. A model
  without a price is never offered: a conversation does not open on it.
- **A ledger row per model call** — tokens (and whether the service estimated
  them: the mock, or a provider that sent no usage, which is estimated and
  charged, never free), cost, latency, the model the provider says it served.
  Charged even when the turn then fails.
- **Budgets:** per conversation (`PASAR_CONVERSATION_BUDGET_USD`, $0.50) and
  per tenant per UTC day (`daily_budget_usd`); checked before each call, so a
  budget is overshot by at most the calls already in flight, and what overshoots
  is recorded.
- **Reports:** `GET /v1/usage?group_by=model|day|conversation`, the CLI's
  `usage`, and Prometheus counters.

**Unit economics.** RUNBOOK 1s measured what a conversation sends: an agent's
225 conversations, about 7.8M prompt tokens in 1,600 requests and 0.13M out —
per conversation 34,667 prompt tokens in 7.1 requests and 578 out, 97.6% of the
prompt tokens from DeepSeek's cache. At list prices (OpenCode Go's per-token
metering for its models), with nine in ten prompt tokens cached, by the
service's own cost function:

| model | a conversation | 1,000 conversations | none cached | its hard cases passed (RESULTS §8) |
|---|---|---|---|---|
| `mimo-v2.6-flash`, thinking off | $0.00074 | $0.74 | $0.0050 | 0.991 |
| `claude-haiku-5-5` (Go) | $0.00095 | $0.95 | $0.0038 | 0.991 |
| `deepseek-flash`, thinking off | $0.0019 | $1.92 | $0.011 | 0.991 |
| `deepseek-v4-pro`, thinking off | $0.0082 | $8.24 | $0.048 | 0.944 |
| `kimi-k2.6` (Go) | $0.011 | $10.60 | $0.035 | 0.815 |
| `grok-4.3`, reasoning none | $0.012 | $12.02 | $0.045 | 0.685 |

The model is the whole of the variable cost. The service's own is three orders
of magnitude below it: one process here does ~20 turns a second on two vCPUs it
shares with its database and its load (below) — at an assumed $0.04 a vCPU-hour,
about $0.000001 a turn. What usually dominates a real deployment's unit cost is
neither: it is the human handling an escalation (tests/test_serving.py's cost
inversion), which is why every escalation is an outbox ticket the ledger can be
joined to.

### 7. The stack, and why

| part | choice | why |
|---|---|---|
| API | FastAPI 0.162, Pydantic 2, uvicorn | typed bodies → validation and the OpenAPI document from one source; sync handlers in a bounded thread pool keep the benchmark's synchronous code unchanged |
| data | SQLAlchemy 2.1 Core, psycopg 3 | explicit SQL, no ORM state to go stale between a model call and a write; one schema for SQLite and PostgreSQL |
| database | PostgreSQL 16; SQLite 3 (WAL) | row locks, snapshot reads and JSONB; SQLite for a laptop and the tests |
| migrations | Alembic | revisions generated from the schema, checked against it |
| metrics | prometheus-client | the counters an operator alerts on, one registry per app |
| packaging | Docker (python:3.12-slim, non-root, the CLI as entrypoint), Compose, Docker Hub | one image, one command to PostgreSQL + migration + API; `docker run` from the registry for amd64 and arm64 |
| CI | GitHub Actions | the benchmark on 3.10–3.13 with no dependencies; the service on 3.12 against a PostgreSQL service container; the image built and run; published when all of it passes |

---

## Measured

`scripts/load_test.py`: uvicorn in its own process, real HTTP, the mock model,
customers from a `synthetic:50` world (300 customers). Half the turns read (where
is my order), half write (a voucher, or a human: an insert under the customer's
lock and an outbox ticket), each turn 3–4 model calls and 2–3 tool calls. On
**2 vCPUs shared by the load generator, the server and PostgreSQL 16** (local
socket), Python 3.12, 2026-10-31 — v75 before its last two fixes (#65, #66),
which add no statement to a turn shorter than 20 seconds; one run each, and
repeats of one configuration differed by up to 8%.

| database | server processes | tenants | clients | model latency | turns/s | turn latency p50 / p95 / p99 (ms) | write tool call p50 / p99 (ms) | replay |
|---|---|---|---|---|---|---|---|---|
| SQLite | 1 | 1 | 1 | 0 | 32.6 | 26 / 47 / 55 | 5 / 11 | 420 identical |
| PostgreSQL | 1 | 1 | 1 | 0 | 15.5 | 61 / 87 / 104 | 12 / 19 | 420 identical |
| SQLite | 1 | 1 | 16 | 0 | 31.5 | 142 / 1,961 / 3,348 | 5 / 939 | 1,265 identical |
| PostgreSQL | 1 | 1 | 16 | 0 | 19.6 | 741 / 1,120 / 1,288 | 227 / 575 | 1,263 identical |
| PostgreSQL | 1 | 16 | 16 | 0 | 20.8 | 699 / 1,057 / 1,199 | 150 / 305 | 1,264 identical |
| PostgreSQL | 2 | 1 | 16 | 0 | 27.7 | 541 / 780 / 944 | 161 / 387 | 1,265 identical |
| PostgreSQL, one customer | 1 | 1 | 16 | 0 | 17.7 | 506 / 2,072 / 3,513 | 792 / 3,452 | 1,266 identical |
| SQLite, one customer | 1 | 1 | 16 | 0 | 26.2 | 170 / 2,371 / 4,130 | 8 / 343 | 1,266 identical |
| PostgreSQL | 1 | 1 | 32 | 400 ms | 12.2 | 2,408 / 3,205 / 3,523 | 16 / 93 | 1,262 identical |
| PostgreSQL | 1 | 1 | 64 | 400 ms | 12.1 | 4,822 / 6,430 / 7,532 | 18 / 94 | 1,925 identical |
| PostgreSQL, gate 64 | 1 | 1 | 64 | 400 ms | 17.0 | 3,035 / 5,966 / 8,140 | 1,207 / 6,190 | 1,926 identical |

What it says:

1. **The service's own cost of a turn, unloaded: 26 ms on SQLite, 61 ms on
   PostgreSQL** — 68.6 statements in 9 transactions for a turn of this kind
   (counted), which SQLite answers in process and PostgreSQL over a socket.
   Next to a model call (hundreds of milliseconds to seconds) it is small; next
   to each other it is why PostgreSQL should be near the service.
2. **One process is CPU-bound, at ~20 turns a second here.** Read tool calls,
   which take no lock, wait ~107 ms at 16 clients: Python's interpreter lock and
   two shared cores, not the database. A second process gives +41% on the same
   two cores; on separate cores it scales with processes, because every piece of
   state two processes share is in the database (the lease, the locks, the
   ledger).
3. **SQLite has one writer**: faster at one client, and a long tail at 16 —
   p99 3.3 s against PostgreSQL's 1.3 s. It is for one machine.
4. **Contention where it was designed to be.** Sixteen clients on one customer
   write one at a time (write p99 3.5 s), while their reads do not wait; the
   tenant's single id counter is visible but not yet the limit (the same load
   over 16 tenants: write p99 575 → 305 ms, +6% turns).
5. **With a model's latency, the provider gate is the limit, as configured.**
   16 calls in flight at 400 ms is 40 calls a second, ~13 turns: measured 12.2
   at 32 clients and 12.1 at 64, the extra clients waiting at the gate (p50
   2.4 s → 4.8 s) rather than overrunning the provider. Widened to 64 the
   process is CPU-bound again (17.0), and the one lock every write of a tenant
   shares shows: a write holds the tenant's id counter from its first new id to
   its commit, and with 64 threads on two cores that hold stretches — writes
   p50 1,207 ms, reads (no lock) 145 ms, 64 clients on 64 different customers.
6. **Under all of it, every tool call replayed identically.**

### With a real model

The load test runs the service's own mock agent, so that what it measures is
the service. `scripts/service_sweep.py` runs a real one: the benchmark's sweep
with the agent behind the HTTP API — the same tasks, the benchmark's simulated
customer (a second model, as in the sweep), and the task's verifier reading
the tenant's world and action log from the database
(`pasarbench/service/episodes.py`). Each episode is traced in the benchmark's
format, its steps rebuilt from the service's own records, so
`compare_cells.py` pairs it, task by task, with the sweep's recording of the
same agent.

```bash
pip install -r requirements-service.txt
python scripts/service_sweep.py --self-test        # a stand-in provider on 127.0.0.1: no key, no cost
set -a; . ./.env; set +a
python scripts/service_sweep.py --model deepseek-flash --customer 3 \
  --suite generated --traps duplicate_refund_escalate,out_of_window_offer_voucher,cancel_while_processing \
  -k 3 --run-id S-dsk-flash-parts --workers 8 --compare traces/V-dsk-flash-parts/full
```

What it reports, from the service's records: the pass rates; the agent's cost
per conversation and per resolved one, from the ledger; the prompt tokens and
the cache's share; and a turn's time split three ways — the model's (the calls
the service timed), the tools' (their transactions) and the service's (the
rest of what the client waited for: HTTP, admission, the lease, the
transactions around them). Before any of it, three checks: the benchmark's own
prompt and tools in every conversation; every trace re-scored to the
service's verdict by the benchmark's tools, replaying the calls the service
ran; every tenant replayed by the service, identically. The service is
configured with the benchmark's caps — 30 model calls a conversation
(`PASAR_CONVERSATION_MAX_STEPS`), 120,000 tokens, 180 seconds a turn, 20
customer turns — so a turn stops where the episode would have; the money guard
is off, as the sweep has none; the server's process holds the agent's key and
no other.

**Not measured yet: RUNBOOK 1z**, deepseek-flash and deepseek-v4-pro on 1s's
39 parts, k=3, each against its recording. The recordings' own numbers, at the
price book's rates, are what to expect: $0.0016 a conversation for flash (117
of 117 passed) and $0.0043 for v4-pro (93 of 117), with 93% and 98% of the
prompt from DeepSeek's cache. What has run: the self-test, the whole path on
the stand-in; and all 215 tasks of the first two editions through the kit with
each task's reference solution as the agent — every one passed, every check
holding (215 prompts as the benchmark's, 215 traces re-scored to the same
verdict, 640 tool calls replayed identically). That run is what found
[WHAT_FAILED #67](WHAT_FAILED.md).

---

## Operating it

| variable | default | |
|---|---|---|
| `PASAR_DATABASE_URL` | `sqlite:///data/pasar.db` | `postgresql+psycopg://user:pass@host:5432/db` |
| `PASAR_ADMIN_TOKEN` | — | the operator's token; unset, `/v1/admin` answers 404 (use the CLI) |
| `PASAR_MODELS` | `mock,scripted:*` | the models a conversation may ask for |
| `PASAR_DEFAULT_MODEL` | `mock` | |
| `PASAR_MODELS_FILE` | — | more models (`deploy/models.example.json`) |
| `PASAR_MAX_CONCURRENT_TURNS` / `PASAR_THREADS` | 48 / 64 | turns at once, worker threads, per process |
| `PASAR_PROVIDER_CONCURRENCY` | 16 | model calls in flight per provider, per process: set from the provider's rate limit ÷ processes |
| `PASAR_RATE` / `PASAR_BURST` | 10 / 40 | requests a second per tenant, per process |
| `PASAR_MAX_STEPS_PER_TURN` / `..._TOOL_CALLS_PER_TURN` / `PASAR_MAX_TURN_SECONDS` | 16 / 24 / 120 | a turn's caps |
| `PASAR_CONVERSATION_BUDGET_USD` | 0.50 | |
| `PASAR_CONVERSATION_MAX_STEPS` | 0 | model calls a conversation may make in all; 0 is no cap (the kit sets the benchmark's 30) |
| `PASAR_LEASE_SECONDS` | 60 | renewed every third of it while a turn runs; a dead process frees its conversation within one |
| `PASAR_IDEMPOTENCY_HOURS` | 24 | |
| `PASAR_STRICT_MONEY` | 1 | the platform's money guard |

Day to day: `migrate` before a new version starts (migrations are additive
first, then a release that stops using a column, then one that drops it);
`create-key` / `revoke-key` to rotate a tenant's key without downtime; `gc`
daily for expired Idempotency-Keys; `replay <tenant>` after a deploy that
touches the tools or the policy, and after anything touches the database by
hand; the outbox relay acknowledges what it delivered. Back up PostgreSQL as
usual; the world, the conversations and the ledger are all in it.

### The image on Docker Hub

CI publishes the image once every job has passed (`.github/workflows/tests.yml`,
job `publish`): from a push to `main` as `latest` and `sha-<commit>`, and from
a version tag `v1.2.3` as `1.2.3` and `1.2`. Each tag holds linux/amd64 and
linux/arm64 (Apple silicon, Graviton), with an SBOM and a provenance
attestation. The amd64 image passes the smoke test first; if it fails, nothing
is pushed. The job is skipped until it is turned on, once:

1. **A token.** Docker Hub → Account settings → Personal access tokens →
   Generate new token, with access **Read & Write**, the least that can push.
   The first push creates the repository `pasarbench-service`, public or
   private as the account's default privacy says (My Hub → Settings →
   Default privacy).
2. **The repository's settings.** GitHub → the repository → Settings →
   Secrets and variables → Actions: a *variable* `DOCKERHUB_USERNAME` (the
   Docker Hub user name; it names the image, so it is not a secret) and a
   *secret* `DOCKERHUB_TOKEN` (the token; the one step that logs in is all
   that sees it). With the GitHub CLI: `gh variable set DOCKERHUB_USERNAME
   --body <name>`, then `gh secret set DOCKERHUB_TOKEN`, which asks for the
   token rather than taking it on the command line.

The Docker actions in that job are pinned to their commits, the version beside
each pin. A pull request never publishes, and neither does a fork, which has
no such variable.

---

## Not built yet, in the order they matter

1. **Ids without a tenant-wide lock.** A PostgreSQL sequence per tenant (no row
   lock; gaps on rollback, which replay does not mind — it gives each call its
   recorded ids). Measured above as the next limit once the gate is widened.
2. **Model calls off the worker threads.** An async client for the providers
   would let one process hold hundreds of waiting turns instead of
   `PASAR_THREADS`; the tools and their transactions stay synchronous.
3. **Limits shared across processes.** The token bucket, the provider gate and
   the breaker are per process (N processes allow N times the rate); Redis (or
   PostgreSQL advisory locks) for one limit per deployment.
4. **Tokens streamed from the provider.** Events are per step today; the
   backends do not stream.
5. **The customer's own identity.** The storefront asserts which customer a
   conversation is for; the agent still verifies her digits per the policy. A
   signed customer token from the storefront would bind the two.
6. **Retention and erasure.** Conversations hold what customers said; a
   per-tenant retention period and an erase-a-customer path (the ledger and
   audit rows keep ids, not text).
7. **An outbox relay** that delivers tickets to a helpdesk with retries, rather
   than an endpoint a relay polls.
8. **Prompt caching per conversation on providers that support a key**, and the
   policy's preload/JIT choice per tenant — both measured in the benchmark,
   neither wired here.
