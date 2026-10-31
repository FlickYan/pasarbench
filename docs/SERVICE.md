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
  [below](#with-a-real-model)). Run with deepseek-flash and deepseek-v4-pro,
  116 and 100 of 117 conversations passed through the service, against 117
  and 93 in their recordings: no difference the sign test resolves. The service
  took 18 ms of a turn at the median, under 1% of it, and a conversation cost
  $0.0015 and $0.0052.
- **It knows where its CPU goes.** A turn of the load test costs the server 19
  ms of CPU, three fifths less than in v79, and one process does 42 turns a
  second on two shared cores where it did 19 — the same turns, replayed
  identically ([below](#where-a-turns-cpu-went-v80)). What was cut was found
  with a sampler of the service's own that an operator runs on a live process
  (`GET /v1/admin/profile`), checked against threads whose CPU is known.
- **A turn waiting on its model holds no thread.** A thousand customers
  waiting on a ten-second model at once are served on 94 threads and 179 MB,
  where v80 needed 1,984 threads and 261 MB, at the same throughput and with
  shorter turns ([below](#turns-that-wait-without-a-thread-v81)).
- **One limit for the deployment.** Two servers in front of a provider that
  takes 8 calls at once: with each limit a process's own, as through v81, they
  had 16 in flight, the provider refused 59–61% of their calls, and under a
  third of the turns were answered; with the limits shared through the
  database (v82), never more than 8, nothing refused, every turn answered —
  6.4–6.5 a second against 3.9–4.2 ([below](#one-limit-for-the-deployment-v82)). A
  tenant's rate holds whichever server it reaches, and a provider failing for
  one server is not called from the other 11–16 ms on.

An independent review of the service before release reported thirteen defects
— a model's bad arguments could crash a turn, a key could be freed after its
turn had written, PostgreSQL returned rows in another order — twelve reproduced
with a script, the thirteenth with two local servers. Its second pass, over the
fixes, found three more that the fixes had made, and NUL characters on
PostgreSQL. Each is fixed and pinned by a test ([WHAT_FAILED](WHAT_FAILED.md)
#59–#66). The review of v82's shared limits reproduced seven more — what they
did when the database was not migrated, not answering or busy behind a
writer — and its second pass one that a fix had made; each is fixed and
tested before release (#71).

```
pip install -r requirements-service.txt      # the benchmark itself still needs nothing
python -m tests.test_service                 # 146 checks on SQLite; PASAR_TEST_POSTGRES_URL, 50 more
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
Runtime.abegin_turn
  │  admission: at most PASAR_MAX_CONCURRENT_TURNS here                   503 Retry-After
  │  the lease: UPDATE ... WHERE (free or expired) AND open RETURNING *   409
  │  budgets, on the row the lease returned: the conversation's, the day  429
  ▼
Turn.arun ── txn: answer what a crash left unanswered; her message ───────▶ event turn.started
  │            (each txn in a worker thread; between them the turn holds none)
  ├─▶ model call ··· awaited, outside any transaction, through the provider's gate:
  │     a slot (16 in flight) · retries with full jitter · circuit breaker
  │   txn: the ledger row (always) · counters + the lease renewed (one UPDATE)
  │        · the assistant message ──────────────────────────────────────▶ event model.call
  │
  ├─▶ each tool call: store.run_tool, ONE transaction
  │     lock the customer (FOR UPDATE) · her slice, in order: the one kept
  │     from the last read if her write_seq has not moved · run the
  │     benchmark's tool on a copy · guards (money, isolation, values) · diff
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
- **Ids as the benchmark makes them.** One sequence per tenant (on SQLite, a
  counter row), so a conversation on a fresh tenant names what it creates
  exactly as the benchmark's episode does (`RET0001`, `REF0002`); each call's
  ids are kept. A sequence gives its next value without holding anything to
  commit (v79): the counter row it replaced made every write of a tenant that
  created a row wait for every other (Measured, below). What a sequence gives
  is never taken back, so a call rolled back and run again leaves a gap, which
  replay does not mind: it gives each call the ids it was given.
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
- **A turn waiting on its model holds no thread** (v81). A turn is a coroutine:
  each of its transactions runs in a worker thread for the milliseconds it
  takes, and the seconds of a model call are awaited, so a process holds as
  many waiting turns as it admits, not as it has threads. An admitted turn's
  transactions have `PASAR_THREADS` threads of their own, apart from every
  other handler's, so a burst of new requests never queues in front of the
  turns already running, which hold leases and slots until they finish.
- **Locks where the invariants are.** A write locks its customer (one row,
  `SELECT ... FOR UPDATE`), so two conversations of one customer write one at a
  time and two customers never wait for each other. Versioned rows
  (`UPDATE ... WHERE version = v`) catch a read-modify-write gone stale; the
  lock is what stops two inserts that each read the same total (two refunds).
  The tests race both on PostgreSQL: without the lock the version check still
  turns a stale cancellation into a retry that keeps both changes, and two
  refunds both pass. Since v79 that holds for a new row's id too: it comes
  from the tenant's sequence, taken without a lock, where the counter row it
  replaced was one lock all of a tenant's creating writes shared (Measured;
  `test_ids` holds one customer's write open and times another's).
- **Reads take no lock** and see one snapshot (REPEATABLE READ).
- **Backpressure, not queues.** Per process: at most `PASAR_MAX_CONCURRENT_TURNS`
  turns (else 503 at once, Retry-After). Per provider, at most
  `PASAR_PROVIDER_CONCURRENCY` calls in flight — the deployment's, each process
  taking its share (v82) — and a turn waits up to 30 s for a slot, then ends
  with a fallback reply, `provider_unavailable`. Per tenant, a bucket every
  process shares (429). A provider that fails five times in a row is not
  called for 30 s, from any process (circuit breaker); transient errors are
  retried with full jitter, three tries, and no call is sent while its
  provider's breaker is open.
- **One turn at a time per conversation**, across processes: a lease row with
  an expiry (60 s), taken with the row it protects (`UPDATE ... RETURNING`),
  renewed by a heartbeat every 20 s while the turn runs and checked by every
  write the turn makes; a dead process frees its conversation within a minute.
- **Fewer round trips, and less CPU in each.** The catalog (products, sellers,
  livestream claims) is read-only to the tools and cached per process; a
  customer's slice is kept by the write it was read at, so a tool call whose
  customer's world has not changed since reads one row instead of twelve
  tables; the step's counters and the lease renewal are one statement; every
  statement is built once; a pooled connection is checked by its socket rather
  than a round trip. A turn of this workload is 42.8 statements in 8.3
  transactions (68.8 before v80, a ping before each transaction included).

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
  committed step as it happens, each with its event id. The turn is a task of
  its own, started before the response is, so a client that drops the stream —
  or is gone before it starts — loses nothing: the turn runs to its end (a
  server stopping waits for it), and `GET .../events?after=<id>` returns every
  event since.
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
  `/v1/admin/profile` samples where the process's CPU goes, every thread, and
  `/v1/admin/nodes` lists the processes that share the deployment's limits
  (Operating it).
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
| concurrency | eight refunds raced; the lock removed and the double refund shown; a stale cancellation retried by the version check; one customer's write held open while another's takes a new id without waiting (10 s with v78's counter row), and sixteen at once each get their own, none lost or repeated (PostgreSQL) |
| isolation | another customer's order invisible, a credit naming it refused, two tenants' rows apart |
| protocol | idempotency (replay, mismatch, in progress), the lease (409), admission (503), rate limit (429), SSE order and recovery, problem details, auth and revocation, pagination |
| failure | a provider down: retries, fallback reply, breaker open, nothing charged; a crash mid-turn: the lease holds, then the unanswered call is answered; a lease lost mid-turn: nothing more written, the answer kept under the key |
| a model's arguments | 1e400, a voucher of 0, a credit of 10**19, a 300-character tool name, credit not in the account's currency, credit against an order in another currency counted in SGD, an empty order id, NUL characters from the customer and the model — each answered, the turn replying, on both databases |
| long turns | a model call twice the lease: the heartbeat keeps the lease and the key, a retry and another message are both turned away, and the retry then gets the answer (the test fails with the heartbeat off); a stream on a server that ran no startup |
| order and logs | a customer's rows in the benchmark's order after updates, on both databases; an exception logged without its message |
| money | prices exact to the microdollar, costs additive, effective dates, budgets for a conversation and a tenant's day |
| a customer's slice, kept | a second read with her write_seq unchanged uses the first's slice (one read of her tables, the same answer); a write makes the next read read again, and it sees the write; rows keyed by plain str; `copy_world` equal to `deepcopy`, a nested value included, and nothing the copy changes reaching the original; off at 0; on both databases |
| the pool | a connection back within seconds is lent with no round trip, one idle longer is pinged, and one whose session the server ended (`pg_terminate_backend`) is replaced either way before it is lent: the next transaction runs, on a new session (PostgreSQL) |
| turns that wait | 24 turns at once on 4 worker threads, three model calls of 0.5 s each, done together in 1.7 s, not the 9 they would take a thread each; a running turn finishes in 0.6 s while every other handler's thread is held for three; a turn cancelled while it waits for a thread gives its slot back; the gate's slots shared by threads and coroutines, its retries the same from either |
| limits a deployment shares | 16 calls at once shared out 6, 5 and 5, and given back by a process that leaves at once and by one that dies within three heartbeats; a limit resized under waiting threads and coroutines, and shrunk below the slots in use; a tenant's bucket for two processes refusing the 4th request where each process's own admitted 6, remembering a refusal until the time it gave, draining a bucket spent under a slower rate at the new one, two processes with different rates on one bucket admitting no more than the looser allows, and on a database not migrated counting alone, as v81 did; a stop held on a heartbeat that never ends going within its timeout, and no row written after it; a breaker opening in one process held open in the other and adopted once; a call not sent, nor sent again, once its provider's breaker is open, and not counted the provider's failure; two servers on one database refusing a tenant's 3rd request whichever it reaches; `auto` sharing on PostgreSQL, and on SQLite for `serve --workers 2`; a malformed `PASAR_WORKERS` stopping `serve` alone; on PostgreSQL, with heartbeats 30 s apart, a join, a leave and an opening heard at once, a listener whose session is ended listening again with nothing logged outside the service's lines, and the tables UNLOGGED — on both databases |
| a model call's transport | the same request through urllib and httpx side by side — the body byte for byte, the key, the content type, the user agent — the same answer, three calls over one connection kept alive; the same failures (a 503 and a 400 word for word, retried and not; a timeout, a refused connection); a redirect followed nowhere, the key never reaching the host it names |
| the profiler | against threads whose CPU is known: a long call charged 0.392 s of its 0.400, short bursts between waits 0.165 of 0.166, a thread waiting throughout nothing, and the stacks' sum the process's CPU; the endpoint is the operator's, runs one at a time (409), ends on DELETE; the CLI's summary and folded stacks |
| schema | migration to head equals `schema.py`, 0003's shared tables included; down and up again; 0002 on a database that has tenants: each gets its sequence after its counter, and down again the counter takes the last value back |
| the API's shape | the route list pinned; every `/v1` operation documents its problem responses |
| the image | on every push, run as a user runs it (`scripts/smoke_image.sh`): its CLI migrates and makes a tenant; its HEALTHCHECK says healthy; it runs as a user other than root and holds no `.env`; over real HTTP a turn, the turn retried under its key (the same bytes, no second turn), a stream, the audit log (`scripts/smoke_api.py`); every call replayed; stopped by SIGTERM. Published only after it passes |

`scripts/load_test.py` measures a real server (uvicorn in its own process,
real HTTP) — its throughput, its latencies and the CPU a turn costs it — and
replays what it made; `scripts/limits_check.py` measures two of them on one
database, against a stand-in provider that counts what it is sent (v82). `scripts/smoke_api.py` checks any running
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

Measured through the service since (RUNBOOK 1z, [below](#with-a-real-model)),
on 1s's 117 parts with the cache's own share: $0.0015 a conversation for
deepseek-flash and $0.0052 for deepseek-v4-pro.

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
| API | FastAPI 0.162, Pydantic 2, uvicorn | typed bodies → validation and the OpenAPI document from one source; a turn as a coroutine — a model call awaited, its transactions in a bounded thread pool — so the benchmark's synchronous tools and store run unchanged and a waiting turn holds no thread; every other handler synchronous, in the same pool |
| model calls | httpx (async) | the benchmark's backends build each request and read each answer; the service sends it from a coroutine over connections kept alive, as urllib would have sent it — the same body, headers and errors |
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
   over 16 tenants: write p99 575 → 305 ms, +6% turns) — v79 removes it
   (below).
5. **With a model's latency, the provider gate is the limit, as configured.**
   16 calls in flight at 400 ms is 40 calls a second, ~13 turns: measured 12.2
   at 32 clients and 12.1 at 64, the extra clients waiting at the gate (p50
   2.4 s → 4.8 s) rather than overrunning the provider. Widened to 64 the
   process is CPU-bound again (17.0), and the one lock every write of a tenant
   shares shows: a write holds the tenant's id counter from its first new id to
   its commit, and with 64 threads on two cores that hold stretches — writes
   p50 1,207 ms, reads (no lock) 145 ms, 64 clients on 64 different customers.
   v79 removes that lock (below): writes 346 ms at the median.
6. **Under all of it, every tool call replayed identically.**

### Ids without the tenant's lock (v79)

The table above was measured with a counter row per tenant: a write that made
a row took the tenant's next id with an `UPDATE` in its own transaction, and
held that row's lock to its commit — so every write of a tenant that created
a row, any customer's, waited for every other. Since v79 PostgreSQL gives each
tenant a sequence, which hands out its next value at once and holds nothing
(`store.id_sequence`; migration 0002 makes one for each tenant already there).
Measured on the same two vCPUs, 2026-10-31: the old code and the new
alternating, three runs of each (median, and the range):

| PostgreSQL, one process | ids from | turns/s | write tool call p50 / p99 (ms) | read p50 (ms) | turn p95 / p99 (ms) |
|---|---|---|---|---|---|
| 1 tenant, 16 clients | the counter row | 19.3 (17.7–19.6) | 217 / 507 | 112 | 1,128 / 1,434 |
| | a sequence | 18.7 (17.9–19.1) | 170 / 329 | 120 | 1,119 / 1,271 |
| 16 tenants, 16 clients (one run each) | the counter row | 18.6 | 173 / 350 | 119 | 1,168 / 1,380 |
| | a sequence | 19.7 | 158 / 352 | 116 | 1,112 / 1,251 |
| 1 tenant, 64 clients, 400 ms model, gate 64 | the counter row | 15.0 (14.9–15.5) | 1,446 / 5,765 | 153 | 6,513 / 8,120 |
| | a sequence | 16.4 (16.3–17.6) | 346 / 672 | 247 | 4,352 / 4,722 |

What it says:

1. **The counter was the cost.** One tenant now writes as sixteen did (170 /
   329 ms, against sixteen tenants' 158 / 352) where it paid 217 / 507; sixteen
   tenants, each of which had a counter of its own already, did not change.
2. **Where it showed most, it is gone.** At 64 clients on 64 customers a write
   waited 1,446 ms at the median and now takes 346; its p99 went from 5.8 s to
   0.7 s; a turn's p99 from 8.1 s to 4.7 s; and the process does 9% more turns,
   the two ranges apart.
3. **The process is CPU-bound now, everywhere.** Reads, which never waited for
   the counter, got slower at 64 clients (153 → 247 ms): the writes no longer
   sit idle on a lock, so more of them share the two cores at once; at 16
   clients the throughput is within noise. The next limit is the interpreter
   and the cores: a second process (+41%, above) — and, this said, the model
   calls off the worker threads, which v80 measured to be the wrong next step:
   the CPU a turn spends was the limit, and most of it was avoidable (below).
4. **Nothing else moved.** Every run replayed identically (3,791 to 3,808 calls
   a configuration), no conflict was retried, every request was answered 200,
   and the parity test's 313 tasks on PostgreSQL still name every row as the
   benchmark does.

### Where a turn's CPU went (v80)

v79 left one process CPU-bound everywhere, and its list put the model calls
off the worker threads next. That would not have moved this limit: an async
model call lets a process hold more turns that are waiting, and spends as much
CPU on each. So v80 measured where a turn's CPU goes, and took most of it out.
The load test now reads the CPU the server used, its own and the whole
machine's (Linux, `/proc`), and divides by the turns served.

**Where it went.** At 16 clients a turn cost the server 49.5 ms of CPU and the
machine 76: PostgreSQL and the system ~21, the load generator ~6. The same
turns run one at a time in the server's own process, without HTTP or
contention, cost 34 ms. cProfile on those, with a CPU clock (its call counts
exact, its times inflated by its own cost), and the service's sampler on the
server under load (`GET /v1/admin/profile`, Operating it) found:

- **SQLAlchemy built every statement anew**, 60.4 a turn. Building one — each
  value coerced into a parameter, the statement's cache key computed again —
  cost more CPU than its round trip: 290–380 µs a statement, against 160 µs
  for the same statement built once with named parameters and 80 µs through
  psycopg alone (this machine, one statement at a time).
- **Each tool call read its customer's slice again**: eleven to twelve of those
  statements, for each of 2.1 calls a turn, whether or not her world had
  changed since the call before.
- **`copy.deepcopy` of a slice** took 360 µs for 16 rows, most of it rebuilding
  SQLAlchemy's column names — a subclass of `str` — one key at a time.
- **A ping before every transaction** (`pool_pre_ping`), 8.4 a turn: ~80 µs of
  CPU and ~120 µs of wall time each.

**What changed** — nothing a turn does:

1. Statements are built once, with named parameters, and rows are read by
   position into plain `str` keys (`store.py`, `runtime.py`).
2. A customer's slice is kept by the write it was read at (`store.Slices`).
   Every change to her world advances her `write_seq` in the transaction that
   makes it, so a call that finds it where a slice was read uses that slice —
   the same rows and versions, by construction — and runs its tool on a copy.
   It relies on nothing else writing the world tables: nothing in the service
   does, replay reports a row changed behind its back, and
   `PASAR_SLICE_CACHE=0` turns it off where another system does.
3. A slice is copied as what it is, rows of plain values, not by `deepcopy`
   (`copy_world`).
4. A pooled connection is checked by its socket — a session the server ended
   leaves it readable — and pinged only after ten seconds idle.
5. The mock agent counts its tool schemas' tokens once a list, not once a call:
   a cost of the load test, not of the service.

A turn is 42.8 statements now, in the same 8.3 transactions, with no ping on a
connection in use; one at a time in-process it costs 12.4 ms of CPU (16.8
without the kept slice). Measured on the same two vCPUs, 2026-10-31: v79's
code and v80's alternating, three runs of each (median, and the range); "no
kept slice" is v80 with `PASAR_SLICE_CACHE=0`.

| one process | code | turns/s | the server's CPU a turn (ms) | the machine's (ms) | turn p50 / p95 / p99 (ms) | write tool call p50 / p99 (ms) | read p50 (ms) |
|---|---|---|---|---|---|---|---|
| PostgreSQL, 1 client | v79 | 14.7 (14.1–15.0) | 44.6 (44.0–46.1) | 63.7 | 63 / 97 / 118 | 13 / 22 | 10 |
| | v80 | 27.1 (25.8–29.5) | 19.3 (18.0–19.8) | 33.1 | 34 / 46 / 52 | 3 / 6 | 1 |
| SQLite, 1 client | v79 | 28.9 (28.7–31.4) | 31.8 (29.4–32.3) | 34.4 | 31 / 48 / 59 | 6 / 12 | 5 |
| | v80 | 73.5 (68.8–80.9) | 11.4 (10.3–12.2) | 13.3 | 11 / 21 / 27 | 1 / 2 | <1 |
| PostgreSQL, 16 clients | v79 | 18.7 (17.8–18.7) | 49.5 (49.4–51.7) | 76.3 | 778 / 1,130 / 1,336 | 176 / 335 | 122 |
| | v80 | 42.1 (38.4–43.0) | 19.4 (19.3–21.4) | 36.6 | 330 / 505 / 618 | 34 / 104 | 11 |
| | v80, no kept slice | 33.7 (33.1–34.6) | 24.7 (24.3–25.3) | 46.0 | 426 / 619 / 740 | 90 / 187 | 61 |
| PostgreSQL, 64 clients, 400 ms model, gate 64 | v79 | 16.5 (16.0–16.6) | 52.2 (51.5–54.9) | 88.0 | 3,266 / 4,472 / 4,871 | 342 / 676 | 254 |
| | v80 | 28.7 (28.3–30.3) | 21.1 (20.4–22.4) | 56.2 | 1,686 / 2,154 / 2,356 | 22 / 166 | 12 |
| | v80, no kept slice | 25.6 (24.6–27.2) | 27.4 (25.2–28.0) | 65.3 | 1,940 / 2,644 / 2,926 | 113 / 395 | 80 |

What it says:

1. **A turn costs the server three fifths less CPU, and one process does about
   twice the turns.** At 16 clients, 19.4 ms of CPU a turn where it was 49.5,
   and 42.1 turns a second where it was 18.7; at 64 clients with a model's
   latency, 28.7 where it was 16.5 — no range overlapping its counterpart's.
   Alone, a turn takes 34 ms on PostgreSQL where it took 63, and 11 on SQLite
   where it took 31. The machine's CPU a turn halved too: PostgreSQL runs 30%
   fewer statements and no pings.
2. **The latencies followed.** A read tool call takes no lock; at 16 clients it
   waited 122 ms at the median — for the interpreter, not the database — and
   now takes 11. A write took 176 ms and takes 34; a turn 778 and 330 (p99 1,336
   and 618). At 64 clients a turn's median halved, 3.3 s to 1.7, of which the
   model's three or four calls are 1.2–1.6 s.
3. **Keeping the slice is a sixth of the CPU saved, and most of a write's time
   under its lock.** Without it a turn costs 24.7 ms and the process does 33.7
   turns a second. A write reads its customer's slice after it takes her lock;
   kept, it reads one row there instead of twelve tables, and holds the lock
   34 ms at the median instead of 90 — the time her other conversations' writes
   wait.
4. **What is left is mostly the statements.** In two profiles of v80 at 16
   clients (1,200 turns, 10 ms looks), a turn's tool calls are 29% of its CPU —
   reading a slice, on the quarter of the calls that do, 7%; recording the
   call, 6% — and its other steps (the model calls with their ledger rows, its
   first and last transactions) 47%; the event loop and HTTP 7.5%. The next
   cuts are fewer statements a step, and a second process (+41%, above); with a
   real model, the cap on turns at once (v81, below).
5. **Nothing else moved.** Every one of the 30 runs replayed identically (840 to
   1,264 calls each), no conflict was retried, and every request was answered
   200; the 313 reference solutions leave the world and the action log as the
   benchmark's episodes do, on both databases, with the slice kept.

The profiler that found it is the operator's too (Operating it). Its first
version put 93% of the CPU on the database driver's wait, where a thread waits
and spends none ([WHAT_FAILED #68](WHAT_FAILED.md)); what it charges now is
checked against threads whose CPU is known.

### Turns that wait without a thread (v81)

Through v80 a turn ran on a worker thread from its first transaction to its
last and held it through every model call, so a process admitted no more turns
than it had threads. With a real model — two or three seconds a turn — that
cap, not the CPU, came first (v80, above). v81 makes a turn a coroutine.

- **A turn** (`Turn.arun`) runs each of its transactions — the customer's
  message, a model call's record, a tool call, the last — in a worker thread
  for the milliseconds it takes, the same code as before, and awaits the
  seconds between them; its lease's heartbeat is a task. A process holds as
  many waiting turns as it admits — `PASAR_MAX_CONCURRENT_TURNS`, 256 by
  default where it was 48 — on `PASAR_THREADS` threads.
- **A model call is awaited.** The benchmark's backends build a request
  (`request()`) and read an answer (`parse()`) apart from sending it; `chat()`,
  what the benchmark's runs call, is `parse(_post(*request()))` as before. The
  service sends the same request from a coroutine, with httpx over
  connections kept alive (`models.Transport`): the body byte for byte, the
  headers, the errors and which of them are retried, no redirect followed —
  `test_transport` sends one through urllib and httpx side by side.
- **One limit for threads and coroutines** (`resilience.Slots`): the provider
  gate and the admission count both against the same slots, and a release
  wakes whichever waits.
- **Running turns have threads of their own.** An admitted turn's
  transactions take turns on `PASAR_THREADS` threads apart from every other
  handler's, so a burst of new requests never queues in front of the turns
  that hold leases and slots (a test takes every other thread for three
  seconds; a running turn still finishes in 0.6).
- **The event loop's lag is a metric** (`pasar_event_loop_lag_seconds`): every
  waiting turn's next step is a callback on the one loop. The load test
  reports it, and the server's own time of each turn beside the client's.

Measured on the same two vCPUs, 2026-11-01, PostgreSQL, one process: v80's
code and v81's alternating, one to three runs of each (median, and the range);
"a mock call ±30%" is the mock's latency strayed as a provider's does
(`--mock-jitter 0.3`), and "arriving over 20 s" each client's first turn sent
at a random moment of the first twenty seconds (`--ramp-s 20`).

| PostgreSQL, one process | code | turns/s | the server's CPU a turn (ms) | turn p50 / p95 / p99 (ms) | write tool call p50 / p99 (ms) | threads, peak | memory, peak (MB) |
|---|---|---|---|---|---|---|---|
| 1,000 clients arriving over 20 s, a 10 s mock call ±30% | v80, 1,000 threads | 15.4 (15.2–15.7) | 28.1 (25.0–31.1) | 33,452 / 49,337 / 57,457 | 12 / 379 | 1,984 | 261 |
|  | v81 | 16.4 (16.4–16.5) | 24.1 (23.4–24.8) | 32,341 / 46,212 / 52,195 | 6 / 248 | 94 | 179 |
| 1,000 clients at once, every mock call 10 s | v80, 1,000 threads | 17.9 (17.7–18.0) | 24.9 (24.6–25.1) | 31,065 / 48,117 / 52,195 | 27 / 576 | 2,005 | 262 |
|  | v81, one pool for all | 17.6 | 24.3 | 31,170 / 58,652 / 60,964 | 16 / 312 | 69 | 179 |
|  | v81 | 17.5 (17.3–17.7) | 25.1 (23.7–26.5) | 31,398 / 57,765 / 60,298 | 22 / 529 | 110 | 179 |
| 200 clients, 600 ms a mock call | v80, 64 threads | 26.8 (26.3–27.3) | 19.7 (19.6–22.0) | 6,127 / 6,767 / 7,143 | 11 / 349 | 133 | 112 |
|  | v80, 256 threads | 34.6 (32.0–34.9) | 22.3 (22.0–24.2) | 4,351 / 6,526 / 7,542 | 83 / 296 | 393 | 137 |
|  | v81 | 35.0 (31.1–35.4) | 21.8 (21.1–23.9) | 4,510 / 6,639 / 8,180 | 124 / 428 | 98 | 122 |
| 16 clients, no model latency | v80 | 46.4 (40.0–46.9) | 18.0 (17.6–20.4) | 303 / 449 / 540 | 32 / 89 | 37 | 92 |
|  | v81 | 44.5 (43.4–48.1) | 18.3 (17.0–18.9) | 297 / 490 / 603 | 24 / 94 | 21 | 92 |
| 64 clients, 400 ms a mock call, gate 64 | v80 | 31.0 (28.5–31.7) | 20.1 (19.4–22.3) | 1,553 / 2,116 / 2,394 | 22 / 111 | 133 | 104 |
|  | v81 | 31.0 (27.6–32.7) | 19.9 (19.1–24.3) | 1,541 / 2,113 / 2,303 | 19 / 140 | 49 | 100 |


What it says:

1. **What v81 removes is threads, and the cap they set.** At 200 clients and
   600 ms a model call, v80 on 64 threads ran 64 turns and queued the rest — a
   turn took 1.9 s on the server and 6.1 s for its client — and did 26.8 turns
   a second; given 256 threads it did 34.6, on 393 of them. v81 does 35.0 on its
   64, with 98 threads in all. At 1,000 customers waiting on a ten-second model,
   arriving over twenty seconds with calls of 7 to 13, v81 serves 16.4 turns a
   second on 94 threads and 179 MB where v80 served 15.4 on 1,984 threads and
   261 MB — at less CPU a turn (24 ms against 28: two thousand threads are not
   free) and with shorter turns (p95 46 s against 49, p99 52 against 57).
2. **Nothing lost where the CPU is the limit.** At 16 clients with no model
   latency, and at 64 with 400 ms, v81 is within its runs' noise of v80 (44.5
   turns a second and 46.4; 31.0 and 31.0).
3. **All at once is a first-come queue's worst case.** With 1,000 clients
   started together and every call exactly ten seconds, the throughput is the
   same (17.5, 17.9), but v81's p95 is 58 s against v80's 48, and the server's
   own time of a turn says the same (55 s and 46): the turns come back in waves,
   each wave queues for the threads in the order it arrived, and a latency
   that never varies hands the next wave back in that order — the same turns
   last every time — where v80's thousand threads, scheduled by the operating
   system, shuffle it. A pool of its own for running turns does not change it
   (58.7 s with one pool for all), the loop's lag does not add up to it (p99
   0.5 s), and 16 threads made it longer (70 s). Arrivals and latencies that
   vary, as a real model's do, undo it: row one ([WHAT_FAILED #69](WHAT_FAILED.md)).
   The same herd at 200 clients leaves v81's p99 a little long (8.2 s against
   7.5).
4. **The loop keeps up.** Its lag is 1–10 ms at the median everywhere; its p99
   is 25–250 ms at 16 and 64 clients and 500 ms at 200 and 1,000, where 4–10%
   of its timers ran more than 100 ms late.
5. **Nothing else moved.** Every run replayed identically, every turn the
   server answered was a 200, and the parity test's 313 tasks still match on
   both databases. At the staggered thousand the load generator lost 5–11 of
   3,000 requests, on both versions — "server disconnected", uvicorn closing a
   connection idle 5 s as a client starved of its own CPU sent on it — and
   counted none; the server now keeps an idle connection 75 s
   (`PASAR_KEEP_ALIVE_SECONDS`), and the load test counts every request it
   loses — run again, v81 lost none of 3,000 where v80, still at 5 s, lost 16
   (#69).

### One limit for the deployment (v82)

A process is limited by its CPU alone since v81, so more turns come from more
processes — `serve --workers N`, or replicas behind a load balancer — and
through v81 every limit was a process's own: N processes allowed a tenant N
times its rate, a provider N times the calls it takes at once, and had N
breakers, each of which had to see a provider fail before it stopped calling
it. v82 shares them through the database the processes already share
(`limits.py`). `PASAR_LIMITS=auto`, the default, shares them on PostgreSQL and
keeps them each process's own on SQLite, one machine's file (below), unless
`serve --workers N` runs several processes on it, which share them;
`deployment` and `process` say which outright, `process` being v81's.

- **Who is serving.** Each process writes its row of `service_nodes` as it
  starts and every `PASAR_HEARTBEAT_SECONDS` (5), saying until when to count
  it: three of its own heartbeats on. One that shuts down deletes its row,
  after any heartbeat in flight and waiting at most 5 s for the database; one
  that dies stops being counted 15 s on.
- **A provider's calls, shared out.** Each process takes its share of
  `PASAR_PROVIDER_CONCURRENCY` by its rank among the rows — 16 over three
  processes is 6, 5 and 5 — with no statement a call. A share that shrinks
  takes no slot from a call that holds one (`Slots.resize`).
- **A tenant's bucket, one for all.** One row a tenant (`rate_limits`) and one
  statement a request, on the database's clock: the bucket holds requests,
  drains at the rate a second, and admits while, drained to now, it holds at
  most burst − 1 (GCRA, counted in requests). Sent through psycopg on a pooled
  connection in autocommit, a request's check costs about 100 µs of CPU
  (99–105), where through SQLAlchemy a statement of its shape cost 208. In
  requests and not seconds, so that processes with other settings agree on
  what a bucket holds: during a rolling change of `PASAR_RATE` each admission
  adds one whoever makes it, and the deployment admits no more than the
  looser setting allows; a bucket filled under a slower rate drains at the new
  one at once. A refusal is remembered until the time it gives — the bucket
  only drains with time — so a client sending past its limit as fast as it
  can costs the database a statement for each request that arrives as a
  refusal runs out, not one a request.
- **A breaker's opening, for all.** A process whose breaker opens writes until
  when (`provider_breakers`), and the others hold theirs open as long — each
  opening once, so that a call there that succeeds after still closes it. And
  a call is not sent, nor sent again, while its provider's breaker is open —
  opened while it waited for its slot or its backoff, here or elsewhere — and a
  call the breaker stops is not counted the provider's failure.
- **Heard at once on PostgreSQL.** A join, a leave or an opening is a NOTIFY;
  each process LISTENs on a connection of its own, out of the pool, and counts
  again when it hears. On SQLite, or behind a pooler in transaction mode that
  drops notifications, the next heartbeat carries it.
- **Nothing worth a disk write.** The three tables are UNLOGGED on PostgreSQL:
  a commit that wrote only to them does not wait for the log to reach the
  disk, and after a crash they start again empty — every bucket full, every
  process back at its next heartbeat, every breaker closed.
- **A database not yet migrated** to 0003, or a statement that fails: a v82
  process keeps each limit its own, as v81 did — a tenant's bucket for 5 s at
  a time, then the database asked again — and says so, in its log, in
  `pasar_limits_errors_total` and in `GET /v1/admin/nodes`. A database that
  does not answer at all holds a request's check as it holds the request's
  every other statement: none has a client-side bound yet (roadmap).
- **What stays a process's own** is what one process holds: its turns at once
  (`PASAR_MAX_CONCURRENT_TURNS`) and its threads.

`scripts/limits_check.py` measures it: two servers on one PostgreSQL database,
each client talking to one of them, in front of a stand-in provider — an
OpenAI-compatible server on the same machine that answers as the mock agent
does, after 400 ms ±30%, and counts what it is sent. Each part ran with every
limit a process's own and with them shared, on the same two vCPUs,
2026-11-01; the ranges are the runs'.

| two servers, one database | each limit the process's own (`PASAR_LIMITS=process`, as v81) | shared (`deployment`, v82) |
|---|---|---|
| ***A provider that takes 8 calls at once***, and refuses past it (429), each server told 8; 32 clients for 30 s; five runs | | |
| calls in flight, the two servers together, at most | 16 | 8 |
| calls the provider refused | 862–891 of 1,450–1,492: 59–61% | none, of 605–614 |
| turns answered | 116–127: 3.9–4.2 a second | 191–194, every one: 6.4–6.5 a second |
| turns that ended unanswered | 250–291 | none |
| an answered turn, p50 / p95 (s) | 2.9–3.1 / 4.0–4.1 | 4.9 / 6.5–6.7 |
| ***A tenant allowed 5 requests a second***, bursts of 5; 8 clients as fast as they can for 10 s; three runs | | |
| requests admitted | 104–106: 10.4–10.6 a second | 52–53: 5.2–5.3 a second, of the 55 allowed |
| requests answered, 200 or 429 | 7,063–8,736 | 7,700–8,784 |
| ***A provider that fails every call*** (503), 45 s; 14 clients on one server, 1 on the other, 16 calls at once allowed; three runs | | |
| calls sent to it | 59–62 | 33–38 |
| … by the light server | 18 | 3–8 |
| the light server's breaker | opened by its own failures, 2.3–4.1 s after the other's | held open for the other's, 11–16 ms after it opened |
| the light server's turns fail fast from | 4.0–6.1 s | 2.4–2.9 s |
| ***Processes coming and going***, heartbeats 5 s apart; three runs | | |
| one killed (SIGKILL): the other takes the whole limit | — | 14.9–15.1 s on |
| one started: counted by the other | — | 0.05 s after it is ready |
| one stopped (SIGTERM): counted out | — | 0.05–0.09 s on |

What it costs one process — the load test, alternating, three runs on
PostgreSQL and two on SQLite (median, and the range):

| one process | each limit its own | shared |
|---|---|---|
| PostgreSQL, 16 clients, no model latency: turns a second | 46.3 (40.6–47.2) | 46.0 (40.8–48.6) |
| … the server's CPU a turn (ms) | 17.8 (17.4–20.4) | 17.6 (16.7–20.1) |
| SQLite, 1 client: turns a second | 72.1 (71.8–72.4) | 68.3 (66.3–70.3) |
| … the server's CPU a turn (ms) | 11.4 (11.4–11.4) | 12.0 (11.8–12.1) |

What it says:

1. **Limits that are each process's own are no limit for the deployment.**
   Each server kept to its 8, and together they had 16 calls in flight at a
   provider that takes 8. The provider refused 59–61% of their calls, and the
   turns those calls belonged to — tried three times, refused each time —
   ended unanswered: 250–291 of them, against 116–127 answered. With the limit
   shared, the two never had more than 8 in flight, the provider refused
   nothing, and every turn was answered, at the rate the provider allows:
   6.4–6.5 a second against 3.9–4.2. An answered turn took longer, 4.9 s at the
   median against 2.9–3.1, because it waited for a slot instead of being
   refused one. In one of four earlier runs — the gate then still counted a call its
   breaker stopped as the provider's failure — breakers opened on the
   refusals and stayed open: the provider was hardly called, refusing 29 of
   573, and 3,075 turns were turned away unanswered at once, each sent again
   as soon as it came back.
2. **A tenant's rate is the tenant's, whichever server it reaches.** At 5 a
   second with bursts of 5, each server admitted its own 52–53 in ten seconds —
   10.4–10.6 a second between them; sharing the bucket they admitted 52–53, of
   the 55 the rate allows. The refusals cost no statement each: about as many
   requests were answered in the ten seconds, 7,700–8,784, as with each
   server's own bucket in memory (7,063–8,736).
3. **A provider that fails for one server fails for the other.** With every
   call failing, the heavy server's breaker opened 2.0–2.4 s in, and the light
   one held its own open 11–16 ms later; its turns failed fast from 2.4–2.9 s.
   With a breaker of its own it had to see five of its own calls fail first,
   three attempts each, and opened 2.3–4.1 s after the other, its turns
   failing fast from 4.0–6.1 s. The deployment sent the dead provider 33–38
   calls where it sent 59–62; most of them were the heavy server's calls in
   flight as the outage began, which no breaker stops. When the openings
   ended, 30 s on, the first trial to fail opened the other server's breaker
   again within 10–60 ms.
4. **A process's count follows the others.** One killed with no word stopped
   being counted 14.9–15.1 s on — three heartbeats of 5 s — and the other then
   took the provider's whole 8; one started or stopped was counted at once,
   within 0.1 s (LISTEN/NOTIFY).
5. **One process pays nothing it can see on PostgreSQL**: 46.0 turns a second
   with the limits shared against 46.3, and 17.6 ms of the server's CPU a turn
   against 17.8, inside the runs' noise — a request's statement is 104 µs, 0.6%
   of a turn. On SQLite, where the statement is a write transaction of its
   own, half a millisecond of a turn's 11.4: 68.3 turns a second at one client
   against 72.1. Every run replayed identically.

What sharing does not do:

- **A share follows the count, not the load.** A process with more turns
  waiting than its share does not borrow the slots another leaves idle; with
  clients spread unevenly over the processes, the deployment calls its
  provider less than it may. (Roadmap.)
- **While the count changes**, the deployment can exceed a provider's limit
  by a process's old share, for as long as the calls begun under it take: a
  share that shrinks waits for its calls to finish, and the others learn of a
  new process when it starts. Past `PASAR_PROVIDER_CONCURRENCY` processes each
  still takes one slot, more than the limit.
- **Each process makes its own trial call** when an opening ends — two
  servers, two trials — and the first to fail opens it again for the others.
- **A rolling change of the settings admits at the looser.** Two processes on
  one bucket, one at 10 requests a second and one at 1, admitted 139 in ten
  seconds where the faster allows 140 and the slower 50: the bucket drains at
  the rate of whichever process writes it, and the one with the faster rate
  writes it most.
- **A request's check is a statement.** On PostgreSQL, 104 µs of CPU and a
  round trip: no difference outside the load test's noise. On SQLite it is a
  write transaction of its own, and SQLite has one writer: half a millisecond
  of a turn's 11.4 at one client, and a wait behind every long write — while
  a tenant of 4,000 customers was made, in one 5–7 s transaction, a client
  got 91 answers in place of 562 (#71). So on SQLite each limit is a
  process's own unless `PASAR_LIMITS=deployment` says otherwise, for several
  processes on one machine's file.

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
python scripts/service_sweep.py --self-test
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
no other. `--report` writes a finished run's report again from its traces —
nothing run, no key read.

**Measured on 2026-10-31 (RUNBOOK 1z).** 1s's 39 parts, k=3, against customer
3 on qwen3.8-flash, each agent paired by task with its own recording of
2026-10-12; the service on SQLite on a MacBook Air, eight conversations at a
time, DeepSeek over the internet from Singapore.

| agent, thinking off | passed, through the service | its recording | paired by task | the agent's $ a conversation (p95) | prompt tokens a conversation, from the cache | a turn, p50 / p95 | the service's ms of a turn, p50 / p95 |
|---|---|---|---|---|---|---|---|
| deepseek-flash | 116 of 117 (pass^3 38 of 39) | 117 of 117 | 0 tasks better, 1 worse, p=1.000 | $0.0015 ($0.0021) | 33,786, 93.6% | 2,251 / 5,507 ms | 18 / 36 |
| deepseek-v4-pro | 100 of 117 (28 of 39) | 93 of 117 | 5 better, 1 worse, p=0.219 | $0.0052 ($0.0087) | 32,450, 95.5% | 2,942 / 7,459 ms | 19 / 35 |

What it says:

1. **The service does not change what the agent does — as far as 117
   conversations each can tell.** Neither difference resolves (the sign test,
   `compare_cells.py`), and every check held, on all 234: the benchmark's own
   prompt and tool schemas; every trace re-scored by the benchmark's tools to
   the service's verdict; every tool call the two models made — 565 and 461 —
   replayed identically. Every failure is the one 1s found: the duplicate
   refund (P10), where the agent reads that the refund was paid and closes the
   case instead of passing it to a human.
2. **The model is the turn.** 99.2% and 99.3% of a turn's time was the
   model's; the tools took 0.1%; the service — HTTP, admission, the lease and
   its transactions — 18 and 19 ms at the median, under 1%: the same order as
   the 26 ms the load test measured unloaded (above, on two shared vCPUs).
3. **A conversation costs $0.0015 with deepseek-flash and $0.0052 with
   deepseek-v4-pro**, from the ledger at the price book's (peak) rates: $1.53
   and $5.23 a thousand, the simulated customer not counted — in production
   she is a person. The cache is most of the economics: v4-pro's conversations
   cost 23% more than its recording's at the same rates, nearly all of it
   because 95.5% of the prompt came from DeepSeek's cache against 97.7% then,
   with the prompt byte for byte the same; flash's share matched its
   recording's (93.6% and 93.3%), and so did its cost ($0.0015 and $0.0016).

The same kit with each task's reference solution as the agent ran all 215
tasks of the first two editions with every check holding; that run is what
found [WHAT_FAILED #67](WHAT_FAILED.md).

---

## Operating it

| variable | default | |
|---|---|---|
| `PASAR_DATABASE_URL` | `sqlite:///data/pasar.db` | `postgresql+psycopg://user:pass@host:5432/db` |
| `PASAR_ADMIN_TOKEN` | — | the operator's token; unset, `/v1/admin` answers 404 (use the CLI) |
| `PASAR_MODELS` | `mock,scripted:*` | the models a conversation may ask for |
| `PASAR_DEFAULT_MODEL` | `mock` | |
| `PASAR_MODELS_FILE` | — | more models (`deploy/models.example.json`) |
| `PASAR_MAX_CONCURRENT_TURNS` / `PASAR_THREADS` | 256 / 64 | turns at once, per process — a turn waiting on its model holds no thread (v81) — and the worker threads that run its transactions and every other handler |
| `PASAR_PROVIDER_CONCURRENCY` | 16 | model calls in flight per provider, for the deployment: each process takes its share (v82) |
| `PASAR_RATE` / `PASAR_BURST` | 10 / 40 | requests a second per tenant, and at once, for the deployment |
| `PASAR_LIMITS` | `auto` | `deployment` on PostgreSQL, and on SQLite for `serve --workers N`; else `process`: each limit a process's own, as before v82 — N processes allow N times each |
| `PASAR_HEARTBEAT_SECONDS` | 5 | how often a process says it is serving; one silent for three is not counted |
| `PASAR_MAX_STEPS_PER_TURN` / `..._TOOL_CALLS_PER_TURN` / `PASAR_MAX_TURN_SECONDS` | 16 / 24 / 120 | a turn's caps |
| `PASAR_CONVERSATION_BUDGET_USD` | 0.50 | |
| `PASAR_CONVERSATION_MAX_STEPS` | 0 | model calls a conversation may make in all; 0 is no cap (the kit sets the benchmark's 30) |
| `PASAR_LEASE_SECONDS` | 60 | renewed every third of it while a turn runs; a dead process frees its conversation within one |
| `PASAR_IDEMPOTENCY_HOURS` | 24 | |
| `PASAR_STRICT_MONEY` | 1 | the platform's money guard |
| `PASAR_KEEP_ALIVE_SECONDS` | 75 | how long `serve` keeps an idle connection: longer than a load balancer's usual 60, so the balancer closes first (v81) |
| `PASAR_SLICE_CACHE` | 2000 | customers' slices kept per process, by the write each was read at; 0 where another system writes the world tables (Measured, v80) |

Day to day: `migrate` before a new version starts (migrations are additive
first, then a release that stops using a column, then one that drops it);
`create-key` / `revoke-key` to rotate a tenant's key without downtime; `gc`
daily for expired Idempotency-Keys; `replay <tenant>` after a deploy that
touches the tools or the policy, and after anything touches the database by
hand; the outbox relay acknowledges what it delivered. Back up PostgreSQL as
usual; the world, the conversations and the ledger are all in it.

### Several processes

`serve --workers N` (or `PASAR_WORKERS`, which `deploy/compose.yaml` passes
on), or N replicas of the image on one PostgreSQL database, share every limit
but a process's own turns and threads (v82); on SQLite the workers of one
`serve` do, and separate processes on one file with `PASAR_LIMITS=deployment`. `GET /v1/admin/nodes` answers for the
process the request reached: the processes as the database has them, which
it counts, its share of each provider's calls at once, the breakers open, and
whether it hears the others at once; the metrics are
`pasar_service_nodes`, `pasar_provider_concurrency_share`,
`pasar_breaker_openings_total{by}` and `pasar_limits_errors_total{part}`, and
`python -m pasarbench.service nodes` prints the rows without the operator's
token. Migrate to 0003 before v82 starts, or as it does: until then a v82 process
keeps each limit its own and says so. Behind a pooler in transaction mode
(PgBouncer's), notifications do not reach a listener, and the processes hear
of each other at their next heartbeat. A process killed is counted 15 s more,
and its share of each provider waits that long: stop processes with SIGTERM,
as an orchestrator does, and the others take it up at once — a process
stopping waits at most 5 s for the database to hear it.

### Where its CPU goes

`GET /v1/admin/profile?seconds=30` samples the process the request reaches —
every thread, every 10 ms — and answers where its CPU went, as JSON;
`DELETE /v1/admin/profile` ends one early. `python -m pasarbench.service
profile p.json --turns N` sums it up by function and by layer, per turn, and
`--folded p.folded` writes its stacks for a flame graph (speedscope.app reads
them as they are). `scripts/load_test.py --profile p.json` profiles a load
test. It holds code locations only — never a value from a frame — and is the
operator's.

A thread is charged the CPU it used between two looks, to the calls that ran
throughout, so waiting is charged nothing and nothing is charged to a call
that did not spend it (`profiling.py` says why it cannot be the stack a thread
is found in, and [WHAT_FAILED #68](WHAT_FAILED.md) what that did). What it
costs: at the default 10 ms with 16 turns running, the sampler's own CPU was
6% of the process's, and the process did about a tenth fewer turns while it
ran; `interval_ms=50` costs less and resolves less.

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

Built since this list was first written: ids without a tenant-wide lock
(v79), a turn's CPU cut by three fifths (v80), turns that wait on their model
without a thread (v81), and one limit for the deployment (v82; all above).

1. **A database that goes silent.** No statement has a client-side bound:
   behind a partition a request waits as long as the network holds its
   connection, and so does a turn's every transaction (v82's review found it
   in the shared bucket; it was already true of everything else).
   `tcp_user_timeout` and keepalives on the pool's connections, and a
   `statement_timeout` for each kind of transaction, would turn it into an
   error the service already answers.
2. **Fewer statements a step.** A turn is 42.8 statements (v80); an event could
   be written with the row it reports, in one statement on PostgreSQL.
3. **Tokens streamed from the provider.** Events are per step today; the
   backends do not stream.
4. **The customer's own identity.** The storefront asserts which customer a
   conversation is for; the agent still verifies her digits per the policy. A
   signed customer token from the storefront would bind the two.
5. **Retention and erasure.** Conversations hold what customers said; a
   per-tenant retention period and an erase-a-customer path (the ledger and
   audit rows keep ids, not text).
6. **An outbox relay** that delivers tickets to a helpdesk with retries, rather
   than an endpoint a relay polls.
7. **Prompt caching per conversation on providers that support a key**, and the
   policy's preload/JIT choice per tenant — both measured in the benchmark,
   neither wired here.
8. **Prices by the hour.** DeepSeek's page has a peak and an off-peak rate
   (RUNBOOK 1y ran at the second); the price book holds one rate a model and a
   date, the peak, so a conversation run off-peak is charged above its bill.
   1z's figures are at the peak rate, whatever hour they ran in.
9. **A share that follows the load.** A process's share of a provider's
   limit is fixed by how many processes there are, not by their turns (v82):
   with clients spread unevenly, one process's turns wait for slots another
   leaves idle. Slots leased from the database and renewed with the heartbeat
   would let a busy process borrow them, at a statement a lease.
