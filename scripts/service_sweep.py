"""
The benchmark's sweep with the agent behind the service's HTTP API: the same
tasks, the same simulated customer and the same verdicts as pasarbench.sweep,
and -- measured by the service on the way -- what a real model costs there, and
how much of a turn's time is the service's.

    set -a; . ./.env; set +a              # the agent's key, and the customer's
    python scripts/service_sweep.py --model deepseek-flash \\
        --simulator openai --sim-model qwen3.8-flash \\
        --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \\
        --sim-extra-body '{"reasoning_effort":"low"}' --customer 3 \\
        --suite generated \\
        --traps duplicate_refund_escalate,out_of_window_offer_voucher,cancel_while_processing \\
        -k 3 --run-id S-dsk-flash-parts --workers 8 \\
        --compare traces/V-dsk-flash-parts/full

    python scripts/service_sweep.py --self-test     no key, no cost: the whole path, checked
    python scripts/service_sweep.py --run-id S-dsk-flash-parts --report \\
        --compare traces/V-dsk-flash-parts/full     its report again, from its traces

(RUNBOOK 1z.) It starts the service in a process of its own on 127.0.0.1, its
database in the run's directory (or --database-url), and gives that process one
key: the agent's, from its provider's own variable (DEEPSEEK_API_KEY for the
deepseek-* models) -- no other variable that holds a key. The customer is
called from here, with her provider's key, exactly as the sweep calls her.
Each episode gets a tenant of its own whose world is the task's; the
conversation runs turn by turn over HTTP (pasarbench/service/episodes.py)
until she ends it or the benchmark's caps are reached -- the service holds
them inside a turn, as the loop does: --max-steps model calls, --token-budget
tokens, 180 seconds a turn, and 20 customer turns. Then the task's verifier
reads the tenant's world and action log from the database.

Each episode is written as a trace in the benchmark's format, in
traces/<run-id>/full, so compare_cells.py, rescore.py and inspect_trace.py read
it like any other run. At the end, three checks: the model was shown the
benchmark's own system prompt and tool schemas; every trace re-scores -- the
benchmark's tools replaying the calls the service ran -- to the verdict the
service's database gave; and every tenant replays identically (the service's
own replay). Then the report: pass rates, the agent's cost per conversation
from the ledger, tokens and the cache's share, and each turn's time split into
the model's, the tools' and the service's -- into traces/<run-id>/report.md,
service.json and summary.json (the sweep's format).

The platform's money guard is off unless --strict-money: the benchmark has
none, and with it the two would not be running the same tools. --max-usd stops
starting episodes once the agent has spent that much (default $5; each
conversation is capped at --conversation-usd by the service itself). Ctrl-C
lets the episodes in flight finish; --resume runs the rest, and again any that
a provider's error ended.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DASHSCOPE_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# What the server process may inherit: the system's, its proxies and
# certificates. No variable that holds a key -- the agent's is added by name.
INHERIT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
           "TEMP", "TMP", "SYSTEMROOT", "COMSPEC", "PATHEXT", "WINDIR", "VIRTUAL_ENV",
           "PYTHONPATH", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
           "CURL_CA_BUNDLE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "ALL_PROXY",
           "https_proxy", "http_proxy", "no_proxy", "all_proxy")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(p / 100 * len(xs) + 0.5)) - 1))]


def parse(argv: list[str] | None) -> argparse.Namespace:
    from pasarbench.harness.types import Budget
    from pasarbench.sweep import SUITES
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-flash",
                    help="a model the service offers (built in, or in --models-file); "
                         "'scripted' runs each task's reference solution")
    ap.add_argument("--models-file", default="", help="more models, as PASAR_MODELS_FILE")
    ap.add_argument("--simulator", default="openai", choices=["openai", "anthropic", "silent"])
    ap.add_argument("--sim-model", default="qwen3.8-flash")
    ap.add_argument("--sim-url", default=DASHSCOPE_INTL)
    ap.add_argument("--sim-extra-body", default='{"reasoning_effort":"low"}')
    from pasarbench.harness.simulator import ENDING
    ap.add_argument("--customer", type=int, default=3, choices=sorted(ENDING),
                    help="the simulated customer's version (RUNBOOK 1s runs 3)")
    ap.add_argument("--gate-facts", action="store_true")
    ap.add_argument("--suite", default="core", choices=list(SUITES))
    ap.add_argument("--tasks", default="")
    ap.add_argument("--traps", default="")
    ap.add_argument("--languages", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("-k", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--token-budget", type=int, default=Budget().max_tokens)
    ap.add_argument("--policy-mode", default="preload",
                    choices=["preload", "jit", "preload-named"])
    ap.add_argument("--strict-money", action="store_true",
                    help="the platform's money guard on (the benchmark has none)")
    ap.add_argument("--workers", type=int, default=4, help="conversations at once")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--trace-root", default="traces")
    ap.add_argument("--database-url", default="",
                    help="default: SQLite in the run's directory")
    ap.add_argument("--max-usd", type=float, default=5.0,
                    help="start no episode once the agent has spent this much")
    ap.add_argument("--conversation-usd", type=float, default=1.0,
                    help="the service's cap on one conversation")
    ap.add_argument("--compare", default="",
                    help="a recorded cell to compare with (scripts/compare_cells.py)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="write the run's report again from its traces; nothing is run")
    ap.add_argument("--self-test", action="store_true",
                    help="a local stand-in provider, no key: check the whole path")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# The agent, the customer, the server
# ---------------------------------------------------------------------------

def agent_for(args):
    """(Agent, the variables the server needs for the agent's key)."""
    from pasarbench.harness.backends import _settings
    from pasarbench.service.episodes import Agent
    from pasarbench.service.models import APIS, ModelError, Registry
    from pasarbench.sweep import _local, provider_key_names
    if args.model == "scripted":
        return Agent("scripted", "scripted", None), ()
    try:
        spec = Registry((args.model,), models_file=args.models_file).resolve(args.model)
    except ModelError as e:
        raise SystemExit(str(e)) from None
    if spec.api not in APIS:                                  # the service's mock
        return Agent(spec.name, spec.api, spec.model, spec.temperature), ()
    extra, temperature = _settings(spec.extra_body, spec.temperature)
    backend = {"chat": "openai-compat", "messages": "anthropic",
               "responses": "responses"}[spec.api] + f":{spec.model}"
    agent = Agent(spec.name, backend, spec.model, temperature, extra)
    if _local(spec.base_url):
        return agent, ()
    names = provider_key_names(spec.base_url)
    if not names:
        raise SystemExit(f"{spec.base_url}: no provider known for this endpoint, so the "
                         f"service would not know which variable holds its key")
    if not any(os.environ.get(n, "").strip() for n in names):
        raise SystemExit(f"set {names[0]}, the agent's key: it is passed to the service's "
                         f"process, and to no other")
    return agent, names


def customer_for(args, agent_extra: dict | None, agent_url: str):
    from pasarbench.sweep import make_simulator
    ns = SimpleNamespace(sim_url=args.sim_url, base_url=agent_url, sim_api_key="", api_key="",
                         sim_model=args.sim_model, sim_extra_body=args.sim_extra_body,
                         extra_body=json.dumps(agent_extra) if agent_extra else "",
                         gate_facts=args.gate_facts, customer=args.customer)
    return make_simulator(args.simulator, ns)


def server_env(args, db_url: str, token: str, key_names: tuple[str, ...]) -> dict[str, str]:
    from pasarbench.harness.types import Budget
    b = Budget()
    env = {k: os.environ[k] for k in INHERIT if k in os.environ}
    for n in key_names:
        if os.environ.get(n, "").strip():
            env[n] = os.environ[n]
    threads = max(64, args.workers * 2)
    env.update(
        PASAR_DATABASE_URL=db_url, PASAR_ADMIN_TOKEN=token,
        PASAR_MODELS=("mock,scripted:*" if args.model == "scripted" else args.model),
        PASAR_DEFAULT_MODEL=("mock" if args.model == "scripted" else args.model),
        PASAR_POLICY_MODE=args.policy_mode, PASAR_STRICT_MONEY="1" if args.strict_money else "0",
        # The benchmark's caps, held inside a turn as the loop holds them.
        PASAR_MAX_STEPS_PER_TURN=str(args.max_steps),
        PASAR_CONVERSATION_MAX_STEPS=str(args.max_steps),
        PASAR_CONVERSATION_MAX_TOKENS=str(args.token_budget),
        PASAR_MAX_TOOL_CALLS_PER_TURN=str(b.max_tool_calls),
        PASAR_MAX_TURN_SECONDS=str(b.max_wall_seconds),
        PASAR_MAX_TURNS=str(b.max_user_turns + 1),
        PASAR_CONVERSATION_BUDGET_USD=str(args.conversation_usd),
        PASAR_RATE="100000", PASAR_BURST="100000", PASAR_THREADS=str(threads),
        PASAR_MAX_CONCURRENT_TURNS=str(min(threads, max(args.workers, 1) * 2)),
        PASAR_PROVIDER_CONCURRENCY=str(max(16, args.workers)),
        PASAR_LOG_LEVEL="WARNING", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    if args.models_file:
        env["PASAR_MODELS_FILE"] = str(Path(args.models_file).resolve())
    return env


class Server:
    """The service, in a process of its own, for the run."""

    def __init__(self, env: dict[str, str], log: Path):
        import httpx
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        py = sys.executable
        subprocess.run([py, "-m", "pasarbench.service", "migrate"], env=env, cwd=ROOT,
                       check=True, stdout=subprocess.DEVNULL)
        self.log = log.open("ab")
        self.proc = subprocess.Popen([py, "-m", "pasarbench.service", "serve", "--port",
                                      str(self.port)], env=env, cwd=ROOT,
                                     stdout=subprocess.DEVNULL, stderr=self.log)
        for _ in range(300):
            try:
                if httpx.get(f"{self.base}/readyz", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        self.stop()
        raise SystemExit(f"the service did not start; its log: {log}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def finished(path: Path, agent, sim_name: str, policy_mode: str) -> bool:
    """A trace that ran to the end, from this setup, and not ended by a server
    or an API: --resume keeps it."""
    from pasarbench.service.episodes import VIA
    from pasarbench.sweep import _finished
    from pasarbench.tools import CHECKS
    done = _finished(path)
    if done is None or done["footer"].get("stop_reason") == "backend_error":
        return False
    h = done["header"]
    if (h.get("requested_model") != agent.requested_model or h.get("via") != VIA
            or h.get("simulator") != sim_name or h.get("policy_mode") != policy_mode
            or h.get("tool_checks") != CHECKS):
        raise SystemExit(f"--resume: {path.parent} holds episodes of a different setup "
                         f"({h.get('requested_model')!r}, {h.get('simulator')!r}): "
                         f"use a new --run-id")
    return True


def run_all(args, tasks, agent, key_names, simulator_factory, cell: Path, db_url: str,
            log: Path) -> tuple[list, dict]:
    import httpx

    from pasarbench.harness.trace import TraceWriter
    from pasarbench.harness.types import Budget
    from pasarbench.service import episodes
    from pasarbench.service.store import make_engine

    budget = Budget(max_steps=args.max_steps, max_tokens=args.token_budget)
    sim_name = getattr(simulator_factory(tasks[0]), "name", "?")
    jobs = [(t, i) for t in tasks for i in range(args.k)]
    if args.resume:
        jobs = [(t, i) for t, i in jobs
                if not finished(cell / f"{t.task_id}__r{i}.jsonl", agent, sim_name,
                                args.policy_mode)]
    from pasarbench.sweep import PROVIDER_KEYS
    token = secrets.token_urlsafe(24)            # the run's own operator token, for its tenants
    env = server_env(args, db_url, token, key_names)
    providers = {n for names in PROVIDER_KEYS.values() for n in names}
    held = sorted(k for k in env if k.endswith("_API_KEY") or k in providers)
    server = Server(env, log)
    operator = {"Authorization": f"Bearer {token}"}
    engine = make_engine(db_url)
    spent = [0]
    lock = threading.Lock()
    halt = threading.Event()
    outcomes: list = []
    crashed: list = []
    skipped = Counter()

    def one(job):
        task, seed = job
        with lock:
            if halt.is_set() or spent[0] >= args.max_usd * 10 ** 12:
                skipped["--max-usd reached" if not halt.is_set() else "stopped"] += 1
                return None
        tenant = "e" + secrets.token_hex(10)
        writer = TraceWriter(root=args.trace_root, run_id=f"{args.run_id}/full")
        with httpx.Client(base_url=server.base, timeout=600) as client:
            try:
                out = episodes.run(task, seed, client=client, operator=operator, engine=engine,
                                   agent=agent, simulator=simulator_factory(task),
                                   budget=budget, writer=writer, tenant_id=tenant,
                                   policy_mode=args.policy_mode)
            except Exception as e:  # noqa: BLE001 -- one episode's failure is reported, not fatal
                writer.close()
                out = episodes.Outcome(task.task_id, seed, tenant,
                                       error=f"{type(e).__name__}: {e}"[:500])
                begun = cell / f"{task.task_id}__r{seed}.jsonl"
                if begun.exists():               # a trace without its footer: "unfinished"
                    out.trace = str(begun)
        with lock:
            spent[0] += out.cost_picousd
            outcomes.append(out)
            if not out.trace:                    # failed before its trace was opened
                crashed.append({"task": task.task_id, "seed": seed, "error": out.error})
            print(f"  {len(outcomes):>4}/{len(jobs)}  {task.task_id}__r{seed}  "
                  f"{'pass' if out.passed else 'FAIL'}  {out.stop:<14} "
                  f"${out.cost_picousd / 10 ** 12:.4f}"
                  + (f"  {out.error}" if out.error else ""), flush=True)
        return out

    print(f"{len(jobs)} episodes to run" + (" (--resume: the rest are done)" if args.resume
                                             else "") +
          f"; the service's process holds: {', '.join(held) or 'no key'}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(one, j) for j in jobs]
            try:
                for f in futures:
                    f.result()
            except KeyboardInterrupt:
                halt.set()
                print("\nstopping: the episodes in flight finish; --resume runs the rest",
                      flush=True)
                for f in futures:
                    try:
                        f.result()
                    except KeyboardInterrupt:
                        raise SystemExit(130) from None
        replays = checks_after(server.base, operator, cell, engine)
    finally:
        server.stop()
        engine.dispose()
    return outcomes, {"skipped": dict(skipped), "replay": replays, "server_holds": held,
                      "crashed": crashed}


def checks_after(base: str, operator: dict[str, str], cell: Path, engine) -> dict:
    """Every tenant of the cell's traces, replayed by the service."""
    import httpx
    tenants = sorted({h["service"]["tenant_id"] for h in headers(cell)})
    reports = []
    with httpx.Client(base_url=base, timeout=1800) as c:
        for t in tenants:
            r = c.get(f"/v1/admin/tenants/{t}/replay", headers=operator)
            reports.append(r.json() if r.status_code == 200 else {"ok": False, "calls": 0,
                                                                 "identical": 0, "tenant_id": t})
    return {"tenants": len(reports), "ok": sum(bool(r.get("ok")) for r in reports),
            "calls": sum(r.get("calls", 0) for r in reports),
            "identical": sum(r.get("identical", 0) for r in reports),
            "failed": [r.get("tenant_id") for r in reports if not r.get("ok")][:10]}


def headers(cell: Path) -> list[dict]:
    out = []
    for f in sorted(cell.glob("*.jsonl")):
        first = f.read_text(encoding="utf-8").split("\n", 1)[0]
        if first.strip():
            h = json.loads(first)
            if h.get("type") == "header" and "service" in h:
                out.append(h)
    return out


# ---------------------------------------------------------------------------
# The report: everything from the traces, so --resume reports the whole run
# ---------------------------------------------------------------------------

def report(cell: Path, served: list[str] | None = None, replay: dict | None = None) -> dict:
    from pasarbench.rescore import rescore_file
    from pasarbench.sweep import language_breakdown, market_breakdown, trap_breakdown
    from pasarbench.rescore import all_tasks
    from pasarbench.verifier import pass_hat_k

    tasks = all_tasks()
    eps, unfinished = [], 0
    for f in sorted(cell.glob("*.jsonl")):
        recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        head = next((r for r in recs if r.get("type") == "header"), {})
        foot = next((r for r in reversed(recs) if r.get("type") == "footer"), None)
        if foot is None:
            unfinished += 1
            continue
        ev = [r for r in recs if r.get("type") == "event"]
        total = next((e for e in reversed(ev) if e["kind"] == "service_episode"), {})
        users = [e for e in ev if e["kind"] == "user_turn"]
        rs = rescore_file(f)
        eps.append({"file": f.name, "task_id": head["task_id"], "passed": bool(foot["passed"]),
                    "stop": foot["stop_reason"], "failures": foot.get("failures") or [],
                    "turns": [e for e in ev if e["kind"] == "service_turn"],
                    "cost": int(total.get("cost_picousd", 0)),
                    "prompt": int(total.get("prompt_tokens", 0)),
                    "cached": int(total.get("cached_tokens", 0)),
                    "completion": int(total.get("completion_tokens", 0)),
                    "model_calls": int(total.get("model_calls", 0)),
                    "tool_calls": int(total.get("tool_calls", 0)),
                    "customer_turns": len(users),
                    "customer": (users[-1].get("sim_usage") if users else None) or {},
                    "prompt_ok": total.get("prompt_ok"), "tools_ok": total.get("tools_ok"),
                    "gaps": int(total.get("gaps", 0)),
                    "rescore": rs.status, "rescored": rs.current,
                    "head": head})
    n = len(eps)
    by_task: dict[str, list] = {}
    for e in eps:
        by_task.setdefault(e["task_id"], []).append(SimpleNamespace(passed=e["passed"]))
    turns = [t for e in eps for t in e["turns"]]
    http = [t["http_ms"] for t in turns]
    over = [t["http_ms"] - t["model_ms"] - t["tool_ms"] for t in turns]
    costs = [e["cost"] for e in eps]
    passed = [e for e in eps if e["passed"]]
    prompt, cached = sum(e["prompt"] for e in eps), sum(e["cached"] for e in eps)
    agree = sum(e["rescore"] == "rescored" and e["rescored"] == e["passed"] for e in eps)
    head = eps[0]["head"] if eps else {}
    sel = [tasks[t] for t in by_task if t in tasks]

    def usd(picousd: float | None) -> str | None:
        return None if picousd is None else f"${picousd / 10 ** 12:.6f}"

    def mean(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    model = (head.get("service") or {}).get("model") or "?"
    out = {
        "cell": str(cell), "episodes": n, "unfinished": unfinished,
        # Counts, so a rate is printed once from them (pass_hat_k's are rounded
        # to four places; rounded again to three, 116 of 117 read 0.992).
        "passed": len(passed), "tasks_all_passed": sum(all(v.passed for v in vs)
                                                       for vs in by_task.values()),
        "model": "scripted" if model.startswith("scripted:") else model,
        "backend": head.get("backend"),
        "simulator": head.get("simulator"), "policy_mode": head.get("policy_mode"),
        **(pass_hat_k(by_task) if by_task else {}),
        "stops": dict(Counter(e["stop"] for e in eps)),
        "per_trap": trap_breakdown(by_task, sel) if sel else {},
        "per_language": language_breakdown(by_task, sel) if sel else {},
        "per_market": market_breakdown(by_task, sel) if sel else {},
        "cost": {"total": usd(sum(costs)), "per_conversation_mean": usd(mean(costs)),
                 "per_conversation_p50": usd(pct(costs, 50)),
                 "per_conversation_p95": usd(pct(costs, 95)),
                 "per_passed_conversation": usd(sum(costs) / len(passed)) if passed else None},
        "tokens_per_conversation": {"prompt": mean([e["prompt"] for e in eps]),
                                    "cached": mean([e["cached"] for e in eps]),
                                    "completion": mean([e["completion"] for e in eps]),
                                    "cache_share": round(cached / prompt, 4) if prompt else None},
        "per_conversation": {"model_calls": mean([e["model_calls"] for e in eps]),
                             "tool_calls": mean([e["tool_calls"] for e in eps]),
                             "customer_turns": mean([e["customer_turns"] for e in eps]),
                             "turns": mean([len(e["turns"]) for e in eps])},
        "turn_ms": {"p50": pct(http, 50), "p95": pct(http, 95), "p99": pct(http, 99)},
        "time_share": ({"model": round(sum(t["model_ms"] for t in turns) / sum(http), 4),
                        "tools": round(sum(t["tool_ms"] for t in turns) / sum(http), 4),
                        "service": round(sum(over) / sum(http), 4)} if sum(http) else {}),
        "service_ms_per_turn": {"p50": pct(over, 50), "p95": pct(over, 95), "p99": pct(over, 99)},
        "customer_tokens_per_conversation": {
            "prompt": mean([e["customer"].get("prompt", 0) for e in eps]),
            "completion": mean([e["customer"].get("completion", 0) for e in eps])},
        "checks": {
            "prompt_and_tools_as_the_benchmark": sum(bool(e["prompt_ok"] and e["tools_ok"])
                                                     for e in eps),
            "rescored_to_the_same_verdict": agree,
            "not_rescorable": dict(Counter(e["rescore"] for e in eps
                                           if e["rescore"] != "rescored")),
            "rescored_differently": [e["file"] for e in eps if e["rescore"] == "rescored"
                                     and e["rescored"] != e["passed"]][:10],
            "steps_not_rebuilt": sum(e["gaps"] for e in eps),
            "replay": replay},
        "served_models": served,
        "episodes_list": [{k: e[k] for k in ("file", "task_id", "passed", "stop", "cost",
                                             "prompt", "cached", "completion", "model_calls",
                                             "tool_calls", "customer_turns", "rescore")}
                          for e in eps],
    }
    return out


def markdown(r: dict) -> str:
    c, t, s = r["cost"], r["tokens_per_conversation"], r["service_ms_per_turn"]
    share = r["time_share"] or {}
    chk = r["checks"]
    rp = chk.get("replay") or {}
    lines = [
        f"**{r['model']}** through the service ({r['backend']}), customer `{r['simulator']}`, "
        f"policy {r['policy_mode']}: {r['episodes']} episodes, {r.get('n_tasks', 0)} tasks, "
        f"k={r.get('k', 0)}.",
        "",
        "| pass^1 | pass^k | agent $ / conversation (mean, p95) | $ / passed conversation "
        "| prompt tokens / conversation (cached) | turn p50 / p95 | of a turn: model, tools, "
        "service | service ms / turn p50 / p95 |",
        "|---|---|---|---|---|---|---|---|",
        f"| {r['passed'] / r['episodes']:.3f} ({r['passed']}/{r['episodes']}) "
        f"| {r['tasks_all_passed'] / max(1, r.get('n_tasks', 0)):.3f} "
        f"({r['tasks_all_passed']}/{r.get('n_tasks', 0)}) "
        f"| {c['per_conversation_mean']}, {c['per_conversation_p95']} "
        f"| {c['per_passed_conversation']} "
        f"| {t['prompt']:,.0f} ({(t['cache_share'] or 0):.1%}) "
        f"| {r['turn_ms']['p50']:,.0f} / {r['turn_ms']['p95']:,.0f} ms "
        f"| {share.get('model', 0):.1%}, {share.get('tools', 0):.1%}, "
        f"{share.get('service', 0):.1%} | {s['p50']:,.0f} / {s['p95']:,.0f} |"
        if r["episodes"] and r["turn_ms"]["p50"] is not None else "| (no episode finished) |",
        "",
        f"- stops: {', '.join(f'{k} {v}' for k, v in sorted(r['stops'].items())) or '-'}"
        + (f"; unfinished: {r['unfinished']}" if r["unfinished"] else ""),
        f"- the agent's spend, from the ledger: {c['total']}; per conversation p50 "
        f"{c['per_conversation_p50']}; {r['per_conversation']['model_calls']} model calls, "
        f"{r['per_conversation']['tool_calls']} tool calls and "
        f"{r['per_conversation']['customer_turns']} customer turns a conversation",
        f"- the simulated customer (a test's cost, not the service's): "
        f"{r['customer_tokens_per_conversation']['prompt']} prompt and "
        f"{r['customer_tokens_per_conversation']['completion']} completion tokens a conversation",
        f"- checks: the benchmark's own prompt and tools shown in "
        f"{chk['prompt_and_tools_as_the_benchmark']} of {r['episodes']}; re-scored by the "
        f"benchmark's tools to the service's verdict: {chk['rescored_to_the_same_verdict']} of "
        f"{r['episodes']}"
        + (f" (not re-scorable: {chk['not_rescorable']})" if chk["not_rescorable"] else "")
        + (f"; replayed by the service: {rp.get('ok')} of {rp.get('tenants')} tenants, "
           f"{rp.get('identical')} of {rp.get('calls')} calls identical" if rp else ""),
    ]
    if r.get("served_models"):
        lines.append(f"- served: {', '.join(r['served_models'])}")
    if r.get("per_trap"):
        lines.append("- per trap: " + ", ".join(f"{k} {v:.2f}" for k, v in r["per_trap"].items()))
    return "\n".join(lines) + "\n"


def served_models(db_url: str, tenants: list[str]) -> list[str]:
    from sqlalchemy import select

    from pasarbench.service import schema as S
    from pasarbench.service.store import make_engine
    engine = make_engine(db_url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(select(S.usage_ledger.c.served_model).distinct().where(
                S.usage_ledger.c.tenant_id.in_(tenants))).all()
    finally:
        engine.dispose()
    return sorted({r[0] for r in rows if r[0]})


def ok(r: dict) -> bool:
    chk = r["checks"]
    rp = chk.get("replay") or {}
    return (r["episodes"] > 0 and not r["unfinished"]
            and chk["prompt_and_tools_as_the_benchmark"] == r["episodes"]
            and chk["rescored_to_the_same_verdict"] + sum(chk["not_rescorable"].values())
            == r["episodes"] and not chk["rescored_differently"]
            and not chk["steps_not_rebuilt"]
            and (not rp or rp.get("ok") == rp.get("tenants")))


def sweep(args) -> int:
    from pasarbench.sweep import select_tasks
    if not args.run_id:
        args.run_id = "S-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tasks = select_tasks(args)
    if not tasks:
        raise SystemExit("no task selected")
    agent, key_names = agent_for(args)
    if args.model == "scripted":
        url = ""
    else:
        from pasarbench.service.models import Registry
        url = Registry((args.model,), models_file=args.models_file).resolve(args.model).base_url
    simulator_factory = customer_for(args, agent.extra_body, url)
    run_dir = Path(args.trace_root) / args.run_id
    cell = run_dir / "full"
    if cell.exists() and any(cell.glob("*.jsonl")) and not args.resume:
        raise SystemExit(f"{cell} already holds episodes: --resume, or a new --run-id")
    cell.mkdir(parents=True, exist_ok=True)
    db_url = args.database_url or f"sqlite:///{(run_dir / 'service.db').resolve()}"
    print(f"{args.run_id}: {len(tasks)} tasks x k={args.k}, {agent.model} through the service, "
          f"customer {getattr(simulator_factory(tasks[0]), 'name', '?')}", flush=True)
    outcomes, extra = run_all(args, tasks, agent, key_names, simulator_factory, cell, db_url,
                              run_dir / "service.log")
    tenants = sorted({h["service"]["tenant_id"] for h in headers(cell)})
    r = report(cell, served_models(db_url, tenants) if tenants else [], extra["replay"])
    r["skipped"] = extra["skipped"]
    r["server_holds"] = extra["server_holds"]
    r["crashed"] = extra["crashed"]
    r["setup"] = {"model": args.model, "simulator": args.simulator, "sim_model": args.sim_model,
                  "sim_url": args.sim_url, "customer": args.customer, "suite": args.suite,
                  "traps": args.traps or None, "tasks": args.tasks or None, "k": args.k,
                  "max_steps": args.max_steps, "token_budget": args.token_budget,
                  "strict_money": args.strict_money, "workers": args.workers,
                  "database": db_url.split(":", 1)[0].split("+")[0]}
    return write_report(args, run_dir, cell, r)


def report_only(args) -> int:
    """--report: the run's report.md, service.json and summary.json written again
    from its traces -- nothing is run or sent, no key is read. The replay, the
    served models and what was not run are the run's own, from its service.json."""
    run_dir = Path(args.trace_root) / args.run_id
    cell = run_dir / "full"
    if not args.run_id or not any(cell.glob("*.jsonl")):
        raise SystemExit(f"--report: no traces in {cell}")
    old_path = run_dir / "service.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    r = report(cell, old.get("served_models") or [], (old.get("checks") or {}).get("replay"))
    r["skipped"] = old.get("skipped") or {}
    r["server_holds"] = old.get("server_holds") or []
    r["crashed"] = old.get("crashed") or []
    r["setup"] = old.get("setup") or {}
    args.token_budget = (r["setup"] or {}).get("token_budget", args.token_budget)
    return write_report(args, run_dir, cell, r)


def write_report(args, run_dir: Path, cell: Path, r: dict) -> int:
    md = markdown(r)
    if args.compare:
        cmp = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_cells.py"),
                              args.compare, str(cell)], capture_output=True, text=True, cwd=ROOT)
        md += f"\nCompared with `{args.compare}` (scripts/compare_cells.py):\n\n```text\n" \
              f"{(cmp.stdout or cmp.stderr).strip()}\n```\n"
    (run_dir / "service.json").write_text(json.dumps(r, indent=2, default=str))
    (run_dir / "report.md").write_text(md)
    summary = {"strategy": "full", "exposure": "default", "cell": "full",
               **{k: r.get(k) for k in ("pass^1", "pass^k", "n_tasks", "k")},
               "stop_reasons": r["stops"], "per_trap": r["per_trap"],
               "per_language": r["per_language"], "per_market": r["per_market"],
               "trace_dir": str(cell), "token_budget": args.token_budget,
               "setup": r["setup"], "via": "pasarbench-service"}
    (run_dir / "summary.json").write_text(json.dumps([summary], indent=2))
    print("\n" + md)
    if r["skipped"]:
        print(f"not run: {r['skipped']} -- --resume runs them")
    print(f"traces, report.md, service.json: {run_dir}")
    errors = r["stops"].get("backend_error", 0)
    if errors:
        print(f"!! {errors} episode(s) ended by a server or API error: --resume runs them again")
    if r["crashed"]:
        print(f"!! {len(r['crashed'])} episode(s) failed before their trace began, first: "
              f"{r['crashed'][0]}")
        return 1
    if not ok(r):
        print("!! a check did not hold: see 'checks' in service.json")
        return 1
    return 3 if errors else 0


# ---------------------------------------------------------------------------
# --self-test: a stand-in provider on 127.0.0.1, every path checked, no key
# ---------------------------------------------------------------------------

def stand_in(openings: dict[str, str]):
    """An OpenAI-compatible server on 127.0.0.1: as the agent, each task's
    reference solution (the service's ScriptedAgent, behind real HTTP); as
    the customer, one question and then the end. It records the Authorization
    it was sent: no key may ever reach it."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from pasarbench.harness.types import Message, ToolCall, approx_tokens
    from pasarbench.service.models import ScriptedAgent, solutions

    auth: Counter = Counter()

    def wire(m: dict) -> Message:
        return Message(role=m["role"], content=m.get("content") or "",
                       tool_calls=[ToolCall(name=f["function"]["name"],
                                            arguments=json.loads(f["function"]["arguments"]
                                                                 or "{}"), id=f["id"])
                                   for f in m.get("tool_calls") or []],
                       tool_call_id=m.get("tool_call_id"), name=m.get("name"))

    def answer(body: dict) -> dict:
        msgs = [wire(m) for m in body["messages"]]
        if body.get("tools"):
            opening = next(m.content for m in msgs if m.role == "user")
            resp = ScriptedAgent(solutions()[openings[opening]]).chat(msgs, body["tools"])
            content, calls = resp.content, resp.tool_calls
            prompt = sum(approx_tokens(m.content) for m in msgs) + len(json.dumps(body["tools"])) // 4
            cached = approx_tokens(msgs[0].content) if msgs[0].role == "system" else 0
        else:
            said = sum(m.role == "assistant" for m in msgs)   # her own turns, as she sees them
            content = ("Thanks. Is that everything done now?" if said <= 1
                       else "Great, thank you. ###END###")
            calls, prompt, cached = [], sum(approx_tokens(m.content) for m in msgs), 0
        completion = approx_tokens(content) + 24 * len(calls)
        return {"id": "stand-in", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                             "message": {"role": "assistant", "content": content or None,
                                         "tool_calls": [{"id": tc.id, "type": "function",
                                                         "function": {"name": tc.name,
                                                                      "arguments": json.dumps(
                                                                          tc.arguments)}}
                                                        for tc in calls] or None}}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                          "total_tokens": prompt + completion,
                          "prompt_cache_hit_tokens": cached}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            auth[self.headers.get("Authorization", "")] += 1
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            raw = json.dumps(answer(body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):  # quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, auth


def self_test(args) -> int:
    from pasarbench.sweep import SUITES
    # Eight traps, each in a different language where one can be had: the
    # payloads' digests are checked in every script the tasks are written in.
    languages = sorted({t.language for t in SUITES["all"]})
    traps = sorted({t.trap for t in SUITES["all"]})[:8]
    tasks, openings = [], {}
    for i, trap in enumerate(traps):
        pool = sorted((t for t in SUITES["all"] if t.trap == trap and t.opening not in openings),
                      key=lambda t: (t.language != languages[i % len(languages)], t.task_id))
        if pool:
            openings[pool[0].opening] = pool[0].task_id
            tasks.append(pool[0])
    server, auth = stand_in(openings)
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    tmp = Path(tempfile.mkdtemp(prefix="pasar-selftest-"))
    models = tmp / "models.json"
    models.write_text(json.dumps({"models": [{
        "name": "stand-in", "provider": "deepseek", "model": "stand-in-agent", "api": "chat",
        "base_url": url, "price_model": "deepseek-flash",
        "extra_body": {"thinking": {"type": "disabled"}}}]}))
    args.model, args.models_file = "stand-in", str(models)
    args.simulator, args.sim_model, args.sim_url, args.sim_extra_body = (
        "openai", "stand-in-customer", url, "{}")
    args.tasks = ",".join(t.task_id for t in tasks)
    args.suite, args.traps, args.languages, args.sample, args.k = "all", "", "", 0, 1
    args.trace_root, args.run_id, args.resume, args.compare = str(tmp), "selftest", False, ""
    args.database_url = ""
    code = sweep(args)
    server.shutdown()
    r = json.loads((tmp / "selftest" / "service.json").read_text())
    expected = {
        "every episode passed (each task's reference solution, behind real HTTP)":
            r.get("pass^1") == 1.0 and r["episodes"] == len(tasks),
        "every conversation ran two turns: the customer asked once more, then ended it":
            all(e["customer_turns"] == 2 for e in r["episodes_list"]),
        "the ledger charged every conversation (priced as deepseek-flash)":
            all(e["cost"] > 0 for e in r["episodes_list"]),
        "the provider's cache counted": (r["tokens_per_conversation"]["cache_share"] or 0) > 0,
        "the stand-in was sent no key, only the local placeholder":
            set(auth) == {"Bearer EMPTY"},
        "the service's process held no key": not r["server_holds"],
        "the report's checks held": code == 0,
    }
    print("self-test:")
    for what, held in expected.items():
        print(f"  [{'ok ' if held else 'BAD'}] {what}")
    if all(expected.values()):
        shutil.rmtree(tmp, ignore_errors=True)
        print("self-test: all checks passed")
        return 0
    print(f"self-test: FAILED -- the run is kept in {tmp}")
    return 1


def main(argv: list[str] | None = None) -> int:
    import importlib.util
    args = parse(argv)
    missing = [m for m in ("fastapi", "sqlalchemy", "alembic", "httpx", "uvicorn",
                           "prometheus_client") if importlib.util.find_spec(m) is None]
    if missing:
        # Said before anything starts or is sent, not as a traceback halfway in.
        print(f"The service's dependencies are not installed for {sys.executable} "
              f"({', '.join(missing)} missing). In the repo:\n\n"
              f"    python3 -m venv .venv\n    . .venv/bin/activate\n"
              f"    pip install -r requirements-service.txt\n")
        return 2
    if args.report:
        return report_only(args)
    return self_test(args) if args.self_test else sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
