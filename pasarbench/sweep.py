"""
Sweep runner.

    python -m pasarbench.sweep --backend scripted
    python -m pasarbench.sweep --backend openai --model Qwen/Qwen3.8-27B \
        --base-url https://<app>.modal.run/v1 --strategies full,window8,trim3 -k 3

Emits a markdown table straight into your README and writes traces to
traces/<run_id>/<strategy>/<task>.jsonl.

The scripted backend gives identical numbers across strategies by design --
it replays a fixed plan and ignores context entirely. That is the point: it
proves the sweep machinery works before you spend a cent. Real separation
appears the moment a model is in the loop.

COST WARNING. n_tasks x n_strategies x k episodes, each many model calls, plus
a simulator call per user turn. Subsample while iterating; run the full grid
only for final numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import os
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import HarnessAgent
from .db import Database
from .harness.backends import (AnthropicBackend, MuteBackend,
                               OpenAICompatBackend, ScriptedBackend)
from .analyze import attribute_failures, markdown_report
from .diagnose import tool_scaling_report
from .harness.closing import GUARD_VERSION
from .harness.context import BASE_STRATEGIES, make_strategy, note_discipline
from .harness.exposure import build_exposure
from .harness.loop import run_episode
from .harness.simulator import CUSTOMER, ENDING, LLMUser, SilentUser
from .harness.trace import TraceWriter, summarise_run
from .harness.types import Budget
from .generate import generate, stratified_sample
from .run import SOLUTIONS
from .tasks import TASKS as CORE_TASKS
from .tasks import Task
from .tools import CHECKS as TOOL_CHECKS
from .verifier import pass_hat_k, verify
from .models import check_agent_simulator
from types import SimpleNamespace


GEN_TASKS, GEN_SOLUTIONS = generate()
# New tasks of the closing check's five traps, to confirm it on tasks that did
# not produce it (RUNBOOK 1m). Never in `all`: the suite stays the suite.
EDITION2_TASKS, EDITION2_SOLUTIONS = generate(edition=2)
# The hard edition: cases that combine the suite's traps (RUNBOOK 1s). Never
# in `all` either.
EDITION3_TASKS, EDITION3_SOLUTIONS = generate(edition=3)
ALL_SOLUTIONS = {**SOLUTIONS, **GEN_SOLUTIONS, **EDITION2_SOLUTIONS, **EDITION3_SOLUTIONS}
ALL_TASKS = CORE_TASKS + GEN_TASKS
SUITES = {"core": CORE_TASKS, "generated": GEN_TASKS, "all": ALL_TASKS,
          "edition2": EDITION2_TASKS, "edition3": EDITION3_TASKS}


def select_tasks(args) -> list[Task]:
    pool = SUITES[args.suite]
    if args.languages:
        want = set(args.languages.split(","))
        pool = [t for t in pool if t.language in want]
    if args.tasks:
        ids = set(args.tasks.split(","))
        pool = [t for t in pool if t.task_id in ids]
    if getattr(args, "traps", ""):
        # Against the suite asked for: each edition has traps of its own, and
        # one the suite does not have would select nothing without a word.
        want = {x.strip() for x in args.traps.split(",") if x.strip()}
        have = {t.trap for t in SUITES[args.suite]}
        unknown = sorted(want - have)
        if unknown:
            raise SystemExit(f"--traps: no such trap {unknown} in --suite {args.suite}; its "
                             f"traps are {sorted(have)}")
        pool = [t for t in pool if t.trap in want]
    if args.sample:
        pool = stratified_sample(pool, per_trap=args.sample)
    return pool


# Which variable holds which provider's key. A key goes only to the endpoint it
# belongs to (v55; WHAT_FAILED #47): before v55 the agent took the first of a
# fixed list -- DeepSeek's key whatever --base-url said -- so pointing the agent
# at a second provider with .env loaded would have sent DeepSeek's key there.
PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "api.deepseek.com": ("DEEPSEEK_API_KEY",),
    "dashscope.aliyuncs.com": ("DASHSCOPE_API_KEY",),
    "dashscope-intl.aliyuncs.com": ("DASHSCOPE_API_KEY",),
    "api.openai.com": ("OPENAI_API_KEY",),
    "generativelanguage.googleapis.com": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "api.anthropic.com": ("ANTHROPIC_API_KEY",),
    "api.mistral.ai": ("MISTRAL_API_KEY",),
    "api.x.ai": ("XAI_API_KEY", "GROK_API_KEY"),
    # Z.ai, GLM's international platform, and Zhipu's own in China: one
    # company's two platforms, each key for its own.
    "api.z.ai": ("ZAI_API_KEY", "Z_AI_API_KEY", "GLM_API_KEY", "ZHIPU_API_KEY",
                 "ZHIPUAI_API_KEY"),
    "open.bigmodel.cn": ("ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"),
    # Xiaomi MiMo: api.xiaomimimo.com, and its token-plan hosts.
    "xiaomimimo.com": ("MIMO_API_KEY", "XIAOMI_MIMO_API_KEY", "XIAOMI_API_KEY"),
    "openrouter.ai": ("OPENROUTER_API_KEY",),
    "api.together.xyz": ("TOGETHER_API_KEY",),
    "api.groq.com": ("GROQ_API_KEY",),
}
# This run's keys, for whatever endpoint the command names: the agent's, and
# the customer's (which the claim reader and the translator share). They come
# before a provider's own variable; scripts/add_model.py sets them for its runs.
AGENT_KEY = "PASARBENCH_API_KEY"
SIM_KEY = "PASARBENCH_SIM_API_KEY"
ANTHROPIC_URL = "https://api.anthropic.com/v1"


def provider_key_names(url: str) -> tuple[str, ...]:
    """The variables that may hold the key of the provider `url` is on, in the
    order they are read: all of them that provider's. () for an endpoint no
    provider here is known by."""
    host = (urlparse(url).hostname or "").lower()
    for h, names in PROVIDER_KEYS.items():
        if host == h or host.endswith("." + h):
            return names
    return ()


def provider_key_env(url: str) -> str | None:
    """The variable that holds `url`'s provider's key: the first of its names
    that is set, else the first of them; None for an unknown endpoint."""
    names = provider_key_names(url)
    return next((n for n in names if os.environ.get(n, "").strip()),
                names[0] if names else None)


def key_for(url: str, override: str | None = AGENT_KEY) -> str:
    """The key for one endpoint: "EMPTY" for a local server; else `override`
    (this run's key), if set; else the first set of the variables of the
    provider the endpoint is on. Never another provider's: "" when there is
    none.

    Prefer a variable to `--api-key sk-...` on a SHARED CLUSTER. Command-line
    arguments land in /proc/<pid>/cmdline, which is world-readable -- any other
    user on the node can read your key out of `ps aux`. Environment variables
    live in /proc/<pid>/environ, readable only by you and root. They also stay
    out of ~/.bash_history."""
    if _local(url):
        return "EMPTY"
    # A value with a space or a Windows line end round it (an .env saved on
    # Windows) is the same key: a header cannot carry the line end.
    if override and os.environ.get(override, "").strip():
        return os.environ[override].strip()
    return next((os.environ[n].strip() for n in provider_key_names(url)
                 if os.environ.get(n, "").strip()), "")


def key_hint(url: str, override: str | None = AGENT_KEY) -> str:
    """What to set for `url`'s key."""
    names = provider_key_names(url)
    host = urlparse(url).hostname or url
    if names:
        return (f"set {names[0]}" + (f" (or {', '.join(names[1:])})" if names[1:] else "")
                + (f", or {override} for this run" if override else ""))
    return (f"set {override}: no provider variable is known for {host}" if override
            else f"no provider variable is known for {host}")


def check_key(args, url: str) -> None:
    if not _local(url) and args.api_key in ("", "EMPTY"):
        raise SystemExit(
            f"No API key for {url}: {key_hint(url)}.\n"
            f"No other provider's key is sent there.\n"
            f"Failing here rather than sending the request and handing you a 401.")


def _extra_body(raw: str) -> dict:
    """Parse --extra-body. Fail loudly: a silently ignored provider flag is how
    you end up running with thinking mode on and not knowing."""
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--extra-body is not valid JSON: {e}")
    if not isinstance(d, dict):
        raise SystemExit("--extra-body must be a JSON object")
    return d


def make_backend(kind: str, args) -> callable:
    if kind == "scripted":
        return lambda t: ScriptedBackend(ALL_SOLUTIONS.get(t.task_id, []))
    if kind == "mute":
        return lambda t: MuteBackend()
    if kind == "openai":
        route = fold_router(args)
        return lambda t: OpenAICompatBackend(model=route(t), base_url=args.base_url,
                                             api_key=args.api_key,
                                             temperature=args.temperature,
                                             max_tokens=args.max_tokens,
                                             extra_body=_extra_body(args.extra_body))
    if kind == "anthropic":
        return lambda t: AnthropicBackend(model=args.model, api_key=args.api_key,
                                          temperature=args.temperature)
    raise ValueError(kind)


def fold_router(args):
    """Which model answers a task. Without --fold-models, always --model.

    With it, every task goes to the model that was NOT trained on its fold --
    `--fold-models A=Qwen/Qwen3.8-27B:pasar-rft-A,B=Qwen/Qwen3.8-27B:pasar-rft-B`
    sends fold-A tasks to the model trained on B and vice versa -- so one sweep
    evaluates both fine-tunes on held-out tasks only, and the trace header
    records which model answered."""
    if not args.fold_models:
        return lambda t: args.model
    if not args.folds:
        raise SystemExit("--fold-models needs --folds (the split the models were "
                         "trained on)")
    from .rl.split import load_folds, other_fold
    # The suite's own tasks are checked against the split: the 215 against
    # folds.json, the 62 newer ones and the hard edition's 36 against theirs
    # (rl/split.py --suite).
    pool = SUITES[getattr(args, "suite", "all")]
    folds = load_folds(args.folds, pool)
    trained_on = dict(kv.split("=", 1) for kv in args.fold_models.split(","))
    if set(trained_on) != set(folds["folds"]):
        raise SystemExit(f"--fold-models must name one model per fold "
                         f"{folds['folds']}, got {sorted(trained_on)}")

    def route(t: Task) -> str:
        fold = folds["task_fold"].get(t.task_id)
        if fold is None:
            raise SystemExit(f"{t.task_id} is not in {args.folds}")
        return trained_on[other_fold(fold)]
    return route


def _local(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0"))


def check_served(base_url: str, names: list[str]) -> str:
    """Refuse to start unless the server will answer every name with that model.

    SGLang serves a LoRA adapter under `<base>:<adapter>`, and it answers a
    request naming the bare adapter with the BASE model, without an error --
    although its /v1/models lists the adapter under that bare name. An
    evaluation of the fine-tunes would quietly score the model they started
    from. So on SGLang (its model cards say `owned_by: sglang`) a name must be
    `<served base>:<loaded adapter>` or the base itself; elsewhere (vLLM routes
    a bare adapter name to the adapter) it must be listed as it is."""
    import urllib.error
    import urllib.request
    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(req, timeout=30) as r:
            cards = json.loads(r.read()).get("data", [])
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise SystemExit(f"cannot list the models at {url} ({e}); is the agent "
                         f"server up? (serve_sglang.sh rft)") from None
    ids = {c.get("id") for c in cards}
    bases = {c.get("id") for c in cards if not c.get("parent")}
    adapters = {c.get("id") for c in cards if c.get("parent")}
    sglang = any(c.get("owned_by") == "sglang" for c in cards)

    def served(name: str) -> bool:
        if not sglang:
            return name in ids
        if ":" in name:
            base, adapter = name.split(":", 1)
            return base in bases and adapter in adapters
        return name in bases

    missing = [n for n in names if not served(n)]
    if missing:
        bare = [n for n in missing if sglang and n in adapters]
        raise SystemExit(
            f"the server at {base_url} would not answer {missing} with that model"
            + (f": SGLang answers a bare adapter name with the BASE model. Ask for "
               f"<base>:<adapter>, e.g. {sorted(bases)[0] if bases else 'Qwen/Qwen3.8-27B'}"
               f":{bare[0]}." if bare else
               f". It serves {sorted(bases)} with adapters {sorted(adapters)}"
               + (" (name one as <base>:<adapter>)." if sglang else ".")))
    return f"the server answers every model named with its own weights: {', '.join(names)}"


def make_simulator(kind: str, args) -> callable:
    """The simulator should be a CHEAPER, DIFFERENT model from the agent.
    Sharing weights with the agent under test quietly inflates scores."""
    if kind == "silent":
        return lambda t: SilentUser()
    if kind == "openai":
        sim_url = args.sim_url or args.base_url
        sim_key = args.sim_api_key or key_for(sim_url, SIM_KEY)
        if _local(sim_url):
            # A local server needs no key, and a real one set in the environment
            # for some provider has no business being sent to it.
            sim_key = "EMPTY"
        elif sim_url == args.base_url:
            sim_key = sim_key or args.api_key
        elif not sim_key:
            # HARD FAIL, never fall back. Reusing the agent's key here does not
            # just produce a confusing 401 -- it TRANSMITS ONE PROVIDER'S SECRET
            # TO A DIFFERENT COMPANY'S SERVERS. That is a credential disclosure,
            # not an inconvenience.
            raise SystemExit(
                f"The simulator is on {sim_url} but no simulator key is set: "
                f"{key_hint(sim_url, SIM_KEY)}.\n\n"
                f"Refusing to fall back to the agent's key: that would send your "
                f"{args.base_url} credential to a different provider.")

        if "dashscope" in sim_url:
            region = ("Singapore/international" if "intl" in sim_url
                      else "Beijing/China")
            print(f"   simulator endpoint is the {region} DashScope region. "
                  f"Model Studio keys are REGION-SPECIFIC -- a key from the "
                  f"other console returns 401 invalid_api_key.", flush=True)
        # max_tokens is small on purpose: a customer turn is one or two
        # sentences. Anything larger just pays for a model to ramble.
        sim = OpenAICompatBackend(model=args.sim_model, base_url=sim_url,
                                  api_key=sim_key, temperature=0.7, max_tokens=256,
                                  extra_body=_extra_body(
                                      args.sim_extra_body or args.extra_body))
        return lambda t: LLMUser(sim, t.persona, t.hidden_facts, t.language,
                                 gate_facts=args.gate_facts,
                                 customer=getattr(args, "customer", CUSTOMER))
    if kind == "anthropic":
        sim_key = args.sim_api_key or key_for(ANTHROPIC_URL, SIM_KEY)
        if not sim_key:
            raise SystemExit(f"The simulator is on {ANTHROPIC_URL} but no simulator key is "
                             f"set: {key_hint(ANTHROPIC_URL, SIM_KEY)}.")
        sim = AnthropicBackend(model=args.sim_model, api_key=sim_key, temperature=0.7)
        return lambda t: LLMUser(sim, t.persona, t.hidden_facts, t.language,
                                 gate_facts=args.gate_facts,
                                 customer=getattr(args, "customer", CUSTOMER))
    raise ValueError(kind)


def make_reader(args):
    """The reader guardrail's model (claim_reader.py; RUNBOOK 1u), behind its
    cache, or None for any other guardrail. Its key comes the way the
    simulator's does: the agent's own only when it is the agent's endpoint,
    never one provider's for another."""
    if args.guardrail != "reader":
        return None
    from .claim_reader import CACHE, ClaimReader, make_backend
    url = args.reader_url or args.sim_url or args.base_url
    if _local(url):
        key = "EMPTY"
    elif url == args.base_url:
        key = args.api_key
    else:
        key = args.sim_api_key or key_for(url, SIM_KEY)
        if not key:
            raise SystemExit(f"The claim reader is on {url} but no key is set for it: "
                             f"{key_hint(url, SIM_KEY)}.\n"
                             f"Refusing to fall back to the agent's key: that would send "
                             f"your {args.base_url} credential to a different provider.")
    backend = make_backend(args.reader_model, url, key, _extra_body(args.reader_extra_body))
    return ClaimReader(backend, args.reader_cache or CACHE)


def _finished(path: Path) -> dict | None:
    """Header and footer of a trace that ran to the end, else None. A crash
    mid-episode leaves a header with no footer, and that episode is re-run."""
    try:
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if not recs or recs[-1].get("type") != "footer" or recs[0].get("type") != "header":
        return None
    return {"header": recs[0], "footer": recs[-1],
            "has_messages": any(r.get("type") == "messages" for r in recs)}


def _resumed(task: Task, done: dict):
    """Stand-ins for (verdict, result) built from a finished trace's footer, so
    a resumed sweep's summary counts the episodes it did not have to re-run."""
    f = done["footer"]
    b = f.get("budget") or {}
    v = SimpleNamespace(passed=bool(f.get("passed")), failures=f.get("failures") or [])
    res = SimpleNamespace(budget={"tokens": b.get("tokens", 0), "steps": b.get("steps", 0)},
                          stop_reason=SimpleNamespace(value=f.get("stop_reason", "?")),
                          steps=[], state=SimpleNamespace(messages=[]), resumed=True)
    return task, v, res


def run_cell(strategy_name: str, backend_factory, simulator_factory, tasks: list[Task],
             k: int, budget: Budget, policy_mode: str, trace_root: str, run_id: str,
             summarizer=None, exposure_spec: str = "", workers: int = 1,
             save_messages: bool = False, resume: bool = False,
             guardrail: str = "off", closing: str = "off",
             closing_room: bool = False, claim_reader=None) -> dict:
    strategy = make_strategy(strategy_name, summarizer)
    exposure = build_exposure(exposure_spec, ALL_SOLUTIONS) if exposure_spec else None
    cell = strategy_name if not exposure_spec else f"{strategy_name}+{exposure_spec}"
    tw = TraceWriter(root=trace_root, run_id=f"{run_id}/{cell}")
    by_task = defaultdict(list)
    tokens, steps, stops = [], [], Counter()
    schema_toks, tool_counts = [], []
    notes_written, notes_possible = 0, 0

    def one_episode(job):
        task, seed = job
        # A private writer per episode: TraceWriter holds a single file handle,
        # so sharing one across threads interleaves records from different
        # episodes into the same file.
        w = TraceWriter(root=trace_root, run_id=f"{run_id}/{cell}")
        db = Database.fresh(task.db_patch)
        res = run_episode(task, db, backend_factory(task),
                          simulator=simulator_factory(task),
                          context=strategy, budget=budget, trace=w,
                          policy_mode=policy_mode, exposure=exposure,
                          run_index=seed, guardrail=guardrail, closing=closing,
                          closing_room=closing_room, claim_reader=claim_reader)
        v = verify(task, db, n_turns=res.budget["steps"],
                   tokens=res.budget["tokens"])
        if save_messages:
            w.messages([m.to_dict() for m in res.state.messages],
                       tool_names=[st.tool_names for st in res.steps],
                       context_messages=[st.context_messages for st in res.steps])
        w.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
        w.close()
        return task, v, res

    jobs = [(t, i) for t in tasks for i in range(k)]
    results = []
    if resume:
        # A rented GPU that dies at episode 1,400 of 1,720 should cost the
        # episode in flight, not the run. Finished traces are kept -- but only
        # if they came from the same agent and customer, or the summary would
        # average two different experiments.
        cell_dir = Path(trace_root) / run_id / cell
        todo = []
        retry = 0
        for task, seed in jobs:
            done = _finished(cell_dir / f"{task.task_id}__r{seed}.jsonl")
            if done is None:
                todo.append((task, seed))
                continue
            if done["footer"].get("stop_reason") == "backend_error":
                # Ended by the server or the API, not by the agent: a dead
                # key, a crashed server, a timeout. Kept, it would count as a
                # failure the model never made; re-running it is the point
                # of running the stage again.
                todo.append((task, seed))
                retry += 1
                continue
            head = done["header"]
            want = backend_factory(task)
            want_sim = simulator_factory(task)
            # tool_checks: an episode run before v27, when the tools took any
            # argument value, is not the same experiment (WHAT_FAILED #35).
            if (head.get("requested_model") != getattr(want, "model", None)
                    or head.get("simulator") != getattr(want_sim, "name", "?")
                    or head.get("context") != strategy.name
                    or head.get("policy_mode", policy_mode) != policy_mode
                    or head.get("guardrail", "off") != guardrail
                    or head.get("closing", "off") != closing
                    or (closing == "guard"
                        and head.get("closing_guard", 1) != GUARD_VERSION)
                    or head.get("closing_room", False) != closing_room
                    or head.get("claim_reader") != getattr(claim_reader, "model", None)
                    or head.get("tool_checks", 1) != TOOL_CHECKS):
                raise SystemExit(
                    f"--resume: {cell_dir} already holds episodes from a different "
                    f"setup (model {head.get('requested_model')!r}, simulator "
                    f"{head.get('simulator')!r}, policy {head.get('policy_mode')!r}, "
                    f"guardrail {head.get('guardrail', 'off')!r}, closing "
                    f"{head.get('closing', 'off')!r}"
                    + (f" v{head.get('closing_guard', 1)}" if closing == "guard" else "")
                    + (" with room" if head.get("closing_room") else "")
                    + f", tool checks "
                    f"v{head.get('tool_checks', 1)}). Use a new --run-id.")
            if save_messages and not done["has_messages"]:
                todo.append((task, seed))
                continue
            results.append(_resumed(task, done))
        if results or retry:
            print(f"    resuming: {len(results)} finished, {len(todo)} to run"
                  + (f" ({retry} of them ended by a server or API error)" if retry else ""),
                  flush=True)
        jobs = todo
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one_episode, j) for j in jobs]
            for n, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if n % 25 == 0 or n == len(jobs):
                    print(f"    {n}/{len(jobs)} episodes", flush=True)
    else:
        for n, j in enumerate(jobs, 1):
            results.append(one_episode(j))
            if n % 25 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)} episodes", flush=True)

    for task, v, res in results:
        by_task[task.task_id].append(v)
        tokens.append(res.budget["tokens"])
        steps.append(res.budget["steps"])
        stops[res.stop_reason.value] += 1
        if res.steps:
            schema_toks.append(sum(st.schema_tokens for st in res.steps)
                               / len(res.steps))
            tool_counts.append(res.steps[0].n_tools)
        if getattr(strategy, "extra_tools", None) and not getattr(res, "resumed", False):
            d = note_discipline(res.state.messages)
            notes_written += d["wrote_any"]
            notes_possible += 1
    tw.close()

    s = pass_hat_k(dict(by_task))
    return {"strategy": strategy_name, "exposure": exposure_spec or "default",
            "cell": cell,
            "n_tools": round(sum(tool_counts) / len(tool_counts)) if tool_counts else 0,
            "mean_schema_tokens": (round(sum(schema_toks) / len(schema_toks), 1)
                                   if schema_toks else 0),
            **s,
            "mean_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "mean_steps": round(sum(steps) / len(steps), 2) if steps else 0,
            "stop_reasons": dict(stops),
            "token_budget": budget.max_tokens,
            "per_trap": trap_breakdown(by_task, tasks),
            "per_language": language_breakdown(by_task, tasks),
            "per_market": market_breakdown(by_task, tasks),
            # The CELL, not the strategy: traces are written to run_id/cell, and
            # with --exposure every arm shares a strategy name. Recording the
            # strategy pointed all five tool-scaling rows at one directory that
            # does not exist, and anything that followed it read zero episodes.
            "trace_dir": str(Path(trace_root) / run_id / cell),
            # For NoteTaking: how often the agent actually used the scratchpad.
            # Without this the strategy's result is uninterpretable -- an agent
            # that never writes notes silently degrades it to a naked window.
            "note_discipline": (round(notes_written / notes_possible, 3)
                                if notes_possible else None)}


def language_breakdown(by_task, tasks) -> dict[str, float]:
    """THE multilingual number. Because locale twins share byte-identical
    checks and an identical world, a gap between `en` and any other row is
    attributable to language and nothing else.

    Reporting that a gap exists is the easy half. The half that is worth an
    interview is diagnosing WHERE it comes from -- malformed tool arguments
    extracted from non-Latin script, retrieval failing on English-centric
    embeddings, or compaction silently switching language and dropping detail.
    The traces have all three; go read them."""
    lang = {t.task_id: t.language for t in tasks}
    acc = defaultdict(list)
    for tid, results in by_task.items():
        for r in results:
            acc[lang[tid]].append(r.passed)
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}


def market_breakdown(by_task, tasks) -> dict[str, float]:
    mk = {t.task_id: t.market for t in tasks}
    acc = defaultdict(list)
    for tid, results in by_task.items():
        for r in results:
            acc[mk[tid]].append(r.passed)
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}


def trap_breakdown(by_task, tasks) -> dict[str, float]:
    """Per-trap pass rate. ALWAYS read this before the aggregate -- a headline
    number hides which policy rule the model cannot follow, and that is the
    finding."""
    by_trap = defaultdict(list)
    lookup = {t.task_id: t.trap for t in tasks}
    for tid, results in by_task.items():
        for r in results:
            by_trap[lookup[tid]].append(r.passed)
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(by_trap.items())}


def markdown_table(rows: list[dict]) -> str:
    head = ("| cell | pass^1 | pass^k | mean tokens | mean steps | stops |\n"
            "|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        stops = ", ".join(f"{k}:{v}" for k, v in sorted(r["stop_reasons"].items()))
        body += (f"| `{r.get('cell', r['strategy'])}` | {r['pass^1']:.3f} | {r['pass^k']:.3f} | "
                 f"{r['mean_tokens']:,} | {r['mean_steps']} | {stops} |\n")
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="scripted",
                    choices=["scripted", "mute", "openai", "anthropic"])
    ap.add_argument("--simulator", default="silent",
                    choices=["silent", "openai", "anthropic"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sim-model", default="gpt-4o-mini")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key", default="",
                    help="prefer the provider's variable (DEEPSEEK_API_KEY for "
                         "api.deepseek.com, OPENAI_API_KEY for api.openai.com, ...) or "
                         "PASARBENCH_API_KEY for this run; a key passed here is visible "
                         "to other users via `ps` on a shared machine")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--sim-url", default="", help="simulator endpoint, if different")
    ap.add_argument("--sim-api-key", default="",
                    help="prefer the provider's variable (DASHSCOPE_API_KEY for "
                         "DashScope) or PASARBENCH_SIM_API_KEY for this run; a key "
                         "passed here is visible in `ps`")
    ap.add_argument("--extra-body", default="",
                    help='provider-specific JSON merged into the request body. '
                         'DeepSeek V4 REQUIRES \'{"thinking":{"type":"disabled"}}\' '
                         'or reasoning tokens inflate every token measurement and '
                         'can exhaust max_tokens before a tool call is emitted.')
    ap.add_argument("--sim-extra-body", default="",
                    help="same, for the simulator only")
    ap.add_argument("--gate-facts", action="store_true",
                    help="withhold each hidden fact from the simulator until the "
                         "agent asks for it, as judged by simqa.ASK_PATTERNS. Built "
                         "to stop leaks that turned out to be the detector's "
                         "(WHAT_FAILED #26); a request the patterns miss stalls "
                         "the customer, so it is off by default and discarded "
                         "for run G.")
    ap.add_argument("--strategies", default="full,window8,window4,trim3,notes4",
                    help="also summarize<N>, which needs --summarizer-model")
    ap.add_argument("--summarizer-model", default="",
                    help="cheap model for the summarize strategy; a different, "
                         "smaller model than the agent under test")
    ap.add_argument("--summarizer-url", default="")
    ap.add_argument("--exposure", default="",
                    help="comma list: oracle,all-20,all-100,random-100,search-300. "
                         "Include BOTH all-N and random-N or the arms cannot "
                         "separate token cost from selection difficulty.")
    ap.add_argument("--policy-mode", default="preload",
                    choices=["preload", "jit", "preload-named"],
                    help="preload-named: the policy with the tools it describes but "
                         "never names named (docs/RUNBOOK.md, 1g)")
    ap.add_argument("--guardrail", default="off", choices=["off", "claims", "reader"],
                    help="claims: hold back a reply that claims a write action no "
                         "tool call has done, and tell the agent instead "
                         "(pasarbench/harness/guardrail.py; docs/RUNBOOK.md, 1i). "
                         "reader: the same, the reply read by a model in any language "
                         "in place of the English list (RUNBOOK 1u)")
    ap.add_argument("--reader-model", default="qwen3.8-flash",
                    help="the claim reader's model (--guardrail reader)")
    ap.add_argument("--reader-url", default="",
                    help="its endpoint; default the simulator's")
    ap.add_argument("--reader-extra-body", default='{"reasoning_effort":"low"}')
    ap.add_argument("--reader-cache", default="",
                    help="default data/claims/reader-cache.json, the calibration's: "
                         "one answer per reader and reply text")
    ap.add_argument("--closing", default="off", choices=["off", "check", "record", "guard"],
                    help="check: when the customer leaves, give the agent one note "
                         "to go over the case against the policy and make any call "
                         "it requires that was not made "
                         "(pasarbench/harness/closing.py; docs/RUNBOOK.md, 1j). "
                         "record: the same note after the case record -- every "
                         "identity check and action the conversation attempted, "
                         "and what came of each (RUNBOOK 1n). guard: 1j's note, and "
                         "in the closing phase a write that repeats one already "
                         "made is not run (RUNBOOK 1o)")
    ap.add_argument("--closing-room", action="store_true",
                    help="bound the closing phase by its own six steps, not by what the "
                         "conversation left of the episode's steps, tool calls and "
                         "tokens (v49; in 1q the budget cut 27 of 456 phases short). "
                         "The conversation runs as before; the trace header records it"),
    ap.add_argument("-k", type=int, default=1)
    ap.add_argument("--suite", default="core", choices=list(SUITES),
                    help="edition2: new tasks of the closing check's five traps, outside "
                         "the suite (RUNBOOK 1m); edition3: the hard edition, cases that "
                         "combine traps, outside it too (RUNBOOK 1s)")
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified: N tasks PER TRAP. Use while iterating; "
                         "run the full suite only for final numbers.")
    ap.add_argument("--languages", default="", help="e.g. en,id,th")
    ap.add_argument("--tasks", default="", help="comma-separated ids, blank = all")
    ap.add_argument("--traps", default="", help="comma-separated traps, blank = all")
    ap.add_argument("--customer", type=int, default=CUSTOMER, choices=sorted(ENDING),
                    help="the simulated customer's rules: 2 (the default since v31) "
                         "waits for what she agreed to be done before she ends the "
                         "conversation; 1 could end it on her yes, before the agent "
                         "acted (WHAT_FAILED #38); 3 is 2 that also leaves after a "
                         "promise she cannot keep in the conversation, and never sends "
                         "photos her persona says she lacks (#43). Recorded in the "
                         "simulator's name.")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--token-budget", type=int, default=Budget().max_tokens,
                    help="per-EPISODE token cap; an episode that hits it fails. "
                         "Raise it for tool scaling: at 100 tools ~10k schema "
                         "tokens ride on every call, the default binds, and "
                         "pass^1 starts measuring cost instead of tool choice.")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent episodes. 8-16 is usually safe; too many "
                         "triggers provider rate limits, which surface as "
                         "backend_error stop reasons rather than as a crash.")
    ap.add_argument("--trace-root", default="traces")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--save-messages", action="store_true",
                    help="write each episode's full conversation into its trace; "
                         "post-training builds its examples from these")
    ap.add_argument("--resume", action="store_true",
                    help="keep finished episodes already in this run and cell, "
                         "run only the rest")
    ap.add_argument("--folds", default="",
                    help="split file from `python -m pasarbench.rl.split`")
    ap.add_argument("--fold-models", default="",
                    help="A=<model trained on A>,B=<model trained on B>: each task "
                         "is answered by the model that did not train on it. On "
                         "SGLang name an adapter <base>:<adapter>")
    ap.add_argument("--allow-same-family", action="store_true",
                    help="permit an agent and simulated customer from one model "
                         "family (only to measure that bias on purpose)")
    args = ap.parse_args()

    if args.backend == "openai" and _local(args.base_url):
        # A local server gets the placeholder key: serve_sglang.sh listens on
        # localhost without one, and a provider key picked up from the
        # environment has no business leaving the machine for a local process.
        args.api_key = "EMPTY"
    elif args.backend == "openai":
        args.api_key = args.api_key or key_for(args.base_url)
        check_key(args, args.base_url)
    elif args.backend == "anthropic":
        args.api_key = args.api_key or key_for(ANTHROPIC_URL)
        check_key(args, ANTHROPIC_URL)
    fold_models = ([kv.split("=", 1)[-1] for kv in args.fold_models.split(",")]
                   if args.fold_models else [])
    if args.backend in ("openai", "anthropic") and args.simulator in ("openai", "anthropic"):
        for note in dict.fromkeys(check_agent_simulator(m, args.sim_model,
                                                        args.allow_same_family)
                                  for m in [args.model, *fold_models]):
            print(note)
    # An adapter named as --model (GRPO's rollouts of one adapter) is checked
    # as the fold models are: a wrong name would roll out another policy.
    served = [*([args.model] if ":" in args.model else []), *fold_models]
    if served and args.backend == "openai" and _local(args.base_url):
        print(check_served(args.base_url, served))

    if args.closing_room and args.closing == "off":
        raise SystemExit("--closing-room gives the closing phase room: it needs --closing")
    tasks = select_tasks(args)
    if not tasks:
        raise SystemExit("no tasks selected")
    print(f"{len(tasks)} tasks | {len({t.trap for t in tasks})} traps | "
          f"{len({t.language for t in tasks})} languages | "
          f"{len(tasks) * args.k * len(args.strategies.split(','))} episodes")

    run_id = args.run_id or f"{args.backend}-{args.policy_mode}"
    budget = Budget(max_steps=args.max_steps, max_tokens=args.token_budget)
    bf = make_backend(args.backend, args)
    sf = make_simulator(args.simulator, args)

    summarizer = None
    if args.summarizer_model:
        url = args.summarizer_url or args.base_url
        # The agent's key only on the agent's endpoint; elsewhere, the key of
        # the provider the summarizer is on.
        key = args.api_key if url == args.base_url else key_for(url, None)
        if not key:
            raise SystemExit(f"The summarizer is on {url} but no key is set for it: "
                             f"{key_hint(url, None)}. The agent's key is not sent to "
                             f"another provider.")
        summarizer = OpenAICompatBackend(model=args.summarizer_model, base_url=url,
                                         api_key=key, temperature=0.0)

    exposures = [e.strip() for e in args.exposure.split(",")] if args.exposure else [""]
    reader = make_reader(args)
    rows = []
    for name in [n.strip() for n in args.strategies.split(",")]:
        for exp in exposures:
            label = name + (f"+{exp}" if exp else "")
            print(f"running {label} over {len(tasks)} tasks x k={args.k} ...",
                  flush=True)
            rows.append(run_cell(name, bf, sf, tasks, args.k, budget,
                                 args.policy_mode, args.trace_root, run_id,
                                 summarizer, exp, args.workers,
                                 save_messages=args.save_messages,
                                 resume=args.resume, guardrail=args.guardrail,
                                 closing=args.closing, closing_room=args.closing_room,
                                 claim_reader=reader))
            if reader is not None:
                reader.save()

    print("\n" + markdown_table(rows))

    if any(len(r["per_language"]) > 1 for r in rows):
        print("pass rate by language -- UNPAIRED, so NOT yet a controlled")
        print("comparison. With --sample these rates cover different traps per")
        print("language. Run the full suite and use diagnose.paired_language_gap")
        print("for the comparison the twin design actually licenses:")
        for r in rows:
            base = r["per_language"].get("en")
            cells = []
            for lang, rate in r["per_language"].items():
                delta = f" ({rate - base:+.3f})" if base is not None and lang != "en" else ""
                cells.append(f"{lang}={rate:.3f}{delta}")
            print(f"  {r.get('cell', r['strategy']):16s} " + "  ".join(cells))
        print()

    print("per-trap pass rate (read this before the aggregate):")
    for r in rows:
        print(f"\n  {r.get('cell', r['strategy'])}:")
        for trap, rate in r["per_trap"].items():
            flag = "  <-- weak" if rate < 0.5 else ""
            print(f"    {rate:.2f}  {trap}{flag}")

    nd = [r for r in rows if r.get("note_discipline") is not None]
    for r in nd:
        print(f"\nnote discipline for `{r['strategy']}`: "
              f"{r['note_discipline']:.0%} of episodes wrote at least one note")
        if r["note_discipline"] < 0.5:
            print("  !! below 50% -- this strategy has degraded to a plain sliding")
            print("     window for most episodes. Its number is not interpretable.")

    if args.exposure:
        print("\n" + tool_scaling_report(
            [{**r, "exposure": r["exposure"]} for r in rows],
            run_dir=Path(args.trace_root) / run_id))
        # Only a tool-scaling sweep needs the pair: one that runs search alone,
        # as the naming and guardrail runs do, is not comparing registry sizes.
        sizes = [re.fullmatch(r"(?:all|random)-(\d+)", e) for e in exposures]
        scaling = any(m and int(m.group(1)) > 20 for m in sizes)
        if scaling and not ({"all-100", "random-100"} <= set(exposures) or
                            {"all-300", "random-300"} <= set(exposures)):
            print("\n!! no all-N / random-N pair in this sweep. Without both, a")
            print("   drop at N tools cannot be attributed to token cost rather")
            print("   than selection difficulty, and the result is not readable.")

    base = args.strategies.split(",")[0].strip()
    if len(rows) > 1 and len(exposures) == 1 and any(r["strategy"] == base for r in rows):
        print("\n" + markdown_report(rows, baseline=base))

    # What produced this run, for the report: which model answered, which
    # customer asked, at what temperature, and -- for a held-out evaluation --
    # which model was trained on which fold of which split.
    for r in rows:
        r["setup"] = {"model": args.model, "sim_model": args.sim_model,
                      "temperature": args.temperature, "k": args.k,
                      "fold_models": args.fold_models or None,
                      "folds": args.folds or None}
    run_path = os.path.join(args.trace_root, run_id)
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(run_path, "report.md"), "w") as f:
        f.write(markdown_report(rows, baseline=base) if len(rows) > 1 else "")
    print(f"\ntraces + summary + report: {run_path}")
    print("\nNext: pasarbench.analyze.what_compaction_lost(<full trace_dir>, "
          "<strategy trace_dir>)\n  -- that is the finding, not the table above.")

    note, stop = _server_errors(rows)
    if note:
        print(note)
        if stop:
            raise SystemExit(3)


def _server_errors(rows: list[dict]) -> tuple[str | None, bool]:
    """What to say about episodes the server or the API ended, and whether to
    stop with exit code 3. Such an episode measured the infrastructure, not the
    agent, and counts as a failure until it runs again. Every one is reported:
    two of 252 -- a network blip before the first step -- once went unsaid,
    and a comparison by task counted them as the agent's (RUNBOOK 1l). More
    than 1% stops a pipeline here; a few do not."""
    errors = sum(r["stop_reasons"].get("backend_error", 0) for r in rows)
    total = sum(sum(r["stop_reasons"].values()) for r in rows)
    if not errors:
        return None, False
    return (f"\n!! {errors} of {total} episodes ended with a server or API error "
            f"(backend_error): a dead key, no balance, a crashed or overloaded "
            f"server, the network. Each counts as a failure until it runs again. "
            f"The first trace with one says which. Fix it and run the same command "
            f"with --resume: only those episodes run again."), errors > 0.01 * total


if __name__ == "__main__":
    main()
