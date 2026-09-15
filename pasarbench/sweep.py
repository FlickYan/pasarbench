"""
Sweep runner.

    python -m pasarbench.sweep --backend scripted
    python -m pasarbench.sweep --backend openai --model Qwen/Qwen3-8B \
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
import os
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import HarnessAgent
from .db import Database
from .harness.backends import (AnthropicBackend, MuteBackend,
                               OpenAICompatBackend, ScriptedBackend)
from .analyze import attribute_failures, markdown_report
from .diagnose import tool_scaling_report
from .harness.context import BASE_STRATEGIES, make_strategy, note_discipline
from .harness.exposure import build_exposure
from .harness.loop import run_episode
from .harness.simulator import LLMUser, SilentUser
from .harness.trace import TraceWriter, summarise_run
from .harness.types import Budget
from .generate import generate, stratified_sample
from .run import SOLUTIONS
from .tasks import TASKS as CORE_TASKS
from .tasks import Task
from .verifier import pass_hat_k, verify


GEN_TASKS, GEN_SOLUTIONS = generate()
ALL_SOLUTIONS = {**SOLUTIONS, **GEN_SOLUTIONS}
ALL_TASKS = CORE_TASKS + GEN_TASKS


def select_tasks(args) -> list[Task]:
    pool = {"core": CORE_TASKS, "generated": GEN_TASKS, "all": ALL_TASKS}[args.suite]
    if args.languages:
        want = set(args.languages.split(","))
        pool = [t for t in pool if t.language in want]
    if args.tasks:
        ids = set(args.tasks.split(","))
        pool = [t for t in pool if t.task_id in ids]
    if args.sample:
        pool = stratified_sample(pool, per_trap=args.sample)
    return pool


API_KEY_ENV = ("PASARBENCH_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
               "DASHSCOPE_API_KEY", "TOGETHER_API_KEY", "GROQ_API_KEY")

# The simulator usually lives on a DIFFERENT provider from the agent, so it
# needs its own key resolved separately -- otherwise you end up passing it on
# the command line, where other users on a shared node can read it.
SIM_KEY_ENV = ("PASARBENCH_SIM_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY",
               "TOGETHER_API_KEY", "GROQ_API_KEY")


def resolve_api_key() -> str:
    """First key found in the environment, in priority order.

    Prefer this over `--api-key sk-...` on a SHARED CLUSTER. Command-line
    arguments land in /proc/<pid>/cmdline, which is world-readable -- any other
    user on the node can read your key out of `ps aux`. Environment variables
    live in /proc/<pid>/environ, readable only by you and root. They also stay
    out of ~/.bash_history.
    """
    for name in API_KEY_ENV:
        v = os.environ.get(name)
        if v:
            return v
    return "EMPTY"


def resolve_sim_api_key() -> str:
    for name in SIM_KEY_ENV:
        v = os.environ.get(name)
        if v:
            return v
    return ""


def check_key(args) -> None:
    remote = not any(h in args.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    if remote and args.api_key in ("", "EMPTY"):
        raise SystemExit(
            f"No API key found for {args.base_url}.\n"
            f"Set one of: {', '.join(API_KEY_ENV)}\n"
            f"  export DEEPSEEK_API_KEY=sk-...\n"
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
        return lambda t: OpenAICompatBackend(model=args.model, base_url=args.base_url,
                                             api_key=args.api_key,
                                             temperature=args.temperature,
                                             max_tokens=args.max_tokens,
                                             extra_body=_extra_body(args.extra_body))
    if kind == "anthropic":
        return lambda t: AnthropicBackend(model=args.model, temperature=args.temperature)
    raise ValueError(kind)


def make_simulator(kind: str, args) -> callable:
    """The simulator should be a CHEAPER, DIFFERENT model from the agent.
    Sharing weights with the agent under test quietly inflates scores."""
    if kind == "silent":
        return lambda t: SilentUser()
    if kind == "openai":
        sim_url = args.sim_url or args.base_url
        sim_key = args.sim_api_key or resolve_sim_api_key()
        if sim_url == args.base_url:
            sim_key = sim_key or args.api_key
        elif not sim_key:
            # HARD FAIL, never fall back. Reusing the agent's key here does not
            # just produce a confusing 401 -- it TRANSMITS ONE PROVIDER'S SECRET
            # TO A DIFFERENT COMPANY'S SERVERS. That is a credential disclosure,
            # not an inconvenience.
            raise SystemExit(
                f"The simulator is on {sim_url} but no simulator key is set.\n"
                f"Set one of: {', '.join(SIM_KEY_ENV)}\n"
                f"  export PASARBENCH_SIM_API_KEY=sk-...\n\n"
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
        return lambda t: LLMUser(sim, t.persona, t.hidden_facts, t.language)
    if kind == "anthropic":
        sim = AnthropicBackend(model=args.sim_model, temperature=0.7)
        return lambda t: LLMUser(sim, t.persona, t.hidden_facts, t.language)
    raise ValueError(kind)


def run_cell(strategy_name: str, backend_factory, simulator_factory, tasks: list[Task],
             k: int, budget: Budget, policy_mode: str, trace_root: str, run_id: str,
             summarizer=None, exposure_spec: str = "", workers: int = 1) -> dict:
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
                          run_index=seed)
        v = verify(task, db, n_turns=res.budget["steps"],
                   tokens=res.budget["tokens"])
        w.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
        w.close()
        return task, v, res

    jobs = [(t, i) for t in tasks for i in range(k)]
    results = []
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
        if getattr(strategy, "extra_tools", None):
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
            "per_trap": trap_breakdown(by_task, tasks),
            "per_language": language_breakdown(by_task, tasks),
            "per_market": market_breakdown(by_task, tasks),
            "trace_dir": str(Path(trace_root) / run_id / strategy_name),
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
    head = ("| strategy | pass^1 | pass^k | mean tokens | mean steps | stops |\n"
            "|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        stops = ", ".join(f"{k}:{v}" for k, v in sorted(r["stop_reasons"].items()))
        body += (f"| `{r['strategy']}` | {r['pass^1']:.3f} | {r['pass^k']:.3f} | "
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
    ap.add_argument("--api-key", default=resolve_api_key(),
                    help="prefer an env var (%s); a key passed here is visible "
                         "to other users via `ps` on a shared machine"
                         % "/".join(API_KEY_ENV[:3]))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--sim-url", default="", help="simulator endpoint, if different")
    ap.add_argument("--sim-api-key", default="",
                    help="prefer PASARBENCH_SIM_API_KEY / DASHSCOPE_API_KEY in "
                         "the environment; a key passed here is visible in `ps`")
    ap.add_argument("--extra-body", default="",
                    help='provider-specific JSON merged into the request body. '
                         'DeepSeek V4 REQUIRES \'{"thinking":{"type":"disabled"}}\' '
                         'or reasoning tokens inflate every token measurement and '
                         'can exhaust max_tokens before a tool call is emitted.')
    ap.add_argument("--sim-extra-body", default="",
                    help="same, for the simulator only")
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
    ap.add_argument("--policy-mode", default="preload", choices=["preload", "jit"])
    ap.add_argument("-k", type=int, default=1)
    ap.add_argument("--suite", default="core", choices=["core", "generated", "all"])
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified: N tasks PER TRAP. Use while iterating; "
                         "run the full suite only for final numbers.")
    ap.add_argument("--languages", default="", help="e.g. en,id,th")
    ap.add_argument("--tasks", default="", help="comma-separated ids, blank = all")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent episodes. 8-16 is usually safe; too many "
                         "triggers provider rate limits, which surface as "
                         "backend_error stop reasons rather than as a crash.")
    ap.add_argument("--trace-root", default="traces")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if args.backend in ("openai", "anthropic"):
        check_key(args)

    tasks = select_tasks(args)
    if not tasks:
        raise SystemExit("no tasks selected")
    print(f"{len(tasks)} tasks | {len({t.trap for t in tasks})} traps | "
          f"{len({t.language for t in tasks})} languages | "
          f"{len(tasks) * args.k * len(args.strategies.split(','))} episodes")

    run_id = args.run_id or f"{args.backend}-{args.policy_mode}"
    budget = Budget(max_steps=args.max_steps)
    bf = make_backend(args.backend, args)
    sf = make_simulator(args.simulator, args)

    summarizer = None
    if args.summarizer_model:
        summarizer = OpenAICompatBackend(model=args.summarizer_model,
                                         base_url=args.summarizer_url or args.base_url,
                                         api_key=args.api_key, temperature=0.0)

    exposures = [e.strip() for e in args.exposure.split(",")] if args.exposure else [""]
    rows = []
    for name in [n.strip() for n in args.strategies.split(",")]:
        for exp in exposures:
            label = name + (f"+{exp}" if exp else "")
            print(f"running {label} over {len(tasks)} tasks x k={args.k} ...",
                  flush=True)
            rows.append(run_cell(name, bf, sf, tasks, args.k, budget,
                                 args.policy_mode, args.trace_root, run_id,
                                 summarizer, exp, args.workers))

    print("\n" + markdown_table(rows))

    if any(len(r["per_language"]) > 1 for r in rows):
        print("pass rate by language (locale twins share identical checks, so")
        print("any gap here is language and nothing else):")
        for r in rows:
            base = r["per_language"].get("en")
            cells = []
            for lang, rate in r["per_language"].items():
                delta = f" ({rate - base:+.3f})" if base is not None and lang != "en" else ""
                cells.append(f"{lang}={rate:.3f}{delta}")
            print(f"  {r['strategy']:10s} " + "  ".join(cells))
        print()

    print("per-trap pass rate (read this before the aggregate):")
    for r in rows:
        print(f"\n  {r['strategy']}:")
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
            [{**r, "exposure": r["exposure"]} for r in rows]))
        if not ({"all-100", "random-100"} <= set(exposures) or
                {"all-300", "random-300"} <= set(exposures)):
            print("\n!! no all-N / random-N pair in this sweep. Without both, a")
            print("   drop at N tools cannot be attributed to token cost rather")
            print("   than selection difficulty, and the result is not readable.")

    base = args.strategies.split(",")[0].strip()
    if len(rows) > 1 and len(exposures) == 1 and any(r["strategy"] == base for r in rows):
        print("\n" + markdown_report(rows, baseline=base))

    run_path = os.path.join(args.trace_root, run_id)
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(run_path, "report.md"), "w") as f:
        f.write(markdown_report(rows, baseline=base) if len(rows) > 1 else "")
    print(f"\ntraces + summary + report: {run_path}")
    print("\nNext: pasarbench.analyze.what_compaction_lost(<full trace_dir>, "
          "<strategy trace_dir>)\n  -- that is the finding, not the table above.")


if __name__ == "__main__":
    main()
