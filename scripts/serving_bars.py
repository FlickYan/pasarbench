#!/usr/bin/env python3
"""
Phase 3: what serving the agent costs, three ways (docs/RUNBOOK.md, phase 3).

The serving stage (`scripts/gpu_pipeline.sh serving`) serves the base agent and
the customer on its machine and runs the same tasks three times, both servers
started afresh for each bar and the agent's flags the bar's:

    preload-radix     the policy in every prompt, SGLang's radix cache on: every run so far
    jit-radix         no policy in the prompt; the agent looks rules up (search_policy)
    preload-noradix   the policy in every prompt, the agent's radix cache off
                      (AGENT_ARGS=--disable-radix-cache; the customer keeps hers)

For each: three warm-up conversations from outside the suite, not counted; the
agent server's /metrics before and after; the suite at the sweeps'
concurrency, 24 conversations at once, interleaved as a server's queue is. A
bar is measured whole: started again, it starts over. `report` diffs the
counters (serving.metrics), reads pass^1, escalations and tokens from the
traces, prices the card's time, and sets each bar against preload-radix
(serving.report): no speed claim without quality held.

    python scripts/serving_bars.py plan                        # the bars and their flags
    python scripts/serving_bars.py run --bar preload-radix     # the stage runs these
    python scripts/serving_bars.py done --bar preload-radix    # exit 0 once measured
    python scripts/serving_bars.py report --usd-per-hour 6.88 > logs/serving.md

THE CACHE THIS BENCHMARK FLATTERS. 199 of the 215 tasks share six customers,
one a market, and the system prompt names the customer before the policy:
here a conversation's first request finds its market's whole system prompt in
the cache; in a queue where every customer differs it would find only the
lines above her id. The report splits the cached tokens into those of a
conversation's first request -- shared with other conversations -- and those
of its later requests, which reuse its own history, in production as here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from argparse import Namespace
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# bar: (the policy mode, the agent server's own flags)
BARS = {"preload-radix": ("preload", ""),
        "jit-radix": ("jit", ""),
        "preload-noradix": ("preload", "--disable-radix-cache")}
# The order the stage measures them in: the bar whose server is likeliest to
# refuse its flags, last, so the other two are measured whatever it does.
ORDER = ("preload-radix", "jit-radix", "preload-noradix")
BASELINE = "preload-radix"
AGENT_URL = "http://127.0.0.1:8000"
OUT = Path("data/serving")
WARMUP = 3
WORKERS = 24
ESCALATE = "escalate_to_human"


def run_id(prefix: str, bar: str) -> str:
    return f"{prefix}-serve-{bar}"


def files(prefix: str, bar: str) -> dict[str, Path]:
    r = run_id(prefix, bar)
    return {"before": OUT / f"{r}.before.prom", "after": OUT / f"{r}.after.prom",
            "bar": OUT / f"{r}.json", "cell": Path("traces") / r / "full"}


def _get(path: str, url: str = AGENT_URL, timeout: float = 30) -> str:
    with urllib.request.urlopen(url + path, timeout=timeout) as r:   # noqa: S310 (localhost)
        return r.read().decode("utf-8")


def server_radix(url: str = AGENT_URL) -> bool | None:
    """Whether the agent's server runs its radix cache, as the server itself
    says (SGLang's server info), or None if it will not say."""
    for path in ("/get_server_info", "/server_info"):
        try:
            info = json.loads(_get(path, url, timeout=10))
        except Exception:                                   # noqa: BLE001
            continue
        args = info.get("server_args", info) if isinstance(info, dict) else {}
        if isinstance(args, dict) and "disable_radix_cache" in args:
            return not bool(args["disable_radix_cache"])
    return None


def warmup_tasks(n: int = WARMUP) -> list[str]:
    """Conversations outside the suite, with customers of their own: the
    warm-up pays for the cold server and caches none of the measured
    conversations' prompts."""
    from pasarbench.sweep import SUITES
    measured = {t.task_id for t in SUITES["all"]}
    users = {t.user_id for t in SUITES["all"]}
    pool = [t for t in SUITES["edition2"] if t.task_id not in measured and t.user_id not in users]
    picked, markets = [], set()
    for t in pool:                      # a market each, while there are markets left
        if t.market not in markets:
            picked.append(t.task_id)
            markets.add(t.market)
        if len(picked) == n:
            break
    return picked


def expected(sample: int, k: int) -> int:
    from pasarbench.sweep import select_tasks
    return len(select_tasks(Namespace(suite="all", languages="", tasks="", traps="",
                                      sample=sample))) * k


def _sweep(a, rid: str, suite: str, k: int, workers: int, extra: list[str]) -> int:
    """The sweep the pipeline's baseline runs (train_rft._sweep: the same
    agent, customer, flags and temperature 0), its policy mode the bar's."""
    import train_rft
    ns = Namespace(model=a.model, base_url=a.base_url, sim_model=a.sim_model,
                   sim_url=a.sim_url, suite=suite, workers=workers,
                   extra_body=a.extra_body, customer=a.customer,
                   sample=a.sample if suite == "all" else 0, traps="", dry_run=a.dry_run)
    try:
        train_rft._sweep(ns, rid, k, 0.0, extra)
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def _gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0].strip() if out else "?"
    except (OSError, subprocess.SubprocessError):
        return "?"


def read_episodes(cell: Path) -> list[dict[str, Any]]:
    """One row a conversation that ran to its end: its verdict, whether it
    escalated, and its tokens -- those of its first request apart."""
    out = []
    for f in sorted(Path(cell).glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        head = next((r for r in recs if r.get("type") == "header"), {})
        foot = next((r for r in reversed(recs) if r.get("type") == "footer"), None)
        if foot is None:
            continue
        steps = [r for r in recs if r.get("type") == "step"]
        use = [s.get("usage") or {} for s in steps]
        calls = [(tc, tr) for s in steps
                 for tc, tr in zip(s.get("tool_calls") or [], s.get("tool_results") or [])]
        first = use[0] if use else {}
        out.append({
            "task_id": head.get("task_id", f.stem.split("__")[0]),
            "trap": head.get("trap", "?"),
            "policy_mode": head.get("policy_mode"),
            "passed": bool(foot.get("passed")),
            "stop": foot.get("stop_reason"),
            "escalated": any(tc.get("name") == ESCALATE and tr.get("ok") for tc, tr in calls),
            "searched": sum(tc.get("name") == "search_policy" for tc, _ in calls),
            "requests": len(steps),
            "prompt": sum(u.get("prompt") or 0 for u in use),
            "completion": sum(u.get("completion") or 0 for u in use),
            "cached": sum(u.get("cached") or 0 for u in use),
            "first_prompt": first.get("prompt") or 0,
            "first_cached": first.get("cached") or 0,
        })
    return out


# --------------------------------------------------------------------------
# run, done

def cmd_run(a) -> int:
    mode, flags = BARS[a.bar]
    f = files(a.prefix, a.bar)
    if not a.dry_run:
        want = "--disable-radix-cache" not in flags
        radix = server_radix(a.agent_url)
        if radix is not None and radix != want:
            raise SystemExit(
                f"the agent's server runs its radix cache {'on' if radix else 'off'}; "
                f"{a.bar} needs it {'on' if want else 'off'} (gpu_pipeline.sh serving starts "
                f"it with AGENT_ARGS={flags!r})")
        if radix is None and os.environ.get("AGENT_ARGS", "").strip() != flags:
            raise SystemExit(
                f"the agent's server will not say whether its radix cache is on, and "
                f"AGENT_ARGS is {os.environ.get('AGENT_ARGS', '')!r}, not {flags!r}: start "
                f"the servers as gpu_pipeline.sh serving does")
        # A bar is measured whole: what an interrupted one left is not half of
        # a measurement, and its counters died with its server.
        shutil.rmtree(f["cell"].parent, ignore_errors=True)
        for key in ("before", "after", "bar"):
            f[key].unlink(missing_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
    warm = run_id(a.prefix, f"warm-{a.bar}")
    code = _sweep(a, warm, "edition2", 1, WARMUP,
                  ["--policy-mode", mode, "--tasks", ",".join(warmup_tasks())])
    if code:
        raise SystemExit(f"the warm-up of {a.bar} failed ({code}); see the lines above")
    if not a.dry_run:
        shutil.rmtree(Path("traces") / warm, ignore_errors=True)
        f["before"].write_text(_get("/metrics", a.agent_url), encoding="utf-8")
    t0, started = time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    code = _sweep(a, run_id(a.prefix, a.bar), "all", a.k, a.workers, ["--policy-mode", mode])
    wall = time.time() - t0
    if a.dry_run:
        return 0
    f["after"].write_text(_get("/metrics", a.agent_url), encoding="utf-8")
    eps = read_episodes(f["cell"])
    broken = sum(e["stop"] == "backend_error" for e in eps)
    want_n = expected(a.sample, a.k)
    complete = code == 0 and broken == 0 and len(eps) == want_n
    meta = {"bar": a.bar, "policy_mode": mode, "agent_args": flags, "radix": not flags,
            "k": a.k, "sample": a.sample, "workers": a.workers, "warmup": WARMUP,
            "conversations": len(eps), "expected": want_n, "server_errors": broken,
            "sweep_exit": code, "wall_seconds": round(wall, 2), "started": started,
            "agent_model": a.model, "sim_model": a.sim_model, "customer": a.customer,
            "gpu": _gpu(), "complete": complete}
    f["bar"].write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if not complete:
        raise SystemExit(
            f"{a.bar} is not a measurement: {len(eps)} of {want_n} conversations finished, "
            f"{broken} ended by a server error, the sweep exited {code}. Run the stage "
            f"again: this bar starts over")
    print(f"{a.bar}: {len(eps)} conversations in {wall / 60:.1f} min -> {f['bar']}")
    return 0


def measured(prefix: str, bar: str, k: int, sample: int = 0) -> bool:
    f = files(prefix, bar)
    try:
        meta = json.loads(f["bar"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (bool(meta.get("complete")) and meta.get("k") == k
            and meta.get("sample", 0) == sample and f["before"].exists()
            and f["after"].exists()
            and len(read_episodes(f["cell"])) == meta.get("expected"))


def cmd_done(a) -> int:
    return 0 if measured(a.prefix, a.bar, a.k, a.sample) else 1


def cmd_plan(a) -> int:
    """The bars in the order the stage measures them, each with the agent
    server's flags, a tab between: what gpu_pipeline.sh serving reads."""
    for bar in ORDER:
        print(f"{bar}\t{BARS[bar][1]}")
    return 0


# --------------------------------------------------------------------------
# report

def bar_row(prefix: str, bar: str, usd_per_hour: float, human_usd: float
            ) -> dict[str, Any] | str:
    """A measured bar's numbers, or why there are none."""
    from pasarbench.serving import MetricsSnapshot
    from pasarbench.serving.cost import ServingConfig
    f = files(prefix, bar)
    try:
        meta = json.loads(f["bar"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "not measured"
    if not meta.get("complete"):
        return (f"not a measurement: {meta.get('conversations')} of {meta.get('expected')} "
                f"conversations, {meta.get('server_errors')} server errors")
    before = MetricsSnapshot.from_text(f["before"].read_text(encoding="utf-8"), "before")
    after = MetricsSnapshot.from_text(f["after"].read_text(encoding="utf-8"), "after")
    d = before.diff(after)
    pc, ttft = d.prefix_cache_hit_rate(), d.latency()
    tp = d.throughput(meta["wall_seconds"])
    eps = read_episodes(f["cell"])
    n = len(eps)
    passed = sum(e["passed"] for e in eps)
    esc = sum(e["escalated"] for e in eps)
    cfg = ServingConfig(name=bar, n_conversations=n, passed=passed, escalated=esc,
                        prompt_tokens=sum(e["prompt"] for e in eps),
                        completion_tokens=sum(e["completion"] for e in eps),
                        wall_seconds=meta["wall_seconds"], gpu=meta.get("gpu", "?"),
                        gpu_hourly_usd=usd_per_hour)
    br = cfg.breakdown(human_usd)
    cached = sum(e["cached"] for e in eps)
    return {"name": bar, "meta": meta, "episodes": eps, "config": cfg, "breakdown": br,
            "n": n, "pass_rate": passed / n if n else 0.0, "escalated": esc,
            "prefix_hit_rate": pc.get("hit_rate"), "prefix": pc, "ttft": ttft,
            "ttft_p50": ttft.get("p50"), "ttft_p99": ttft.get("p99"),
            "output_tok_per_s": tp.get("output_tok_per_s"), "throughput": tp,
            "usd_per_resolved": br.get("usd_per_resolved"),
            "cached": cached, "first_cached": sum(e["first_cached"] for e in eps),
            "prompt": sum(e["prompt"] for e in eps),
            "requests": sum(e["requests"] for e in eps),
            "searched": sum(e["searched"] for e in eps)}


def _fmt(v: Any, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def report(prefix: str, usd_per_hour: float, human_usd: float) -> str:
    from pasarbench.diagnose import paired_episodes
    from pasarbench.serving import per_trap_guard, three_bar_report
    from pasarbench.serving.cost import compare, markdown
    rows: dict[str, Any] = {b: bar_row(prefix, b, usd_per_hour, human_usd) for b in ORDER}
    done = [rows[b] for b in ORDER if isinstance(rows[b], dict)]
    out = ["# Serving the agent three ways (RUNBOOK phase 3)", ""]
    if not done:
        out.append("No bar is measured yet: `bash scripts/gpu_pipeline.sh serving` (or "
                   "`modal run --detach scripts/modal_pipeline.py --stage serving`).")
        return "\n".join(out)
    m = done[0]["meta"]
    out += [f"`{m['agent_model']}` served by SGLang on {m['gpu']}, the customer "
            f"(`{m['sim_model']}`, customer {m['customer']}) on the same card. Each bar: "
            f"both servers started afresh, {m['warmup']} warm-up conversations from outside "
            f"the suite not counted, then {m['expected']} conversations "
            f"({m['k']} a task{', a sample' if m.get('sample') else ''}), {m['workers']} at "
            f"once. The card at ${usd_per_hour:.2f}/h -- it serves the customer too, so its "
            f"cost is an upper bound on the agent's; a human contact at ${human_usd:.2f} "
            f"(serving/cost.py: an illustrative rate).", ""]
    missing = [f"`{b}`: {rows[b]}" for b in ORDER if not isinstance(rows[b], dict)]
    if missing:
        out += ["Not in this report: " + "; ".join(missing) + ".", ""]

    out += ["## What the server did", "",
            "| bar | policy | agent's radix cache | minutes | requests | prompt tokens a "
            "conversation | cached (server) | cached on a first request | cached from its own "
            "history | TTFT p50 / p99, s | output tok/s |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in done:
        share = r["first_cached"] / r["cached"] if r["cached"] else None
        own = (r["cached"] - r["first_cached"]) / r["prompt"] if r["prompt"] else None
        hit = "-" if r["prefix_hit_rate"] is None else f"{r['prefix_hit_rate']:.1%}"
        first = "-" if share is None else f"{share:.1%}"
        history = "-" if own is None else f"{own:.1%}"
        radix = "on" if r["meta"]["radix"] else "off"
        out.append(
            f"| `{r['name']}` | {r['meta']['policy_mode']} | {radix} | "
            f"{r['meta']['wall_seconds'] / 60:.1f} | {r['requests']} | "
            f"{r['prompt'] / r['n']:,.0f} | {hit} | {first} | {history} | "
            f"{_fmt(r['ttft_p50'])} / {_fmt(r['ttft_p99'])} | "
            f"{_fmt(r['output_tok_per_s'], 0)} |")
    out += ["", "\"cached (server)\" is the agent server's own count over the bar "
            "(`sglang:cached_tokens_total` over `sglang:prompt_tokens_total`, both diffed); "
            "\"cached on a first request\" is the share of the cached tokens, as each "
            "response reported them, that a conversation's first request found -- the "
            "part shared between conversations, which this suite's six customers make "
            "far more common than a queue of different customers would; \"cached from its "
            "own history\" is the rest, over all prompt tokens: what a queue where no two "
            "conversations share a prompt would still find."
            + (f" TTFT: {done[0]['ttft']['resolution_note']}."
               if done[0]["ttft"].get("resolution_note") else ""), ""]

    base = rows[BASELINE] if isinstance(rows[BASELINE], dict) else None
    out += ["## What the agent did", "",
            "| bar | pass^1 | escalated | search_policy calls | tasks against "
            f"`{BASELINE}`: better / worse, sign test |", "|---|---|---|---|---|"]
    for r in done:
        pair = "-"
        if base is not None and r is not base:
            g = paired_episodes(base["episodes"], r["episodes"])
            if g:
                pair = f"{g['better']} / {g['worse']}, p = {g['p']:.3f} ({g['diff']:+.3f})"
        out.append(f"| `{r['name']}` | {r['pass_rate']:.3f} ({sum(e['passed'] for e in r['episodes'])}"
                   f"/{r['n']}) | {r['escalated']} | {r['searched']} | {pair} |")
    out.append("")

    out.append(three_bar_report([{k: r[k] for k in (
        "name", "prefix_hit_rate", "ttft_p50", "ttft_p99", "output_tok_per_s", "pass_rate",
        "n", "usd_per_resolved")} for r in done], BASELINE))
    if base is not None:
        out += ["", "## By trap: did one rule break?", ""]
        for r in done:
            if r is base:
                continue
            per = {}
            for name, row in (("t", r), ("b", base)):
                by: dict[str, list[bool]] = {}
                for e in row["episodes"]:
                    by.setdefault(e["trap"], []).append(e["passed"])
                per[name] = {t: sum(v) / len(v) for t, v in by.items()}
            g = per_trap_guard(per["t"], per["b"])
            out.append(f"- `{r['name']}` against `{BASELINE}`: {g['verdict']}"
                       + (" -- " + ", ".join(f"{t} {v:+.2f}" for t, v in
                                             g["regressed_traps"].items())
                          if g["regressed_traps"] else ""))
        out.append("  (a trap holds 13-16 tasks: at one conversation each, a drop of 0.10 "
                   "is one or two of them -- a reason to read those conversations, not a "
                   "finding)")
    out += ["", markdown(compare([r["config"] for r in done], human_usd)), "",
            "## What this cannot say", "",
            "- How much a production queue would gain from the cache: its first requests "
            "would find little of a system prompt that names a different customer each "
            "time, before the policy. The later requests' share is the transferable part.",
            "- A quality difference smaller than the margin: the verdicts above say what the "
            "conversations measured can tell apart.",
            "- What the agent alone costs: the card served the customer as well."]
    return "\n".join(out)


def cmd_report(a) -> int:
    print(report(a.prefix, a.usd_per_hour, a.human_usd))
    return 0


def main(argv: list[str] | None = None) -> int:
    from pasarbench.harness.simulator import CUSTOMER
    from pasarbench.serving.cost import HUMAN_AGENT_USD_PER_CONTACT
    from train_rft import AGENT, NO_THINKING, SIM_DEFAULT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one bar: warm up, snapshot, the sweep, snapshot")
    r.add_argument("--bar", required=True, choices=list(BARS))
    r.add_argument("--prefix", default="P", help="the round's run prefix: P, or P2")
    r.add_argument("-k", type=int, default=1)
    r.add_argument("--sample", type=int, default=0,
                   help="N tasks a trap instead of all 215 (to try the stage)")
    r.add_argument("--workers", type=int, default=WORKERS)
    r.add_argument("--customer", type=int,
                   default=int(os.environ.get("PASAR_CUSTOMER") or CUSTOMER))
    r.add_argument("--model", default=AGENT)
    r.add_argument("--base-url", default="http://localhost:8000/v1")
    r.add_argument("--sim-model", default=os.environ.get("SIM_MODEL") or SIM_DEFAULT)
    r.add_argument("--sim-url", default="http://localhost:8001/v1")
    r.add_argument("--extra-body", default=NO_THINKING)
    r.add_argument("--agent-url", default=AGENT_URL, help="where the agent's /metrics are")
    r.add_argument("--dry-run", action="store_true", help="print the sweeps, run nothing")
    r.set_defaults(fn=cmd_run)
    d = sub.add_parser("done", help="exit 0 if this bar is measured, 1 if not")
    d.add_argument("--bar", required=True, choices=list(BARS))
    d.add_argument("--prefix", default="P")
    d.add_argument("-k", type=int, default=1)
    d.add_argument("--sample", type=int, default=0)
    d.set_defaults(fn=cmd_done)
    pl = sub.add_parser("plan", help="the bars, in order, with the agent's flags")
    pl.set_defaults(fn=cmd_plan)
    p = sub.add_parser("report", help="the bars side by side, with their verdicts")
    p.add_argument("--prefix", default="P")
    p.add_argument("--usd-per-hour", type=float, required=True,
                   help="the card's price (scripts/gpu_plan.py price)")
    p.add_argument("--human-usd", type=float, default=HUMAN_AGENT_USD_PER_CONTACT,
                   help="a human contact's cost; serving/cost.py's is illustrative")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
