"""
The language swap (docs/RUNBOOK.md, 1t): a recorded duplicate-refund
conversation held fixed at the moment the agent reads the refund, its language
changed and nothing else, and the agent asked again (pasarbench/swap.py).

    # 1. the prefixes: recorded conversations rebuilt up to the lookup, per language
    python scripts/language_swap.py prefixes traces --out data/swap/W-prefixes.jsonl
    # 2. each in three arms -- as recorded, translated, translated and back -- k times
    python scripts/language_swap.py run data/swap/W-prefixes.jsonl --run-id W-swap \\
        --base-url https://api.deepseek.com/v1 \\
        --translator-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    # 3. the reading
    python scripts/language_swap.py report data/swap/W-swap

Keys come from the environment, as the sweep's do, each by the provider its
endpoint is on: the agent's from DEEPSEEK_API_KEY (or PASARBENCH_API_KEY), the
translator's from DASHSCOPE_API_KEY (or PASARBENCH_SIM_API_KEY) -- and one
provider's key is never sent to another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("data/swap")
CACHE = ROOT / "translations.json"


# --------------------------------------------------------------------------
# prefixes

def _pid(path: Path) -> str:
    """<run>.<cell>.<transcript>, with the transcript's "__r" written ".r"."""
    return f"{path.parent.parent.name}.{path.parent.name}.{path.stem.replace('__', '.')}"


def find_prefixes(roots: list[Path], model: str) -> tuple[list, Counter]:
    """Every rebuildable prefix of `model`'s duplicate-refund conversations
    under the roots, and why the others were not."""
    from pasarbench.rescore import all_tasks
    from pasarbench.swap import TRAP, decision_prefix

    tasks = all_tasks()
    files = sorted({f for r in roots for f in ([r] if r.is_file() else r.rglob("*.jsonl"))})
    out, why = [], Counter()
    for f in files:
        try:
            with f.open(encoding="utf-8") as fh:
                head = json.loads(fh.readline() or "{}")
        except (OSError, ValueError):
            continue
        if head.get("type") != "header" or head.get("trap") != TRAP:
            continue
        if head.get("requested_model") != model:
            why["another agent model"] += 1
            continue
        task = tasks.get(head.get("task_id"))
        if task is None:
            why["task not defined today"] += 1
            continue
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        p, reason = decision_prefix(recs, task, pid=_pid(f), source=str(f))
        if p is None:
            why[reason] += 1
        else:
            out.append(p)
    return out, why


def group(p) -> tuple[str, str]:
    """(the direction, the language other than English): conversations
    recorded in Malay and swapped into English are one group, recorded in
    English and swapped into Malay another."""
    from pasarbench.swap import ENGLISH, local_language
    return ("from English" if p.language in ENGLISH else "into English", local_language(p))


def choose(prefixes: list, per_group: int, seed: int = 0) -> list:
    """Up to `per_group` prefixes of each group -- each language other than
    English, in each direction -- spread over the group's tasks: the first of
    each task in a seeded order, then the second, and so on. Never chosen by
    what the agent did."""
    by: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in sorted(prefixes, key=lambda p: hashlib.sha1(f"{seed}:{p.pid}".encode()).hexdigest()):
        by[group(p)][p.task_id].append(p)
    out = []
    for g in sorted(by):
        queues = [by[g][t] for t in sorted(by[g], key=lambda t: hashlib.sha1(
            f"{seed}:{t}".encode()).hexdigest())]
        got: list = []
        while len(got) < per_group and any(queues):
            for q in queues:
                if q and len(got) < per_group:
                    got.append(q.pop(0))
        out += got
    return out


def cmd_prefixes(a) -> int:
    prefixes, why = find_prefixes([Path(r) for r in a.roots], a.model)
    chosen = choose(prefixes, a.per_group, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in chosen:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False, default=str) + "\n")
    have = Counter(group(p) for p in prefixes)
    print(f"{a.model}: {len(prefixes)} conversations rebuilt to the moment it read the refund; "
          f"{len(chosen)} chosen, at most {a.per_group} a group")
    print(f"  {'group':24s} {'rebuilt':>8s} {'chosen':>7s} {'tasks':>6s}  recorded in   "
          f"escalated in that turn, as recorded")
    for g in sorted(have):
        mine = [p for p in chosen if group(p) == g]
        src = Counter(p.language for p in mine)
        esc = sum(p.escalated for p in mine)
        print(f"  {g[0] + ', ' + g[1]:24s} {have[g]:8d} {len(mine):7d} "
              f"{len({p.task_id for p in mine}):6d}  "
              f"{', '.join(f'{t} {n}' for t, n in sorted(src.items())):12s}  {esc}/{len(mine)}")
    print("  left out: " + (", ".join(f"{r} {n}" for r, n in why.most_common()) or "none"))
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------
# run

def load_prefixes(path: str | Path) -> list:
    from pasarbench.swap import Prefix
    return [Prefix.from_dict(json.loads(l)) for l in Path(path).read_text(
        encoding="utf-8").splitlines() if l.strip()]


def _extra(raw: str) -> dict:
    if not raw:
        return {}
    d = json.loads(raw)
    if not isinstance(d, dict):
        raise SystemExit("an extra body must be a JSON object")
    return d


def _keys(a) -> tuple[str, str]:
    """(the agent's key, the translator's): each from its own variables, and
    never one provider's for another."""
    from pasarbench.sweep import SIM_KEY, _local, key_for, key_hint
    agent = key_for(a.base_url)
    if not agent:
        raise SystemExit(f"no key for {a.base_url}: {key_hint(a.base_url)}")
    if _local(a.translator_url):
        return agent, "EMPTY"
    if a.translator_url == a.base_url:
        return agent, agent
    tr = key_for(a.translator_url, SIM_KEY)
    if not tr:
        raise SystemExit(f"the translator is on {a.translator_url} and no key is set for it: "
                         f"{key_hint(a.translator_url, SIM_KEY)}. The agent's key is not sent "
                         f"to another provider.")
    return agent, tr


class _SwapTrace:
    """A trace writer that names the file after the prefix and adds what the
    arm was to the header; the header's task_id stays the task's."""

    def __init__(self, root: Path, run_id: str, name: str, extra: dict):
        from pasarbench.harness.trace import TraceWriter
        self.w = TraceWriter(root=root, run_id=run_id)
        self.name, self.extra = name, extra

    def open_episode(self, task_id, meta=None, run_index=None, **kw):
        self.w.open_episode(self.name, meta={**(meta or {}), "task_id": task_id,
                                             "swap": self.extra}, run_index=run_index, **kw)

    def __getattr__(self, k):
        return getattr(self.w, k)


def _finished(path: Path) -> bool:
    try:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return bool(lines) and json.loads(lines[-1]).get("type") == "footer" \
            and json.loads(lines[-1]).get("stop_reason") != "backend_error"
    except (OSError, ValueError):
        return False


def cmd_run(a, agent_factory=None, translator_backend=None) -> int:
    """Translate every prefix both ways first (one cache, so every arm and
    every repetition reads the same text), then run the arms."""
    from pasarbench.harness.backends import OpenAICompatBackend
    from pasarbench.harness.exposure import build_exposure
    from pasarbench.rescore import all_tasks
    from pasarbench.swap import ARMS, Translator, arm_language, arm_messages, continue_agent
    from pasarbench.sweep import ALL_SOLUTIONS
    from pasarbench.tasks import task_digest

    prefixes = load_prefixes(a.prefixes)[:a.limit or None]
    if not prefixes:
        raise SystemExit(f"no prefixes in {a.prefixes}")
    models = {p.head.get("requested_model") for p in prefixes}
    if len(models) != 1:
        raise SystemExit(f"the prefixes come from more than one agent model: {sorted(models)}")
    tasks = all_tasks()
    for p in prefixes:
        t = tasks.get(p.task_id)
        if t is None or task_digest(t) != p.head.get("task_digest"):
            raise SystemExit(f"{p.pid}: its task has changed since the prefixes were made "
                             f"(or is not defined today): make them again")
    model = models.pop()
    jobs = [(p, arm, k) for p in prefixes for arm in ARMS for k in range(a.k)]
    root = Path(a.out_root)
    todo = [j for j in jobs if not _finished(root / a.run_id / j[1] / f"{j[0].pid}__r{j[2]}.jsonl")]
    texts = sum(1 for p in prefixes for m in p.messages if (m.get("content") or "").strip()
                and m["role"] in ("user", "assistant"))
    chars = {p.pid: sum(len(m.get("content") or "") for m in p.messages) for p in prefixes}
    approx = sum(2 * chars[p.pid] // 4 for p, _, _ in todo)
    print(f"{len(prefixes)} prefixes x {len(ARMS)} arms x k={a.k}: {len(jobs)} continuations "
          f"of {model}, {len(todo)} still to run; {texts} messages to translate both ways "
          f"with {a.translator_model}. Roughly {approx / 1e6:.1f}M prompt tokens for the agent "
          f"(4 characters a token, two steps a continuation, tool schemas not counted)")
    if a.dry_run:
        return 0
    if agent_factory is None or translator_backend is None:
        agent_key, tr_key = _keys(a)
    if translator_backend is None:
        translator_backend = OpenAICompatBackend(
            model=a.translator_model, base_url=a.translator_url, api_key=tr_key,
            temperature=0.0, max_tokens=2048, extra_body=_extra(a.translator_extra_body))
    tr = Translator(translator_backend, Path(a.cache))
    try:
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            futs = [pool.submit(arm_messages, p, arm, tr) for p in prefixes
                    for arm in ("swap", "back")]
            for f in as_completed(futs):
                f.result()
    finally:
        tr.save()
    print(f"translated: {len(tr.cache)} texts in the cache ({a.cache})")
    shown = write_arms(prefixes, tr, root / a.run_id / "arms.md")
    print(f"every prefix's words in its three arms, to read before the results: {shown}")

    if agent_factory is None:
        def agent_factory(p):
            return OpenAICompatBackend(
                model=model, base_url=a.base_url, api_key=agent_key,
                temperature=float(p.head.get("agent_temperature") or 0.0),
                max_tokens=a.max_tokens, extra_body=p.head.get("agent_extra_body") or {})
    exposures: dict[str, object] = {}

    def exposure(spec):
        if not spec or spec == "default":
            return None
        if spec not in exposures:
            exposures[spec] = build_exposure(spec, ALL_SOLUTIONS)
        return exposures[spec]

    def one(job):
        p, arm, k = job
        task = tasks[p.task_id]
        msgs = arm_messages(p, arm, tr)
        extra = {"pid": p.pid, "arm": arm, "language": arm_language(p, arm),
                 "source_language": p.language, "target": p.target, "source": p.source,
                 "translator": None if arm == "orig" else tr.model,
                 "prefix_steps": p.steps, "recorded_escalated": p.escalated}
        trace = _SwapTrace(root, f"{a.run_id}/{arm}", p.pid, extra)
        out = continue_agent(p, msgs, task, agent_factory(p), trace=trace, run_index=k,
                             max_steps=a.max_steps, exposure=exposure(p.head.get("exposure")))
        trace.event("swap_outcome", escalated=out.escalated, category=out.category,
                    calls=out.calls, reply=out.reply)
        trace.close_episode(out.stop, None, [], out.budget)
        return out

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        for f in as_completed([pool.submit(one, j) for j in todo]):
            f.result()
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} continuations", flush=True)
    print(f"traces: {root / a.run_id}/<arm>/. Next: python scripts/language_swap.py report "
          f"{root / a.run_id}")
    return 0


class _Probe:
    """A backend that answers at once and remembers what it was first shown."""
    reports_usage = False
    name = "probe"

    def __init__(self, model: str):
        self.model, self.seen = model, None

    def chat(self, messages, tools):
        from pasarbench.harness.types import ModelResponse, Usage
        if self.seen is None:
            self.seen = (len(messages), sorted(t["function"]["name"] for t in tools))
        return ModelResponse(content="(probe)", usage=Usage(1, 1))


def cmd_check(a) -> int:
    """Before any spend: continue each prefix with a backend that only looks,
    and compare what its first call is shown with the recording's next step --
    as many messages, and the same tools (with tool search, the ones found
    before the lookup). A prefix that fails is not what the agent saw."""
    from pasarbench.harness.exposure import build_exposure
    from pasarbench.rescore import all_tasks
    from pasarbench.swap import continue_agent
    from pasarbench.sweep import ALL_SOLUTIONS

    tasks, exposures, bad, n = all_tasks(), {}, [], 0
    for p in load_prefixes(a.prefixes):
        spec = p.head.get("exposure")
        ex = (None if spec in (None, "default")
              else exposures.setdefault(spec, build_exposure(spec, ALL_SOLUTIONS)))
        probe = _Probe(p.head.get("requested_model") or "?")
        continue_agent(p, p.messages, tasks[p.task_id], probe, exposure=ex)
        steps = [r for r in (json.loads(l) for l in Path(p.source).read_text(
            encoding="utf-8").splitlines() if l.strip())
            if r.get("type") == "step" and not r.get("closing")]
        want = sorted(steps[p.steps]["tool_names"] if len(steps) > p.steps
                      else steps[p.steps - 1]["tool_names"])
        n += 1
        if probe.seen != (len(p.messages), want):
            bad.append(f"{p.pid}: shown {probe.seen[0]} messages and {len(probe.seen[1])} "
                       f"tools; the recording's next step had {len(want)} tools")
    print(f"{n} prefixes continued with a probe; {len(bad)} shown other than the recording's "
          f"next step" + ("" if not bad else ":\n  " + "\n  ".join(bad[:20])))
    return 1 if bad else 0


def write_arms(prefixes: list, tr, path: Path) -> Path:
    """Each prefix's words as recorded, translated, and translated back, side
    by side: what a reader checks before trusting the arms -- above all in
    Thai and Vietnamese, whose suite translations no native speaker has
    reviewed either."""
    from pasarbench.swap import arm_messages, translatable
    lines = ["# The language swap: every prefix's words in its three arms", ""]
    for p in prefixes:
        arms = {arm: arm_messages(p, arm, tr) for arm in ("orig", "swap", "back")}
        lines += [f"## {p.pid}", "",
                  f"{p.task_id}, recorded in {p.language}, swapped into {p.target}; "
                  f"escalated in that turn as recorded: {p.escalated}", ""]
        for i, m in enumerate(p.messages):
            if not translatable(m):
                continue
            lines.append(f"- **{m['role']}**")
            for arm in ("orig", "swap", "back"):
                text = " ".join(arms[arm][i]["content"].split())
                lines.append(f"  - {arm}: {text}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# report

def read_run(run_dir: Path) -> list[dict]:
    """One row per continuation: its prefix, arm, language and outcome."""
    rows = []
    for f in sorted(run_dir.rglob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        head = next((r for r in recs if r.get("type") == "header"), {})
        foot = recs[-1] if recs and recs[-1].get("type") == "footer" else None
        out = next((r for r in recs if r.get("kind") == "swap_outcome"), None)
        sw = head.get("swap")
        if not sw or foot is None or out is None:
            continue
        rows.append({**sw, "task_id": head.get("task_id"), "stop": foot.get("stop_reason"),
                     "escalated": bool(out.get("escalated")),
                     "category": out.get("category"), "model": head.get("requested_model")})
    return rows


def _paired(rows: list[dict], first, second, key: str = "escalated") -> dict | None:
    """Prefixes in both arms, paired: second minus first, sign test."""
    from pasarbench.diagnose import paired_episodes
    ea = [{"task_id": r["pid"], "passed": r[key]} for r in rows if first(r)]
    eb = [{"task_id": r["pid"], "passed": r[key]} for r in rows if second(r)]
    return paired_episodes(ea, eb)


def _rate(rows: list[dict], key: str = "escalated") -> str:
    n = len(rows)
    k = sum(bool(r[key]) for r in rows)
    return f"{k}/{n} {k / n:4.0%}" if n else "-"


def _line(label: str, g: dict | None, more: str, fewer: str) -> str:
    if g is None:
        return f"  {label}: fewer than 5 prefixes in both -- not testable"
    v = (more if g["diff"] > 0 else fewer) if g["resolved"] else "INCONCLUSIVE"
    return (f"  {label}: {g['diff']:+.3f}; {g['better']} prefixes higher, {g['worse']} lower, "
            f"sign test p={g['p']:.3f}  ->  {v}")


def cmd_report(a) -> int:
    from pasarbench.swap import ENGLISH
    rows = read_run(Path(a.run_dir))
    if not rows:
        raise SystemExit(f"no continuations in {a.run_dir}")
    broken = [r for r in rows if r["stop"] == "backend_error"]
    rows = [r for r in rows if r["stop"] != "backend_error"]
    for r in rows:
        r["dup"] = r["escalated"] and r["category"] == "duplicate_refund"
        r["english"] = r["language"] in ENGLISH
        r["english_source"] = r["source_language"] in ENGLISH
    pids = {r["pid"] for r in rows}
    print(f"{len(rows)} continuations of {', '.join(sorted({str(r['model']) for r in rows}))} "
          f"from {len(pids)} prefixes" + (f" ({len(broken)} ended by a server or API error, "
                                          f"left out: run again)" if broken else ""))
    print("\nescalated before its next reply to the customer, by arm and source:")
    print(f"  {'':30s} {'orig':>13s} {'swap':>13s} {'back':>13s}")
    for label, keep in (("recorded in English/Singlish", lambda r: r["english_source"]),
                        ("recorded in another language", lambda r: not r["english_source"]),
                        ("all", lambda r: True)):
        mine = [r for r in rows if keep(r)]
        print(f"  {label:30s} " + " ".join(f"{_rate([r for r in mine if r['arm'] == arm]):>13s}"
                                           for arm in ("orig", "swap", "back")))
    first = {r["pid"]: r["recorded_escalated"] for r in rows}
    origs = defaultdict(list)
    for r in rows:
        if r["arm"] == "orig":
            origs[r["pid"]].append(r["escalated"])
    same = sum((2 * sum(v) > len(v)) == first[p] for p, v in origs.items())
    print(f"  the recording escalated in that turn in {sum(first.values())} of {len(first)} "
          f"prefixes; the orig arm, by the majority of its runs, agrees with it on {same} of "
          f"{len(origs)}" + ("" if not origs or same >= 0.75 * len(origs) else
                             "  !! under 75%: the prefixes may not be what the agent saw -- "
                             "read nothing else until this is understood (RUNBOOK 1t)"))

    def eng(r):
        return r["arm"] != "orig" and r["english"]

    def loc(r):
        return r["arm"] != "orig" and not r["english"]
    print("\nTHE TEST: the same conversation in English (or Singlish) against the other "
          "language, both translated, paired by prefix")
    print(_line("all prefixes", _paired(rows, loc, eng), "LANGUAGE: English higher",
                "LANGUAGE: English lower"))
    for label, keep in (("recorded in English or Singlish", lambda r: r["english_source"]),
                        ("recorded in another language", lambda r: not r["english_source"])):
        mine = [r for r in rows if keep(r)]
        print(_line(label, _paired(mine, loc, eng), "English higher", "English lower"))
    print("\nTHE TRANSLATOR: as recorded against there and back, the same language, paired")
    print(_line("all prefixes", _paired(rows, lambda r: r["arm"] == "orig",
                                        lambda r: r["arm"] == "back"),
                "back higher: the translator moves it", "back lower: the translator moves it"))
    print("\nby the other language: English against it, both translated")
    langs = sorted({r["language"] for r in rows if not r["english"]})
    for lang in langs:
        mine = [r for r in rows if r["pid"] in {x["pid"] for x in rows
                                                if x["language"] == lang}]
        print(f"  {lang:7s} {_rate([r for r in mine if eng(r)]):>10s} in English, "
              f"{_rate([r for r in mine if loc(r)]):>10s} in {lang}  "
              f"({len({r['pid'] for r in mine})} prefixes)")
    print("\nwith the category the policy names (duplicate_refund):")
    print(_line("all prefixes", _paired(rows, loc, eng, key="dup"), "English higher",
                "English lower"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefixes", help="rebuild recorded conversations up to the lookup")
    p.add_argument("roots", nargs="+", help="trace directories (or files)")
    p.add_argument("--model", default="deepseek-v4-pro", help="the agent model whose "
                   "conversations to hold fixed")
    p.add_argument("--per-group", type=int, default=8,
                   help="prefixes a group: each language other than English, recorded in "
                        "it and swapped into English, or recorded in English and swapped "
                        "into it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "W-prefixes.jsonl"))
    p.set_defaults(fn=cmd_prefixes)

    r = sub.add_parser("run", help="each prefix in three arms, k times")
    r.add_argument("prefixes")
    r.add_argument("--run-id", default="W-swap")
    r.add_argument("--out-root", default=str(ROOT))
    r.add_argument("-k", type=int, default=3)
    r.add_argument("--base-url", default="https://api.deepseek.com/v1",
                   help="the agent's endpoint; its model, temperature and request flags "
                        "are the recording's")
    r.add_argument("--max-tokens", type=int, default=2048)
    r.add_argument("--max-steps", type=int, default=8,
                   help="agent steps after the prefix before it must have replied")
    r.add_argument("--translator-model", default="qwen3.8-flash")
    r.add_argument("--translator-url",
                   default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    r.add_argument("--translator-extra-body", default='{"reasoning_effort":"low"}')
    r.add_argument("--cache", default=str(CACHE))
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int, default=0, help="the first N prefixes only (a smoke)")
    r.add_argument("--dry-run", action="store_true", help="count, call nothing")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("check", help="before any spend: is each prefix what the agent saw?")
    c.add_argument("prefixes")
    c.set_defaults(fn=cmd_check)

    rp = sub.add_parser("report", help="the reading")
    rp.add_argument("run_dir")
    rp.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
