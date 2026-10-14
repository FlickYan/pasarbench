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

    # 1v: the same prefixes, the policy in the conversation's language
    python scripts/language_swap.py policy data/swap/W-prefixes.jsonl --run-id V-policy
    python scripts/language_swap.py run data/swap/W-prefixes.jsonl --policy --run-id V-policy
    python scripts/language_swap.py report data/swap/V-policy

    # 1w: another agent's own conversations (its endpoint, its key)
    python scripts/language_swap.py prefixes traces --model glm-5.3-flash \\
        --out data/swap/X-glm-prefixes.jsonl
    python scripts/language_swap.py run data/swap/X-glm-prefixes.jsonl --run-id X-glm \\
        --base-url https://api.z.ai/api/paas/v4 --max-tokens 4096

    # 1x: conversations that show every tool, 1v's arms with the tools named and not
    python scripts/language_swap.py prefixes traces --exposure default \\
        --out data/swap/Y-prefixes.jsonl
    python scripts/language_swap.py check data/swap/Y-prefixes.jsonl
    python scripts/language_swap.py run data/swap/Y-prefixes.jsonl --naming --run-id Y-naming
    python scripts/language_swap.py report data/swap/Y-naming

Keys come from the environment, as the sweep's do, each by the provider its
endpoint is on: the agent's from its provider's variable -- DEEPSEEK_API_KEY,
ZAI_API_KEY, MIMO_API_KEY, XAI_API_KEY (or PASARBENCH_API_KEY) -- the
translator's from DASHSCOPE_API_KEY (or PASARBENCH_SIM_API_KEY), and one
provider's key is never sent to another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
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


def exposure_of(p) -> str:
    """The tools a prefix's run showed the agent: "default", every tool, or
    the exposure it ran under -- "search-300", tools behind search."""
    return p.head.get("exposure") or "default"


def cmd_prefixes(a) -> int:
    prefixes, why = find_prefixes([Path(r) for r in a.roots], a.model)
    if getattr(a, "exposure", ""):
        # 1x (v70): only conversations whose run showed the tools this way.
        other = [p for p in prefixes if exposure_of(p) != a.exposure]
        if other:
            why["another tool exposure"] += len(other)
        prefixes = [p for p in prefixes if exposure_of(p) == a.exposure]
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


def _translator_key(a) -> str:
    """The translator's key, from its own provider's variables."""
    from pasarbench.sweep import SIM_KEY, _local, key_for, key_hint
    if _local(a.translator_url):
        return "EMPTY"
    tr = key_for(a.translator_url, SIM_KEY)
    if not tr:
        raise SystemExit(f"the translator is on {a.translator_url} and no key is set for it: "
                         f"{key_hint(a.translator_url, SIM_KEY)}. The agent's key is not sent "
                         f"to another provider.")
    return tr


def _keys(a) -> tuple[str, str]:
    """(the agent's key, the translator's): each from its own variables, and
    never one provider's for another."""
    from pasarbench.sweep import _local, key_for, key_hint
    agent = key_for(a.base_url)
    if not agent:
        raise SystemExit(f"no key for {a.base_url}: {key_hint(a.base_url)}")
    if _local(a.translator_url):
        return agent, "EMPTY"
    if a.translator_url == a.base_url:
        return agent, agent
    return agent, _translator_key(a)


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


def _translator(a, translator_backend=None):
    """The translator behind its cache: the backend given (a test's), or the
    one the arguments name, with its own key."""
    from pasarbench.harness.backends import OpenAICompatBackend
    from pasarbench.swap import Translator
    if translator_backend is None:
        # The policy step needs no agent, and asks for no agent's key.
        tr_key = _keys(a)[1] if getattr(a, "base_url", None) else _translator_key(a)
        # Room for a section of the policy in Thai (1v); a chat message takes
        # a fraction of it.
        translator_backend = OpenAICompatBackend(
            model=a.translator_model, base_url=a.translator_url, api_key=tr_key,
            temperature=0.0, max_tokens=4096, extra_body=_extra(a.translator_extra_body))
    return Translator(translator_backend, Path(a.cache))


def _policy_needs(prefixes: list, both: bool = False) -> list[tuple[str, bool]]:
    """(language, named) for every policy a 1v run reads: each prefix's
    language other than English, with the tools named in it or not as its
    prompt had it -- and the plain one in every language, which is what
    search_policy returns. 1x (`both`) reads both in every language. A prefix
    whose policy cannot be changed by its prompt stops this before anything is
    asked of anyone."""
    from pasarbench.swap import local_language, policy_unchangeable
    stuck = [f"{p.pid}: {why}" for p in prefixes for why in [policy_unchangeable(p)] if why]
    if stuck:
        raise SystemExit(f"{len(stuck)} prefixes cannot have their policy put in another "
                         f"language (RUNBOOK 1v):\n  " + "\n  ".join(stuck[:20]))
    need = {(local_language(p), (p.head.get("policy_mode") or "preload") == "preload-named")
            for p in prefixes}
    if both:
        need |= {(lang, True) for lang, _ in need}
    return sorted(need | {(lang, False) for lang, _ in need})


def _policies(prefixes: list, tr, workers: int = 8,
              both: bool = False) -> tuple[dict, dict, list]:
    """Every policy the prefixes need, translated piece by piece (the pieces
    asked for side by side, then read from the cache): {(language, named):
    text} for the system prompts, {language: sections} for search_policy, and
    [(language, variant, piece, what it lost)] -- a run starts only if that is
    empty."""
    from pasarbench.harness.prompts import policy_text
    from pasarbench.swap import POLICY_BREAK, translate_policy
    from pasarbench.tools import policy_sections
    need = _policy_needs(prefixes, both)
    pieces = {(c.strip(), lang) for lang, named in need
              for c in policy_text(named=named).split(POLICY_BREAK) if c.strip()}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for f in as_completed([pool.submit(tr.policy, c, lang) for c, lang in pieces]):
            f.result()
    texts, bad = {}, []
    for lang, named in need:
        text, lost = translate_policy(policy_text(named=named), lang, tr.policy)
        texts[(lang, named)] = text
        bad += [(lang, "named" if named else "plain", i, x) for i, x in lost]
    english = [re.match(r"P\d+", h).group() for h, _ in policy_sections(policy_text())]
    sections = {}
    for lang in sorted({lang for lang, _ in need}):
        sections[lang] = policy_sections(texts[(lang, False)])
        heads = [h for h, _ in sections[lang]]
        if len(heads) != len(english) or not all(
                re.search(rf"\b{n}\b", h) for n, h in zip(english, heads)):
            bad.append((lang, "plain", -1, ["its sections are not P1 to P11, one each, "
                                            "each under its own heading"]))
    return texts, sections, bad


def _write_policies(texts: dict, root: Path) -> Path:
    """Each translation, to read; and P10 in English and every language side
    by side -- the section the run is about, read before it runs: as the
    plain text has it (what search_policy returns, and the prompt where the
    tools are not named) and, where a prompt names the tools, as that text has
    it -- a translation of its own (v65; until then the plain one alone)."""
    from pasarbench.harness.prompts import policy_text
    from pasarbench.tools import policy_sections
    out = root / "policy-translated"
    out.mkdir(parents=True, exist_ok=True)
    for (lang, named), text in texts.items():
        (out / f"{lang}.{'named' if named else 'plain'}.md").write_text(text, encoding="utf-8")

    def p10(text: str) -> str:
        return next((f"### {h}\n\n{b}" for h, b in policy_sections(text)
                     if h.startswith("P10")), "(no P10 found)")
    lines = ["# P10 in every language, to read before the run (RUNBOOK 1v)", "",
             "## English", "", p10(policy_text()), "",
             "## English, the tools named", "", p10(policy_text(named=True)), ""]
    for (lang, named), text in sorted(texts.items()):
        lines += [f"## {lang}" + (", the tools named" if named else ""), "", p10(text), ""]
    (out / "P10.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def _print_policy_check(texts: dict, bad: list) -> None:
    print(f"the policy in {len({l for l, _ in texts})} languages "
          f"({', '.join(sorted({l for l, _ in texts}))}), {len(texts)} texts with the tools "
          f"named in them or not: each piece in its language, with every text in backticks, "
          f"section number, number, heading, list item and bold kept -- "
          + ("yes" if not bad else f"NO, {len(bad)} pieces lost something or are not "
                                 f"in their language:"))
    for lang, variant, i, lost in bad:
        where = "its sections" if i < 0 else f"piece {i}"
        print(f"  {lang} {variant}, {where}: " + ", ".join(lost))


def cmd_policy(a, translator_backend=None) -> int:
    """1v, before any agent runs: the policy translated into every language the
    prefixes need, each piece checked, written out to be read."""
    prefixes = load_prefixes(a.prefixes)[:a.limit or None]
    if not prefixes:
        raise SystemExit(f"no prefixes in {a.prefixes}")
    tr = _translator(a, translator_backend)
    try:
        texts, _, bad = _policies(prefixes, tr, a.workers, bool(getattr(a, "naming", False)))
    finally:
        tr.save()
    _print_policy_check(texts, bad)
    where = _write_policies(texts, Path(a.out_root) / a.run_id)
    print(f"written to {where}/ -- read {where / 'P10.md'} before the run" + (
        "" if not bad else ". The run will not start until every piece passes."))
    return 1 if bad else 0


def cmd_run(a, agent_factory=None, translator_backend=None) -> int:
    """Translate every prefix both ways first (one cache, so every arm and
    every repetition reads the same text) -- and with --policy, the policy too,
    checked -- then run the arms."""
    from pasarbench.harness.backends import OpenAICompatBackend
    from pasarbench.harness.exposure import build_exposure
    from pasarbench.rescore import all_tasks
    from pasarbench.swap import (ARMS, NAMING_ARMS, POLICY_ARMS, arm_language, arm_messages,
                                 continue_agent, local_language, naming_arm,
                                 naming_arm_messages, policy_arm_language, policy_arm_messages,
                                 shown_digest)
    from pasarbench.sweep import ALL_SOLUTIONS
    from pasarbench.tasks import task_digest
    from pasarbench.tools import policy_shown_as

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
    policy = bool(getattr(a, "policy", False))
    naming = bool(getattr(a, "naming", False))
    if policy and naming:
        raise SystemExit("--policy or --naming, not both: --naming runs 1v's three arms, "
                         "each with the tools named and not (RUNBOOK 1x)")
    arms = NAMING_ARMS if naming else POLICY_ARMS if policy else ARMS
    jobs = [(p, arm, k) for p in prefixes for arm in arms for k in range(a.k)]
    root = Path(a.out_root)
    todo = [j for j in jobs if not _finished(root / a.run_id / j[1] / f"{j[0].pid}__r{j[2]}.jsonl")]
    texts = sum(1 for p in prefixes for m in p.messages if (m.get("content") or "").strip()
                and m["role"] in ("user", "assistant"))
    chars = {p.pid: sum(len(m.get("content") or "") for m in p.messages) for p in prefixes}
    approx = sum(2 * chars[p.pid] // 4 for p, _, _ in todo)
    print(f"{len(prefixes)} prefixes x {len(arms)} arms x k={a.k}: {len(jobs)} continuations "
          f"of {model}, {len(todo)} still to run; {texts} messages to translate both ways "
          f"with {a.translator_model}. Roughly {approx / 1e6:.1f}M prompt tokens for the agent "
          f"(4 characters a token, two steps a continuation, tool schemas not counted)")
    if policy or naming:
        need = _policy_needs(prefixes, naming)
        print(f"  and the policy, in {len({l for l, _ in need})} languages, {len(need)} texts "
              f"(the tools named in it or not), translated piece by piece and checked before "
              f"any continuation runs; every prefix's policy can be put in its language")
    if naming:
        shows = Counter(exposure_of(p) for p in prefixes)
        print(f"  the tools shown: {', '.join(f'{n} {x}' for x, n in sorted(shows.items()))} "
              f"(RUNBOOK 1x reads the prefixes that show every tool, \"default\")")
    if a.dry_run:
        return 0
    if agent_factory is None:
        agent_key, _ = _keys(a)
    tr = _translator(a, translator_backend)
    try:
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            futs = [pool.submit(arm_messages, p, arm, tr) for p in prefixes
                    for arm in ("swap", "back")]
            for f in as_completed(futs):
                f.result()
        if policy or naming:
            policies, sections, bad = _policies(prefixes, tr, a.workers, naming)
    finally:
        tr.save()
    print(f"translated: {len(tr.cache)} texts in the cache ({a.cache})")

    def messages_for(p, arm):
        local = local_language(p)
        if naming:
            return naming_arm_messages(p, arm, tr, {n: policies[(local, n)]
                                                    for n in (False, True)})
        if policy:
            named = (p.head.get("policy_mode") or "preload") == "preload-named"
            return policy_arm_messages(p, arm, tr, policies.get((local, named)))
        return arm_messages(p, arm, tr)
    if policy or naming:
        _print_policy_check(policies, bad)
        where = _write_policies(policies, root / a.run_id)
        if bad:
            raise SystemExit(f"the run does not start: a piece of the policy lost something "
                             f"in translation, or is not in its language (read {where}/; "
                             f"RUNBOOK 1v)")
        print(f"the policy in each language: {where}/ (P10.md: P10 in each, side by side)")
        # Every arm's messages made once before anything is spent: a prefix
        # that cannot be one stops the run here, not in a worker.
        for p in prefixes:
            for arm in arms:
                messages_for(p, arm)
    else:
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
        local = local_language(p)
        named = (p.head.get("policy_mode") or "preload") == "preload-named"
        msgs, mode = messages_for(p, arm), None
        if naming:
            base, shown_named = naming_arm(arm)
            extra = {"design": "naming", "experiment": "1x", "pid": p.pid, "arm": arm,
                     "base_arm": base, "tools_named": shown_named,
                     "recorded_tools_named": named,
                     "language": policy_arm_language(p, base), "local_language": local,
                     "policy_language": local if base == "policy" else "en",
                     "source_language": p.language, "target": p.target, "source": p.source,
                     "translator": tr.model, "prefix_steps": p.steps,
                     "recorded_escalated": p.escalated}
            shown = policy_shown_as(sections[local]) if base == "policy" else nullcontext()
            # The trace's policy mode is what the agent was shown.
            mode = "preload-named" if shown_named else "preload"
        elif policy:
            extra = {"design": "policy", "experiment": "1v", "pid": p.pid, "arm": arm,
                     "language": policy_arm_language(p, arm), "local_language": local,
                     "policy_language": local if arm == "policy" else "en",
                     "source_language": p.language, "target": p.target, "source": p.source,
                     "translator": tr.model, "prefix_steps": p.steps,
                     "recorded_escalated": p.escalated}
            shown = policy_shown_as(sections[local]) if arm == "policy" else nullcontext()
        else:
            extra = {"pid": p.pid, "arm": arm, "language": arm_language(p, arm),
                     "local_language": local,
                     "source_language": p.language, "target": p.target, "source": p.source,
                     "translator": None if arm == "orig" else tr.model,
                     "prefix_steps": p.steps, "recorded_escalated": p.escalated}
            shown = nullcontext()
        # From v70: which prefixes file, and a digest of every message shown,
        # so the conversation published beside the trace can be checked to be
        # the one the agent saw (scripts/export_hf.py).
        extra.update(prefixes=Path(a.prefixes).name, shown_sha1=shown_digest(msgs))
        trace = _SwapTrace(root, f"{a.run_id}/{arm}", p.pid, extra)
        with shown:
            out = continue_agent(p, msgs, task, agent_factory(p), trace=trace, run_index=k,
                                 max_steps=a.max_steps,
                                 exposure=exposure(p.head.get("exposure")), policy_mode=mode)
        trace.event("swap_outcome", escalated=out.escalated, category=out.category,
                    calls=out.calls, reply=out.reply, searched_nothing=out.searched_nothing)
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
                     "policy_mode": head.get("policy_mode"),
                     "exposure": head.get("exposure") or "default",
                     "escalated": bool(out.get("escalated")),
                     "category": out.get("category"), "calls": out.get("calls") or [],
                     "searched_nothing": out.get("searched_nothing") or 0,
                     "model": head.get("requested_model")})
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


# The p below which a test resolves: 0.05, or less where a reading tests
# several at once (RUNBOOK 1w: 0.05 over three agents). Set by report --alpha.
ALPHA = 0.05


def _line(label: str, g: dict | None, more: str, fewer: str) -> str:
    """A sign test's line. Its direction is the sign test's own -- more
    prefixes higher than lower, or fewer -- not the mean difference's, which
    can point the other way."""
    if g is None:
        return f"  {label}: fewer than 5 prefixes in both -- not testable"
    v = ((more if g["better"] > g["worse"] else fewer) if g["p"] < ALPHA
         else "INCONCLUSIVE")
    return (f"  {label}: {g['diff']:+.3f}; {g['better']} prefixes higher, {g['worse']} lower, "
            f"sign test p={g['p']:.3f}  ->  {v}")


def _room(rate: float) -> str:
    """RUNBOOK 1w: an agent whose translated arms escalate in 10% of their
    runs or fewer, or 90% or more, decides alike in every language."""
    out = rate <= 0.1 or rate >= 0.9
    return (f"  ROOM: the translated arms escalated in {rate:.1%} of their runs"
            + (" -- outside 10%-90%: an INCONCLUSIVE above is NO ROOM, not a null "
               "(RUNBOOK 1w)" if out else ""))


def cmd_report(a) -> int:
    """The reading, its tests resolving below --alpha (and only for it)."""
    global ALPHA
    before, ALPHA = ALPHA, getattr(a, "alpha", 0.05) or 0.05
    try:
        return _report(a)
    finally:
        ALPHA = before


def _report(a) -> int:
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
                                          f"left out: run again)" if broken else "")
          + (f"; a test resolves at p < {ALPHA}" if ALPHA != 0.05 else ""))
    if any(r.get("design") == "naming" for r in rows):
        return _report_naming(rows)
    if any(r.get("design") == "policy" for r in rows):
        return _report_policy(rows)
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
    # An agent that decides alike in every language -- always or never --
    # leaves the test nothing to see (RUNBOOK 1w).
    moved = [r for r in rows if r["arm"] != "orig"]
    print(_room(sum(r["escalated"] for r in moved) / len(moved) if moved else 0.0))
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


def _report_policy(rows: list[dict]) -> int:
    """1v's reading (RUNBOOK 1v): 1t's test again, then the policy in the
    conversation's language against in English, then what is left of the gap."""
    from pasarbench.swap import POLICY_ARMS

    def arm(name):
        return lambda r: r["arm"] == name
    print("\nescalated before its next reply to the customer, by arm and source:")
    print(f"  {'':30s} " + " ".join(f"{h:>13s}" for h in POLICY_ARMS))
    sources = (("recorded in English/Singlish", lambda r: r["english_source"]),
               ("recorded in another language", lambda r: not r["english_source"]),
               ("all", lambda r: True))
    for label, keep in sources:
        mine = [r for r in rows if keep(r)]
        print(f"  {label:30s} " + " ".join(f"{_rate([r for r in mine if r['arm'] == x]):>13s}"
                                           for x in POLICY_ARMS))
    print("  local: the conversation in the language other than English, the policy in English; "
          "policy: the same, the policy in that language; english: the conversation in English")
    print("\nTHE REPLICATION: 1t's test on another day -- the conversation in English against "
          "in the other language, the policy in English in both, paired by prefix")
    print(_line("all prefixes", _paired(rows, arm("local"), arm("english")),
                "LANGUAGE again: English higher", "LANGUAGE: English lower"))
    for label, keep in sources[:2]:
        mine = [r for r in rows if keep(r)]
        print(_line(label, _paired(mine, arm("local"), arm("english")), "English higher",
                    "English lower"))
    print("\nTHE TEST: the policy in the conversation's language against in English, the "
          "conversation the same, paired by prefix")
    print(_line("all prefixes", _paired(rows, arm("local"), arm("policy")),
                "POLICY: higher in her language", "POLICY: lower in her language"))
    for label, keep in sources[:2]:
        mine = [r for r in rows if keep(r)]
        print(_line(label, _paired(mine, arm("local"), arm("policy")), "higher", "lower"))
    print("\nTHE GAP: the conversation and the policy in that language, against both in English")
    print(_line("all prefixes", _paired(rows, arm("policy"), arm("english")),
                "English still higher", "English lower"))
    # Read, not tested (v67): the reading tests the gap over all prefixes; a
    # direction's own can differ from the average, and is shown so it is seen.
    for label, keep in sources[:2]:
        mine = [r for r in rows if keep(r)]
        print(_line(label + " (read)", _paired(mine, arm("policy"), arm("english")),
                    "English still higher", "English lower"))
    # Read, not tested, and not foreseen (v67): the prompts that name the tools
    # -- P10's "Escalate (tool: `escalate_to_human`)", RUNBOOK 1g/1h -- against
    # those that do not. They are different prefixes, from different runs.
    print("\nby the policy's text (read, not tested): the prompts that name the tools, and those "
          "that do not; local / policy / english, then the three comparisons")
    for label, named in (("tools named", True), ("tools not named", False)):
        mine = [r for r in rows if (r.get("policy_mode") == "preload-named") == named]
        if not mine:
            continue
        print(f"  {label} ({len({r['pid'] for r in mine})} prefixes): "
              + " / ".join(_rate([r for r in mine if r["arm"] == x]) for x in POLICY_ARMS))
        for name, first, second, more, fewer in (
                ("  the replication", "local", "english", "English higher", "English lower"),
                ("  the test", "local", "policy", "higher in her language", "lower in her language"),
                ("  the gap", "policy", "english", "English still higher", "English lower")):
            print(_line(name, _paired(mine, arm(first), arm(second)), more, fewer))
    print("\nby the language other than English: escalated with the conversation in it, under "
          "the policy in English / in it; and the conversation in English")
    for lang in sorted({r["local_language"] for r in rows}):
        mine = [r for r in rows if r["local_language"] == lang]
        print(f"  {lang:7s} {_rate([r for r in mine if r['arm'] == 'local']):>10s} / "
              f"{_rate([r for r in mine if r['arm'] == 'policy']):>10s};  "
              f"{_rate([r for r in mine if r['arm'] == 'english']):>10s} in English  "
              f"({len({r['pid'] for r in mine})} prefixes)")
    print("\nsearch_policy, which returns the policy in the language the arm's prompt has it "
          "in, called in:")
    for x in POLICY_ARMS:
        mine = [r for r in rows if r["arm"] == x]
        print(f"  {x:8s} {sum('search_policy' in r['calls'] for r in mine)} of {len(mine)} "
              f"continuations; {sum(r.get('searched_nothing', 0) for r in mine)} searches "
              f"found nothing")
    print("\nwith the category the policy names (duplicate_refund):")
    print(_line("the test", _paired(rows, arm("local"), arm("policy"), key="dup"),
                "higher", "lower"))
    return 0


def _interaction(rows: list[dict], a1: str, b1: str, a2: str, b2: str,
                 key: str = "escalated") -> dict | None:
    """Per prefix, (b1 - a1) - (b2 - a2), each arm's mean over its runs: how
    much more the change from a1 to b1 moves it than the change from a2 to b2.
    A sign test over the prefixes where that is not zero; prefixes with all
    four arms only."""
    from pasarbench.diagnose import _sign_p
    by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["pid"]][r["arm"]].append(bool(r[key]))
    diffs = []
    for arms in by.values():
        if all(arms.get(x) for x in (a1, b1, a2, b2)):
            m = {x: sum(arms[x]) / len(arms[x]) for x in (a1, b1, a2, b2)}
            diffs.append((m[b1] - m[a1]) - (m[b2] - m[a2]))
    if len(diffs) < 5:
        return None
    better = sum(d > 1e-9 for d in diffs)
    worse = sum(d < -1e-9 for d in diffs)
    return {"n": len(diffs), "diff": sum(diffs) / len(diffs), "better": better,
            "worse": worse, "p": _sign_p(worse, better)}


# RUNBOOK 1x: the test reads the prefixes whose run showed the agent every
# tool; behind search, a policy that does not name a tool leaves the agent
# nothing to look for (1g), whatever its language.
NAMING_TESTED = "default"


def _report_naming(rows: list[dict]) -> int:
    """1x's reading (RUNBOOK 1x): whether the policy's language moves the
    decision through the tools it names -- 1v's test with the tools named
    against without, within each prefix."""
    from pasarbench.swap import NAMING_ARMS, POLICY_ARMS

    def arm(name):
        return lambda r: r["arm"] == name
    tested = [r for r in rows if r["exposure"] == NAMING_TESTED]
    print("\nescalated before its next reply to the customer, by arm, the tools not named in "
          "the policy / named:")
    print(f"  {'':10s} {'not named':>13s} {'named':>13s}")
    for base in POLICY_ARMS:
        print(f"  {base:10s} " + " ".join(f"{_rate([r for r in rows if r['arm'] == f'{base}-{n}']):>13s}"
                                          for n in ("plain", "named")))
    own = {r["pid"]: bool(r.get("recorded_tools_named")) for r in rows}
    print(f"  local: her language, the policy in English; policy: her language, the policy in "
          f"it; english: the conversation in English. Every prefix in every arm, whatever its "
          f"own prompt had: it named the tools in {sum(own.values())} of {len(own)}. The tools "
          f"shown: " + ", ".join(f"{x} in {len({r['pid'] for r in rows if r['exposure'] == x})}"
                                 f" prefixes" for x in sorted({r["exposure"] for r in rows}))
          + ".")
    print("\nTHE TEST: naming the tools and the policy's language -- per prefix, the policy in "
          "her language against in English with the tools named, minus the same without "
          f"them; sign test over the prefixes that show every tool ({NAMING_TESTED})")
    if not tested:
        print("  not testable here: no prefix shows every tool")
    else:
        print(_line("all prefixes", _interaction(tested, "local-named", "policy-named",
                                                 "local-plain", "policy-plain"),
                    "NAMING: the policy in her language moves it more with the tools named",
                    "NAMING, the other way: it moves it more without them"))
        moved = [r for r in tested if r["base_arm"] in ("local", "policy")]
        print(_room(sum(r["escalated"] for r in moved) / len(moved) if moved else 0.0)
              .replace("translated arms", "four arms in her language"))
    for label, mine in (("the prefixes that show every tool, each line a sign test read "
                         "with the test", tested),) + tuple(
            (f"the prefixes with the tools {x}, read and never in the test (RUNBOOK 1x): each "
             f"line a sign test", [r for r in rows if r["exposure"] == x])
            for x in sorted({r["exposure"] for r in rows} - {NAMING_TESTED})):
        if not mine:
            continue
        print(f"\n{label}:")
        if mine is not tested:
            print(_line("the test's comparison", _interaction(
                mine, "local-named", "policy-named", "local-plain", "policy-plain"),
                "more with the tools named", "more without them"))
        for name, first, second, more, fewer in (
                ("the policy in her language against in English", "local", "policy",
                 "higher in her language", "lower in her language"),
                ("1t's gap: the conversation in English against in hers", "local", "english",
                 "English higher", "English lower"),
                ("what is left of it: both in English against both in hers", "policy",
                 "english", "English still higher", "English lower")):
            print(f"  {name}:")
            for n, nlabel in (("named", "the tools named"), ("plain", "the tools not named")):
                print(_line(f"    {nlabel}", _paired(mine, arm(f"{first}-{n}"),
                                                     arm(f"{second}-{n}")), more, fewer))
        print("  naming the tools, named against not:")
        for base in POLICY_ARMS:
            print(_line(f"    {base}", _paired(mine, arm(f"{base}-plain"), arm(f"{base}-named")),
                        "higher named", "lower named"))
    print("\nTHE TEST by direction (read):")
    for label, keep in (("recorded in English/Singlish", lambda r: r["english_source"]),
                        ("recorded in another language", lambda r: not r["english_source"])):
        mine = [r for r in tested if keep(r)]
        print(_line(label, _interaction(mine, "local-named", "policy-named", "local-plain",
                                        "policy-plain"), "more with the tools named",
                    "more without them"))
    print("\nby the language other than English (read): local / policy / english, the tools "
          "not named; then named")
    for lang in sorted({r["local_language"] for r in rows}):
        mine = [r for r in rows if r["local_language"] == lang]
        cells = [" / ".join(_rate([r for r in mine if r["arm"] == f"{b}-{n}"])
                            for b in POLICY_ARMS) for n in ("plain", "named")]
        print(f"  {lang:7s} {cells[0]};  {cells[1]}  ({len({r['pid'] for r in mine})} prefixes)")
    print("\nsearch_policy, which returns the policy in the language the arm's prompt has it "
          "in, called in:")
    for x in NAMING_ARMS:
        mine = [r for r in rows if r["arm"] == x]
        print(f"  {x:14s} {sum('search_policy' in r['calls'] for r in mine)} of {len(mine)} "
              f"continuations; {sum(r.get('searched_nothing', 0) for r in mine)} searches "
              f"found nothing")
    print("\nwith the category the policy names (duplicate_refund):")
    print(_line("the test", _interaction(tested, "local-named", "policy-named", "local-plain",
                                         "policy-plain", key="dup"),
                "more with the tools named", "more without them"))
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
    p.add_argument("--exposure", default="",
                   help="only conversations whose run showed the tools this way: \"default\", "
                        "every tool (RUNBOOK 1x), or e.g. \"search-300\"")
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
    r.add_argument("--policy", action="store_true",
                   help="RUNBOOK 1v: the arms local, policy and english -- the policy in the "
                        "conversation's language against in English")
    r.add_argument("--naming", action="store_true",
                   help="RUNBOOK 1x: 1v's three arms, each with the tools named in the policy "
                        "and not")
    r.set_defaults(fn=cmd_run)

    po = sub.add_parser("policy", help="1v, before any agent runs: the policy translated "
                                       "into every language the prefixes need, and checked")
    po.add_argument("prefixes")
    po.add_argument("--run-id", default="V-policy")
    po.add_argument("--out-root", default=str(ROOT))
    po.add_argument("--translator-model", default="qwen3.8-flash")
    po.add_argument("--translator-url",
                    default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    po.add_argument("--translator-extra-body", default='{"reasoning_effort":"low"}')
    po.add_argument("--cache", default=str(CACHE))
    po.add_argument("--workers", type=int, default=8)
    po.add_argument("--limit", type=int, default=0)
    po.add_argument("--naming", action="store_true",
                    help="1x: both texts, the tools named and not, in every language")
    po.set_defaults(fn=cmd_policy)

    c = sub.add_parser("check", help="before any spend: is each prefix what the agent saw?")
    c.add_argument("prefixes")
    c.set_defaults(fn=cmd_check)

    rp = sub.add_parser("report", help="the reading")
    rp.add_argument("run_dir")
    rp.add_argument("--alpha", type=float, default=0.05,
                    help="the p below which a test resolves: 0.0167 for each of 1w's three "
                         "agents (0.05 over three)")
    rp.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
