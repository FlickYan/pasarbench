"""
The recorded runs as a Hugging Face dataset (v44).

    python scripts/export_hf.py traces --out hf-dataset
    hf auth login                       # once; or: huggingface-cli login
    hf upload <you>/pasarbench-traces hf-dataset . --repo-type dataset

One JSON line per episode, one file per run (data/<run>.jsonl): the fields to
filter on -- run, cell, task, trap, market, language, agent model, customer,
setup, the verdict the run recorded, steps and tokens -- beside the whole trace
as it was written. runs.jsonl lists the runs; README.md is the dataset card,
written from what was exported.

The language swap's runs too (v70; RUNBOOK 1t, 1v, 1w, 1x), from data/swap:
one line per continuation (swap/<run>.jsonl) -- its arm, languages, whether
the policy named the tools, the recorded conversation it continued, whether
it escalated, and its trace -- and one per arm of each prefix
(swap_prompts/<run>.jsonl): every message the agent was shown, rebuilt from
the prefixes and the run's translation cache as the run built them, no call
made, and checked against the digest each continuation records from v70.

Nothing leaves this machine that should not. Before writing anything, every
trace is checked for API keys, access tokens and home-directory paths: one
found stops the export and names the file -- remove it at the source, never
here. Email addresses are replaced with "[email]" and counted: the customers
are simulated, but an address one of them made up may be somebody's real one.

`passed` is the verdict of the checker the day the run happened. Checks
corrected since change some of them; scripts/rescore.py replays an episode's
calls and scores it under today's.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SECRETS = {
    "an API key (sk-...)": re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    "a Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "an AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "a bearer token": re.compile(r"Bearer\s+[A-Za-z0-9._-]{16,}"),
    "an api_key value": re.compile(r"(?i)api[_-]?key[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_-]{12,}"),
    "a home-directory path": re.compile(r"(/Users/|/home/|C:\\\\Users\\\\)[^/\\\s\"]+"),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
FIELDS = ("run", "cell", "episode", "task_id", "run_index", "trap", "market", "language",
          "agent_model", "simulator", "policy_mode", "exposure", "context", "closing",
          "guardrail", "started_at", "passed", "stop_reason", "steps", "tokens", "failures",
          "trace")
SWAP_FIELDS = ("run", "experiment", "arm", "pid", "task_id", "run_index", "market", "trap",
               "agent_model", "translator", "recorded_language", "target", "language",
               "local_language", "policy_language", "tools_named", "exposure", "source_run",
               "source_cell", "source_episode", "prefix_steps", "recorded_escalated",
               "escalated", "category", "calls", "reply", "searched_nothing", "stop_reason",
               "steps", "tokens", "started_at", "trace")
PROMPT_FIELDS = ("run", "experiment", "arm", "pid", "task_id", "language", "policy_language",
                 "tools_named", "translator", "messages")


def problems(text: str) -> list[str]:
    """What in a trace must not be published: the kinds found, each once."""
    return [kind for kind, pat in SECRETS.items() if pat.search(text)]


def row(run: str, cell: str, path: Path, text: str) -> dict:
    """One episode's line: its setup and outcome as columns, the trace whole."""
    recs = [json.loads(l) for l in text.splitlines() if l.strip()]
    head = recs[0] if recs and recs[0].get("type") == "header" else {}
    foot = recs[-1] if recs and recs[-1].get("type") == "footer" else {}
    budget = foot.get("budget") or {}
    return {"run": run, "cell": cell, "episode": path.stem,
            "task_id": head.get("task_id") or path.stem.rsplit("__", 1)[0],
            "run_index": head.get("run_index"), "trap": head.get("trap"),
            "market": head.get("market"), "language": head.get("language"),
            "agent_model": head.get("requested_model"), "simulator": head.get("simulator"),
            "policy_mode": head.get("policy_mode"), "exposure": head.get("exposure"),
            "context": head.get("context"), "closing": head.get("closing") or "off",
            "guardrail": head.get("guardrail") or "off", "started_at": head.get("started_at"),
            "passed": foot.get("passed"), "stop_reason": foot.get("stop_reason"),
            "steps": budget.get("steps"), "tokens": budget.get("tokens"),
            "failures": [str(f) for f in foot.get("failures") or []], "trace": text}


def collect(root: Path, runs: list[str] | None = None) -> tuple[dict, Counter]:
    """Every episode under root, by run, after the checks; and the emails replaced."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    emails: Counter = Counter()
    bad: list[str] = []
    for path in sorted(root.rglob("*.jsonl")):
        rel = path.relative_to(root).parts
        if len(rel) != 3:          # traces/<run>/<cell>/<episode>.jsonl only
            continue
        run, cell = rel[0], rel[1]
        if runs and run not in runs:
            continue
        text = path.read_text(encoding="utf-8")
        found = problems(text)
        if found:
            bad.append(f"{path}: {', '.join(found)}")
            continue
        text, n = EMAIL.subn("[email]", text)
        if n:
            emails[f"{run}/{cell}/{path.name}"] += n
        r = row(run, cell, path, text)
        if r["trap"] is None and r["passed"] is None:
            continue               # not an episode trace
        by_run[run].append(r)
    if bad:
        raise SystemExit("not exported -- remove these at the source and run again:\n  "
                         + "\n  ".join(bad))
    return by_run, emails


# --------------------------------------------------------------------------
# the language swap (RUNBOOK 1t, 1v, 1w, 1x)

def experiment(run: str, swap: dict) -> str | None:
    """The RUNBOOK section a swap run answers: recorded by the run from v70;
    before, by its design and its name as the RUNBOOK gives them (W- 1t,
    V-policy 1v, X- 1w)."""
    if swap.get("experiment"):
        return swap["experiment"]
    if swap.get("design") == "policy":
        return "1v"
    return {"W": "1t", "X": "1w"}.get(run.split("-", 1)[0])


def _source(path: str | None) -> tuple:
    """(run, cell, episode) of the recorded trace a prefix came from, as the
    episodes config names them."""
    parts = Path(path or "").parts           # <traces>/<run>/<cell>/<episode>.jsonl
    if len(parts) >= 3:
        return parts[-3], parts[-2], Path(parts[-1]).stem
    return None, None, None


def swap_row(run: str, text: str) -> dict | None:
    """One continuation's line, or None for a file that is not a finished one
    -- the continuations `language_swap.py report` counts."""
    recs = [json.loads(l) for l in text.splitlines() if l.strip()]
    head = next((r for r in recs if r.get("type") == "header"), {})
    foot = recs[-1] if recs and recs[-1].get("type") == "footer" else None
    out = next((r for r in recs if r.get("type") == "event"
                and r.get("kind") == "swap_outcome"), None)
    sw = head.get("swap")
    if not isinstance(sw, dict) or foot is None or out is None:
        return None
    budget = foot.get("budget") or {}
    src = _source(sw.get("source"))
    return {"run": run, "experiment": experiment(run, sw), "arm": sw.get("arm"),
            "pid": sw.get("pid"), "task_id": head.get("task_id"),
            "run_index": head.get("run_index"), "market": head.get("market"),
            "trap": head.get("trap"), "agent_model": head.get("requested_model"),
            "translator": sw.get("translator"), "recorded_language": sw.get("source_language"),
            "target": sw.get("target"), "language": sw.get("language"),
            "local_language": sw.get("local_language"),
            "policy_language": sw.get("policy_language") or "en",
            "tools_named": (sw["tools_named"] if "tools_named" in sw else
                            (head.get("policy_mode") or "preload") == "preload-named"),
            "exposure": head.get("exposure") or "default", "source_run": src[0],
            "source_cell": src[1], "source_episode": src[2],
            "prefix_steps": sw.get("prefix_steps"),
            "recorded_escalated": sw.get("recorded_escalated"),
            "escalated": bool(out.get("escalated")), "category": out.get("category"),
            "calls": [str(c) for c in out.get("calls") or []], "reply": out.get("reply") or "",
            "searched_nothing": out.get("searched_nothing") or 0,
            "stop_reason": foot.get("stop_reason"), "steps": budget.get("steps"),
            "tokens": budget.get("tokens"), "started_at": head.get("started_at"),
            "simulator": head.get("simulator"), "trace": text, "_swap": sw}


def collect_swap(root: Path) -> tuple[dict, Counter]:
    """Every continuation under root (data/swap/<run>/<arm>/...), by run,
    after the same checks as the traces'; and the emails replaced."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    emails: Counter = Counter()
    bad: list[str] = []
    for path in sorted(root.rglob("*.jsonl")):
        rel = path.relative_to(root).parts
        if len(rel) != 3:          # <run>/<arm>/<continuation>.jsonl only
            continue
        run = rel[0]
        text = path.read_text(encoding="utf-8")
        found = problems(text)
        if found:
            bad.append(f"{path}: {', '.join(found)}")
            continue
        text, n = EMAIL.subn("[email]", text)
        r = swap_row(run, text)
        if r is None:
            continue
        if n:
            emails[f"swap/{run}/{rel[1]}/{path.name}"] += n
        by_run[run].append(r)
    if bad:
        raise SystemExit("not exported -- remove these at the source and run again:\n  "
                         + "\n  ".join(bad))
    return by_run, emails


class _CacheOnly:
    """A translator that never calls: what the run's cache holds, or a stop."""
    reports_usage = False
    name = "cache-only"

    def __init__(self, model: str):
        self.model = model

    def chat(self, messages, tools):
        raise LookupError("a translation the run's cache does not hold")


def swap_prompts(root: Path, by_run: dict) -> tuple[dict, Counter]:
    """What each arm of each prefix showed the agent: rebuilt from the
    prefixes (the files beside the runs) and the run's translation cache as the run
    built them, no call made; checked against the digest each continuation
    records from v70, and a translated policy against the text the run wrote
    beside its traces. Anything that cannot be rebuilt stops the export."""
    from pasarbench.harness.prompts import policy_text
    from pasarbench.swap import Prefix, Translator, shown_digest, shown_messages, translate_policy
    index: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for f in sorted(root.glob("*.jsonl")):          # the prefixes files, as `prefixes` wrote them
        for line in f.read_text(encoding="utf-8").splitlines():
            d = json.loads(line) if line.strip() else None
            if isinstance(d, dict) and "pid" in d and "messages" in d:
                index[d["pid"]].append((f.name, d))
    out: dict[str, list[dict]] = defaultdict(list)
    emails: Counter = Counter()
    bad: list[str] = []
    for run, rows in sorted(by_run.items()):
        models = sorted({r["translator"] for r in rows if r["translator"]})
        if len(models) > 1:
            bad.append(f"{run}: translated by more than one model ({', '.join(models)})")
            continue
        tr = Translator(_CacheOnly(models[0] if models else "-"), root / "translations.json")
        policies: dict[tuple[str, bool], str] = {}

        def policy_for(lang: str, named: bool, run=run, tr=tr, policies=policies) -> str:
            if (lang, named) not in policies:
                text, _ = translate_policy(policy_text(named=named), lang, tr.policy)
                kept = root / run / "policy-translated" / \
                    f"{lang}.{'named' if named else 'plain'}.md"
                if kept.exists() and kept.read_text(encoding="utf-8") != text:
                    raise LookupError(f"the policy in {lang} the cache rebuilds is not the one "
                                      f"{kept} holds")
                policies[(lang, named)] = text
            return policies[(lang, named)]
        seen = set()
        for r in rows:
            if (r["pid"], r["arm"]) in seen:
                continue
            seen.add((r["pid"], r["arm"]))
            sw, where = r["_swap"], f"{run}/{r['arm']}/{r['pid']}"
            # The prefix the run continued: its file where the run names it
            # (v70), and the one whose source, task, steps and recording match.
            cands = {json.dumps(d, sort_keys=True): d for name, d in index.get(r["pid"], [])
                     if (not sw.get("prefixes") or name == sw["prefixes"])
                     and d.get("source") == sw.get("source") and d["task_id"] == r["task_id"]
                     and d["steps"] == r["prefix_steps"]
                     and d["escalated"] == r["recorded_escalated"]}
            if len(cands) != 1:
                bad.append(f"{where}: {'no' if not cands else 'more than one'} prefix of that "
                           f"id, source, task and recording in the prefixes files in {root}")
                continue
            p = Prefix.from_dict(next(iter(cands.values())))
            try:
                msgs = shown_messages(p, sw, tr, policy_for)
            except (LookupError, ValueError) as e:
                bad.append(f"{where}: {e}")
                continue
            if sw.get("shown_sha1") and shown_digest(msgs) != sw["shown_sha1"]:
                bad.append(f"{where}: rebuilt, it is not what the continuation was shown "
                           f"(its digest differs)")
                continue
            text = json.dumps(msgs, ensure_ascii=False)
            found = problems(text)
            if found:
                bad.append(f"{where}: {', '.join(found)}")
                continue
            text, n = EMAIL.subn("[email]", text)
            if n:
                emails[f"swap_prompts/{where}"] += n
            out[run].append({"run": run, "experiment": r["experiment"], "arm": r["arm"],
                             "pid": r["pid"], "task_id": r["task_id"], "language": r["language"],
                             "policy_language": r["policy_language"],
                             "tools_named": r["tools_named"], "translator": r["translator"],
                             "messages": text})
    if bad:
        raise SystemExit("the language swap not exported -- what each continuation was shown "
                         "could not be rebuilt as the run built it:\n  " + "\n  ".join(bad[:30])
                         + (f"\n  ... and {len(bad) - 30} more" if len(bad) > 30 else ""))
    return out, emails


def _repo_url() -> str | None:
    """The code's public address, from this checkout's git remote."""
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.match(r"(?:git@github\.com:|https://github\.com/)([^/]+/[^/]+?)(?:\.git)?$", url)
    return f"https://github.com/{m.group(1)}" if m else None


def _swap_card(swap: dict, emails: Counter) -> list[str]:
    """The card's section on the language swap, from what was exported."""
    rows = [r for rs in swap.values() for r in rs]
    table = []
    for run, rs in sorted(swap.items()):
        checked = ("against its digest" if all(r["_swap"].get("shown_sha1") for r in rs)
                   else "rebuilt; no digest before v70")
        for arm in sorted({r["arm"] for r in rs}):
            these = [r for r in rs if r["arm"] == arm]
            k = sum(r["escalated"] for r in these)
            table.append(f"| {run} | {these[0]['experiment'] or '-'} | "
                         f"{', '.join(sorted({str(r['agent_model']) for r in these}))} | {arm} | "
                         f"{len(these):,} | {k:,} ({k / len(these):.0%}) | {checked} |")
    present = sorted({r["experiment"] for r in rows if r["experiment"]})
    language = [x for x, there in (("1t", "1t" in present),
                                   ("1w, on other agents", "1w" in present)) if there]
    arms = ([f"only the conversation's language ({'; '.join(language)})"] if language else []) \
        + (["the policy's language too (1v)"] if "1v" in present else []) \
        + (["whether the policy names the tools (1x)"] if "1x" in present else [])
    change = (", ".join(arms[:-1]) + ", or " + arms[-1] if len(arms) > 1 else
              "".join(arms) or "the conversation")
    return [
        f"## The language swap (RUNBOOK {', '.join(present) or 'the language swap'})",
        "",
        f"{len(rows):,} continuations in {len(swap)} runs. Each holds a recorded "
        "duplicate-refund conversation fixed at the moment the agent reads a refund already "
        "paid -- the system prompt, every message, every tool payload and the world as they "
        f"were -- and lets the agent continue, in arms that change {change}, until it next "
        "speaks to the customer. The outcome is `escalated`: whether it escalated before that "
        "reply, which the policy's P10 asks of a refund already paid. There is no `passed`.",
        "",
        "| run | RUNBOOK | agent model | arm | continuations | escalated | what it was shown, "
        "checked |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *table,
        "",
        "Config `swap`, one line per continuation:",
        "",
        "| field | what it is |",
        "| --- | --- |",
        "| run, experiment | the run, and the RUNBOOK section it answers |",
        "| arm | orig, swap, back (1t, 1w); local, policy, english (1v); each of those with "
        "-plain or -named (1x) |",
        "| pid | the prefix: the recorded conversation held fixed, the same in every arm and "
        "run that continued it |",
        "| task_id, run_index, market, trap | the task, the continuation's repetition, its "
        "market and trap |",
        "| agent_model, translator | the agent continuing, and the model that translated the "
        "conversation and the policy |",
        "| recorded_language, target, language, local_language, policy_language | the "
        "language it was recorded in and swapped into; this arm's conversation; the language "
        "other than English; the policy's |",
        "| tools_named, exposure | whether this arm's policy named the tools; the tools the "
        "run showed: every one (default) or behind search |",
        "| source_run, source_cell, source_episode | the recorded conversation, in the "
        "`episodes` config |",
        "| prefix_steps, recorded_escalated | the agent's steps up to the lookup; whether the "
        "recording escalated before its next reply |",
        "| escalated, category, calls, reply, searched_nothing | escalated before its next "
        "reply, the category it gave, the tools it called, what it then said, policy searches "
        "that found nothing |",
        "| stop_reason, steps, tokens, started_at | how it ended and what it used |",
        "| trace | the continuation's whole trace: its header (the arm under `swap`), steps "
        "and footer |",
        "",
        "Config `swap_prompts`, one line per arm of each prefix in a run -- join on run, pid "
        "and arm: run, experiment, arm, pid, task_id, language, policy_language, tools_named "
        "and translator as above, and `messages`, JSON: every message the agent was shown, in "
        "order -- the system prompt with its policy, her turns and its replies in the arm's "
        "language, every tool call and payload.",
        "",
        "- The translations are a model's, qwen3.8-flash at temperature 0 behind one cache: "
        "checked for the codes, numbers, sections and emphasis each must keep, and read before "
        "the runs; not reviewed by native speakers.",
        "- `messages` is rebuilt from the prefixes and the run's own translation cache, as the "
        "run built it, with no call made. Each continuation from v70 records a digest of what "
        "it was shown, and the export checks it; for the earlier runs a translated policy is "
        "checked against the text the run wrote beside its traces.",
        "- Pair the arms by `pid` within a run. The reading fixed before each run, and its "
        "result, are in docs/RUNBOOK.md (1t, 1v, 1w, 1x) and RESULTS.md section 9. Email "
        f"addresses are replaced here too ({sum(emails.values())}).",
        "",
    ]


def card(by_run: dict, emails: Counter, repo: str | None, swap: dict | None = None,
         swap_emails: Counter | None = None) -> str:
    """The dataset card, from what was exported."""
    eps = [r for rs in by_run.values() for r in rs]
    langs = sorted({r["language"] for r in eps if r["language"]})
    hf_langs = sorted({l.split("-")[0] for l in langs} - {"sg"} | ({"en"} if "sg-en" in langs
                                                                  else set()))
    models = Counter(r["agent_model"] for r in eps)
    size = ("n<1K" if len(eps) < 1000 else "1K<n<10K" if len(eps) < 10_000
            else "10K<n<100K" if len(eps) < 100_000 else "100K<n<1M")
    code = f"[{repo}]({repo})" if repo else "the PasarBench repository"
    lines = [
        "---",
        "license: mit",
        "pretty_name: PasarBench traces",
        "language:",
        *[f"- {l}" for l in hf_langs],
        "tags:",
        *[f"- {t}" for t in ("agents", "tool-use", "customer-service", "benchmark",
                             "multilingual", "southeast-asia", "llm-evaluation")],
        "size_categories:",
        f"- {size}",
        "configs:",
        "- config_name: episodes",
        "  default: true",
        "  data_files: data/*.jsonl",
        "- config_name: runs",
        "  data_files: runs.jsonl",
        *(["- config_name: swap", "  data_files: swap/*.jsonl",
           "- config_name: swap_prompts", "  data_files: swap_prompts/*.jsonl"] if swap else []),
        "---",
        "",
        "# PasarBench traces",
        "",
        f"Every recorded episode of PasarBench, a verifiable customer-service agent benchmark "
        f"for Southeast Asian e-commerce: {len(eps):,} episodes in {len(by_run)} runs, "
        f"{len({(r['run'], r['cell']) for r in eps})} run cells, {len(langs)} language "
        f"varieties ({', '.join(langs)})"
        + (f"; and {sum(len(v) for v in swap.values()):,} continuations of the language swap "
           f"in {len(swap)} runs, below" if swap else "")
        + f". The code, the tasks and the write-up of every run "
        f"are in {code}; each run's question and result is a section of its docs/RUNBOOK.md.",
        "",
        "An episode is one conversation between an agent model and a simulated customer, "
        "with tools over a mock order database. It passes when the database ends in the "
        "right state -- the refund issued, the escalation filed, the forbidden action never "
        "taken -- not when the reply reads well.",
        "",
        "## Agent models",
        "",
        "| agent model | episodes |",
        "| --- | --- |",
        *[f"| {m or 'scripted'} | {n:,} |" for m, n in models.most_common()],
        "",
        "## Fields",
        "",
        "| field | what it is |",
        "| --- | --- |",
        "| run, cell | the run (one experiment) and its cell (one setup within it) |",
        "| episode | the trace's file name: task id and repetition |",
        "| task_id, trap, market, language | the task, the policy trap it tests, its market "
        "and the language the customer writes in |",
        "| agent_model, simulator | the agent under test and the simulated customer |",
        "| policy_mode, exposure, context | how the policy and the tools were shown, and how "
        "the conversation was kept |",
        "| closing, guardrail | the closing check and the claim guardrail, when a run had them |",
        "| passed, failures | the verdict the run recorded, and the checks it failed |",
        "| stop_reason, steps, tokens | how the episode ended and what it used |",
        "| trace | the whole trace, one JSON record a line: header, steps (each model call, "
        "its tool calls and whether each succeeded), events (customer turns, notes) and "
        "footer |",
        "",
        "## Read this before using the verdicts",
        "",
        "- `passed` is the verdict of the checker the day the run happened. Checks corrected "
        "since (docs/WHAT_FAILED.md) change some of them; `scripts/rescore.py` replays an "
        "episode's calls and scores it under today's checks.",
        "- Runs differ in setup on purpose: the tool exposure, the policy mode, the closing "
        "check, and the simulated customer's rules (its version 2 from v31 on). Compare cells "
        "only within a run, paired by task, as the runbook does.",
        "- The customers and every customer record are synthetic. Email addresses written "
        f"in the conversations are replaced with `[email]` ({sum(emails.values())} in "
        f"{len(emails)} episodes).",
        "",
        *(_swap_card(swap, swap_emails or Counter()) if swap else []),
        "## Load",
        "",
        "```python",
        "from datasets import load_dataset",
        "eps = load_dataset(\"<this dataset>\", \"episodes\", split=\"train\")",
        "import json; records = [json.loads(l) for l in eps[0][\"trace\"].splitlines()]",
        *(["swap = load_dataset(\"<this dataset>\", \"swap\", split=\"train\")",
           "shown = load_dataset(\"<this dataset>\", \"swap_prompts\", split=\"train\")",
           "messages = json.loads(shown[0][\"messages\"])"] if swap else []),
        "```",
        "",
        "License: MIT, as the code.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", help="the traces directory, e.g. traces")
    ap.add_argument("--out", default="hf-dataset", help="a new or empty directory")
    ap.add_argument("--runs", default="", help="only these runs, comma-separated")
    ap.add_argument("--swap", default="data/swap",
                    help="the language swap's runs (RUNBOOK 1t, 1v, 1w, 1x); \"\" leaves "
                         "them out")
    ap.add_argument("--dry-run", action="store_true",
                    help="check and count, write nothing")
    a = ap.parse_args(argv)

    root, out = Path(a.traces), Path(a.out)
    if not root.is_dir():
        raise SystemExit(f"no directory {root}")
    if out.exists() and any(out.iterdir()) and not a.dry_run:
        raise SystemExit(f"{out} is not empty: pick a new directory")
    by_run, emails = collect(root, [r for r in a.runs.split(",") if r] or None)
    if not by_run:
        raise SystemExit(f"no episode traces under {root}")
    n = sum(len(v) for v in by_run.values())
    print(f"{n:,} episodes in {len(by_run)} runs; no keys, tokens or home paths; "
          f"{sum(emails.values())} email addresses replaced in {len(emails)} episodes")
    swap, prompts, swap_emails = {}, {}, Counter()
    if a.swap and Path(a.swap).is_dir():
        swap, swap_emails = collect_swap(Path(a.swap))
        prompts, more = swap_prompts(Path(a.swap), swap)
        swap_emails.update(more)
        print(f"the language swap: {sum(len(v) for v in swap.values()):,} continuations in "
              f"{len(swap)} runs ({', '.join(sorted(swap))}), and what each arm of each prefix "
              f"showed the agent, {sum(len(v) for v in prompts.values()):,} rebuilt from the "
              f"runs' caches with no call made; no keys, tokens or home paths; "
              f"{sum(swap_emails.values())} email addresses replaced")
    elif a.swap:
        print(f"the language swap: no directory {a.swap}, left out")
    if a.dry_run:
        return 0

    (out / "data").mkdir(parents=True, exist_ok=True)
    manifest = []
    for run, rows in sorted(by_run.items()):
        with (out / "data" / f"{run}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n")
        for cell in sorted({r["cell"] for r in rows}):
            these = [r for r in rows if r["cell"] == cell]
            judged = [r["passed"] for r in these if r["passed"] is not None]
            starts = sorted(r["started_at"] for r in these if r["started_at"])
            manifest.append({
                "kind": "episodes", "run": run, "cell": cell, "episodes": len(these),
                "agent_models": sorted({str(r["agent_model"]) for r in these}),
                "simulators": sorted({str(r["simulator"]) for r in these}),
                "first_started": starts[0] if starts else None,
                "last_started": starts[-1] if starts else None,
                "passed_recorded": round(sum(judged) / len(judged), 4) if judged else None,
                "escalated": None, "file": f"data/{run}.jsonl"})
    if swap:
        (out / "swap").mkdir(exist_ok=True)
        (out / "swap_prompts").mkdir(exist_ok=True)
    for run, rows in sorted(swap.items()):
        with (out / "swap" / f"{run}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in SWAP_FIELDS}, ensure_ascii=False) + "\n")
        with (out / "swap_prompts" / f"{run}.jsonl").open("w", encoding="utf-8") as f:
            for r in prompts.get(run, []):
                f.write(json.dumps({k: r[k] for k in PROMPT_FIELDS}, ensure_ascii=False) + "\n")
        for arm in sorted({r["arm"] for r in rows}):
            these = [r for r in rows if r["arm"] == arm]
            starts = sorted(r["started_at"] for r in these if r["started_at"])
            manifest.append({
                "kind": "swap", "run": run, "cell": arm, "episodes": len(these),
                "agent_models": sorted({str(r["agent_model"]) for r in these}),
                "simulators": sorted({str(r["simulator"]) for r in these}),
                "first_started": starts[0] if starts else None,
                "last_started": starts[-1] if starts else None, "passed_recorded": None,
                "escalated": round(sum(r["escalated"] for r in these) / len(these), 4),
                "file": f"swap/{run}.jsonl"})
    with (out / "runs.jsonl").open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(card(by_run, emails, _repo_url(), swap, swap_emails),
                                   encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out}/: README.md, runs.jsonl ({len(manifest)} cells), "
          f"data/ ({len(by_run)} files)"
          + (f", swap/ and swap_prompts/ ({len(swap)} files each)" if swap else "")
          + f", {size / 2**20:.0f} MB")
    print(f"next: hf upload <you>/pasarbench-traces {out} . --repo-type dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
