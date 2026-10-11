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


def _repo_url() -> str | None:
    """The code's public address, from this checkout's git remote."""
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.match(r"(?:git@github\.com:|https://github\.com/)([^/]+/[^/]+?)(?:\.git)?$", url)
    return f"https://github.com/{m.group(1)}" if m else None


def card(by_run: dict, emails: Counter, repo: str | None) -> str:
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
        "---",
        "",
        "# PasarBench traces",
        "",
        f"Every recorded episode of PasarBench, a verifiable customer-service agent benchmark "
        f"for Southeast Asian e-commerce: {len(eps):,} episodes in {len(by_run)} runs, "
        f"{len({(r['run'], r['cell']) for r in eps})} run cells, {len(langs)} language "
        f"varieties ({', '.join(langs)}). The code, the tasks and the write-up of every run "
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
        "## Load",
        "",
        "```python",
        "from datasets import load_dataset",
        "eps = load_dataset(\"<this dataset>\", \"episodes\", split=\"train\")",
        "import json; records = [json.loads(l) for l in eps[0][\"trace\"].splitlines()]",
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
                "run": run, "cell": cell, "episodes": len(these),
                "agent_models": sorted({str(r["agent_model"]) for r in these}),
                "simulators": sorted({str(r["simulator"]) for r in these}),
                "first_started": starts[0] if starts else None,
                "last_started": starts[-1] if starts else None,
                "passed_recorded": round(sum(judged) / len(judged), 4) if judged else None,
                "file": f"data/{run}.jsonl"})
    with (out / "runs.jsonl").open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(card(by_run, emails, _repo_url()), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out}/: README.md, runs.jsonl ({len(manifest)} cells), "
          f"data/ ({len(by_run)} files), {size / 2**20:.0f} MB")
    print(f"next: hf upload <you>/pasarbench-traces {out} . --repo-type dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
