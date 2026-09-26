"""
Audit a tool-scaling run from its traces. No API calls, no cost.

    python scripts/audit_tool_arms.py traces/I-tools

1. VISIBILITY -- did any arm let the agent call a tool it was not shown?
   Until the loop enforced it, the model received only the visible schemas but
   ANY registered tool ran if its name was guessed. This replays what each arm
   showed at every step and counts calls outside it. An arm with guessed calls
   measured name-guessing as well as the thing it was built to measure.

2. SEARCH FAILURES -- for every failed episode in a search-N arm, and every tool
   the verifier required that the visible core did not include: did the agent
   never search for it, search and get a ranking without it, see it and not use
   it, or use it and fail on something else? Each query in the trace is replayed
   through the same deterministic ranker, so what the agent was SHOWN is exact.
   Only the "was the query aimed at the tool" step is a keyword heuristic, and
   every query is printed so you can overrule it.

REPLAY FIDELITY. The oracle and random-N definitions changed (WHAT_FAILED #18).
Each step records how many tools were shown and their schema-token cost; the
replay rebuilds the set under the current definition and the pre-fix one and
keeps whichever reproduces the recorded cost EXACTLY. An episode neither
reproduces is reported as unreconstructable and left out -- never guessed.

This classifies. It does not establish the mechanism: rule 2 of the report
still applies, and the per-episode lines at the bottom are the traces to read.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pasarbench.harness.exposure import (ARG_SOURCES, BASE_LOOKUPS,  # noqa: E402
                                         CORE_VISIBLE, _rank_tools,
                                         build_exposure, needed_tools,
                                         register_distractors, schema_tokens)
from pasarbench.sweep import ALL_SOLUTIONS, ALL_TASKS                   # noqa: E402
from pasarbench.tools import DEFAULT_TOOLS, TOOLS                       # noqa: E402

from pasarbench.harness.prompts import POLICY_PATH                     # noqa: E402

BY_ID = {t.task_id: t for t in ALL_TASKS}

# The policy the agent reads. A tool it names verbatim can be called without
# searching; a tool it only describes in prose cannot. In the first real run
# that split explained both halves of the search arm: every hidden call was to
# a tool the policy names, every never-searched failure to one it does not.
POLICY_TEXT = POLICY_PATH.read_text(encoding="utf-8")


def _named(tool: str) -> str:
    return "named in policy" if tool in POLICY_TEXT else "NOT named in policy"

# What a query aimed at a tool tends to contain. A triage aid, not a verdict:
# every query is printed next to its classification.
INTENT = {
    "escalate_to_human": ("escalat", "human", "supervisor", "specialist",
                          "manager", "hand off", "handoff", "transfer",
                          "live agent", "senior"),
    "issue_goodwill_voucher": ("voucher", "goodwill", "coupon", "compensat",
                               "gesture", "discount"),
}


def _aimed(query: str, tool: str) -> bool:
    q = query.lower()
    keys = INTENT.get(tool) or tuple(w[:5] for w in tool.split("_") if len(w) > 3)
    return any(k in q for k in keys)


# Every definition the oracle / random-N arms have had (WHAT_FAILED #18). A
# trace is replayed under each; the one that reproduces the recorded tool count
# and schema cost at EVERY step is the one that produced it.
def _legacy_needed(solution) -> set[str]:
    """v1: only what the reference solution calls."""
    return ({n for n, _ in solution} | BASE_LOOKUPS) & set(TOOLS)


def _v2_needed(solution) -> set[str]:
    """v2: plus the lookup that reveals each id the solution hard-codes."""
    need = {n for n, _ in solution} | BASE_LOOKUPS
    for _, a in solution:
        need |= {ARG_SOURCES[k] for k in a if k in ARG_SOURCES}
    return need & set(TOOLS)


RULES = {"current": needed_tools, "v2": _v2_needed, "legacy": _legacy_needed}
GLOBAL = object()   # search ranked over the whole registry before the fix


def _random_set(task, n: int, rule, pool: list[str], seed: int = 0) -> list[str]:
    needed = rule(ALL_SOLUTIONS.get(task.task_id, []))
    rng = random.Random(f"{seed}:{task.task_id}")
    rest = [t for t in pool if t not in needed]
    rng.shuffle(rest)
    return sorted(needed | set(rest[:max(0, n - len(needed))]))


class Arm:
    """Every visibility an arm could have had, for replay."""

    def __init__(self, spec: str):
        self.spec = spec
        self.kind = spec.split("-")[0] if spec != "default" else "default"
        if self.kind == "random":
            self.pool = list(DEFAULT_TOOLS) + register_distractors(300, 0)
        elif self.kind in ("all", "search"):
            self.ex = build_exposure(spec, ALL_SOLUTIONS, seed=0)

    def candidates(self, task) -> dict[str, tuple[list[str], object]]:
        """name -> (visible set at step 1, search universe)."""
        sol = ALL_SOLUTIONS.get(task.task_id, [])
        if self.kind == "default":
            return {"exact": (sorted(DEFAULT_TOOLS), None)}
        if self.kind == "oracle":
            return {k: (sorted(r(sol)), None) for k, r in RULES.items()}
        if self.kind == "random":
            n = int(self.spec.split("-")[1])
            return {k: (_random_set(task, n, r, self.pool), None)
                    for k, r in RULES.items()}
        if self.kind == "all":
            return {"exact": (sorted(self.ex.tools_for(None, None, task)), None)}
        core = [t for t in CORE_VISIBLE if t in TOOLS]
        return {"current": (core, self.ex.universe), "legacy": (core, GLOBAL)}


def _walk(steps: list[dict], base: list[str], universe, search: bool) -> dict:
    visible = list(base)
    verified, hidden_calls, queries, calls = 0, [], [], []
    through, rejected = [], []          # hidden calls that ran / were refused
    shown: dict[str, int] = {}          # tool -> best rank it was ever shown at
    for st in steps:
        if (len(visible) == st.get("n_tools")
                and schema_tokens(visible) == st.get("schema_tokens")):
            verified += 1
        pre = set(visible)
        for tc, tr in zip(st.get("tool_calls", []), st.get("tool_results", [])):
            calls.append((tc["name"], bool(tr.get("ok"))))
            if tc["name"] not in pre:
                hidden_calls.append(tc["name"])
                # Before the loop enforced visibility a guessed call RAN; after,
                # it comes back "unknown tool". The trace says which happened.
                if "unknown tool" in (tr.get("error") or ""):
                    rejected.append((tc["name"], len(calls)))
                else:
                    through.append(tc["name"])
            if (search and tc["name"] == "search_tools" and tr.get("ok")
                    and "search_tools" in pre):
                q = (tc.get("arguments") or {}).get("query", "")
                lim = (tc.get("arguments") or {}).get("limit") or 6
                queries.append(q)
                uni = None if universe is GLOBAL else universe
                for rank, (name, _) in enumerate(_rank_tools(q, lim, uni), 1):
                    shown[name] = min(rank, shown.get(name, rank))
                    if name not in visible:
                        visible.append(name)
    # A rejected guess is recovered if the agent later found that tool and used
    # it successfully -- the difference between "the name was a hint" and "the
    # name was the only route it had".
    recovered = [n for n, i in rejected
                 if any(c == n and ok for c, ok in calls[i:])]
    return {"verified": verified, "hidden_calls": hidden_calls, "queries": queries,
            "shown": shown, "calls": calls, "core": set(base),
            "through": through, "rejected": [n for n, _ in rejected],
            "recovered": recovered}


def replay(arm: Arm, recs: list[dict]) -> dict | None:
    head = next((r for r in recs if r.get("type") == "header"), {})
    task = BY_ID.get(head.get("task_id"))
    if task is None:
        return None
    steps = [r for r in recs if r.get("type") == "step"]
    foot = next((r for r in reversed(recs) if r.get("type") == "footer"), {})
    said = next((st.get("model_content") for st in reversed(steps)
                 if (st.get("model_content") or "").strip()), "")
    texts = [st["model_content"] for st in steps if (st.get("model_content") or "").strip()]

    walks = {k: _walk(steps, base, uni, arm.kind == "search")
             for k, (base, uni) in arm.candidates(task).items()}
    full = [k for k, w in walks.items() if w["verified"] == len(steps)]
    base_ep = {"task": task, "head": head, "foot": foot, "steps": len(steps),
               "said": said, "texts": texts}
    if not full:
        # No definition reproduces every step. Keep nothing from this episode:
        # hidden-call counts after a divergence would be guesses.
        best = max(walks.values(), key=lambda w: w["verified"])
        return {**base_ep, "unreconstructable": True, "verified": best["verified"]}
    w = walks[full[0]]
    return {**base_ep, **w, "how": full[0] if len(full) == 1 else "either"}


from pasarbench.claims import CLAIMS, ENGLISH, false_claims   # noqa: E402,F401


def classify(ep: dict, tool: str) -> str:
    called_ok = any(n == tool and ok for n, ok in ep["calls"])
    guessed = tool in ep["hidden_calls"]
    if called_ok:
        return "used it, failed elsewhere" + (" (guessed name)" if guessed else "")
    if tool in ep["shown"]:
        return "was shown it, did not use it"
    if any(_aimed(q, tool) for q in ep["queries"]):
        return "searched for it, ranker missed it"
    if any(any(ord(c) > 127 for c in q) for q in ep["queries"]):
        return "searched in another language (ranker is English-only)"
    return "never searched for it"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="e.g. traces/I-tools")
    ap.add_argument("--show", type=int, default=40, help="per-episode lines to print")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_dir():
        here = sorted(p.name for p in Path("traces").iterdir() if p.is_dir()) \
            if Path("traces").is_dir() else []
        raise SystemExit(
            f"no run directory {run}  (looking from {Path.cwd()})\n"
            + (f"runs here: {', '.join(here)}" if here else
               "no traces/ folder here -- run this from the repo root, the "
               "folder that contains scripts/ and traces/"))
    cells = sorted(p for p in run.iterdir() if p.is_dir())
    if not cells:
        raise SystemExit(f"no arm directories under {args.run}")

    print("== 1. visibility: calls to tools the arm did not show ==\n")
    print(f"{'cell':18s} {'episodes':>8s} {'steps verified':>15s}  {'replay':8s} "
          f"{'hidden':>6s} {'ran':>4s} {'refused':>7s} {'…found & used':>13s} "
          f"{'inflation ≤':>11s}  most common")
    search_eps: dict[str, list[dict]] = {}
    all_eps: dict[str, list[dict]] = {}
    for cell in cells:
        spec = cell.name.split("+", 1)[1] if "+" in cell.name else "default"
        arm = Arm(spec)
        eps = []
        for f in sorted(cell.glob("*.jsonl")):
            recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            ep = replay(arm, recs)
            if ep:
                ep["id"] = f.stem
                eps.append(ep)
        ok = [e for e in eps if not e.get("unreconstructable")]
        bad = len(eps) - len(ok)
        hc = Counter(n for e in ok for n in e["hidden_calls"])
        affected = sum(1 for e in ok if e["hidden_calls"])
        # Only a guess that RAN can have bought a pass. A refused one cannot.
        passed = sum(1 for e in ok if e["through"] and e["foot"].get("passed"))
        ran = sum(len(e["through"]) for e in ok)
        refused = sum(len(e["rejected"]) for e in ok)
        rec = sum(len(e["recovered"]) for e in ok)
        only = sorted({e["how"] for e in ok} - {"either"})
        how = "/".join(only) or ("exact" if ok else "-")
        top = ", ".join(f"{n} x{c}" for n, c in hc.most_common(3))
        if hc:
            named = sum(c for n, c in hc.items() if n in POLICY_TEXT)
            top += f"  [{named}/{sum(hc.values())} to tools the policy names]"
        v = f"{sum(e['verified'] for e in eps)}/{sum(e['steps'] for e in eps)}"
        print(f"{cell.name:18s} {len(eps):8d} {v:>15s}  {how:8s} "
              f"{sum(hc.values()):6d} {ran:4d} {refused:7d} {rec:13d} "
              f"{passed:11d}  {top}")
        all_eps[cell.name] = ok
        if bad:
            print(f"{'':18s} !! {bad} episode(s) reproduced by no known definition "
                  f"at every step; left out rather than guessed")
        if arm.kind == "search":
            search_eps[cell.name] = ok
    print("\nhidden: calls to a tool the arm had not shown. ran: the old loop "
          "executed them.\n  refused: the enforced loop returned 'unknown tool'. "
          "…found & used: refused,\n  then found by search and used successfully "
          "later in the episode.\ninflation ≤: episodes where a hidden call RAN and "
          "the episode passed -- the\n  most guessed names can have added to the "
          "pass rate. Zero on an enforced run.")
    print("\nsteps verified: steps whose recorded tool count AND schema cost the "
          "replay\n  reproduces exactly. Anything short of all of them means the "
          "counts are not\n  trustworthy for the episodes that diverged, which "
          "are excluded.\nreplay: which arm definition reproduced the trace -- "
          "legacy (v1), v2, current,\n  or exact for arms that never changed. "
          "See WHAT_FAILED #18.")

    for cell, eps in search_eps.items():
        failed = [e for e in eps if not e["foot"].get("passed")]
        print(f"\n== 2. {cell}: {len(failed)} failed of {len(eps)} ==\n")
        rows, per_lang, lines = defaultdict(Counter), defaultdict(Counter), []
        for e in failed:
            req = {a.tool for a in e["task"].checks.required_actions}
            for tool in sorted(req - e["core"]):
                why = classify(e, tool)
                rows[tool][why] += 1
                per_lang[e["head"].get("language", "?")][why] += 1
                ff = (e["foot"].get("failures") or [""])[0]
                rank = (f" (best rank {e['shown'][tool]})"
                        if why.startswith("was shown") else "")
                said = " ".join((e.get("said") or "").split())[:170]
                lines.append(f"  {e['id']:24s} {e['task'].trap:34s} "
                             f"{e['head'].get('language', '?'):6s} {tool}: {why}{rank}\n"
                             f"      queries: {e['queries'] or '(none)'}"
                             + (f"\n      verifier: {str(ff)[:110]}"
                                if why.startswith("used") else "")
                             # Did it tell the customer it acted, without acting?
                             + (f"\n      last said: {said!r}"
                                if why == "never searched for it" and said else ""))
        if not rows:
            print("  no failed episode needed a tool outside the visible core")
            continue
        kinds = sorted({k for c in rows.values() for k in c})
        for tool, c in sorted(rows.items()):
            print(f"  {tool}   ({_named(tool)})")
            for k in kinds:
                if c[k]:
                    print(f"      {c[k]:3d}  {k}")
        print("\n  by language:")
        for lang, c in sorted(per_lang.items()):
            print(f"      {lang:6s} " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
        print(f"\n  per episode (first {args.show}) -- these are the traces to read:")
        print("\n".join(lines[:args.show]))

    print("\n== 3. said it, didn't do it: failed episodes whose text claims a "
          "required action\n   the database never saw (English phrasings; other "
          "languages counted, not read) ==\n")
    for cell, eps in all_eps.items():
        failed = [e for e in eps if not e["foot"].get("passed")]
        eng = [e for e in failed if e["head"].get("language") in ENGLISH]
        hits = [(e, c) for e in eng for c in false_claims(e)]
        print(f"  {cell:18s} failed {len(failed):3d}  read {len(eng):3d}  "
              f"claimed-not-done {len(hits):3d}")
        for e, (tool, quote) in hits:
            print(f"      {e['id']:24s} {tool}: \"…{quote}…\"")


if __name__ == "__main__":
    main()
