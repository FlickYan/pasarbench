"""
Multilingual diagnosis.

"Non-English scores lower" is not a finding. Everyone reports it, it surprises
nobody, and it suggests no action.

The finding is the DECOMPOSITION. Because locale twins in this suite share an
identical world and byte-identical checks, the entire gap between `en` and any
other language is caused by the surface form of the customer's words -- and
that gap can be attributed across mechanisms that are each separately
measurable from the traces:

  1. MALFORMED TOOL ARGUMENTS
     Models emit broken JSON far more often when extracting values out of
     non-Latin script. The harness already turns these into a distinct
     recoverable tool error, so they are countable rather than inferred.

  2. RETRIEVAL MISS
     `search_policy` and `search_tools` are keyword matchers over ENGLISH text.
     An agent that searches in the customer's language scores zero hits. This
     is a real property of the system, not a bug to hide, and it is the
     mechanism people most often mistake for "the model is worse at Thai".

  3. LANGUAGE DRIFT
     The agent replies in English to a Thai customer. Invisible to every state
     check -- the task can pass -- which is exactly why the judge's
     `language_match` criterion exists.

  4. TOKENISATION INFLATION
     Thai and Vietnamese tokenise far worse than English. The same conversation
     costs more tokens, so budgets bind earlier and episodes get truncated. A
     gap caused by a budget ceiling is a HARNESS artefact, not a model
     weakness, and conflating the two is the single most common error in
     multilingual agent evals.

Mechanism 4 is the one to check first, because if it dominates, the fix is to
raise the budget and re-run, and every other conclusion drawn before that is
wrong.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

LATIN_LANGS = {"en", "sg-en", "ms", "id"}
NON_LATIN_LANGS = {"th", "vi"}     # vi is Latin-script but heavily diacritic;
                                   # tokenisers treat it much like non-Latin


def load_episodes(run_dir: str | Path) -> list[dict[str, Any]]:
    """Flatten trace files into per-episode records with the signals we need."""
    out = []
    for f in sorted(Path(run_dir).glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        head = next((r for r in recs if r.get("type") == "header"), {})
        foot = next((r for r in reversed(recs) if r.get("type") == "footer"), {})
        steps = [r for r in recs if r.get("type") == "step"]
        events = [r for r in recs if r.get("type") == "event"]

        malformed = search_calls = search_misses = 0
        prose_chars = 0
        for st in steps:
            prose_chars += len(st.get("model_content") or "")
            for tr in st.get("tool_results", []):
                err = (tr.get("error") or "").lower()
                if "not valid json" in err or "bad arguments" in err:
                    malformed += 1
                if tr["name"] in ("search_policy", "search_tools"):
                    search_calls += 1
                    if not tr.get("ok") or tr.get("empty"):
                        search_misses += 1

        user_chars = sum(len(e.get("text") or "") for e in events
                         if e.get("kind") == "user_turn")
        out.append({
            "transcript_id": f.stem,
            "task_id": head.get("task_id", f.stem.split("__")[0]),
            "language": head.get("language", "?"),
            "market": head.get("market", "?"),
            "trap": head.get("trap", "?"),
            "passed": bool(foot.get("passed")),
            "stop_reason": foot.get("stop_reason", "?"),
            "tokens": (foot.get("budget") or {}).get("tokens", 0),
            "steps": (foot.get("budget") or {}).get("steps", 0),
            "malformed_args": malformed,
            "search_calls": search_calls,
            "search_misses": search_misses,
            "prose_chars": prose_chars,
            "user_chars": user_chars,
            "first_failure": (foot.get("failures") or [None])[0],
            "simulator": head.get("simulator", "?"),
        })
    return out


def _rate(items: list[dict], pred) -> float:
    return round(sum(1 for x in items if pred(x)) / len(items), 4) if items else 0.0


def by_language(episodes: list[dict[str, Any]],
                judge_labels: list[dict[str, Any]] | None = None
                ) -> dict[str, dict[str, Any]]:
    """Per-language signal table. The columns ARE the candidate mechanisms."""
    lang_labels: dict[str, list[bool]] = defaultdict(list)
    if judge_labels:
        by_id = {r["transcript_id"]: r for r in judge_labels}
        for e in episodes:
            r = by_id.get(e["transcript_id"])
            if r and r.get("labels", {}).get("language_match") is not None:
                lang_labels[e["language"]].append(bool(r["labels"]["language_match"]))

    groups: dict[str, list[dict]] = defaultdict(list)
    for e in episodes:
        groups[e["language"]].append(e)

    out = {}
    for lang, items in sorted(groups.items()):
        n = len(items)
        tok = [x["tokens"] for x in items]
        chars = [max(x["user_chars"] + x["prose_chars"], 1) for x in items]
        drift = lang_labels.get(lang)
        out[lang] = {
            "n": n,
            "pass_rate": _rate(items, lambda x: x["passed"]),
            "malformed_arg_rate": _rate(items, lambda x: x["malformed_args"] > 0),
            "search_miss_rate": (
                round(sum(x["search_misses"] for x in items)
                      / max(sum(x["search_calls"] for x in items), 1), 4)),
            "language_drift_rate": (
                round(1 - sum(drift) / len(drift), 4) if drift else None),
            "budget_exhaustion_rate": _rate(
                items, lambda x: str(x["stop_reason"]).startswith("max_")),
            "mean_tokens": round(sum(tok) / n, 1),
            "tokens_per_char": round(sum(tok) / sum(chars), 4),
        }
    return out


def paired_language_gap(episodes: list[dict[str, Any]], baseline: str = "en"
                        ) -> dict[str, Any]:
    """THE controlled language comparison. Only compares locale TWINS.

    `by_language` aggregates every episode of a language together, which is
    confounded: with a stratified sample, different languages draw different
    traps, so a Thai rate and a Vietnamese rate can be computed over entirely
    different tasks. The whole "identical world, identical checks" guarantee
    applies to a task and ITS OWN twin -- nowhere else.

    This pairs `HRWR-ID` against `HRWR-ID.id`, compares only within pairs, and
    reports how many pairs it actually found. If that count is small, the
    number means nothing and the function says so instead of printing a rate.
    """
    by_task: dict[str, list[bool]] = defaultdict(list)
    lang_of: dict[str, str] = {}
    base_of: dict[str, str] = {}
    for e in episodes:
        tid = e.get("task_id") or e["transcript_id"]
        by_task[tid].append(e["passed"])
        lang_of[tid] = e["language"]
        base_of[tid] = tid.split(".")[0]

    rate = {t: sum(v) / len(v) for t, v in by_task.items()}
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for tid, base in base_of.items():
        groups[base][lang_of[tid]] = tid

    out: dict[str, Any] = {}
    for lang in sorted({l for l in lang_of.values() if l != baseline}):
        pairs = [(g[baseline], g[lang]) for g in groups.values()
                 if baseline in g and lang in g]
        if not pairs:
            out[lang] = {"pairs": 0,
                         "note": "no twins present -- run the FULL suite, not a "
                                 "stratified sample, or this language cannot be "
                                 "compared at all"}
            continue
        deltas = [rate[b] - rate[o] for b, o in pairs]
        # fsum: plain sum() of floats changed in Python 3.12, and a mean that
        # lands on a rounding boundary (0.9125) printed 0.912 on one machine
        # and 0.913 on another. Exact summation makes the report the same file
        # wherever it is regenerated.
        out[lang] = {
            "pairs": len(pairs),
            "baseline_rate": round(math.fsum(rate[b] for b, _ in pairs) / len(pairs), 3),
            "language_rate": round(math.fsum(rate[o] for _, o in pairs) / len(pairs), 3),
            "paired_gap": round(math.fsum(deltas) / len(deltas), 3),
            "pairs_where_worse": sum(1 for d in deltas if d > 0),
            "pairs_where_better": sum(1 for d in deltas if d < 0),
            "reliable": len(pairs) >= 10,
            "note": ("" if len(pairs) >= 10 else
                     f"only {len(pairs)} pair(s) -- too few to read as a rate; "
                     f"treat as anecdote until the full suite is run"),
        }
    return {"baseline": baseline, "languages": out,
            "warning": ("Unpaired per-language rates compare DIFFERENT tasks "
                        "and are confounded by which traps landed in which "
                        "language. Only this paired table is controlled.")}


def attribute_gap(table: dict[str, dict[str, Any]], baseline: str = "en"
                  ) -> dict[str, Any]:
    """For each language, the gap against `en` and which mechanisms co-move.

    THIS DOES NOT ESTABLISH CAUSATION and says so. Co-movement narrows the
    search to the traces worth reading; the causal claim comes from reading
    them, and from the ablations suggested below. Presenting a correlation
    table as an attribution is the mistake this docstring exists to prevent.
    """
    base = table.get(baseline)
    if not base:
        raise ValueError(f"no baseline language {baseline!r}")

    rows = {}
    for lang, m in table.items():
        if lang == baseline:
            continue
        gap = round(base["pass_rate"] - m["pass_rate"], 4)
        signals = {
            "malformed_args": round(m["malformed_arg_rate"] - base["malformed_arg_rate"], 4),
            "search_miss": round(m["search_miss_rate"] - base["search_miss_rate"], 4),
            "language_drift": (round(m["language_drift_rate"] - (base["language_drift_rate"] or 0), 4)
                               if m["language_drift_rate"] is not None else None),
            "budget_exhaustion": round(m["budget_exhaustion_rate"] - base["budget_exhaustion_rate"], 4),
            "token_inflation": (round(m["tokens_per_char"] / base["tokens_per_char"], 3)
                                if base["tokens_per_char"] else None),
        }
        ranked = sorted(((k, v) for k, v in signals.items()
                         if v is not None and k != "token_inflation" and v > 0.02),
                        key=lambda kv: -kv[1])
        rows[lang] = {
            "pass_gap": gap,
            "signals": signals,
            "ranked_mechanisms": [k for k, _ in ranked],
            "next_step": _next_step(gap, signals),
        }
    return {"baseline": baseline, "languages": rows,
            "caveat": ("These are CO-MOVEMENTS, not causal attributions. Use them "
                       "to choose which traces to read and which ablation to run; "
                       "do not present this table as an explanation on its own.")}


def _next_step(gap: float, s: dict[str, Any]) -> str:
    if gap <= 0.02:
        return "no meaningful gap to explain"
    if s["budget_exhaustion"] > 0.05:
        return ("CHECK THIS FIRST: episodes are hitting the budget ceiling more "
                "often than in English. That is a harness artefact, not a model "
                "weakness. Raise max_steps and max_tokens, re-run, and discard "
                "any conclusion drawn before you do")
    if (s.get("token_inflation") or 1) > 1.4:
        return ("tokenisation inflation above 1.4x. Same conversation, far more "
                "tokens. Re-run with a proportionally larger budget before "
                "attributing anything to the model")
    if s["search_miss"] > 0.15:
        return ("retrieval is missing. `search_policy` and `search_tools` are "
                "English keyword matchers -- the agent is likely querying in the "
                "customer's language. Test the fix: translate the query to "
                "English before searching, or index the policy multilingually")
    if s["malformed_args"] > 0.05:
        return ("malformed tool arguments. The model is failing to extract values "
                "cleanly out of non-Latin script. Check whether the failures "
                "cluster on specific argument types (ids, amounts, dates)")
    if (s.get("language_drift") or 0) > 0.1:
        return ("the agent is replying in the wrong language. Invisible to every "
                "state check, so this gap only shows up in the judge -- and it is "
                "a customer-facing failure even when the task passes")
    return ("no single mechanism dominates. Read the traces for the tasks that "
            "pass in English and fail here; the difference is legible per-task "
            "because the world and checks are identical")


def report(episodes: list[dict[str, Any]],
           judge_labels: list[dict[str, Any]] | None = None,
           baseline: str = "en") -> str:
    table = by_language(episodes, judge_labels)
    attr = attribute_gap(table, baseline)

    cols = ["n", "pass_rate", "malformed_arg_rate", "search_miss_rate",
            "language_drift_rate", "budget_exhaustion_rate", "tokens_per_char"]
    out = ["## Multilingual diagnosis\n",
           "| lang | " + " | ".join(c.replace("_rate", "").replace("_", " ")
                                    for c in cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for lang, m in table.items():
        cells = ["-" if m[c] is None else
                 (f"{m[c]}" if c == "n" else f"{m[c]:.3f}") for c in cols]
        out.append(f"| `{lang}` | " + " | ".join(cells) + " |")

    paired = paired_language_gap(episodes, baseline)
    out.append("\n### Paired comparison (the controlled one)\n")
    # "enough pairs" is not "significant": the sign test over the twin pairs
    # that differ is what says whether a gap is more than noise.
    out.append("| lang | pairs | en rate | this rate | paired gap | pairs worse / better "
               "| sign test p |")
    out.append("|---|---|---|---|---|---|---|")
    for lang, r in paired["languages"].items():
        if not r["pairs"]:
            out.append(f"| `{lang}` | 0 | - | - | - | no twins | - |")
            continue
        p = _sign_p(r["pairs_where_worse"], r["pairs_where_better"])
        few = "" if r["reliable"] else " (too few pairs)"
        out.append(f"| `{lang}` | {r['pairs']} | {r['baseline_rate']:.3f} | "
                   f"{r['language_rate']:.3f} | {r['paired_gap']:+.3f} | "
                   f"{r['pairs_where_worse']} / {r['pairs_where_better']} | "
                   f"{'**' if p < 0.05 else ''}{p:.3f}{'**' if p < 0.05 else ''}{few} |")
    out.append(f"\n> {paired['warning']}")

    out.append("\n### Gap attribution\n")
    # Only a gap the paired test resolves has a mechanism to look for. The
    # unpaired gap below 0.02 was the old bar, and it sent readers hunting for
    # the cause of a 2-point difference the sign test calls noise.
    resolved = {lang for lang, r in paired["languages"].items()
                if r["pairs"] and r["paired_gap"] > 0
                and _sign_p(r["pairs_where_worse"], r["pairs_where_better"]) < 0.05}
    shown = 0
    for lang, row in attr["languages"].items():
        if lang not in resolved:
            continue
        shown += 1
        mechs = ", ".join(f"`{m}`" for m in row["ranked_mechanisms"]) or "none isolated"
        infl = row["signals"].get("token_inflation")
        out.append(f"**`{lang}`** -- paired gap {paired['languages'][lang]['paired_gap']:+.3f} "
                   f"vs `{baseline}`; co-moving: {mechs}"
                   + (f"; tokenisation {infl}x" if infl else ""))
        out.append(f"  - {row['next_step']}\n")
    if not shown:
        if not any(r["pairs"] for r in paired["languages"].values()):
            out.append("No locale twins in this run, so no language gap can be "
                       "measured, let alone attributed. Run the full suite.")
        else:
            out.append(f"No language does worse than `{baseline}` by the paired "
                       f"test (sign test p < 0.05), so there is no gap to "
                       f"attribute. A mechanism for a gap that is not there is "
                       f"not a finding.")
        return "\n".join(out)

    out.append("> " + attr["caveat"])
    out.append("\nLocale twins share an identical world and byte-identical "
               "checks, so the gap itself is attributable to language. The "
               "MECHANISM still has to be established by reading traces.")
    return "\n".join(out)


# --------------------------------------------------------------------------

def _arm_episodes(row: dict[str, Any], run_dir: str | Path | None
                  ) -> list[dict[str, Any]]:
    """Episodes for one arm, located by CELL.

    Summaries written before the sweep fix record trace_dir as run/strategy,
    which with --exposure is the same non-existent directory for every arm.
    So the cell name decides, and a trace_dir that disagrees with it is ignored.
    """
    cell = row.get("cell") or row.get("strategy", "")
    cands = []
    if run_dir:
        cands.append(Path(run_dir) / cell)
    td = row.get("trace_dir")
    if td and Path(td).name == cell:
        cands.append(Path(td))
    for c in cands:
        if c.is_dir():
            return load_episodes(c)
    return []


def _sign_p(worse: int, better: int) -> float:
    """Exact two-sided sign test on the tasks that differ."""
    from math import comb
    n = worse + better
    if n == 0:
        return 1.0
    k = min(worse, better)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_episodes(ea: list[dict[str, Any]], eb: list[dict[str, Any]],
                    metric: str = "passed") -> dict[str, Any] | None:
    """Two sets of episodes over shared tasks: per-task mean difference and an
    exact sign test over the tasks that differ. The episode sets may come from
    different runs -- that is how the noise floor is measured."""
    ra, rb = defaultdict(list), defaultdict(list)
    for e in ea:
        ra[e["task_id"]].append(e[metric])
    for e in eb:
        rb[e["task_id"]].append(e[metric])
    ra = {k: sum(v) / len(v) for k, v in ra.items()}
    rb = {k: sum(v) / len(v) for k, v in rb.items()}
    keys = sorted(set(ra) & set(rb))
    if len(keys) < 5:
        return None
    diffs = [rb[k] - ra[k] for k in keys]
    worse = sum(1 for x in diffs if x < -1e-9)
    better = sum(1 for x in diffs if x > 1e-9)
    p = _sign_p(worse, better)
    return {"unit": "task", "n": len(keys), "diff": sum(diffs) / len(diffs),
            "worse": worse, "better": better, "p": p, "resolved": p < 0.05}


def paired_arm_gap(a: dict[str, Any], b: dict[str, Any],
                   run_dir: str | Path | None = None,
                   metric: str = "passed") -> dict[str, Any] | None:
    """Arm B against arm A on the SAME tasks: mean per-task difference, and an
    exact sign test over the tasks where the two arms differ.

    The unit is the task. Each task runs k seeds in every arm, and seeds of one
    task are not independent draws of the suite -- treating 96 episodes as 96
    samples overstates precision roughly k-fold. A bootstrap is also wrong
    here: when a handful of tasks all differ in one direction, a percentile
    interval can exclude zero while the sign test on those same tasks is
    nowhere near significance. Falls back to per-trap rates without traces,
    and says so. `metric` is any per-episode field: "passed" (the default) or
    "tokens", for a cost comparison with the same pairing.
    """
    ea, eb = _arm_episodes(a, run_dir), _arm_episodes(b, run_dir)
    if ea and eb:
        return paired_episodes(ea, eb, metric)
    elif metric == "passed":
        unit = "trap"
        ra, rb = a.get("per_trap") or {}, b.get("per_trap") or {}
    else:
        return None                   # no per-trap fallback for other metrics
    keys = sorted(set(ra) & set(rb))
    if len(keys) < 5:
        return None
    diffs = [rb[k] - ra[k] for k in keys]
    worse = sum(1 for d in diffs if d < -1e-9)
    better = sum(1 for d in diffs if d > 1e-9)
    p = _sign_p(worse, better)
    return {"unit": unit, "n": len(keys), "diff": sum(diffs) / len(diffs),
            "worse": worse, "better": better, "p": p, "resolved": p < 0.05}


def _gap_text(g: dict[str, Any] | None) -> str:
    if not g:
        return "not testable (too few shared tasks)"
    return (f"{g['diff']:+.3f}; {g['worse']} {g['unit']}s worse, {g['better']} "
            f"better, sign test p={g['p']:.3f} over {g['n']} {g['unit']}s")


BUDGET_STOPS = ("max_tokens", "max_steps", "max_tool_calls", "max_wall", "max_turns")


def tool_scaling_report(rows: Iterable[dict[str, Any]],
                        run_dir: str | Path | None = None) -> str:
    """Rows: {exposure, cell, pass^1, mean_tokens, mean_schema_tokens, n_tools,
    stop_reasons, per_trap}. `run_dir` lets the report open each arm's traces.

    The reading is in the CONTRASTS, not the trend, and every contrast is a
    paired sign test over tasks before it is read at all:
      oracle vs all-20      cost of the unneeded real tools, no distractors
      random-N vs all-N     same count and schema cost, different COMPOSITION
                            -> random tracks all-N  => the NUMBER of tools
                               (count, length, dilution -- not separable here)
                            -> random tracks oracle => the unneeded REAL tools
      search-N vs all-20    accuracy AND total tokens, not per-call schema

    Two things are checked BEFORE any contrast is read, because each silently
    produced a wrong reading once:
      1. The oracle must actually be a ceiling. If the real registry beats it,
         the arm hides something a real agent needs, and contrasts against it
         are meaningless -- the earlier version printed "no degradation"
         because every arm beat a broken oracle.
      2. Budget stops are cost, not choice. At 100 tools every call carries
         ~10k schema tokens and episodes run into the per-episode token budget;
         those count as failures by design, but they must not be read as the
         agent picking the wrong tool.
    """
    from .harness.types import Budget

    rows = list(rows)
    eps = {r["exposure"]: _arm_episodes(r, run_dir) for r in rows}
    out = ["## Tool scaling\n",
           "| exposure | tools | schema tokens | total tokens | pass^1 | "
           "budget stops | pass^1, finished only |",
           "|---|---|---|---|---|---|---|"]
    finished: dict[str, tuple[int, int] | None] = {}
    for r in rows:
        stops = r.get("stop_reasons") or {}
        n = sum(stops.values())
        hit = sum(v for k, v in stops.items() if k in BUDGET_STOPS)
        e = eps.get(r["exposure"]) or []
        done = [x for x in e if x["stop_reason"] not in BUDGET_STOPS]
        finished[r["exposure"]] = ((sum(x["passed"] for x in done), len(done))
                                   if done else None)
        fin = (f"{finished[r['exposure']][0] / finished[r['exposure']][1]:.3f} "
               f"({finished[r['exposure']][1]})" if finished[r["exposure"]] else "-")
        out.append(f"| `{r['exposure']}` | {r.get('n_tools', '-')} | "
                   f"{r.get('mean_schema_tokens', 0):,.0f} | "
                   f"{r.get('mean_tokens', 0):,.0f} | {r.get('pass^1', 0):.3f} | "
                   f"{f'{hit}/{n}' if n else '-'} | {fin} |")

    idx = {r["exposure"]: r for r in rows}
    out.append("\n### Reading\n")

    # 1. Is the ceiling a ceiling?
    orc, a20 = idx.get("oracle"), idx.get("all-20")
    ceiling_ok = True
    if orc and a20 and orc["pass^1"] < a20["pass^1"] - 0.05:
        ceiling_ok = False
        zeros = sorted(t for t, v in (orc.get("per_trap") or {}).items()
                       if v == 0 and (a20.get("per_trap") or {}).get(t, 0) >= 0.5)
        out.append(f"- **The oracle arm is not a ceiling.** It scores "
                   f"{orc['pass^1']:.3f}, below `all-20` at {a20['pass^1']:.3f}. "
                   f"An arm the full registry beats is hiding something a real "
                   f"agent needs, so every contrast against it — including the "
                   f"random-N control, which shares its guarantee — is withheld.")
        if zeros:
            out.append(f"  Traps at exactly 0.00 under oracle that `all-20` "
                       f"passes: {', '.join('`' + z + '`' for z in zeros)}. Exact "
                       f"zeros across seeds are structural: look for a lookup "
                       f"tool the arm does not expose.")

    # 2. Budget stops are cost, not choice.
    for r in rows:
        cap = r.get("token_budget") or Budget().max_tokens
        stops = r.get("stop_reasons") or {}
        hit = sum(v for k, v in stops.items() if k in BUDGET_STOPS)
        if not hit:
            continue
        n = sum(stops.values())
        line = (f"- `{r['exposure']}`: {hit} of {n} episodes stopped on the "
                f"per-episode budget (token cap {cap:,}). These fail by "
                f"design and measure COST — {r.get('mean_schema_tokens', 0):,.0f} "
                f"schema tokens ride on every call — not tool choice.")
        f, f20 = finished.get(r["exposure"]), finished.get("all-20")
        if f and f20 and r["exposure"] != "all-20":
            line += (f" On episodes that finished: {f[0] / f[1]:.3f} ({f[1]}) "
                     f"vs `all-20` {f20[0] / f20[1]:.3f} ({f20[1]}) — an UPPER "
                     f"bound, since the episodes cut off are the long, harder "
                     f"ones. For a clean read of tool choice, re-run with a "
                     f"`--token-budget` that does not bind.")
        elif not f:
            line += " Finished-only pass rate needs the traces (pass run_dir)."
        out.append(line)

    # 3. Contrasts at N tools. Nothing is attributed until the loss itself is
    #    shown to be more than noise.
    #
    #    all-N and random-N have the same number of tools and the same schema
    #    cost; they differ in COMPOSITION. all-N holds every unneeded real
    #    write tool (plausible wrong actions); random-N mostly swaps those for
    #    distractors. So:
    #      random-N tracks all-N   -> the loss comes with the NUMBER of tools
    #                                 (count, context length, dilution -- these
    #                                 move together here and are not separable)
    #      random-N tracks oracle  -> the loss comes from the unneeded REAL
    #                                 tools: choosing among plausible actions
    #    An earlier version mapped these the other way round (WHAT_FAILED #21).
    if ceiling_ok and orc:
        for n in (50, 100, 300):
            a, rnd = idx.get(f"all-{n}"), idx.get(f"random-{n}")
            if not (a and rnd):
                continue
            g_all = paired_arm_gap(orc, a, run_dir)
            g_rnd = paired_arm_gap(orc, rnd, run_dir)
            g_mix = paired_arm_gap(rnd, a, run_dir)
            ratio = (a.get("mean_tokens", 0) / orc["mean_tokens"]
                     if orc.get("mean_tokens") else None)
            vs20 = (f" and {a20['mean_tokens']:,.0f} for all-20 "
                    f"({a.get('mean_tokens', 0) / a20['mean_tokens']:.1f}x — what "
                    f"the distractors alone cost)"
                    if a20 and a20.get("mean_tokens") else "")
            cost = (f" Cost is not in doubt: {a.get('mean_tokens', 0):,.0f} tokens "
                    f"per episode against {orc['mean_tokens']:,.0f} for oracle "
                    f"({ratio:.1f}x){vs20}.") if ratio else ""
            if not (g_all and g_all["resolved"] and g_all["diff"] < 0):
                out.append(f"- **INCONCLUSIVE at {n} tools** — no accuracy loss "
                           f"distinguishable from noise. `all-{n}` vs oracle: "
                           f"{_gap_text(g_all)}. `random-{n}` vs oracle: "
                           f"{_gap_text(g_rnd)}.{cost}")
            elif g_mix and g_mix["resolved"] and g_mix["diff"] < 0:
                out.append(f"- at {n} tools the loss comes from the unneeded REAL "
                           f"tools: `random-{n}` recovers it at the same count "
                           f"({_gap_text(g_mix)}).{cost}")
            elif g_rnd and g_rnd["resolved"] and g_rnd["diff"] < 0:
                out.append(f"- at {n} tools the loss comes with the NUMBER of "
                           f"tools: it persists in `random-{n}`, which swaps most "
                           f"unneeded real tools for distractors "
                           f"({_gap_text(g_rnd)}).{cost}")
            else:
                out.append(f"- at {n} tools the loss is real ({_gap_text(g_all)}) "
                           f"but its source is unresolved at this sample size."
                           f"{cost}")

            # Per trap before aggregate: a loss shared by BOTH N-tool arms on
            # the same trap is a lead worth reading even when the aggregate is
            # inconclusive -- and only a lead, at six episodes a trap.
            op = orc.get("per_trap") or {}
            # 0.3, not 0.33: rates are rounded to 2dp, and 1.00 - 0.67 is
            # 0.3299... in float -- a 2-of-6 loss would silently not count.
            conc = sorted(t for t, v in op.items()
                          if v - (a.get("per_trap") or {}).get(t, v) >= 0.3
                          and v - (rnd.get("per_trap") or {}).get(t, v) >= 0.3)
            if conc:
                out.append("  Both " + f"{n}-tool arms lose on: " + ", ".join(
                    f"`{t}` (oracle {op[t]:.2f}, all {a['per_trap'][t]:.2f}, "
                    f"random {rnd['per_trap'][t]:.2f})" for t in conc)
                    + ". A lead to read in the traces, not a result.")

    # 4. Retrieval, against the real registry: accuracy AND total cost.
    for r in rows:
        if not r["exposure"].startswith("search-"):
            continue
        base = idx.get(f"all-{r['exposure'].split('-')[1]}") or a20
        if not base:
            continue
        g = paired_arm_gap(base, r, run_dir)
        tok, btok = r.get("mean_tokens", 0), base.get("mean_tokens", 0)
        line = (f"- `{r['exposure']}` vs `{base['exposure']}`: {_gap_text(g)}. "
                f"Total tokens per episode {tok:,.0f} vs {btok:,.0f}"
                + (f" ({(tok / btok - 1):+.0%})" if btok else "")
                + f", steps {r.get('mean_steps', 0):.1f} vs "
                  f"{base.get('mean_steps', 0):.1f}, schema tokens per call "
                  f"{r.get('mean_schema_tokens', 0):,.0f} vs "
                  f"{base.get('mean_schema_tokens', 0):,.0f}.")
        # Per-call schema savings are not the cost: every extra search is a
        # round trip that re-sends the whole conversation.
        if g and g["resolved"] and g["diff"] < 0 and btok and tok > btok:
            line += (" **Dominated at this registry size: less accurate AND more "
                     "expensive in total** — the per-call schema saving is spent "
                     "on the extra round trips.")
        out.append(line)
        lost = sorted((t, base["per_trap"].get(t, 0) - v)
                      for t, v in (r.get("per_trap") or {}).items()
                      # per_trap is rounded to 2dp: 0.83 - 0.33 is 0.4999... in float
                      if base.get("per_trap") and base["per_trap"].get(t, 0) - v >= 0.5 - 1e-6)
        if lost:
            out.append(f"  Loses ≥0.5 on: {', '.join(f'`{t}`' for t, _ in lost)}. "
                       f"Before naming a cause, read those traces: did the agent "
                       f"search for the tool and the ranker miss it, or never "
                       f"search at all? `python scripts/audit_tool_arms.py "
                       f"traces/<run>` separates the two, from the traces alone.")
    return "\n".join(out)
