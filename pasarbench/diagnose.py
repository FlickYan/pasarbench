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

    out.append("\n### Gap attribution\n")
    for lang, row in attr["languages"].items():
        if row["pass_gap"] <= 0.02:
            continue
        mechs = ", ".join(f"`{m}`" for m in row["ranked_mechanisms"]) or "none isolated"
        infl = row["signals"].get("token_inflation")
        out.append(f"**`{lang}`** -- gap {row['pass_gap']:+.3f} vs `{baseline}`; "
                   f"co-moving: {mechs}"
                   + (f"; tokenisation {infl}x" if infl else ""))
        out.append(f"  - {row['next_step']}\n")

    out.append("> " + attr["caveat"])
    out.append("\nLocale twins share an identical world and byte-identical "
               "checks, so the gap itself is attributable to language. The "
               "MECHANISM still has to be established by reading traces.")
    return "\n".join(out)


# --------------------------------------------------------------------------

def tool_scaling_report(rows: Iterable[dict[str, Any]]) -> str:
    """Rows: {exposure, pass^1, mean_tokens, mean_schema_tokens, n_tools}.

    The reading is in the CONTRASTS, not the trend:
      oracle vs all-20      cost of the real registry, no distractors
      all-N  vs random-N    same token cost, guaranteed reachability
                            -> tracks all-N   => dilution/token cost
                            -> tracks oracle  => selection difficulty
      search-N vs all-N     does retrieval recover the loss, and at what
                            latency cost from the extra round trip
    """
    rows = list(rows)
    out = ["## Tool scaling\n",
           "| exposure | tools | schema tokens | total tokens | pass^1 |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| `{r['exposure']}` | {r.get('n_tools', '-')} | "
                   f"{r.get('mean_schema_tokens', 0):,.0f} | "
                   f"{r.get('mean_tokens', 0):,.0f} | {r.get('pass^1', 0):.3f} |")

    idx = {r["exposure"]: r for r in rows}
    out.append("\n### Reading\n")
    for n in (50, 100, 300):
        a, rnd, orc = idx.get(f"all-{n}"), idx.get(f"random-{n}"), idx.get("oracle")
        if not (a and rnd and orc):
            continue
        d_all = orc["pass^1"] - a["pass^1"]
        d_rnd = orc["pass^1"] - rnd["pass^1"]
        if d_all <= 0.01:
            verdict = f"no degradation at {n} tools"
        elif d_rnd >= 0.7 * d_all:
            verdict = (f"at {n} tools the loss survives when the needed tools are "
                       f"guaranteed present -> SELECTION difficulty, not token cost")
        elif d_rnd <= 0.3 * d_all:
            verdict = (f"at {n} tools the loss disappears once reachability is "
                       f"guaranteed -> TOKEN COST and dilution, not selection")
        else:
            verdict = f"at {n} tools both effects contribute roughly equally"
        out.append(f"- {verdict} (all {d_all:+.3f}, random {d_rnd:+.3f} vs oracle)")

    for n in (100, 300):
        a, se = idx.get(f"all-{n}"), idx.get(f"search-{n}")
        if a and se:
            out.append(f"- `search-{n}` recovers {se['pass^1'] - a['pass^1']:+.3f} "
                       f"over `all-{n}` and injects "
                       f"{a.get('mean_schema_tokens', 0) - se.get('mean_schema_tokens', 0):,.0f} "
                       f"fewer schema tokens per call -- weigh that against the "
                       f"extra round trip in the latency numbers")
    return "\n".join(out)
