"""
Ablation analysis.

A sweep produces a table. A table is not a finding. The finding is:

    "window4 saves 61% of tokens and costs 4 points of pass^1 -- and ALL of
     that loss is concentrated in three traps, every one of which needs a fact
     established more than four units earlier. Sliding windows do not fail
     uniformly; they fail on long-range dependencies, and here is the trace."

This module gets you from the first sentence to the second:

    pareto_frontier      which strategies are actually worth considering
    degradation_matrix   per-trap delta against the baseline
    what_compaction_lost tasks that pass under `full` and fail under X
    attribute_failures   why episodes ended, from the traces
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------
# Pareto
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Point:
    label: str
    cost: float        # tokens, dollars or milliseconds -- lower is better
    quality: float     # pass^1 or pass^k -- higher is better

    def dominates(self, other: "Point") -> bool:
        return (self.cost <= other.cost and self.quality >= other.quality
                and (self.cost < other.cost or self.quality > other.quality))


def pareto_frontier(points: Iterable[Point]) -> list[Point]:
    """Non-dominated set, sorted by cost.

    Everything off the frontier is strictly worse on both axes than something
    else and should be dropped from the recommendation -- not shown as a
    trade-off. Presenting a dominated option as a "trade-off" is the most
    common way an ablation table misleads its reader.
    """
    pts = list(points)
    front = [p for p in pts if not any(q.dominates(p) for q in pts if q is not p)]
    return sorted(front, key=lambda p: (p.cost, -p.quality))


def frontier_report(rows: list[dict[str, Any]], cost_key: str = "mean_tokens",
                    quality_key: str = "pass^1") -> dict[str, Any]:
    pts = [Point(r["strategy"], float(r[cost_key]), float(r[quality_key])) for r in rows]
    front = pareto_frontier(pts)
    front_labels = {p.label for p in front}
    baseline = next((p for p in pts if p.label == "full"), None)

    trade = []
    for p in sorted(front, key=lambda x: x.cost):
        if baseline and p.label != "full" and baseline.cost:
            trade.append({
                "strategy": p.label,
                "token_saving": round(1 - p.cost / baseline.cost, 4),
                "quality_delta": round(p.quality - baseline.quality, 4),
                "points_per_10pct_saved": (
                    round((p.quality - baseline.quality) / ((1 - p.cost / baseline.cost) * 10), 4)
                    if p.cost < baseline.cost else None),
            })
    return {
        "frontier": [p.label for p in front],
        "dominated": [p.label for p in pts if p.label not in front_labels],
        "vs_baseline": trade,
    }


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------

def degradation_matrix(rows: list[dict[str, Any]], baseline: str = "full"
                       ) -> dict[str, Any]:
    """Per-trap delta against the baseline, for every strategy.

    Read the rows, not the column totals. A strategy that loses 4 points spread
    evenly across 16 traps is a different animal from one that loses 4 points
    entirely inside two traps, and only the second tells you anything about the
    mechanism.
    """
    base = next((r for r in rows if r["strategy"] == baseline), None)
    if base is None:
        raise ValueError(f"no baseline row {baseline!r}")
    bt = base["per_trap"]

    out: dict[str, Any] = {}
    for r in rows:
        if r["strategy"] == baseline:
            continue
        deltas = {t: round(r["per_trap"].get(t, 0.0) - v, 3) for t, v in bt.items()}
        hurt = {t: d for t, d in deltas.items() if d < -0.05}
        helped = {t: d for t, d in deltas.items() if d > 0.05}
        total = sum(deltas.values())
        concentration = (round(sum(hurt.values()) / total, 3)
                         if total < 0 and hurt else None)
        out[r["strategy"]] = {
            "deltas": deltas,
            "regressions": dict(sorted(hurt.items(), key=lambda kv: kv[1])),
            "improvements": helped,
            "n_traps_hurt": len(hurt),
            "loss_concentration": concentration,
            "verdict": _verdict(len(hurt), len(bt), concentration),
        }
    return out


def _verdict(n_hurt: int, n_traps: int, concentration: float | None) -> str:
    if n_hurt == 0:
        return "no regression: cheaper at no measured cost"
    if n_hurt <= max(2, n_traps // 6):
        return (f"concentrated: loss sits in {n_hurt} trap(s) -- go read those "
                f"traces, the mechanism is legible there")
    return (f"diffuse: {n_hurt} traps degraded -- the strategy is dropping "
            f"something the agent needs generally, not a specific dependency")


# --------------------------------------------------------------------------
# What compaction lost
# --------------------------------------------------------------------------

def what_compaction_lost(baseline_dir: str | Path, strategy_dir: str | Path
                         ) -> dict[str, Any]:
    """Tasks that PASS under the baseline and FAIL under the strategy.

    This is the actual deliverable of the context ablation. Aggregate numbers
    say a strategy is worse; this says which specific tasks it broke and what
    the verifier objected to, which is one step from the mechanism.
    """
    base = _footers(baseline_dir)
    strat = _footers(strategy_dir)

    broke, fixed = [], []
    for tid, b in base.items():
        s = strat.get(tid)
        if s is None:
            continue
        if b["passed"] and not s["passed"]:
            broke.append({"task_id": tid,
                          "stop_reason": s["stop_reason"],
                          "first_failure": (s["failures"] or ["?"])[0],
                          "baseline_tokens": b["budget"].get("tokens"),
                          "strategy_tokens": s["budget"].get("tokens")})
        elif not b["passed"] and s["passed"]:
            fixed.append(tid)

    reasons = Counter(x["stop_reason"] for x in broke)
    kinds = Counter(_classify(x["first_failure"]) for x in broke)
    return {
        "broken": broke,
        "n_broken": len(broke),
        "unexpectedly_fixed": fixed,
        "stop_reasons": dict(reasons),
        "failure_kinds": dict(kinds),
        "reading": _reading(kinds),
    }


def _classify(failure: str) -> str:
    f = failure.lower()
    if "missing required action" in f:
        return "never took a required action"
    if "forbidden action" in f:
        return "took a forbidden action"
    if "ordering violated" in f:
        return "did things in the wrong order"
    if f.startswith("db:"):
        return "wrong final state"
    return "other"


def _reading(kinds: Counter) -> str:
    if not kinds:
        return "no regressions to explain"
    top = kinds.most_common(1)[0][0]
    return {
        "never took a required action":
            "the agent forgot a step it had already planned -- compaction "
            "dropped the plan. Check whether the summary or note board retained "
            "the outstanding action.",
        "took a forbidden action":
            "the agent lost a constraint it had already established, most often "
            "the eligibility verdict or the payment method. This is the "
            "expensive failure: it is a policy violation, not an omission.",
        "did things in the wrong order":
            "the agent re-derived state it had already computed and reran steps "
            "out of sequence. Usually means the compacted view does not record "
            "what was already done.",
        "wrong final state":
            "the agent acted on a stale or re-derived fact -- typically an "
            "amount or a currency it recomputed after losing the original.",
    }.get(top, "mixed causes; read the traces individually")


def _footers(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    out = {}
    for f in sorted(Path(run_dir).glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        foot = next((r for r in reversed(recs) if r.get("type") == "footer"), None)
        if foot:
            out[f.stem] = foot
    return out


# --------------------------------------------------------------------------
# Failure attribution across a run
# --------------------------------------------------------------------------

def attribute_failures(run_dir: str | Path) -> dict[str, Any]:
    foots = _footers(run_dir)
    failed = {k: v for k, v in foots.items() if not v.get("passed")}
    return {
        "episodes": len(foots),
        "failed": len(failed),
        "stop_reasons": dict(Counter(v["stop_reason"] for v in failed.values())),
        "failure_kinds": dict(Counter(
            _classify((v.get("failures") or ["?"])[0]) for v in failed.values())),
        "budget_exhausted": sum(1 for v in failed.values()
                                if v["stop_reason"].startswith("max_")),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def markdown_report(rows: list[dict[str, Any]], baseline: str = "full") -> str:
    f = frontier_report(rows)
    deg = degradation_matrix(rows, baseline)

    out = ["## Context ablation\n",
           "| strategy | pass^1 | pass^k | mean tokens | vs baseline | frontier |",
           "|---|---|---|---|---|---|"]
    base = next(r for r in rows if r["strategy"] == baseline)
    for r in rows:
        save = (f"{1 - r['mean_tokens'] / base['mean_tokens']:+.1%}"
                if base["mean_tokens"] else "-")
        mark = "**yes**" if r["strategy"] in f["frontier"] else "dominated"
        out.append(f"| `{r['strategy']}` | {r['pass^1']:.3f} | {r['pass^k']:.3f} | "
                   f"{r['mean_tokens']:,.0f} | {save} | {mark} |")

    if f["dominated"]:
        out.append(f"\nDominated (worse on both axes, drop from the "
                   f"recommendation): {', '.join('`' + d + '`' for d in f['dominated'])}")

    out.append("\n### Where each strategy loses\n")
    for name, d in deg.items():
        out.append(f"**`{name}`** -- {d['verdict']}")
        if d["regressions"]:
            for trap, delta in list(d["regressions"].items())[:4]:
                out.append(f"  - `{trap}` {delta:+.2f}")
        out.append("")

    langs = [r for r in rows if len(r.get("per_language", {})) > 1]
    if langs:
        out.append("### Pass rate by language\n")
        keys = sorted({k for r in langs for k in r["per_language"]})
        out.append("| strategy | " + " | ".join(keys) + " |")
        out.append("|" + "---|" * (len(keys) + 1))
        for r in langs:
            cells = [f"{r['per_language'].get(k, float('nan')):.3f}" for k in keys]
            out.append(f"| `{r['strategy']}` | " + " | ".join(cells) + " |")
        out.append("\nLocale twins share identical checks and an identical world, "
                   "so any gap in this table is language and nothing else.")
    return "\n".join(out)
