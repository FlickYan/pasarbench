"""
Serving verdicts.

THE GUARD THIS MODULE EXISTS TO ENFORCE
---------------------------------------
Never report a throughput or latency win without proving quality held.

FP8 weights, FP8 KV cache, aggressive batching and prefix caching all make the
numbers you were measuring go up. The damage they can do shows up in TASK
SUCCESS long before it shows up in perplexity, and on a policy-following
workload it concentrates: the model still writes fluent, plausible customer
service prose while quietly getting the refund method wrong on COD orders.
Perplexity will not see that. `pass^k` per trap will.

So `serving_verdict` refuses to say WIN unless the quality difference is
NON-INFERIOR at a stated margin -- not merely "about the same", which is what
an overlapping-error-bars eyeball check actually means.

AND THE THIRD OUTCOME IS REAL
-----------------------------
With 186 tasks, a 2-point regression is not detectable. The honest verdict in
that case is INCONCLUSIVE with a required sample size attached, not a win.
`required_n` computes it, and it is usually a bigger number than people expect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

Z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def proportion_diff_ci(p1: float, n1: int, p2: float, n2: int,
                       conf: float = 0.95) -> tuple[float, float, float]:
    """(diff, lo, hi) for p1 - p2, normal approximation."""
    if n1 <= 0 or n2 <= 0:
        return (0.0, 0.0, 0.0)
    diff = p1 - p2
    se = math.sqrt(max(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2, 0.0))
    z = Z.get(conf, 1.96)
    return (diff, diff - z * se, diff + z * se)


def required_n(baseline_rate: float, margin: float, conf: float = 0.95,
               power: float = 0.80) -> int:
    """Tasks per arm needed to rule out a regression larger than `margin`.

    Two-proportion, equal arms. This number is routinely 3-10x the suite you
    have, which is exactly why INCONCLUSIVE has to be an available verdict.
    """
    z_a, z_b = Z.get(conf, 1.96), 0.84 if power >= 0.8 else 0.52
    p = baseline_rate
    if margin <= 0:
        return 0
    n = ((z_a + z_b) ** 2 * 2 * p * (1 - p)) / (margin ** 2)
    return int(math.ceil(n))


@dataclass
class QualityCheck:
    treatment: str
    baseline: str
    diff: float
    ci: tuple[float, float]
    margin: float
    non_inferior: bool
    conclusive: bool
    n: int
    needed_n: int
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {"treatment": self.treatment, "baseline": self.baseline,
                "diff": round(self.diff, 4), "ci95": [round(x, 4) for x in self.ci],
                "margin": self.margin, "non_inferior": self.non_inferior,
                "conclusive": self.conclusive, "n": self.n,
                "needed_n": self.needed_n, "verdict": self.verdict}


def non_inferiority(treat_rate: float, treat_n: int,
                    base_rate: float, base_n: int,
                    margin: float = 0.02, name_t: str = "treatment",
                    name_b: str = "baseline") -> QualityCheck:
    diff, lo, hi = proportion_diff_ci(treat_rate, treat_n, base_rate, base_n)
    non_inf = lo > -margin
    # "Conclusive" means the interval is tight enough to distinguish
    # non-inferiority from a real regression. A CI spanning both is not a
    # result, however nice the point estimate looks.
    conclusive = (hi - lo) < 2 * margin or lo > -margin or hi < -margin
    need = required_n(base_rate, margin)

    if non_inf and diff >= 0:
        v = f"quality held or improved ({diff:+.3f}, CI lower {lo:+.3f} > -{margin})"
    elif non_inf:
        v = (f"quality non-inferior at margin {margin}: point estimate {diff:+.3f} "
             f"but CI lower bound {lo:+.3f} rules out a regression worse than "
             f"{margin}")
    elif hi < -margin:
        v = (f"REAL REGRESSION: {diff:+.3f}, entire CI below -{margin}. Do not "
             f"ship this serving change on throughput grounds")
    else:
        v = (f"INCONCLUSIVE: {diff:+.3f} with CI [{lo:+.3f}, {hi:+.3f}] spanning "
             f"the margin. With n={treat_n} you cannot tell a {margin:.0%} "
             f"regression from noise; you need ~{need} tasks per arm")
    return QualityCheck(name_t, name_b, diff, (lo, hi), margin, non_inf,
                        conclusive, treat_n, need, v)


# --------------------------------------------------------------------------

def serving_verdict(treat: dict[str, Any], base: dict[str, Any],
                    margin: float = 0.02) -> dict[str, Any]:
    """Combine a speed result with a quality guard.

    `treat`/`base`: {name, pass_rate, n, ttft_p50, ttft_p99, output_tok_per_s,
                     usd_per_resolved}
    """
    q = non_inferiority(treat["pass_rate"], treat["n"],
                        base["pass_rate"], base["n"], margin,
                        treat["name"], base["name"])

    def gain(key: str, higher_better: bool) -> float | None:
        a, b = treat.get(key), base.get(key)
        if a in (None, 0) or b in (None, 0):
            return None
        return round((a / b - 1) if higher_better else (1 - a / b), 4)

    speed = {
        "ttft_p50_improvement": gain("ttft_p50", False),
        "ttft_p99_improvement": gain("ttft_p99", False),
        "throughput_gain": gain("output_tok_per_s", True),
        "cost_per_resolved_saving": gain("usd_per_resolved", False),
    }
    improved = any(v is not None and v > 0.05 for v in speed.values())

    if not q.conclusive:
        headline = ("INCONCLUSIVE -- the quality comparison cannot be resolved at "
                    "this sample size, so no serving claim can be made yet")
    elif not q.non_inferior:
        headline = ("REJECT -- quality regressed beyond the margin. Speed gains "
                    "do not buy this back on a policy-following workload")
    elif improved:
        headline = "WIN -- faster or cheaper, with quality proven non-inferior"
    else:
        headline = ("NEUTRAL -- quality held but no material speed or cost gain; "
                    "prefer the simpler configuration")

    return {"headline": headline, "quality": q.to_dict(), "speed": speed,
            "treatment": treat["name"], "baseline": base["name"]}


def per_trap_guard(treat_per_trap: dict[str, float], base_per_trap: dict[str, float],
                   drop: float = 0.10) -> dict[str, Any]:
    """Quantisation damage concentrates. Aggregate pass rate hides it.

    A model that loses 1 point overall but 25 points on `cod_cannot_refund_to_
    original_method` has not degraded slightly -- it has stopped following one
    specific rule, which is a shippable-blocker even at a flat aggregate.
    """
    hits = {t: round(treat_per_trap.get(t, 0.0) - v, 3)
            for t, v in base_per_trap.items()
            if treat_per_trap.get(t, 0.0) - v <= -drop}
    total = sum(treat_per_trap.get(t, 0.0) - v for t, v in base_per_trap.items())
    return {
        "regressed_traps": dict(sorted(hits.items(), key=lambda kv: kv[1])),
        "n_regressed": len(hits),
        "aggregate_delta": round(total / len(base_per_trap), 4) if base_per_trap else 0,
        "verdict": (
            "no single rule broke" if not hits else
            f"CONCENTRATED DAMAGE: {len(hits)} trap(s) lost more than {drop:.0%}. "
            f"The aggregate understates this -- the model has stopped following "
            f"specific rules rather than degrading smoothly. Read those traces "
            f"before shipping."),
    }


# --------------------------------------------------------------------------

def three_bar_report(bars: list[dict[str, Any]], baseline_name: str) -> str:
    """The prefix-caching story: preload+cache vs preload-no-cache vs JIT.

    The non-obvious outcome is that JIT can send far fewer tokens and still
    LOSE on latency, because the extra `search_policy` round trip costs more
    than the cached prefill it saves. That result is workload-specific and it
    is exactly the kind of thing an inference team wants to hear a candidate
    say out loud.
    """
    out = ["## Prefix caching and policy delivery\n",
           "| config | prefix hit | TTFT p50 | TTFT p99 | out tok/s | pass^1 | $/resolved |",
           "|---|---|---|---|---|---|---|"]
    for b in bars:
        hr = b.get("prefix_hit_rate")
        out.append(
            f"| `{b['name']}` | {'-' if hr is None else f'{hr:.3f}'} | "
            f"{_f(b.get('ttft_p50'))} | {_f(b.get('ttft_p99'))} | "
            f"{_f(b.get('output_tok_per_s'), 0)} | {b.get('pass_rate', 0):.3f} | "
            f"{_f(b.get('usd_per_resolved'), 4)} |")

    base = next((b for b in bars if b["name"] == baseline_name), None)
    if base:
        out.append("\n### Verdicts\n")
        for b in bars:
            if b["name"] == baseline_name:
                continue
            v = serving_verdict(b, base)
            out.append(f"**`{b['name']}` vs `{baseline_name}`** -- {v['headline']}")
            out.append(f"  - {v['quality']['verdict']}")
            spd = ", ".join(f"{k.replace('_', ' ')} {val:+.1%}"
                            for k, val in v["speed"].items() if val is not None)
            if spd:
                out.append(f"  - {spd}")
            out.append("")
    out.append("> No speed number in this table is a result on its own. Each row's "
               "claim is only as good as its quality verdict.")
    return "\n".join(out)


def _f(v: Any, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"
