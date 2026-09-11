"""
Cost.

THE METRIC IS COST PER *RESOLVED* CONVERSATION, NOT COST PER CONVERSATION.
-------------------------------------------------------------------------
A cheaper model that resolves fewer conversations can be strictly more
expensive, and quoting tokens-per-dollar hides that completely. This is the
single most important framing in the whole serving section, and it is the one
an e-commerce CS org actually uses.

AND THE HUMAN COST DOMINATES.
-----------------------------
Every escalated conversation consumes a human agent's time, and every failed
one comes back as a second contact. At any realistic agent wage, a few
percentage points of escalation rate outweigh the entire model bill. Which
means:

    a model that is 3x cheaper per token and escalates 5 points more
    is usually the more expensive model

That inversion is the finding worth reporting, and it is invisible to anyone
measuring throughput.

Note that escalation is not the same as failure. Several tasks in this suite
require escalation -- it is the CORRECT action for an out-of-window dispute or
a customs hold. A correctly escalated conversation still costs human time; it
just is not a quality problem. The model below separates the two, because
conflating them punishes the right behaviour.

ALL DEFAULT RATES BELOW ARE ILLUSTRATIVE PLACEHOLDERS. Replace them with your
own before quoting anything. GPU hourly rates in particular vary by 3x across
providers and contract types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---- Placeholders. REPLACE. ----------------------------------------------
GPU_HOURLY_USD = {"A100-80": 1.80, "H100-80": 3.00, "L40S": 1.10, "A10G": 0.75}
HUMAN_AGENT_USD_PER_CONTACT = 2.50     # SEA CS agent, ~8 min handle time
REPEAT_CONTACT_MULTIPLIER = 1.6        # a failed conversation comes back worse


@dataclass
class ServingConfig:
    """One measured configuration: a model, a serving setup, and its results."""
    name: str
    n_conversations: int
    passed: int
    escalated: int                 # includes correctly escalated
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    gpu: str = "H100-80"
    n_gpus: int = 1
    gpu_hourly_usd: float | None = None
    # API alternative: if set, token pricing is used instead of GPU time
    usd_per_mtok_in: float | None = None
    usd_per_mtok_out: float | None = None
    notes: str = ""

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n_conversations if self.n_conversations else 0.0

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.n_conversations if self.n_conversations else 0.0

    @property
    def resolved_without_human(self) -> int:
        """Passed AND not handed to a person. The thing you are buying."""
        return max(0, self.passed - self.escalated)

    def model_cost_usd(self) -> float:
        if self.usd_per_mtok_in is not None:
            return (self.prompt_tokens / 1e6 * self.usd_per_mtok_in
                    + self.completion_tokens / 1e6 * (self.usd_per_mtok_out or 0))
        rate = self.gpu_hourly_usd or GPU_HOURLY_USD.get(self.gpu, 3.00)
        return rate * self.n_gpus * (self.wall_seconds / 3600.0)

    def human_cost_usd(self, per_contact: float = HUMAN_AGENT_USD_PER_CONTACT,
                       repeat_multiplier: float = REPEAT_CONTACT_MULTIPLIER) -> float:
        failed = self.n_conversations - self.passed
        return (self.escalated * per_contact
                + failed * per_contact * repeat_multiplier)

    def breakdown(self, per_contact: float = HUMAN_AGENT_USD_PER_CONTACT
                  ) -> dict[str, Any]:
        model = self.model_cost_usd()
        human = self.human_cost_usd(per_contact)
        resolved = self.resolved_without_human
        total = model + human
        return {
            "config": self.name,
            "conversations": self.n_conversations,
            "pass_rate": round(self.pass_rate, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "resolved_without_human": resolved,
            "model_usd": round(model, 4),
            "human_usd": round(human, 4),
            "total_usd": round(total, 4),
            "model_share": round(model / total, 4) if total else None,
            "usd_per_conversation": round(total / self.n_conversations, 5)
            if self.n_conversations else None,
            "usd_per_resolved": round(total / resolved, 5) if resolved else None,
            "model_usd_per_resolved": round(model / resolved, 5) if resolved else None,
            "tok_per_s_out": round(self.completion_tokens / self.wall_seconds, 1)
            if self.wall_seconds else None,
        }


def compare(configs: list[ServingConfig],
            per_contact: float = HUMAN_AGENT_USD_PER_CONTACT) -> dict[str, Any]:
    rows = [c.breakdown(per_contact) for c in configs]
    usable = [r for r in rows if r["usd_per_resolved"] is not None]
    best_resolved = min(usable, key=lambda r: r["usd_per_resolved"], default=None)
    best_raw = min(usable, key=lambda r: r["usd_per_conversation"], default=None)
    cheapest_model = min(rows, key=lambda r: r["model_usd"], default=None)

    inversion = None
    if best_resolved and cheapest_model and \
            best_resolved["config"] != cheapest_model["config"]:
        inversion = (
            f"`{cheapest_model['config']}` has the lowest model bill but "
            f"`{best_resolved['config']}` is cheaper per RESOLVED conversation. "
            f"The difference is human handling: escalation "
            f"{cheapest_model['escalation_rate']:.1%} vs "
            f"{best_resolved['escalation_rate']:.1%}, pass "
            f"{cheapest_model['pass_rate']:.1%} vs "
            f"{best_resolved['pass_rate']:.1%}. Quoting tokens-per-dollar would "
            f"have picked the more expensive option.")

    return {
        "rows": rows,
        "best_per_resolved": best_resolved["config"] if best_resolved else None,
        "best_per_conversation": best_raw["config"] if best_raw else None,
        "cheapest_model_bill": cheapest_model["config"] if cheapest_model else None,
        "inversion": inversion,
        "human_rate_used": per_contact,
    }


def break_even_human_cost(a: ServingConfig, b: ServingConfig) -> dict[str, Any]:
    """At what human cost per contact do these two configs cost the same?

    Total(c, h) = model(c) + h * (escalated + failed * repeat), so the crossover
    is linear in h and solvable exactly. Report it: "B wins above $1.40 per
    contact" is a decision a manager can actually make, where "B has better
    pass^1" is not.
    """
    def slope(c: ServingConfig) -> float:
        failed = c.n_conversations - c.passed
        return c.escalated + failed * REPEAT_CONTACT_MULTIPLIER

    ma, mb = a.model_cost_usd(), b.model_cost_usd()
    sa, sb = slope(a), slope(b)
    if abs(sa - sb) < 1e-9:
        return {"crossover_usd": None,
                "note": (f"identical human load; `{a.name if ma < mb else b.name}` "
                         f"is cheaper at every human rate")}
    h = (mb - ma) / (sa - sb)
    if h < 0:
        cheaper = a.name if (ma + sa * 1.0) < (mb + sb * 1.0) else b.name
        return {"crossover_usd": None,
                "note": f"no positive crossover; `{cheaper}` wins at any human rate"}
    below = a.name if (ma + sa * (h * 0.5)) < (mb + sb * (h * 0.5)) else b.name
    above = b.name if below == a.name else a.name
    return {
        "crossover_usd": round(h, 4),
        "below_crossover_winner": below,
        "above_crossover_winner": above,
        "note": (f"`{below}` is cheaper while a human contact costs under "
                 f"${h:.2f}; `{above}` wins above it"),
    }


def amortised_capacity(cfg: ServingConfig) -> dict[str, Any]:
    """Self-hosted framing: what does the hardware you already have buy you?

    With owned GPUs the marginal token is nearly free and the real constraint
    is throughput. The number that matters becomes conversations per GPU-hour,
    and the comparison against an API is a capacity question, not a price one.
    """
    hours = cfg.wall_seconds / 3600.0
    gpu_hours = hours * cfg.n_gpus
    return {
        "config": cfg.name,
        "gpu_hours": round(gpu_hours, 4),
        "conversations_per_gpu_hour": round(cfg.n_conversations / gpu_hours, 1)
        if gpu_hours else None,
        "resolved_per_gpu_hour": round(cfg.resolved_without_human / gpu_hours, 1)
        if gpu_hours else None,
        "daily_capacity_resolved": round(
            cfg.resolved_without_human / gpu_hours * 24 * cfg.n_gpus, 0)
        if gpu_hours else None,
    }


def markdown(compare_result: dict[str, Any]) -> str:
    out = ["## Cost per resolved conversation\n",
           f"Human contact priced at ${compare_result['human_rate_used']:.2f}. "
           f"Escalated conversations consume human time even when escalating "
           f"was the CORRECT action.\n",
           "| config | pass | escal. | model $ | human $ | $/conv | **$/resolved** |",
           "|---|---|---|---|---|---|---|"]
    for r in compare_result["rows"]:
        out.append(
            f"| `{r['config']}` | {r['pass_rate']:.3f} | {r['escalation_rate']:.3f} | "
            f"{r['model_usd']:.3f} | {r['human_usd']:.3f} | "
            f"{r['usd_per_conversation']:.4f} | "
            f"**{r['usd_per_resolved']:.4f}** |"
            if r["usd_per_resolved"] is not None else
            f"| `{r['config']}` | {r['pass_rate']:.3f} | {r['escalation_rate']:.3f} | "
            f"{r['model_usd']:.3f} | {r['human_usd']:.3f} | - | - |")
    if compare_result["inversion"]:
        out.append(f"\n> **{compare_result['inversion']}**")
    return "\n".join(out)
