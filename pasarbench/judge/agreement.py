"""
Agreement.

The number everyone quotes is Cohen's kappa. The number almost nobody quotes
alongside it is the one that makes kappa interpretable.

THE KAPPA PARADOX
-----------------
On a criterion where 95% of transcripts are fine, two labellers can agree 94%
of the time and still score kappa = 0.15. Kappa corrects for chance agreement,
and when one category dominates, chance agreement is already near-total, so
almost nothing is left for kappa to credit. Reporting that 0.15 as "poor
agreement" is wrong, and reporting the 94% alone is also wrong.

So every criterion here reports five numbers:

    p_o        raw agreement
    kappa      chance-corrected
    prevalence |p(1,1) - p(0,0)|  -- high means kappa is being suppressed
    bias       |p(1,0) - p(0,1)|  -- high means one rater says yes more often
    pabak      prevalence-adjusted bias-adjusted kappa

Read them together. High p_o + low kappa + high prevalence is a criterion where
almost nothing ever goes wrong, not a broken rubric.

THE CEILING
-----------
A judge cannot meaningfully agree with you more than you agree with yourself.
Relabel a subset a week later, compute test-retest kappa, and report judge
agreement AS A FRACTION OF THAT CEILING. "kappa 0.71, ceiling 0.78, so 91% of
achievable agreement" is a far more honest claim than "kappa 0.71", and almost
nobody does it.

CONFIDENCE INTERVALS
--------------------
200 labels is a small sample and kappa has a wide interval on it. Bootstrap and
report it. A jump from 0.62 to 0.68 whose intervals overlap by half is not an
improvement, and claiming it is will not survive a careful interviewer.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


@dataclass
class Agreement:
    n: int
    p_o: float
    kappa: float
    prevalence: float
    bias: float
    pabak: float
    ci95: tuple[float, float] | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        ci = f" [{self.ci95[0]:.2f}, {self.ci95[1]:.2f}]" if self.ci95 else ""
        return (f"n={self.n:<4d} p_o={self.p_o:.3f}  kappa={self.kappa:+.3f}{ci}  "
                f"prev={self.prevalence:.2f} bias={self.bias:.2f} "
                f"pabak={self.pabak:+.3f}  {self.note}")


def cohens_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    if len(a) != len(b):
        raise ValueError("rater sequences differ in length")
    n = len(a)
    if n == 0:
        return 0.0
    cats = sorted(set(a) | set(b), key=str)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1 - p_e) < 1e-12:
        # Both raters used exactly one category for everything. Chance agreement
        # is total, kappa is undefined. Return 0 and let the prevalence figure
        # explain why rather than pretending to a number.
        return 0.0
    return (p_o - p_e) / (1 - p_e)


def agreement(a: Sequence[Any], b: Sequence[Any], bootstrap: int = 2000,
              seed: int = 0) -> Agreement:
    n = len(a)
    if n == 0:
        return Agreement(0, 0.0, 0.0, 0.0, 0.0, 0.0, note="no data")

    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    k = cohens_kappa(a, b)

    binary = set(a) | set(b) <= {True, False, 0, 1}
    if binary:
        both1 = sum(1 for x, y in zip(a, b) if x and y) / n
        both0 = sum(1 for x, y in zip(a, b) if not x and not y) / n
        a1b0 = sum(1 for x, y in zip(a, b) if x and not y) / n
        a0b1 = sum(1 for x, y in zip(a, b) if not x and y) / n
        prev, bias = abs(both1 - both0), abs(a1b0 - a0b1)
    else:
        prev = bias = 0.0
    pabak = 2 * p_o - 1

    ci = _bootstrap_ci(a, b, bootstrap, seed) if bootstrap else None
    return Agreement(n, round(p_o, 4), round(k, 4), round(prev, 4), round(bias, 4),
                     round(pabak, 4), ci, note=interpret(p_o, k, prev, bias))


def interpret(p_o: float, k: float, prev: float, bias: float) -> str:
    if p_o > 0.85 and k < 0.4 and prev > 0.6:
        return ("KAPPA PARADOX: near-total agreement on a criterion that is "
                "almost always satisfied. The low kappa is prevalence, not "
                "disagreement -- report p_o and pabak, and consider whether "
                "this criterion earns its place in the rubric")
    if bias > 0.2:
        return ("SYSTEMATIC BIAS: one rater says yes far more often. Usually a "
                "rubric-boundary problem, not noise -- tighten the guidance")
    if k >= 0.8:
        return "strong"
    if k >= 0.6:
        return "substantial"
    if k >= 0.4:
        return "moderate -- the rubric boundary is unclear somewhere"
    return "poor -- do not use this criterion until the guidance is fixed"


def _bootstrap_ci(a: Sequence[Any], b: Sequence[Any], iters: int, seed: int
                  ) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a)
    idx = range(n)
    ks = []
    for _ in range(iters):
        pick = [rng.choice(idx) for _ in range(n)]
        ks.append(cohens_kappa([a[i] for i in pick], [b[i] for i in pick]))
    ks.sort()
    lo = ks[int(0.025 * len(ks))]
    hi = ks[min(int(0.975 * len(ks)), len(ks) - 1)]
    return (round(lo, 4), round(hi, 4))


def paired_kappa_diff(ref: Sequence[Any], a: Sequence[Any], b: Sequence[Any],
                      iters: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Is judge B's agreement with the reference really different from A's?

    Both kappas are computed on the SAME transcripts, so their errors are
    correlated and two separate confidence intervals are the wrong test:
    overlapping CIs do not mean "no difference", and non-overlapping ones
    overstate it. Resample transcripts once per iteration, compute both
    kappas on the same resample, and take the interval of the difference.
    If it excludes zero, the difference is real at this sample size.
    """
    if not (len(ref) == len(a) == len(b)):
        raise ValueError("rater sequences differ in length")
    n = len(ref)
    if n == 0:
        return {"n": 0, "kappa_a": 0.0, "kappa_b": 0.0, "diff": 0.0,
                "ci95": None, "resolved": False}
    ka, kb = cohens_kappa(ref, a), cohens_kappa(ref, b)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        pick = [rng.randrange(n) for _ in range(n)]
        r = [ref[i] for i in pick]
        diffs.append(cohens_kappa(r, [b[i] for i in pick])
                     - cohens_kappa(r, [a[i] for i in pick]))
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[min(int(0.975 * iters), iters - 1)]
    return {"n": n, "kappa_a": round(ka, 4), "kappa_b": round(kb, 4),
            "diff": round(kb - ka, 4), "ci95": (round(lo, 4), round(hi, 4)),
            "resolved": lo > 0 or hi < 0}


def confusion(a: Sequence[Any], b: Sequence[Any]) -> dict[str, int]:
    """Where they disagree, not just how much. The asymmetry is the finding:
    a judge that says yes when you said no is a different bug from the reverse."""
    return {
        "both_yes": sum(1 for x, y in zip(a, b) if x and y),
        "both_no": sum(1 for x, y in zip(a, b) if not x and not y),
        "human_yes_judge_no": sum(1 for x, y in zip(a, b) if x and not y),
        "human_no_judge_yes": sum(1 for x, y in zip(a, b) if not x and y),
    }


# --------------------------------------------------------------------------
# Per-criterion + ceiling
# --------------------------------------------------------------------------

def per_criterion(human: list[dict[str, Any]], judge: list[dict[str, Any]],
                  keys: Iterable[str], bootstrap: int = 1000) -> dict[str, Agreement]:
    """Align on shared ids, then score each criterion separately.

    'kappa = 0.71 overall' is not actionable. 'kappa = 0.89 on data leakage,
    0.41 on tone' tells you which half of the rubric to fix or drop.
    """
    h = {r["transcript_id"]: r for r in human}
    j = {r["transcript_id"]: r for r in judge}
    shared = sorted(set(h) & set(j))
    out = {}
    for key in keys:
        pairs = [(h[i]["labels"].get(key), j[i]["labels"].get(key)) for i in shared]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        out[key] = agreement([x for x, _ in pairs], [y for _, y in pairs],
                             bootstrap=bootstrap)
    return out


def ceiling_report(round1: list[dict[str, Any]], round2: list[dict[str, Any]],
                   judge: list[dict[str, Any]], keys: Iterable[str]
                   ) -> dict[str, Any]:
    """Judge agreement expressed as a fraction of your own test-retest ceiling.

    This is the headline of the calibration work. It reframes the result from
    "my judge is imperfect" to "my judge captures 91% of the agreement a human
    achieves with themselves", which is both more honest and more impressive.
    """
    self_k = per_criterion(round1, round2, keys, bootstrap=500)
    judge_k = per_criterion(round1, judge, keys, bootstrap=500)

    rows = {}
    for key in keys:
        s, jj = self_k.get(key), judge_k.get(key)
        if not s or not jj:
            continue
        frac = round(jj.kappa / s.kappa, 3) if s.kappa > 0.05 else None
        rows[key] = {
            "self_kappa": s.kappa, "judge_kappa": jj.kappa,
            "fraction_of_ceiling": frac,
            "verdict": _ceiling_verdict(s.kappa, jj.kappa, frac),
        }
    valid = [r for r in rows.values() if r["fraction_of_ceiling"] is not None]
    return {
        "per_criterion": rows,
        "mean_self_kappa": round(sum(r["self_kappa"] for r in rows.values()) / len(rows), 4) if rows else 0,
        "mean_judge_kappa": round(sum(r["judge_kappa"] for r in rows.values()) / len(rows), 4) if rows else 0,
        "mean_fraction_of_ceiling": round(
            sum(r["fraction_of_ceiling"] for r in valid) / len(valid), 3) if valid else None,
    }


def _ceiling_verdict(self_k: float, judge_k: float, frac: float | None) -> str:
    if self_k < 0.4:
        return ("YOUR OWN CEILING IS LOW. You do not agree with yourself on this "
                "criterion, so no judge can do better. Fix the guidance or cut "
                "the criterion -- do not tune the judge against it")
    if frac is None:
        return "ceiling too low to express a fraction"
    if frac >= 0.9:
        return "judge is at the achievable ceiling; further tuning is noise"
    if frac >= 0.7:
        return "judge is close; remaining gap is worth one prompt iteration"
    return "judge is well below the human ceiling; the prompt is the problem"


# --------------------------------------------------------------------------
# Bias probes
# --------------------------------------------------------------------------

def position_bias(pairwise: list[dict[str, Any]]) -> dict[str, Any]:
    """Each item must have been judged twice with A and B swapped.

    A judge that prefers whichever response is shown first is not comparing
    content. Measured as the rate at which the verdict FLIPS with the order,
    which should be ~0 for a consistent judge and 1.0 for a pure position
    follower.
    """
    consistent = flipped = 0
    first_pref = 0
    for item in pairwise:
        a, b = item.get("verdict_ab"), item.get("verdict_ba")
        if a is None or b is None:
            continue
        # verdicts are "left"/"right"; after swapping, a consistent judge
        # picks the opposite side
        if a != b:
            consistent += 1
        else:
            flipped += 1
        if a == "left":
            first_pref += 1
    n = consistent + flipped or 1
    return {
        "n": consistent + flipped,
        "position_consistency": round(consistent / n, 4),
        "flip_rate": round(flipped / n, 4),
        "first_position_preference": round(first_pref / n, 4),
        "verdict": ("no meaningful position bias" if flipped / n < 0.1 else
                    "POSITION BIAS: the judge is partly reading order, not content. "
                    "Always evaluate both orders and average, or move to a "
                    "pointwise decomposed rubric"),
    }


def length_bias(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Point-biserial correlation between response length and a positive verdict.

    Judges reliably reward longer answers. In customer service that is actively
    wrong -- a correct one-line refusal beats three paragraphs of hedging -- so
    this is worth measuring rather than assuming.
    """
    pts = [(r["chars"], 1 if r["positive"] else 0) for r in records
           if "chars" in r and "positive" in r]
    if len(pts) < 3:
        return {"n": len(pts), "r": None, "verdict": "not enough data"}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in pts)
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    r = num / (dx * dy) if dx and dy else 0.0
    return {
        "n": len(pts), "r": round(r, 4),
        "mean_chars_positive": round(
            sum(x for x, y in pts if y) / max(sum(ys), 1), 1),
        "mean_chars_negative": round(
            sum(x for x, y in pts if not y) / max(len(pts) - sum(ys), 1), 1),
        "verdict": ("no strong length effect" if abs(r) < 0.25 else
                    "LENGTH BIAS: verdicts track response length. Add an explicit "
                    "'length is not quality' instruction and re-measure"),
    }
