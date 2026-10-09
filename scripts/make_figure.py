"""
The four results at a glance, as one figure for the README and the blog.

    python scripts/make_figure.py --traces traces --out docs/img
    python scripts/make_figure.py --numbers docs/img/results.json --out docs/img

The first reads the recorded runs, as every number in the docs is read, and
writes results.json beside four SVGs: light and dark, wide and narrow, for the
README's <picture> to choose between. The second redraws the SVGs from
results.json, for a machine without the traces. No plotting library: the
figure is plain SVG, so the script runs wherever the tests do.

Each panel's subtitle carries the test that licenses it. The figure shows
nothing the RUNBOOK and WRITEUP do not state, with the same numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# The cells each panel reads, as the RUNBOOK names them.
CELLS = {
    "naming": ("K-preload/full+search-300", "K-preload-named/full+search-300"),
    "guardrail": ("L2-off/full+search-300", "L2-claims/full+search-300"),
    "finetune": ("P-base/full", "P-rft/full", "P-ref/full"),
    "language": ("C-clean/full", "D-nozh/full"),
}
LANGUAGE_NAMES = {"id": "Indonesian", "ms": "Malay", "sg-en": "Singlish", "th": "Thai",
                  "vi": "Vietnamese", "zh-MY": "Chinese (MY)", "zh-SG": "Chinese (SG)"}


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

def _sign_p(worse: int, better: int) -> float:
    """Exact two-sided sign test over the pairs that differ."""
    n = worse + better
    if n == 0:
        return 1.0
    k = min(worse, better)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _fisher_p(a: int, n_a: int, b: int, n_b: int) -> float:
    """Exact two-sided Fisher test for a/n_a against b/n_b."""
    total, hits = n_a + n_b, a + b
    def prob(x: int) -> float:
        return math.comb(hits, x) * math.comb(total - hits, n_a - x) / math.comb(total, n_a)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, hits - n_b), min(hits, n_a) + 1)
                        if prob(x) <= observed * (1 + 1e-9)))


def _pass1(eps: list[dict]) -> float:
    return sum(e["passed"] for e in eps) / max(1, len(eps))


def _lenient(cell: Path) -> int:
    """Failures a checking escalation tool could have saved (WHAT_FAILED #35):
    every failure line about the escalation, and a successful escalation under
    a category the schema does not list or naming no real order."""
    import re

    from pasarbench.rescore import rescore_file
    from pasarbench.tools import TOOLS
    cats = set(TOOLS["escalate_to_human"]["schema"]["function"]["parameters"]
               ["properties"]["category"]["enum"])
    real = re.compile(r"^(GO-[0-9A-Za-z]{4,}|O\d{4})$")
    n = 0
    for f in sorted(cell.glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not recs or recs[-1].get("type") != "footer":
            continue
        rs = rescore_file(f)
        if rs.verdict or not rs.failures or not all("escalat" in x for x in rs.failures):
            continue
        bad = any(tc.get("name") == "escalate_to_human" and tr.get("ok")
                  and ((tc.get("arguments") or {}).get("category") not in cats
                       or not real.match(str((tc.get("arguments") or {}).get("order_id") or "")))
                  for r in recs if r.get("type") == "step"
                  for tc, tr in zip(r.get("tool_calls") or [], r.get("tool_results") or []))
        n += bad
    return n


def compute(traces: Path) -> dict:
    from compare_cells import _audit_claims
    from pasarbench.diagnose import load_episodes, paired_episodes, paired_language_gap
    from pasarbench.rescore import all_tasks

    tasks = all_tasks()
    out: dict = {"source": "scripts/make_figure.py --traces", "cells": CELLS}

    a, b = (load_episodes(traces / c, checker="current") for c in CELLS["naming"])
    g = paired_episodes(a, b)
    out["naming"] = {"control": _pass1(a), "named": _pass1(b), "episodes": len(a),
                     "tasks": g["n"], "better": g["better"], "worse": g["worse"], "p": g["p"]}

    cells = [traces / c for c in CELLS["guardrail"]]
    eps = [load_episodes(c, checker="current") for c in cells]
    (read_a, told_a), (read_b, told_b) = (_audit_claims(c, e, tasks) for c, e in zip(cells, eps))
    out["guardrail"] = {"control": told_a, "guarded": told_b,
                        "episodes": [len(eps[0]), len(eps[1])],
                        "failures_read": [read_a, read_b],
                        "p": _fisher_p(told_a, len(eps[0]), told_b, len(eps[1]))}

    base, rft, ref = CELLS["finetune"]
    ft: dict = {"episodes": None}
    for checker, key in (("recorded", "first"), ("current", "fixed")):
        eb, er, ex = (load_episodes(traces / c, checker=checker) for c in (base, rft, ref))
        ft["episodes"] = len(eb)
        ft[key] = {"rft": 100 * (_pass1(er) - _pass1(eb)), "ref": 100 * (_pass1(ex) - _pass1(eb)),
                   "p_rft": paired_episodes(eb, er)["p"], "p_ref": paired_episodes(eb, ex)["p"]}
    ft["ref_best_case"] = ft["fixed"]["ref"] + 100 * _lenient(traces / ref) / ft["episodes"]
    out["finetune"] = ft

    lang = {}
    for c in CELLS["language"]:
        rows = paired_language_gap(load_episodes(traces / c, checker="current"))["languages"]
        lang[c] = {k: {"gap": -100 * r["paired_gap"], "pairs": r["pairs"],
                       "p": _sign_p(r["pairs_where_worse"], r["pairs_where_better"])}
                   for k, r in rows.items() if r.get("pairs")}
    out["language"] = lang
    return out


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "link": "#c3c2b7", "accent": "#2a78d6",
              "rest": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "link": "#52514e", "accent": "#3987e5",
             "rest": "#898781"},
}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
# Where each panel sits. Wide: two by two, for a page. Narrow: one column, so
# the text stays readable on a phone, where the README's <picture> picks it.
LAYOUTS = {
    "wide": {"size": (880, 426), "at": {"naming": (24, 30), "guardrail": (468, 30),
                                        "finetune": (24, 196), "language": (468, 196)},
             "footer": (24, 410)},
    "narrow": {"size": (436, 786), "at": {"naming": (24, 30), "guardrail": (24, 192),
                                          "finetune": (24, 354), "language": (24, 558)},
               "footer": (24, 770)},
}
FILES = {("light", "wide"): "results.svg", ("dark", "wide"): "results-dark.svg",
         ("light", "narrow"): "results-narrow.svg", ("dark", "narrow"): "results-narrow-dark.svg"}


class Svg:
    def __init__(self, t: dict):
        self.t, self.parts = t, []

    def text(self, x, y, s, size=13, color="ink", weight=400, anchor="start"):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                          f'font-weight="{weight}" fill="{self.t[color]}" '
                          f'text-anchor="{anchor}">{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, color="axis", width=1):
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{self.t[color]}" stroke-width="{width}"/>')

    def bar(self, x0, y, length, h, color, tip):
        """A horizontal bar from a baseline: square at the base, 4px rounded
        at the data end."""
        if length <= 0:
            return
        r = min(4, length, h / 2)
        x1 = x0 + length
        d = (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
             f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")
        self.parts.append(f'<path d="{d}" fill="{self.t[color]}"><title>{escape(tip)}</title></path>')

    def dot(self, x, y, color, tip, hollow=False):
        fill = self.t["surface"] if hollow else self.t[color]
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" '
                          f'stroke="{self.t[color] if hollow else self.t["surface"]}" '
                          f'stroke-width="2"><title>{escape(tip)}</title></circle>')

    def heading(self, x, y, title, *subs):
        self.text(x, y, title, size=15, weight=600)
        for i, s in enumerate(subs):
            self.text(x, y + 19 + 16 * i, s, size=12.5, color="ink2")


def _signed(v: float, digits: int = 1) -> str:
    """+2.3, −6.6, 0: a real minus sign, not a hyphen."""
    if round(v, digits) == 0:
        return "0" if digits == 0 else f"{0:.{digits}f}"
    return f"{v:+.{digits}f}".replace("-", "\u2212")


def _p(p: float) -> str:
    if p < 1e-3:
        e = math.floor(math.log10(p))
        m = round(p / 10 ** e)
        if m == 10:
            m, e = 1, e + 1
        sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"p = {m} × 10{sup}"
    return f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"


def _bars(s: Svg, x, y, rows, vmax, fmt, label_w=136, area=196):
    """Two-row horizontal bars from one baseline; the label column on the left
    and the value at each bar's tip."""
    x0 = x + label_w
    for i, (label, value, color, tip) in enumerate(rows):
        yy = y + i * 34
        s.text(x0 - 10, yy + 15, label, size=13, color="ink2", anchor="end")
        length = area * value / vmax
        s.bar(x0, yy, length, 20, color, tip)
        s.text(x0 + max(length, 0) + 8, yy + 15, fmt(value), size=13, weight=600)
    s.line(x0, y - 6, x0, y + len(rows) * 34 - 8, color="axis")


def _naming(s: Svg, x: float, y: float, nm: dict) -> None:
    s.heading(x, y, "Naming the tools the policy describes",
              f"Conversations passed, {nm['tasks']} new tasks, {nm['episodes']} per arm",
              f"{nm['better']} tasks better, {nm['worse']} worse · {_p(nm['p'])}")
    _bars(s, x, y + 54, [
        ("Tools described", nm["control"], "rest", f"Policy describes the tools: {nm['control']:.0%}"),
        ("Tools named", nm["named"], "accent", f"Policy names them: {nm['named']:.0%}")],
        1.0, lambda v: f"{v:.0%}")


def _guardrail(s: Svg, x: float, y: float, gr: dict) -> None:
    total = gr["episodes"][0]
    s.heading(x, y, "Checking what the agent says it did",
              "Conversations that told the customer an action",
              f"happened that never did, of {total} per arm · {_p(gr['p'])}")
    _bars(s, x, y + 54, [
        ("No check", gr["control"], "rest", f"Without the guardrail: {gr['control']} of {total}"),
        ("Claim guardrail", gr["guarded"], "accent", f"With it: {gr['guarded']} of {total}")],
        max(6, gr["control"] + 1), lambda v: f"{v}")


def _finetune(s: Svg, x: float, y: float, ft: dict) -> None:
    """Dumbbells around zero: each model's gain over the base model as first
    scored and with the checker fixed."""
    s.heading(x, y, "A checker quirk the fine-tune learned",
              f"Pass rate minus the base model's, points, {ft['episodes']:,} per arm",
              f"Checker fixed: fine-tuned {_p(ft['fixed']['p_rft'])}, "
              f"reference {_p(ft['fixed']['p_ref'])}")
    lo, hi = -8.0, 6.0
    left, area = x + 136, 220
    px = lambda v: left + area * (v - lo) / (hi - lo)
    top = y + 58
    for v in (-8, -4, 0, 4):
        s.line(px(v), top - 4, px(v), top + 70, color="grid" if v else "axis")
        s.text(px(v), top + 84, _signed(v, 0), size=11, color="muted", anchor="middle")
    rows = [("Fine-tuned", "rft", None), ("API reference", "ref", ft["ref_best_case"])]
    for i, (label, key, best) in enumerate(rows):
        first, fixed = ft["first"][key], ft["fixed"][key]
        yy = top + 14 + i * 34
        s.text(left - 10, yy + 4.5, label, size=13, color="ink2", anchor="end")
        s.line(px(first), yy, px(fixed), yy, color="link", width=2)
        s.dot(px(first), yy, "rest", f"{label}, as first scored: {_signed(first)} points, "
                                     f"{_p(ft['first']['p_' + key])}")
        s.dot(px(fixed), yy, "accent", f"{label}, checker fixed: {_signed(fixed)} points, "
                                       f"{_p(ft['fixed']['p_' + key])}")
        s.text(px(fixed) + (-10 if fixed < first else 10), yy - 9,
               _signed(fixed), size=12, weight=600, anchor="end" if fixed < first else "start")
        if best is not None:
            s.dot(px(best), yy, "accent", f"At best, had the escalation tool refused "
                                          f"invalid inputs (#35): {_signed(best)}", hollow=True)
    # two series and a bound: a legend, always
    ly = top + 104
    for dx, color, hollow, label in ((6, "rest", False, "as first scored"),
                                     (126, "accent", False, "checker fixed"),
                                     (238, "accent", True, "at best (#35)")):
        s.dot(x + dx, ly - 4, color, label, hollow=hollow)
        s.text(x + dx + 10, ly, label, size=12, color="ink2")


def _p_bound(lg: dict) -> str:
    """The smallest p over every language in both runs, rounded down."""
    p_min = min(r["p"] for run in lg.values() for r in run.values())
    return f"p ≥ {math.floor(p_min * 100) / 100:.2f}"


def _language(s: Svg, x: float, y: float, lg: dict) -> list[str]:
    """Dots around zero, one per language, from the full suite's run; the
    subtitle's bound covers both runs."""
    rows = lg[CELLS["language"][0]]
    s.heading(x, y, "No language differed from English",
              "Pass rate minus English's on twin tasks, points,",
              f"every {_p_bound(lg)}, in two independent runs")
    lo, hi = -12.0, 12.0
    left, area = x + 124, 236
    px = lambda v: left + area * (v - lo) / (hi - lo)
    top = y + 58
    order = [k for k in LANGUAGE_NAMES if k in rows]
    for v in (-10, -5, 0, 5, 10):
        s.line(px(v), top - 4, px(v), top + 16 * len(order), color="grid" if v else "axis")
        s.text(px(v), top + 16 * len(order) + 14, _signed(v, 0), size=11,
               color="muted", anchor="middle")
    extreme = max(order, key=lambda k: abs(rows[k]["gap"]))
    for i, k in enumerate(order):
        yy = top + 8 + 16 * i
        r = rows[k]
        s.text(left - 10, yy + 4.5, LANGUAGE_NAMES[k], size=12.5, color="ink2", anchor="end")
        s.dot(px(r["gap"]), yy, "accent",
              f"{LANGUAGE_NAMES[k]}: {_signed(r['gap'])} points over {r['pairs']} pairs, {_p(r['p'])}")
        if k == extreme:
            # the extreme alone is labelled; its p is in the subtitle's bound
            right = r["gap"] > 0
            s.text(px(r["gap"]) + (10 if right else -10), yy + 4.5, _signed(r["gap"]),
                   size=12, weight=600, anchor="start" if right else "end")
    return order


def unsupported(n: dict) -> list[str]:
    """The panel titles the numbers no longer bear out. Two titles state a
    finding rather than name a test; if a re-run moves it, the figure is not
    redrawn under the old words."""
    out = []
    ft = n["finetune"]
    if not (ft["first"]["rft"] > ft["fixed"]["rft"] and ft["fixed"]["p_rft"] >= 0.05):
        out.append("'A checker quirk the fine-tune learned' needs the fine-tune's gain to "
                   "shrink once the checker is fixed, to one the sign test does not resolve: "
                   f"{_signed(ft['first']['rft'])} to {_signed(ft['fixed']['rft'])}, "
                   f"{_p(ft['fixed']['p_rft'])}")
    p_min = min(r["p"] for run in n["language"].values() for r in run.values())
    if p_min < 0.05:
        out.append(f"'No language differed from English' needs every p >= 0.05; one is {_p(p_min)}")
    return out


def render(n: dict, mode: str, layout: str = "wide") -> str:
    t, lay = THEMES[mode], LAYOUTS[layout]
    w, h = lay["size"]
    s = Svg(t)
    nm, gr, ft, lg = n["naming"], n["guardrail"], n["finetune"], n["language"]
    _naming(s, *lay["at"]["naming"], nm)
    _guardrail(s, *lay["at"]["guardrail"], gr)
    _finetune(s, *lay["at"]["finetune"], ft)
    order = _language(s, *lay["at"]["language"], lg)
    s.text(*lay["footer"], "Every number is read from the recorded runs by "
                          "scripts/make_figure.py", size=11, color="muted")

    rows = lg[CELLS["language"][0]]
    total = gr["episodes"][0]
    title = "PasarBench: four results"
    desc = (f"Naming the tools: {nm['control']:.0%} to {nm['named']:.0%} of conversations "
            f"passed ({nm['better']} tasks better, {nm['worse']} worse). Claim guardrail: "
            f"{gr['control']} to {gr['guarded']} conversations of {total} told the customer "
            f"an action happened that never did ({_p(gr['p'])}). Fine-tune: "
            f"{_signed(ft['first']['rft'])} to {_signed(ft['fixed']['rft'])} points over the base model "
            f"once the checker was fixed ({_p(ft['fixed']['p_rft'])}); API reference "
            f"{_signed(ft['first']['ref'])} to {_signed(ft['fixed']['ref'])} "
            f"({_p(ft['fixed']['p_ref'])}), at best {_signed(ft['ref_best_case'])}. Languages: "
            + "; ".join(f"{LANGUAGE_NAMES[k]} {_signed(rows[k]['gap'])}" for k in order)
            + f" points against English, every {_p_bound(lg)}.")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d" '
            f'font-family="{FONT}">\n<title id="t">{escape(title)}</title>\n'
            f'<desc id="d">{escape(desc)}</desc>\n'
            f'<rect width="{w}" height="{h}" rx="8" fill="{t["surface"]}"/>\n'
            + "\n".join(s.parts) + "\n</svg>\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--traces", type=Path, help="the trace root, e.g. traces")
    src.add_argument("--numbers", type=Path, help="a results.json this script wrote")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img")
    a = ap.parse_args(argv)
    numbers = compute(a.traces) if a.traces else json.loads(a.numbers.read_text())
    if wrong := unsupported(numbers):
        print("not drawn -- the numbers no longer bear out a panel title; change the "
              "title with the finding:\n  " + "\n  ".join(wrong))
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    if a.traces:
        (a.out / "results.json").write_text(json.dumps(numbers, indent=1) + "\n")
    written = ["results.json"] if a.traces else []
    for (mode, layout), name in FILES.items():
        (a.out / name).write_text(render(numbers, mode, layout), encoding="utf-8")
        written.append(name)
    print("wrote " + ", ".join(str(a.out / n) for n in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
