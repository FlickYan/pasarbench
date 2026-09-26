"""
Assemble RESULTS.md from whatever has actually been measured.

    python scripts/make_report.py --out RESULTS.md

DESIGN RULE: NEVER FABRICATE.
-----------------------------
Every number in the output comes from a file on disk. Anything not yet
measured prints as `NOT MEASURED` with the exact command that would produce
it. There is no default of 0, no placeholder that looks like a result, and no
interpolation.

This matters more than it sounds. The failure mode is not dishonesty, it is
drift: you paste a scaffold table into a draft, fill three cells, forget the
other five, and ship a blog post containing two numbers you never ran. A
report generator that prints NOT MEASURED in bold is the cheapest defence
against that, and it is also what lets you send the repo to someone mid-project
without misleading them.

Inputs, all optional:
    traces/<run>/summary.json     from pasarbench.sweep
    data/labels/human_round*.jsonl + data/labels/judge_*.jsonl
    data/serving/configs.json     list of ServingConfig kwargs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MISSING = "**NOT MEASURED**"


def _load_json(p: Path) -> Any | None:
    try:
        return json.loads(p.read_text())
    except Exception:                                    # noqa: BLE001
        return None


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _missing(what: str, cmd: str) -> str:
    return f"{MISSING} — {what}\n\n```bash\n{cmd}\n```\n"


# --------------------------------------------------------------------------

def section_suite() -> str:
    from pasarbench.generate import generate
    from pasarbench.locales import NEEDS_NATIVE_REVIEW
    from pasarbench.tasks import TASKS as CORE
    from pasarbench.tools import DEFAULT_TOOLS

    gen, _ = generate()
    langs = sorted({t.language for t in gen} | {t.language for t in CORE})
    return (f"- **{len(CORE) + len(gen)} tasks** "
            f"({len(CORE)} hand-written, {len(gen)} generated)\n"
            f"- **{len({t.trap for t in gen})} traps** across "
            f"**{len({t.market for t in gen})} markets** and "
            f"**{len(langs)} language varieties** ({', '.join(langs)})\n"
            f"- **{len(DEFAULT_TOOLS)} tools**, extendable to 303 with distractors\n"
            f"- Unreviewed translations: {', '.join(sorted(NEEDS_NATIVE_REVIEW))} "
            f"— get these checked before quoting a per-language number\n")


def _episodes(rows: list[dict]) -> int:
    return sum(int(r.get("n_tasks", 0)) * int(r.get("k", 1)) for r in rows)


def _choose(cands: list[tuple[str, int, Any]], pinned: str, flag: str
            ) -> tuple[Any, str] | tuple[None, str]:
    """Pick the run that backs a section, and say which one it was.

    `cands` is (name, episodes, payload) for every qualifying run. The rule used
    to be "first qualifying run in alphabetical order", which silently put a
    smoke test named A-... ahead of the full-suite runs the finding rests on.
    Now: the pinned run if given, else the LARGEST qualifying run -- and the
    report always names it and lists the alternatives, so the choice is visible
    and reversible rather than an accident of directory names.
    """
    if not cands:
        return None, ""
    if pinned:
        hit = [c for c in cands if c[0] == pinned or c[0].split("/")[0] == pinned]
        if not hit:
            raise SystemExit(f"{flag} {pinned!r} does not qualify for this "
                             f"section. Qualifying: {', '.join(c[0] for c in cands)}")
        name, n, payload = max(hit, key=lambda c: c[1])
        why = f"pinned with `{flag}`"
    else:
        name, n, payload = max(cands, key=lambda c: (c[1], c[0]))
        why = (f"the largest of {len(cands)} qualifying run(s); pin another "
               f"with `{flag} <name>`") if len(cands) > 1 else "the only qualifying run"
    others = ", ".join(f"`{c[0]}` ({c[1]})" for c in sorted(cands) if c[0] != name)
    note = (f"*Source: `traces/{name}` — {n} episodes, {why}.*"
            + (f"  \n*Also qualifying: {others}.*" if others else "") + "\n\n")
    return payload, note


def section_context(runs: list[Path], pinned: str = "") -> str:
    from pasarbench.analyze import markdown_report
    cands = []
    for run in runs:
        rows = _load_json(run / "summary.json")
        if rows and len(rows) > 1 and any(r.get("exposure", "default") == "default"
                                          for r in rows):
            cands.append((run.name, _episodes(rows), rows))
    rows, note = _choose(cands, pinned, "--context-run")
    if rows is not None:
        return note + markdown_report(rows, baseline=rows[0]["strategy"])
    return _missing(
        "context ablation not run against a real model",
        "python -m pasarbench.sweep --backend openai --model <m> --suite all \\\n"
        "    --strategies full,window8,window4,trim3,notes4 --sample 2 -k 3")


def section_tools(runs: list[Path], pinned: str = "") -> str:
    from pasarbench.diagnose import tool_scaling_report
    cands = []
    for run in runs:
        rows = _load_json(run / "summary.json")
        if rows and any(r.get("exposure", "default") != "default" for r in rows):
            cands.append((run.name, _episodes(rows), (rows, run)))
    picked, note = _choose(cands, pinned, "--tools-run")
    if picked is not None:
        rows, run = picked
        # The template already heads this section; drop the report's own.
        return note + tool_scaling_report(rows, run_dir=run).replace(
            "## Tool scaling\n", "", 1)
    return _missing(
        "tool-scaling arms not run",
        "python -m pasarbench.sweep --backend openai --model <m> --suite all \\\n"
        "    --strategies full --exposure oracle,all-20,all-100,random-100,search-300")


def section_multilingual(runs: list[Path], judge: list[dict],
                         pinned: str = "") -> str:
    from pasarbench.diagnose import load_episodes, report
    cands = []
    for run in runs:
        for cell in sorted(run.iterdir()):
            # Only the unmodified agent. A window4 or all-100 cell degrades the
            # agent on purpose, so a language gap measured there is confounded
            # with the ablation -- and every ablation run spans a few languages,
            # so without this they all "qualify".
            if not cell.is_dir() or cell.name != "full":
                continue
            eps = load_episodes(cell)
            if eps and len({e["language"] for e in eps}) > 1:
                cands.append((f"{run.name}/{cell.name}", len(eps), eps))
    eps, note = _choose(cands, pinned, "--multilingual-run")
    if eps is not None:
        return note + report(eps, judge or None)
    return _missing(
        "no multi-language run found in traces/",
        "python -m pasarbench.sweep --backend openai --model <m> --suite all -k 3")


def _labellers(rows: list[dict]) -> set[str]:
    return {r.get("labeller") or "?" for r in rows}


def _strictness(rows: list[dict], keys: list[str]) -> float | None:
    """Fraction of criterion judgements this rater marked SATISFIED.

    Two raters can produce the same kappa for opposite reasons, so this is the
    first thing to look at when a second rater is involved. A rater who marks
    nearly everything satisfied is either seeing a genuinely clean sample or is
    not engaging with the rubric, and the two look identical in the agreement
    table. Comparing the rates across raters separates them.
    """
    n = hit = 0
    for r in rows:
        for k in keys:
            v = r.get("labels", {}).get(k)
            if v is not None:
                n += 1
                hit += bool(v)
    return hit / n if n else None


def _retest_gap_hours(r1: list[dict], r2: list[dict]) -> float | None:
    """Median hours between the first and second labelling of a transcript.

    Only meaningful when BOTH rounds are the same person. Then it says whether
    round 2 measures independent re-application of the rubric or recall of the
    round-1 answer. Across two different people there is nothing to recall and
    the gap carries no information, so the caller checks the labellers first.

    Returns None if the `at` timestamps are missing (labels written before
    that field existed) or no transcript appears in both rounds.
    """
    from datetime import datetime

    def when(rec: dict) -> datetime | None:
        try:
            return datetime.fromisoformat(rec["at"])
        except (KeyError, TypeError, ValueError):
            return None

    first = {r["transcript_id"]: when(r) for r in r1}
    gaps = []
    for rec in r2:
        a, b = first.get(rec["transcript_id"]), when(rec)
        if a and b:
            gaps.append(abs((b - a).total_seconds()) / 3600.0)
    if not gaps:
        return None
    gaps.sort()
    mid = len(gaps) // 2
    return gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2


def _reading(note: str) -> str:
    """The verdict word from an agreement note ('moderate', 'KAPPA PARADOX').

    The full note is a sentence of guidance; in a table cell it gets cut
    mid-word. The word is what the table needs; the guidance is in agreement.py.
    """
    return note.split(" -- ")[0].split(":")[0].strip()


def _naive_vs_decomposed(r1: list[dict], naive: list[dict],
                         dec: list[dict]) -> list[str]:
    """The naive baseline, scored like for like.

    Rule 4 says report the naive baseline; this is where it lives. Everything
    goes onto ONE target -- the human's round-1 labels collapsed to the same
    acceptable/not bit the naive judge emits -- and the decomposed judge is
    collapsed by the same rule, so the two kappas are about the same question
    on the same transcripts. They are then compared with a paired bootstrap of
    the difference, because two CIs on shared transcripts are the wrong test.

    Nothing here is allowed to vanish silently. The earlier version compared
    the naive bit against nine-criterion labels, found no shared keys, and
    printed nothing at all; 200 judgements dropped out of the report.
    """
    from pasarbench.judge.agreement import (agreement, confusion,
                                            paired_kappa_diff)
    from pasarbench.judge.rubric import is_complete, overall_acceptable

    human = {r["transcript_id"]: overall_acceptable(r["labels"])
             for r in r1 if is_complete(r.get("labels", {}))}
    # An errored decomposed judgement has blank labels, which the verdict rule
    # would read as "no violations" -- a free pass. Incomplete means excluded.
    dec_bit = {r["transcript_id"]: overall_acceptable(r["labels"])
               for r in dec if is_complete(r.get("labels", {}))}
    # A naive response that did not parse scores 0, and 0 is below any
    # threshold, so it would enter the table as a confident "unacceptable".
    # It is not a judgement. Exclude it and say how many.
    naive_bit, bad = {}, 0
    for r in naive:
        s, lab = r.get("score"), r.get("labels", {})
        if isinstance(s, int) and 1 <= s <= 5 and "overall_acceptable" in lab:
            naive_bit[r["transcript_id"]] = bool(lab["overall_acceptable"])
        else:
            bad += 1

    ids = sorted(set(human) & set(naive_bit) & set(dec_bit))
    if not ids:
        return ["\nNaive baseline: " + MISSING + " — judge output exists but "
                "shares no scorable transcript with the round-1 labels. "
                "Check that `transcript_id`s match."]

    h = [human[i] for i in ids]
    nv = [naive_bit[i] for i in ids]
    dv = [dec_bit[i] for i in ids]
    an, ad = agreement(h, nv, bootstrap=1000), agreement(h, dv, bootstrap=1000)
    p = paired_kappa_diff(h, nv, dv)

    def row(name: str, a) -> str:
        ci = f"[{a.ci95[0]:+.2f}, {a.ci95[1]:+.2f}]" if a.ci95 else "-"
        return (f"| {name} | {a.p_o:.3f} | {a.kappa:+.3f} | {ci} | "
                f"{_reading(a.note)} |")

    out = [f"\n**Naive baseline, like for like** — n = {len(ids)}. Target: "
           f"the human's round-1 labels collapsed to acceptable / not by the "
           f"verdict rule (no critical violation, at most two minor); "
           f"{sum(h) / len(h):.0%} of transcripts are acceptable on it."]
    if bad:
        out.append(f"{bad} naive response(s) had no valid 1–5 score and are "
                   f"excluded rather than counted as a fail.")
    out += ["", "| judge | p_o | kappa | CI95 | reading |", "|---|---|---|---|---|",
            row("naive: one 1–5 score, thresholded", an),
            row("decomposed: nine criteria, collapsed by the same rule", ad)]

    # Which way does the collapsed decomposed judge miss? The two directions
    # have different causes, so read it off the data instead of guessing.
    cf = confusion(h, dv)
    harsh, lenient = cf["human_yes_judge_no"], cf["human_no_judge_yes"]
    out.append(f"\nDecomposed judge's {harsh + lenient} disagreements: "
               f"**{harsh} too harsh** (human acceptable, judge not), "
               f"**{lenient} too lenient** (the reverse).")

    lo, hi = p["ci95"]
    head = (f"\nDecomposed minus naive: {p['diff']:+.3f}, paired 95% CI "
            f"[{lo:+.3f}, {hi:+.3f}] — ")
    if not p["resolved"]:
        out.append(head + f"**not resolved at n = {p['n']}.** On the overall "
                   "verdict the two judges are indistinguishable. What the "
                   "decomposition buys is the per-criterion table above — "
                   "knowing *which* part of the rubric disagrees — not a better "
                   "overall call. Say that; do not claim it beats the baseline.")
    elif p["diff"] > 0:
        out.append(head + "**resolved: the decomposed judge agrees with the "
                   "human better than a single score does**, on the same "
                   "transcripts and the same target.")
    elif harsh >= lenient:
        out.append(head + "**resolved, in the naive judge's favour — and the "
                   "decomposed judge errs harsh.** That is the signature of "
                   "compounding: the verdict rule fails a transcript on ANY "
                   "critical violation (or three minor ones), so a judge with "
                   "a modest false-alarm rate per criterion gets nine chances "
                   "to fail a good transcript. The *harsh* column of the per-criterion table "
                   "shows which criteria fire the false alarms; tightening "
                   "those, not the whole rubric, is the fix. It is also a "
                   "finding worth reporting as is: per-criterion accuracy does "
                   "not survive a disjunctive collapse.")
    else:
        out.append(head + "**resolved, in the naive judge's favour — and the "
                   "decomposed judge errs lenient.** Check the unparsed count "
                   "from run_judges first: an unparsed criterion defaults to "
                   "SATISFIED, and every such default can only move the "
                   "verdict toward acceptable.")
    return out


def section_judge() -> str:
    from pasarbench.judge.agreement import ceiling_report, per_criterion
    from pasarbench.judge.rubric import CRITERIA

    root = Path("data/labels")
    r1 = _load_jsonl(root / "human_round1.jsonl")
    r2 = _load_jsonl(root / "human_round2.jsonl")
    naive = _load_jsonl(root / "judge_naive.jsonl")
    dec = _load_jsonl(root / "judge_decomposed.jsonl")

    if not r1:
        return _missing(
            "no human labels yet — this is the week-5 grind and nothing "
            "substitutes for it",
            "python -m pasarbench.judge.label sample --traces traces/<run>/full --n 200\n"
            "python -m pasarbench.judge.label annotate --round 1")

    keys = [c.key for c in CRITERIA]
    out = [f"Human labels: round 1 = {len(r1)}, round 2 (retest) = {len(r2)}\n"]

    if dec:
        from pasarbench.judge.agreement import confusion
        hmap = {r["transcript_id"]: r["labels"] for r in r1}
        dmap = {r["transcript_id"]: r.get("labels", {}) for r in dec}
        shared = sorted(set(hmap) & set(dmap))
        # harsh / lenient split each criterion's disagreements by direction.
        # kappa says how much a criterion disagrees; direction says why, and
        # which way it pushes the collapsed verdict.
        out.append("| criterion | p_o | kappa | CI95 | prevalence | "
                   "harsh | lenient | reading |")
        out.append("|---|---|---|---|---|---|---|---|")
        for key, a in per_criterion(r1, dec, keys).items():
            ci = f"[{a.ci95[0]:.2f}, {a.ci95[1]:.2f}]" if a.ci95 else "-"
            pairs = [(hmap[i].get(key), dmap[i].get(key)) for i in shared]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            cf = confusion([x for x, _ in pairs], [y for _, y in pairs])
            out.append(f"| `{key}` | {a.p_o:.3f} | {a.kappa:+.3f} | {ci} | "
                       f"{a.prevalence:.2f} | {cf['human_yes_judge_no']} | "
                       f"{cf['human_no_judge_yes']} | "
                       f"{_reading(a.note)} |")
        out.append("\n*harsh* = human satisfied, judge violated. "
                   "*lenient* = the reverse.")
    else:
        out.append(_missing("decomposed judge not run against the labels",
                            "# score the labelled sample with DecomposedJudge, "
                            "write data/labels/judge_decomposed.jsonl"))

    if naive and dec:
        out.extend(_naive_vs_decomposed(r1, naive, dec))
    else:
        out.append("\nNaive baseline: " + MISSING +
                   " — without it the decomposed judge has nothing to beat")

    if r2:
        c = ceiling_report(r1, r2, dec or r1, keys)
        who1, who2 = _labellers(r1), _labellers(r2)
        # One rater in both rounds = test-retest. Two raters = inter-annotator.
        # Same arithmetic, different quantity, opposite bias on the fraction --
        # so the report must say which one it is rather than call both "ceiling".
        unknown = "?" in who1 or "?" in who2
        same_person = not unknown and who1 == who2
        kind = ("rater unknown" if unknown
                else "test-retest (one rater, twice)" if same_person
                else f"inter-annotator ({' / '.join(sorted(who1))} vs "
                     f"{' / '.join(sorted(who2))})")

        out.append(f"\n**Ceiling — {kind}:** mean human-human kappa "
                   f"{c['mean_self_kappa']:+.3f}, mean judge kappa "
                   f"{c['mean_judge_kappa']:+.3f}, "
                   f"**{(c['mean_fraction_of_ceiling'] or 0):.0%} of achievable "
                   f"agreement**")

        s1, s2 = _strictness(r1, keys), _strictness(r2, keys)
        if s1 is not None and s2 is not None:
            out.append(f"\nSatisfied rate: round 1 {s1:.1%}, round 2 {s2:.1%} "
                       f"(gap {abs(s1 - s2):+.1%} in round 2's favour "
                       f"{'—' if abs(s1 - s2) < 0.08 else '— LOOK AT THIS:'} "
                       f"{'raters are similarly strict' if abs(s1 - s2) < 0.08 else 'the raters are not applying the same bar'})")

        if same_person:
            gap = _retest_gap_hours(r1, r2)
            if gap is None:
                out.append("\nRetest gap: unknown (labels carry no `at` "
                           "timestamp). Do not quote the ceiling without "
                           "establishing it.")
            else:
                out.append(f"\nRetest gap: median {gap:.1f} h between first "
                           f"and second labelling of the same transcript.")
            if gap is not None and gap < 12:
                out.append(
                    "\n> **Read the ceiling as a lower bound on the judge, not "
                    "as a measurement of human consistency.** Round 2 ran in "
                    "the same sitting as round 1, so the human-human kappa "
                    "partly reflects memory of the round-1 answer rather than "
                    "independent re-application of the rubric. That biases it "
                    "*upward*, and since it is the denominator, "
                    "`fraction_of_ceiling` is biased *downward* — the judge is "
                    "being reported conservatively. What cannot be claimed is "
                    "the ceiling itself. For a usable one, re-label a freshly "
                    "drawn subset after at least a day:\n"
                    ">\n"
                    "> ```\n"
                    "> mv data/labels/human_round2.jsonl "
                    "data/labels/human_round2_samesession.jsonl\n"
                    "> python -m pasarbench.judge.label annotate --round 2 "
                    "--retest 30 --seed 7\n"
                    "> ```\n"
                    ">\n"
                    "> The new seed is what draws a different subset; reusing "
                    "the default would reshuffle to the same 30 transcripts "
                    "and reproduce the problem.")
        elif unknown:
            out.append(
                "\n> **Who labelled which round is not recorded**, so this "
                "number cannot be interpreted. One rater labelling twice and "
                "two raters labelling once produce the same arithmetic and "
                "opposite biases. Stamp the rounds (`--labeller <name>`, or "
                "edit the `labeller` field in the existing JSONL) and "
                "regenerate.")
        else:
            out.append(
                "\n> **This is inter-annotator agreement, not test-retest.** "
                "A second person labelled round 2 blind, which is the stronger "
                "measurement for a benchmark meant to be used by other people: "
                "it asks whether the rubric is specified well enough that an "
                "independent reader converges on it, rather than whether one "
                "reader is self-consistent. The judge is itself a second "
                "rater, so this is the like-for-like comparison — both are "
                "independent raters trying to reproduce the round-1 labels.\n"
                ">\n"
                "> Bias runs the *opposite* way from a same-sitting retest. "
                "Two people agree less than one person with themselves, so the "
                "denominator is smaller and `fraction_of_ceiling` is flattered "
                "— do not report it as 'fraction of human self-consistency'. "
                "Two checks before quoting it: (1) the satisfied-rate line "
                "above, which catches a second rater who waved everything "
                "through; (2) the per-language table below — a rater who "
                "cannot read a language cannot judge `language_match` in it, "
                "and pressing enter defaults to SATISFIED.")

            by_lang: dict[str, list] = {}
            lang1 = {r["transcript_id"]: r.get("language", "?") for r in r1}
            for rec in r2:
                by_lang.setdefault(lang1.get(rec["transcript_id"], "?"),
                                   []).append(rec)
            if len(by_lang) > 1:
                out.append("\n| language | n in round 2 | round-2 satisfied rate |")
                out.append("|---|---|---|")
                rates = {}
                for lang in sorted(by_lang):
                    rows = by_lang[lang]
                    sr = _strictness(rows, keys)
                    cell = f"{sr:.1%}" if sr is not None else "-"
                    out.append(f"| `{lang}` | {len(rows)} | {cell} |")
                    if sr is not None and len(rows) >= 3:
                        rates[lang] = sr
                # The aggregate satisfied rate averages this away: a rater who
                # waves through four languages and reads three can still land
                # within a few points of the reference overall. The spread is
                # what shows it, so state the reading rather than leaving the
                # table to be eyeballed.
                if len(rates) > 1:
                    hi = max(rates, key=rates.get)
                    lo = min(rates, key=rates.get)
                    spread = rates[hi] - rates[lo]
                    if spread >= 0.15:
                        out.append(
                            f"\n> Spread across languages is {spread:.0%} "
                            f"(`{hi}` {rates[hi]:.0%} vs `{lo}` "
                            f"{rates[lo]:.0%}). A second rater should not be "
                            f"systematically more lenient in one language than "
                            f"another unless the agent really is better in it — "
                            f"and the round-1 labels say whether it is. If the "
                            f"lenient languages are the ones the rater does not "
                            f"read, those rows are not a ceiling, they are a "
                            f"default. Drop them from the ceiling and say so, "
                            f"or have them relabelled by someone who reads the "
                            f"language.")
                    else:
                        out.append(f"\n> Spread across languages is "
                                   f"{spread:.0%} — no sign of a language the "
                                   f"second rater waved through.")
    else:
        out.append("\n**Ceiling:** " + MISSING +
                   " — run `annotate --round 2 --retest 30` blind, either a "
                   "week later or by a second labeller")
    return "\n".join(out)


def section_serving() -> str:
    from pasarbench.serving.cost import ServingConfig, compare, markdown
    cfgs = _load_json(Path("data/serving/configs.json"))
    if not cfgs:
        return _missing(
            "serving configurations not measured",
            "./scripts/serve_vllm.sh agent-32b\n"
            "# snapshot /metrics, run the sweep, snapshot again, then write\n"
            "# data/serving/configs.json as a list of ServingConfig kwargs")
    return markdown(compare([ServingConfig(**c) for c in cfgs]))


def section_training() -> str:
    stats = _load_json(Path("data/rft/stats.json"))
    if not stats:
        return _missing(
            "no rollouts collected — post-training has not started",
            "./scripts/serve_vllm.sh agent-8b\n"
            "python scripts/train_rft.py collect --k 8 --temperature 1.0")
    dead = stats.get("traps_with_zero_signal") or []
    out = [f"Rollouts: {stats['rollouts']}, pass rate {stats['pass_rate']:.3f}, "
           f"mean reward {stats['mean_reward']:+.3f}, "
           f"{stats['gated_by_forbidden']} gated by a forbidden action\n"]
    if dead:
        out.append(f"**{len(dead)} trap(s) never solved**, so RFT cannot teach "
                   f"them and they will still fail after training: "
                   f"{', '.join('`' + d + '`' for d in dead)}\n")
    out.append("| model | pass^1 | pass^4 | $/resolved |")
    out.append("|---|---|---|---|")
    for name in ("base 8B", "RFT 8B", "GRPO 8B", "72B reference"):
        out.append(f"| {name} | {MISSING} | {MISSING} | {MISSING} |")
    out.append("\nThe gap-closing claim goes here once all four rows are filled.")
    return "\n".join(out)


# --------------------------------------------------------------------------

TEMPLATE = """# PasarBench — results

Generated by `scripts/make_report.py`. Every number below comes from a file on
disk. Anything marked {missing} has not been run, and the command to produce it
is shown inline. **Do not fill these in by hand** — re-run and regenerate.

## The suite

{suite}

## 1. Context ablation

{context}

## 2. Tool scaling

{tools}

## 3. Multilingual diagnosis

{multilingual}

## 4. Judge calibration

{judge}

## 5. Serving and cost

{serving}

## 6. Post-training

{training}

---

## Reporting rules this project holds itself to

1. **Per-trap before aggregate.** A 4-point drop spread over 16 traps and a
   4-point drop concentrated in two are different findings, and only the second
   identifies a mechanism.
2. **Co-movement is not causation.** The multilingual table narrows which
   traces to read. The mechanism is established by reading them.
3. **No speed claim without a quality verdict.** At this suite size a 2-point
   regression is undetectable; INCONCLUSIVE is a real and frequent answer.
4. **Report the naive baseline.** A judge without its 1-5 baseline, or a GRPO
   number without its RFT baseline, cannot be evaluated by the reader.
5. **Name what failed.** See `docs/WHAT_FAILED.md`.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--out", default="RESULTS.md")
    ap.add_argument("--context-run", default="",
                    help="run under traces/ for section 1 (default: largest)")
    ap.add_argument("--tools-run", default="",
                    help="run under traces/ for section 2 (default: largest)")
    ap.add_argument("--multilingual-run", default="",
                    help="run, or run/cell, for section 3 (default: largest)")
    args = ap.parse_args()

    runs = sorted(p for p in Path(args.traces).glob("*") if p.is_dir()) \
        if Path(args.traces).exists() else []
    judge = _load_jsonl(Path("data/labels/judge_decomposed.jsonl"))

    sections = {
        "context": ("1. Context ablation",
                    section_context(runs, args.context_run)),
        "tools": ("2. Tool scaling", section_tools(runs, args.tools_run)),
        "multilingual": ("3. Multilingual diagnosis",
                         section_multilingual(runs, judge, args.multilingual_run)),
        "judge": ("4. Judge calibration", section_judge()),
        "serving": ("5. Serving and cost", section_serving()),
        "training": ("6. Post-training", section_training()),
    }
    body = TEMPLATE.format(missing=MISSING, suite=section_suite(),
                           **{k: text for k, (_, text) in sections.items()})
    Path(args.out).write_text(body)

    # Account per SECTION, not per marker. Counting markers in the body
    # counted the legend line that explains the marker, so the count could
    # never reach zero -- and one section can hold several markers (the
    # post-training table has twelve), so a count says nothing about what
    # is left to run anyway.
    gpu = {"serving", "training"}
    done, partly, api, needs_gpu = [], [], [], []
    for key, (name, text) in sections.items():
        if MISSING not in text:
            done.append(name)
        elif not text.lstrip().startswith(MISSING):
            partly.append(name)
        else:
            (needs_gpu if key in gpu else api).append(name)

    print(f"wrote {args.out}")
    for label, names in (("measured      ", done), ("partly        ", partly),
                         ("not run (API) ", api), ("not run (GPU) ", needs_gpu)):
        if names:
            print(f"  {label} {', '.join(names)}")
    if partly or api or needs_gpu:
        print("Each gap prints the command that fills it in the report. Run "
              "those and regenerate; never hand-fill.")
    else:
        print("every section is backed by measured data")


if __name__ == "__main__":
    main()
