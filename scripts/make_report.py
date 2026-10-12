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

ONE CHECKER FOR EVERY NUMBER. Traces record the verdict the checker of the day
gave each episode. When a check is corrected, those verdicts no longer mean the
same thing (WHAT_FAILED #30), so by default every table here is built on
today's checks, re-scored from each episode's recorded tool calls
(pasarbench/rescore.py), and section 7 shows what moved against the recorded
verdicts. `--checker recorded` builds the whole report on the recorded ones.
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
        top = max(c[1] for c in cands)
        tied = sorted(c for c in cands if c[1] == top)
        # A tie is broken alphabetically -- deterministic, but ARBITRARY, and
        # the report says so. Breaking it silently once put a broken run
        # (the gated one) behind a finding (WHAT_FAILED #25).
        name, n, payload = tied[0]
        if len(tied) > 1:
            why = (f"**a tie at {n} episodes with "
                   f"{', '.join('`' + c[0] + '`' for c in tied[1:])} — the pick is "
                   f"arbitrary; pin the run you mean with `{flag} <name>`**")
        elif len(cands) > 1:
            why = (f"the largest of {len(cands)} qualifying run(s); pin another "
                   f"with `{flag} <name>`")
        else:
            why = "the only qualifying run"
    others = ", ".join(f"`{c[0]}` ({c[1]})" for c in sorted(cands) if c[0] != name)
    note = (f"*Source: `traces/{name}` — {n} episodes, {why}.*"
            + (f"  \n*Also qualifying: {others}.*" if others else "") + "\n\n")
    return payload, note


def section_noise(traces: Path, pair: str) -> str:
    """The same configuration, run twice: how far a result moves on its own.

    Named explicitly, never inferred -- the report cannot tell from traces that
    two runs used the same model, simulator and flags, and a noise floor built
    from two DIFFERENT configurations would be a finding dressed as noise.
    """
    if not pair:
        return ""
    from pasarbench.diagnose import load_episodes, paired_episodes
    a, b = [x.strip() for x in pair.split(",")][:2]
    ea, eb = load_episodes(traces / a), load_episodes(traces / b)
    if not ea or not eb:
        return f"\n## Noise floor\n\n{MISSING} — no episodes under `{a}` or `{b}`\n"
    g = paired_episodes(ea, eb)
    if not g:
        return f"\n## Noise floor\n\n{MISSING} — `{a}` and `{b}` share too few tasks\n"
    pa, pb = sum(e["passed"] for e in ea), sum(e["passed"] for e in eb)
    return (f"\n## Noise floor\n\n*The same configuration run twice: `{a}` and "
            f"`{b}` (identical by your assertion, not by inference).*\n\n"
            f"{pa}/{len(ea)} vs {pb}/{len(eb)} passed: {g['diff']:+.3f}, "
            f"{g['better']} tasks better and {g['worse']} worse in the second run, "
            f"sign test p={g['p']:.3f} over {g['n']} shared tasks. A gap between "
            f"arms of about this size is what re-running does on its own; it is "
            f"not a finding until it clears its own paired test.\n")


def _rows_now(rows: list[dict], run: Path) -> list[dict]:
    """summary.json holds each arm's pass rates as the sweep scored them. Under
    today's checker they are recomputed from the arm's episodes, so a table's
    rates and the paired tests beside it come from the same verdicts."""
    from collections import defaultdict

    from pasarbench.diagnose import CHECKER, _arm_episodes
    if CHECKER == "recorded":
        return rows
    out = []
    for r in rows:
        eps = _arm_episodes(r, run)
        if not eps:
            # Nothing to re-score: the row keeps the sweep's rates, and says so
            # (_kept_note) instead of passing them off as today's.
            out.append({**r, "scored": "recorded"})
            continue
        by_task, groups = defaultdict(list), {k: defaultdict(list) for k in
                                             ("trap", "language", "market")}
        for e in eps:
            by_task[e["task_id"]].append(e["passed"])
            for k, g in groups.items():
                g[e[k]].append(e["passed"])
        rate = {k: {x: round(sum(v) / len(v), 3) for x, v in sorted(g.items())}
                for k, g in groups.items()}
        out.append({**r, "pass^1": round(sum(e["passed"] for e in eps) / len(eps), 4),
                    "pass^k": round(sum(all(v) for v in by_task.values()) / len(by_task), 4),
                    "per_trap": rate["trap"], "per_language": rate["language"],
                    "per_market": rate["market"], "pass^1_recorded": r.get("pass^1")})
    return out


def _kept_note(rows: list[dict]) -> str:
    """Arms whose rates could not be re-scored, named under the table."""
    kept = [r.get("strategy") if r.get("exposure", "default") == "default"
            else r.get("exposure") for r in rows if r.get("scored") == "recorded"]
    if not kept:
        return ""
    return (f"\n\n*Not re-scored: {', '.join(f'`{k}`' for k in kept)} — no episodes on "
            f"disk, so {'its' if len(kept) == 1 else 'their'} rates are the ones the "
            f"sweep recorded, under the checker of the day.*\n")


def _conclusion(g: dict) -> str:
    return (("better" if g["diff"] > 0 else "worse") if g["resolved"]
            else "inconclusive")


def _checker_contrast(contrasts: list[tuple[str, dict, dict]], run: Path) -> str:
    """The section's paired tests on the verdicts the run recorded as well,
    and every conclusion the checker change moves (WHAT_FAILED #30). Nothing
    under --checker recorded."""
    from pasarbench.diagnose import CHECKER, paired_arm_gap
    if CHECKER != "current":
        return ""
    moved = []
    for label, a, b in contrasts:
        now = paired_arm_gap(a, b, run)
        was = paired_arm_gap(a, b, run, metric="passed_recorded")
        if not now or not was or now.get("unit") != "task":
            continue
        if (was["worse"], was["better"], round(was["diff"], 9)) != \
                (now["worse"], now["better"], round(now["diff"], 9)):
            moved.append((label, was, now))
    if not moved:
        return ("\n\n*On the verdicts the run recorded, every paired test above comes "
                "out the same.*\n" if contrasts else "")
    out = ["\n\n**The paired tests that differ on the verdicts the run recorded** "
           "(the checker of the day; section 7 says what moved):\n",
           "| contrast | recorded | today's checks |", "|---|---|---|"]
    for label, was, now in moved:
        out.append(f"| {label} | " + " | ".join(
            f"{g['diff']:+.3f}; {g['worse']} worse, {g['better']} better, p={g['p']:.3f}"
            for g in (was, now)) + " |")
    flips = [f"{label}: {_conclusion(was)} (p={was['p']:.3f}) recorded, "
             f"{_conclusion(now)} (p={now['p']:.3f}) now"
             for label, was, now in moved if _conclusion(was) != _conclusion(now)]
    if flips:
        out.append("\n**The checker change moves a conclusion here:** "
                   + "; ".join(flips) + ".")
    return "\n".join(out) + "\n"


def section_context(runs: list[Path], pinned: str = "") -> str:
    from pasarbench.analyze import markdown_report
    cands = []
    for run in runs:
        rows = _load_json(run / "summary.json")
        if rows and len(rows) > 1 and any(r.get("exposure", "default") == "default"
                                          for r in rows):
            cands.append((run.name, _episodes(rows), (rows, run)))
    picked, note = _choose(cands, pinned, "--context-run")
    if picked is not None:
        from pasarbench.diagnose import paired_arm_gap
        rows, run = picked
        rows = _rows_now(rows, run)
        base = rows[0]
        paired = {r["strategy"]: {"pass": paired_arm_gap(base, r, run),
                                  "tokens": paired_arm_gap(base, r, run, metric="tokens")}
                  for r in rows[1:]}
        return (note + markdown_report(rows, baseline=base["strategy"], paired=paired)
                + _kept_note(rows)
                + _checker_contrast([(f"`{r['strategy']}` vs `{base['strategy']}`",
                                      base, r) for r in rows[1:]], run))
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
        rows = _rows_now(rows, run)
        # The contrasts the reading draws, in its order.
        idx = {r.get("exposure"): r for r in rows}
        pairs = [(a, b) for n in (20, 50, 100, 300)
                 for a, b in (("oracle", f"all-{n}"), ("oracle", f"random-{n}"),
                              (f"random-{n}", f"all-{n}"))]
        pairs += [("all-20", x) for x in idx if str(x).startswith("search-")]
        contrasts = [(f"`{b}` vs `{a}`", idx[a], idx[b]) for a, b in pairs
                     if a in idx and b in idx]
        # The template already heads this section; drop the report's own.
        return (note + tool_scaling_report(rows, run_dir=run).replace(
            "## Tool scaling\n", "", 1) + _kept_note(rows)
            + _checker_contrast(contrasts, run))
    return _missing(
        "tool-scaling arms not run",
        "python -m pasarbench.sweep --backend openai --model <m> --suite all \\\n"
        "    --strategies full --exposure oracle,all-20,all-100,random-100,search-300")


def _replication(chosen: list[tuple[str, list[dict], str]]) -> str:
    """The paired language gap in every pinned run, side by side.

    One run's gap is a measurement; the same gap in an independent run is a
    finding. And a gap that flips sign between runs of identical tasks is the
    run-to-run noise, measured.
    """
    from pasarbench.diagnose import _sign_p, paired_language_gap
    per = {name: paired_language_gap(eps)["languages"] for name, eps, _ in chosen}
    names = [n for n, _, _ in chosen]
    langs = sorted({l for v in per.values() for l in v})
    out = ["### Replication: the paired gap in each run\n",
           "Gap = English minus this language on the SAME tasks (positive: this "
           "language does worse). Sign test over the twin pairs that differ.\n",
           "| lang | " + " | ".join(f"`{n}`" for n in names) + " | reading |",
           "|---|" + "---|" * len(names) + "---|"]
    for lang in langs:
        cells, res = [], []
        for n in names:
            r = per[n].get(lang)
            if not r or not r.get("pairs"):
                cells.append("-")
                continue
            p = _sign_p(r["pairs_where_worse"], r["pairs_where_better"])
            res.append((r["paired_gap"], p, n))
            cells.append(f"{r['paired_gap']:+.3f} ({r['pairs_where_worse']}↓ "
                         f"{r['pairs_where_better']}↑, p={p:.2f})")
        sig = [g for g, p, _ in res if p < 0.05]
        if len(res) < 2:
            reading = "one run only — not replicated"
        elif len(sig) == len(res) and len({g > 0 for g in sig}) == 1:
            reading = "**replicates**"
        elif not sig:
            flips = len({g > 0 for g, _, _ in res if abs(g) > 1e-9}) > 1
            reading = "null in every run" + (" — and the sign flips" if flips else "")
            # "null" must not hide a large gap that only just missed: say which
            # run came close, so nobody reads +0.45 at p=0.07 as nothing.
            g, p, n = min(res, key=lambda x: x[1])
            if p < 0.10:
                reading += f"; closest: `{n}` {g:+.3f}, p={p:.2f}"
        else:
            reading = "significant in some runs only — does not replicate"
        out.append(f"| `{lang}` | " + " | ".join(cells) + f" | {reading} |")
    # A run whose customer was gated measures the gate. Say so from the traces
    # themselves, rather than trusting whoever reads the table to remember.
    gated = [n for n, eps, _ in chosen
             if any("+gated" in str(e.get("simulator", "")) for e in eps)]
    if gated:
        out.append("")
        out.append("> " + ", ".join(f"`{n}`" for n in gated)
                   + " ran with the fact gate (`+gated` in the trace headers): the "
                   "customer withholds each fact until the ask-patterns recognise a "
                   "request for it, so a request they miss stalls the customer. "
                   "Check the stall rate by language before reading its gaps: "
                   "`python scripts/inspect_trace.py traces/<run> --leaks-only`. "
                   "In run G the patterns could not read how the agent asks in "
                   "Indonesian and Chinese, and its gaps there measure the gate, "
                   "not the language (WHAT_FAILED #26).")
    return "\n".join(out)


def _language_flips(chosen: list[tuple[str, list[dict], str]]) -> str:
    """The paired language gaps on the verdicts each run recorded, where they
    differ from today's (WHAT_FAILED #30)."""
    from pasarbench.diagnose import CHECKER, _sign_p, paired_language_gap
    if CHECKER != "current":
        return ""
    rows, flips = [], []
    for name, eps, _ in chosen:
        now = paired_language_gap(eps)["languages"]
        was = paired_language_gap([{**e, "passed": e["passed_recorded"]} for e in eps]
                                  )["languages"]
        for lang in sorted(set(now) & set(was)):
            n, w = now[lang], was[lang]
            if not n.get("pairs") or not w.get("pairs"):
                continue
            key = lambda r: (r["paired_gap"], r["pairs_where_worse"], r["pairs_where_better"])
            if key(n) == key(w):
                continue
            pn = _sign_p(n["pairs_where_worse"], n["pairs_where_better"])
            pw = _sign_p(w["pairs_where_worse"], w["pairs_where_better"])
            rows.append(f"| `{name}` | `{lang}` | {w['paired_gap']:+.3f} (p={pw:.2f}) | "
                        f"{n['paired_gap']:+.3f} (p={pn:.2f}) |")
            if (pw < 0.05) != (pn < 0.05):
                flips.append(f"`{name}` `{lang}`: p={pw:.2f} recorded, p={pn:.2f} now")
    if not rows:
        return "\n\n*On the verdicts the runs recorded, every paired gap comes out the same.*"
    out = ["\n\n**Paired gaps that differ on the verdicts the runs recorded** "
           "(English minus the language; section 7 says what moved):\n",
           "| run | lang | recorded | today's checks |", "|---|---|---|---|"] + rows
    if flips:
        out.append("\n**The checker change moves a conclusion here:** "
                   + "; ".join(flips) + ".")
    return "\n".join(out)


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
    pins = [x.strip() for x in pinned.split(",") if x.strip()] if pinned else [""]
    chosen = []
    for pin in pins:
        eps, note = _choose(cands, pin, "--multilingual-run")
        if eps is not None:
            name = next(c[0] for c in cands if c[2] is eps)
            chosen.append((name, eps, note))
    if chosen:
        name, eps, note = chosen[0]
        text = note + report(eps, judge or None)
        if len(chosen) > 1:
            text += "\n\n" + _replication(chosen)
        return text + _language_flips(chosen)
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
                         dec: list[dict], no_variance: bool = False) -> list[str]:
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
    if no_variance:
        # Both kappas are zero by construction, so their difference is too, and
        # "indistinguishable" would be read as a result. Raw agreement is not
        # undefined, and here it can be far apart.
        text = (f"\nWith no variance in the human labels both kappas are zero "
                f"by construction, so their difference says nothing. Raw "
                f"agreement does: the single score matches the human verdict on "
                f"{an.p_o:.0%} of transcripts, the nine-criterion judge on "
                f"{ad.p_o:.0%}.")
        if harsh and not lenient:
            text += (" Every one of the nine-criterion judge's disagreements goes "
                     "the harsh way — the *harsh* column above shows which "
                     "criterion. These judges saw the transcript without the tool "
                     "results (WHAT_FAILED #23); how they do with them is under "
                     "*Judges against the verifier*.")
        out.append(text)
        return out
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


_KEPT_SAMPLES: set[str] = set()


def _sample_verdict(rec: dict) -> bool | None:
    """A labelled transcript's verdict, under the report's checker: re-scored
    from its trace when that is today's checks and the trace is at hand. One
    that is not keeps its recorded verdict, and is counted (_KEPT_SAMPLES)."""
    from pasarbench.diagnose import CHECKER
    if not rec:
        return None
    if CHECKER == "current":
        if rec.get("path") and Path(rec["path"]).is_file():
            from pasarbench.rescore import rescore_file
            return rescore_file(rec["path"]).verdict
        _KEPT_SAMPLES.add(rec.get("transcript_id") or rec.get("path") or "?")
    return rec.get("passed")


def section_planted(path: Path = Path("data/judge_cal/report.json")) -> str:
    """The planted-mistake calibration (RUNBOOK, planted mistakes): one mistake
    planted per copy of a transcript the verifier passed, so the answers are
    known. Per criterion, how often the rubric judge caught the plant and how
    often it flagged the clean originals, against the bar fixed before the run;
    and how often the 1-5 judge still accepted a planted copy."""
    import math
    s = _load_json(path)
    head = "\n\n### Planted mistakes\n\n"
    if not s or "criteria" not in s:
        return head + _missing(
            "the planted-mistake calibration has not been run",
            "python scripts/judge_calibration.py build \"traces/N-off/full+search-300\" "
            "\"traces/M-off/full+search-300\"\n"
            "python scripts/judge_calibration.py run --workers 12   # the judge's API key\n"
            "python scripts/judge_calibration.py report")

    def rate(k: int, n: int) -> str:
        if not n:
            return "—"
        z, p = 1.96, k / n
        d = 1 + z * z / n
        mid, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n
                                                            + z * z / (4 * n * n)) / d
        return f"{k}/{n} ({p:.0%}; {max(0, mid - half):.0%}–{min(1, mid + half):.0%})"
    out = [head + f"*{s['judged']} of {s['items']} items judged, {s['originals']} clean "
           f"originals. Caught: the plant flagged, among the pairs whose original was not. "
           f"False alarms: the originals flagged (`language_match`: those in the other "
           f"languages). A criterion is left to the judge at caught ≥ 80% and false alarms "
           f"≤ 10%, the bar fixed before the run.*\n",
           "| criterion | caught | false alarms | left to the judge | 1–5 judge accepts "
           "original → planted |", "|---|---|---|---|---|"]
    kept = []
    for key, r in s["criteria"].items():
        k, n = ((r["false_alarms_other_languages"], r["other_languages"])
                if key == "language_match" else (r["false_alarms"], s["originals"]))
        naive = (f"{r['naive_accepted_original']}/{r['naive_pairs']} → "
                 f"{r['naive_accepted_planted']}/{r['naive_pairs']}" if r.get("naive_pairs")
                 else "—")
        out.append(f"| `{key}`{' (critical)' if r.get('critical') else ''} | "
                   f"{rate(r['caught'], r['clean_pairs'])} | {rate(k, n)} | "
                   f"{'**yes**' if r.get('trusted') else 'no'} | {naive} |")
        if r.get("trusted"):
            kept.append(key)
    out.append(f"\n{len(kept)} of {len(s['criteria'])} criteria can be left to the judge"
               + (f": {', '.join('`' + k + '`' for k in kept)}." if kept else ".")
               + " Every other one is reported with its numbers and scores nothing. The "
               "originals it flagged are listed in `data/judge_cal/report.txt`: each is a "
               "false alarm or a violation the database cannot see, and they are read "
               "(RUNBOOK, planted mistakes).")
    return "\n".join(out)


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
    # Round 2 is a retest or a second rater depending on who labelled it; the
    # ceiling line below says which, so the header does not guess.
    out = [f"Human labels: round 1 = {len(r1)}, round 2 = {len(r2)}\n"]

    # Before any agreement number: is there anything to agree ABOUT? A kappa
    # needs both raters to say "violated" sometimes. When the human labels
    # almost never do, every statistic below is undefined, and a table of
    # zeros would read as "the judge disagrees" when it means "nothing varied".
    n_j = sum(1 for r in r1 for k in keys if r.get("labels", {}).get(k) is not None)
    n_v = sum(1 for r in r1 for k in keys if r.get("labels", {}).get(k) is False)
    no_variance = bool(n_j) and n_v / n_j < 0.01
    if no_variance:
        sample = {x["transcript_id"]: x for x in _load_jsonl(root / "sample.jsonl")}
        _KEPT_SAMPLES.clear()
        failed = [r for r in r1 if _sample_verdict(sample.get(r["transcript_id"], {})) is False]
        ok_failed = sum(1 for r in failed if not any(
            r["labels"].get(k) is False for k in keys))
        out.append(
            f"> **These labels cannot calibrate a judge.** {n_v} of {n_j} human "
            f"judgments are 'violated' ({n_v / n_j:.2%}). With almost no variance "
            f"every agreement statistic below is undefined: each KAPPA PARADOX "
            f"means *nothing to agree about*, not *the judge disagrees*."
            + (f" {ok_failed} of the {len(failed)} sampled transcripts that FAILED "
               f"the verifier were rated clean on every criterion — this rubric "
               f"scores what the agent said, and these agents fail in what they "
               f"do." if failed else "")
            + " See WHAT_FAILED #24; the judges are evaluated against the "
              "database under *Judges against the verifier*."
            + (f" ({len(_KEPT_SAMPLES)} sampled transcript(s) have no trace on disk "
               f"and keep the verdict recorded at sampling.)" if _KEPT_SAMPLES else "")
            + "\n")

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
        out.extend(_naive_vs_decomposed(r1, naive, dec, no_variance))
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

        s1, s2 = _strictness(r1, keys), _strictness(r2, keys)
        if no_variance:
            # A ceiling is a kappa between two raters; with nothing violated
            # there is no kappa, so no ceiling and no fraction of one. Printing
            # "0% of achievable agreement" here would read as a finding about
            # the judge when it is a fact about the labels.
            rates = (f" Round 1 rated {s1:.1%} of judgments satisfied and round 2 "
                     f"{s2:.1%}." if s1 is not None and s2 is not None else "")
            out.append(f"\n**Ceiling — {kind}: undefined.**{rates} Two raters "
                       f"who almost never say 'violated' have nothing to agree "
                       f"or disagree about (see the notice above), so there is "
                       f"no ceiling to measure a judge against.")
            return "\n".join(out)

        out.append(f"\n**Ceiling — {kind}:** mean human-human kappa "
                   f"{c['mean_self_kappa']:+.3f}, mean judge kappa "
                   f"{c['mean_judge_kappa']:+.3f}, "
                   f"**{(c['mean_fraction_of_ceiling'] or 0):.0%} of achievable "
                   f"agreement**")

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
                    elif min(rates.values()) >= 0.99:
                        # Flat at the top is not reassurance: a rater who
                        # accepted every default looks exactly like this.
                        out.append(f"\n> Round 2 rated nearly every judgment "
                                   f"satisfied in every language, so the spread "
                                   f"({spread:.0%}) cannot tell a careful second "
                                   f"rater from one who accepted every default.")
                    else:
                        out.append(f"\n> Spread across languages is "
                                   f"{spread:.0%} — no sign of a language the "
                                   f"second rater waved through.")
    else:
        out.append("\n**Ceiling:** " + MISSING +
                   " — run `annotate --round 2 --retest 30` blind, either a "
                   "week later or by a second labeller")
    return "\n".join(out)


def _salted_payloads(runs: dict, traces: Path) -> str:
    """#33: v18's generated tracking numbers came from Python's salted hash, so
    no replay can rebuild a shipment lookup from a run before v19. A judge file
    built by the length-checking replay showed the judge a number the agent
    never saw; one built since withholds that result and shows the call's
    arguments. Its rows say which: a withheld result is not among the
    verified ones in `payload_calls`. Counted from the traces, not left to
    prose."""
    import json

    from pasarbench.harness.replay import replay
    from pasarbench.rescore import all_tasks
    tasks, out = all_tasks(), []
    for cell, conds in sorted(runs.items()):
        views = {c: rows for c, rows in conds.items() if "tool results" in c}
        if not views:
            continue
        run, _, arm = cell.partition("__")
        hit, n = set(), 0
        for f in sorted((traces / run / arm).glob("*.jsonl")):
            recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            task = tasks.get(next((r for r in recs if r.get("type") == "header"),
                                  {}).get("task_id"))
            if task is None:
                continue
            n += 1
            if replay(recs, task).unrecoverable:
                hit.add(f.stem)
        if not hit:
            continue

        def shown(r: dict) -> bool:
            v, t = r.get("payload_calls") or (0, 0)
            return not r.get("payload_calls") or v >= t
        wrong = sorted(c for c, rows in views.items()
                       if any(r["transcript_id"] in hit and shown(r) for r in rows))
        if wrong:
            out.append(f"> **`{cell}`, {', '.join(f'*{c}*' for c in wrong)}:** "
                       f"{len(hit)} of {n} episodes looked up a generated shipment, "
                       f"and the replay that built these views showed the judge a "
                       f"tracking number the agent never saw -- v18 drew them from "
                       f"Python's salted hash, and the replay checked only lengths "
                       f"(WHAT_FAILED #33). A judge that compared the numbers compared "
                       f"them against the wrong one. Re-running `scripts/run_judges.py "
                       f"--payloads` withholds those payloads instead.\n")
        else:
            out.append(f"> **`{cell}`, views with tool results:** {len(hit)} of {n} "
                       f"episodes looked up a generated shipment, whose result no replay "
                       f"can rebuild for a run before v19 (WHAT_FAILED #33). These views "
                       f"withhold it -- the judges saw that call's arguments only -- so "
                       f"what the agent said about the shipment has no tool result "
                       f"behind it, and a strict judge may take it for invention.\n")
    return "\n".join(out) + ("\n" if out else "")


def section_judge_vs_verifier(traces: Path = Path("traces")) -> str:
    """The same judges scored against the database instead of human labels,
    under each view of the evidence they were given (run_judges.py --traces).

    The judge files carry the verifier's verdict as recorded when the judges
    ran. Under today's checker the ground truth is re-scored like every other
    number, and the section says how many verdicts that moved."""
    from pasarbench.diagnose import CHECKER
    from pasarbench.judge.vs_verifier import load_runs, markdown
    runs = load_runs(Path("data/judge_vs_verifier"))
    if not runs:
        return ""                     # optional experiment; absence is not a gap
    note = ""
    if CHECKER == "current":
        from pasarbench.rescore import rescore_dir
        moved, total, kept = 0, 0, 0
        for cell, conds in runs.items():
            run, _, arm = cell.partition("__")
            d = traces / run / arm
            now = {r.transcript_id: r.verdict for r in rescore_dir(d)} if d.is_dir() else {}
            for rows in conds.values():
                for r in rows:
                    # Files from before v19 hold only the recorded verdict;
                    # later ones hold both (scripts/run_judges.py).
                    r.setdefault("verifier_passed_recorded", r["verifier_passed"])
                    r["verifier_passed"] = now.get(r["transcript_id"], r["verifier_passed"])
            rows = next(iter(conds.values()))
            total += len(rows)
            moved += sum(r["verifier_passed"] != r["verifier_passed_recorded"] for r in rows)
            kept += sum(r["transcript_id"] not in now for r in rows)
        note = (f"*Ground truth is today's checks: {moved} of the {total} episodes' "
                f"verdicts differ from the ones the runs recorded (section 7). The "
                f"judges' answers are the same either way; only what they are scored "
                f"against changes."
                + (f" {kept} episode(s) have no trace on disk and keep the recorded "
                   f"verdict." if kept else "") + "*\n\n")
    else:
        for conds in runs.values():
            for rows in conds.values():
                for r in rows:
                    if "verifier_passed_recorded" in r:
                        r["verifier_passed"] = r["verifier_passed_recorded"]
    note += _salted_payloads(runs, traces)
    return ("\n\n### Judges against the verifier\n\n"
            "Human labels measure whether a judge reads a transcript the way a "
            "person does. This measures whether it can tell a solved case from "
            "an unsolved one, with the database as ground truth, and how that "
            "depends on what it is shown.\n\n" + note + markdown(runs))


def section_serving() -> str:
    from pasarbench.serving.cost import ServingConfig, compare, markdown
    cfgs = _load_json(Path("data/serving/configs.json"))
    if not cfgs:
        return _missing(
            "serving configurations not measured",
            "./scripts/serve_sglang.sh agent\n"
            "# snapshot /metrics, run the sweep, snapshot again, then write\n"
            "# data/serving/configs.json as a list of ServingConfig kwargs")
    return markdown(compare([ServingConfig(**c) for c in cfgs]))


REFERENCE_CMD = (
    "python -m pasarbench.sweep --backend openai --model deepseek-v4-pro \\\n"
    "  --base-url https://api.deepseek.com/v1 --extra-body '{\"thinking\":{\"type\":\"disabled\"}}' \\\n"
    "  --simulator openai --sim-model RedHatAI/gemma-4-31B-it-FP8-Dynamic --sim-url http://localhost:8001/v1 --sim-extra-body '{}' \\\n"
    "  --suite all -k {k} --temperature 0 --customer {customer} --strategies full --workers 8 --run-id P-ref --resume")


def _customer(simulator: Any) -> int:
    """The simulated customer's version, from the name a run recorded:
    "llm-user/v2:..." since v31, "llm-user:..." for version 1 (WHAT_FAILED #38)."""
    name = str(simulator or "")
    if name.startswith("llm-user/v"):
        version = name[len("llm-user/v"):].split(":", 1)[0]
        if version.isdigit():
            return int(version)
    return 1


def _headers(cell: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(cell.glob("*.jsonl")):
        try:
            with f.open() as fh:
                out[f.stem] = json.loads(fh.readline())
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _passk(eps: list[dict], key: str = "passed") -> tuple[float, float, int]:
    by: dict[str, list[bool]] = {}
    for e in eps:
        by.setdefault(e["task_id"], []).append(e[key])
    k = max(len(v) for v in by.values())
    return (sum(e[key] for e in eps) / len(eps),
            sum(all(v) for v in by.values()) / len(by), k)


def _photo_claims(runs: list[tuple[str, Path, list[dict]]]) -> str:
    """For each (label, cell, episodes): the photo trap's episodes in which the
    customer, told she has no photos, says she sent some, and the failures after
    and without that line -- "" when she never did. Customer v2 does (WHAT_FAILED
    #43)."""
    from pasarbench.rescore import all_tasks
    from pasarbench.simqa import forbids_photos, photo_claim
    tasks = all_tasks()
    parts = []
    for label, cell, eps in runs:
        n = said = after = without = 0
        for e in eps:
            task = tasks.get(e["task_id"])
            if task is None or not forbids_photos(task):
                continue
            try:
                recs = [json.loads(l) for l in (cell / f"{e['transcript_id']}.jsonl")
                        .read_text(encoding="utf-8").splitlines() if l.strip()]
            except (OSError, ValueError):
                continue
            turns = [r.get("text") or "" for r in recs
                     if r.get("type") == "event" and r.get("kind") == "user_turn"]
            hit = photo_claim(task, turns) is not None
            n += 1
            said += hit
            if not e["passed"]:
                after += hit
                without += not hit
        if said:
            parts.append(f"{said} of {n} {label} episodes (failed after it {after} of "
                         f"{said}, without it {without} of {n - said})")
    return "; ".join(parts)


def _voucher_choices(runs: list[tuple[str, Path, list[dict]]]) -> str:
    """For each (label, cell, episodes): the voucher trap's episodes in which
    the customer, offered the voucher, declined it and the agent escalated --
    what P6.3 and P10 ask for, and what the trap's check fails (WHAT_FAILED
    #46) -- "" when it never happened."""
    from pasarbench.diagnose import voucher_choice
    parts = []
    for label, cell, eps in runs:
        _, n = voucher_choice(cell, eps)
        if n["declined"]:
            parts.append(f"{n['declined']} of {n['episodes']} {label} episodes ("
                         f"{n['read_as_pass']} of them failed on the voucher's lines alone)")
    return "; ".join(parts)


def _as_policy_reads(pairs: list[tuple[str, str, Path, list[dict], Path, list[dict]]]
                     ) -> list[str]:
    """For each (what, label, cell a, episodes a, cell b, episodes b): the
    paired comparison again with the voucher trap's escalations after she
    declined the voucher read as P6.3 and P10 read them (WHAT_FAILED #46) --
    only where that changes a verdict. After the fact: never the result."""
    from pasarbench.diagnose import paired_episodes, voucher_choice
    out = []
    for what, label, ca, ea, cb, eb in pairs:
        (pa, na), (pb, nb) = voucher_choice(ca, ea), voucher_choice(cb, eb)
        if not (na["read_as_pass"] or nb["read_as_pass"]):
            continue
        g = paired_episodes(pa, pb)
        out.append(f"{what}: " + (f"{g['diff']:+.3f}; {g['better']} tasks better, "
                                  f"{g['worse']} worse, p={g['p']:.3f}" if g else
                                  "fewer than 5 shared tasks — not testable")
                   + (f" ({label})" if label else ""))
    return out


def section_training(traces: Path = Path("traces"),
                     spec: str = "base=P-base,rft=P-rft,reference=P-ref",
                     stats_path: Path = Path("data/rft/stats.json")) -> str:
    """Base, the fold-swapped fine-tunes and the API reference, one table.

    Everything a reader needs to trust the RFT row is checked here rather than
    asserted in prose: the same customer and temperature in every run, the
    tasks unchanged since the split, and every held-out episode answered by
    the model that did not train on its fold. One round's runs and its
    collection's stats (round 2: P2-*, data/rft2/stats.json)."""
    from pasarbench.diagnose import load_episodes, paired_episodes
    from pasarbench.rl.split import other_fold

    runs = dict(kv.split("=", 1) for kv in spec.split(",") if "=" in kv)
    cells = {role: traces / name / "full" for role, name in runs.items()}
    eps = {role: load_episodes(c) for role, c in cells.items() if c.exists()}
    eps = {r: e for r, e in eps.items() if e}
    heads = {role: _headers(cells[role]) for role in eps}

    if "base" not in eps:
        return _missing(
            "the base model has not been run with the post-training customer",
            "# on the GPU node (docs/RUNBOOK.md, phase 4)\n"
            "bash scripts/setup_node.sh\n"
            "bash scripts/gpu_pipeline.sh smoke\n"
            "bash scripts/gpu_pipeline.sh stage1   # split, train_rft.py baseline and collect, examples")

    folds = _load_json(Path("data/splits/folds.json"))
    labels = {"base": "base", "rft": "RFT, each on its held-out fold",
              "reference": "reference"}
    out = []
    if folds:
        out.append(f"*Split `{folds['split_digest']}`: two folds by family — locale "
                   f"twins together, every trap in both. Each fine-tune is scored "
                   f"only on the fold it did not train on, so between them all "
                   f"{len(folds['task_fold'])} tasks are held out. This measures new "
                   f"worlds, markets and languages for traps the model has seen.*\n")

    out.append("| model | run | tasks | pass^1 | pass^k | tokens per resolved "
               "| vs base, paired by task |")
    out.append("|---|---|---|---|---|---|---|")
    verdicts = []
    for role in ("base", "rft", "reference"):
        if role not in eps:
            out.append(f"| {labels[role]} | `{runs.get(role, '-')}` | {MISSING} | "
                       f"{MISSING} | {MISSING} | {MISSING} | {MISSING} |")
            continue
        e = eps[role]
        models = sorted({h.get("requested_model") or "?" for h in heads[role].values()})
        p1, pk, k = _passk(e)
        resolved = sum(x["passed"] for x in e)
        tpr = sum(x["tokens"] for x in e) / resolved if resolved else float("nan")
        g = paired_episodes(eps["base"], e) if role != "base" else None
        vs = ("—" if g is None else
              f"{g['diff']:+.3f}; {g['better']} tasks better, {g['worse']} worse, "
              f"p={g['p']:.3f}")
        name = labels[role] + (f" `{models[0]}`" if len(models) == 1 else "")
        out.append(f"| {name} | `{runs[role]}` | {len({x['task_id'] for x in e})} | "
                   f"{p1:.3f} | {pk:.3f} (k={k}) | {tpr:,.0f} | {vs} |")
        if g:
            verdicts.append((role, g))

    out.append("")
    for role, g in verdicts:
        what = labels[role].split(",")[0]
        if g["resolved"]:
            out.append(f"- **{what} is {'better' if g['diff'] > 0 else 'worse'} than "
                       f"base** on held-out tasks (p={g['p']:.3f}).")
        else:
            out.append(f"- {what} vs base: **INCONCLUSIVE** — {g['better']} tasks "
                       f"better and {g['worse']} worse is within noise (p={g['p']:.3f}).")

    # The same table on the verdicts the runs recorded, when they differ: a
    # conclusion that flips between the two is a finding about the checker,
    # not the model (WHAT_FAILED #30).
    from pasarbench.diagnose import CHECKER
    moved = {role: sum(x["passed"] != x["passed_recorded"] for x in e)
             for role, e in eps.items()}
    if CHECKER == "current" and any(moved.values()):
        out.append("\n**The same runs, as recorded at run time** — the checker of the "
                   "day, before v19 corrected two of its checks (WHAT_FAILED #30):\n")
        out.append("| model | pass^1 | pass^k | vs base, paired by task | "
                   "episodes re-scored differently |")
        out.append("|---|---|---|---|---|")
        for role in ("base", "rft", "reference"):
            if role not in eps:
                continue
            e = eps[role]
            p1, pk, k = _passk(e, "passed_recorded")
            g = (paired_episodes(eps["base"], e, metric="passed_recorded")
                 if role != "base" else None)
            vs = ("—" if g is None else
                  f"{g['diff']:+.3f}; {g['better']} better, {g['worse']} worse, "
                  f"p={g['p']:.3f}")
            out.append(f"| {labels[role].split(',')[0]} | {p1:.3f} | {pk:.3f} (k={k}) | "
                       f"{vs} | {moved[role]} of {len(e)} |")
        flips = []
        for role, g in verdicts:
            was = paired_episodes(eps["base"], eps[role], metric="passed_recorded")
            if was and (was["resolved"], was["diff"] > 0) != (g["resolved"], g["diff"] > 0):
                flips.append(f"{labels[role].split(',')[0]} vs base: "
                             f"{'resolved' if was['resolved'] else 'inconclusive'} "
                             f"{was['diff']:+.3f} (p={was['p']:.3f}) recorded, "
                             f"{'resolved' if g['resolved'] else 'inconclusive'} "
                             f"{g['diff']:+.3f} (p={g['p']:.3f}) now")
        if flips:
            out.append("\n**The checker change moves a conclusion here:** "
                       + "; ".join(flips) + ". Section 7 lists the traps that moved.")
    if "rft" not in eps:
        out.append(f"\nRFT: {MISSING}\n\n```bash\n"
                   "bash scripts/gpu_pipeline.sh stage1   # if not yet run\n"
                   "# read logs/sim_audit.txt, logs/check-A.txt, data/rft/stats.json\n"
                   "bash scripts/gpu_pipeline.sh stage2   # one LoRA per fold, held-out eval\n```")
    if "reference" not in eps:
        # The base run's customer and repetitions: a reference with another
        # customer would be measuring her. And its tools: one made now runs
        # today's, and a base run under others is not the same world (#37).
        from pasarbench.tools import CHECKS
        customer = max((_customer(h.get("simulator")) for h in heads["base"].values()),
                       default=2)
        tools_base = sorted({h.get("tool_checks", 1) for h in heads["base"].values()})
        if tools_base != [CHECKS]:
            out.append(f"\nReference: {MISSING}. The base run had tools "
                       f"{', '.join(f'v{x}' for x in tools_base)} and a run made now has "
                       f"v{CHECKS} (WHAT_FAILED #37): a reference made now would not be "
                       f"compared with like. Run stage 1 again first (docs/GPU_GUIDE.md).")
        else:
            cmd = (REFERENCE_CMD.replace("{customer}", str(customer))
                   .replace("{k}", str(_passk(eps["base"])[2]))
                   .replace("--run-id P-ref", f"--run-id {runs.get('reference', 'P-ref')}"))
            out.append(f"\nReference: {MISSING}, same customer, through the API:\n\n"
                       f"```bash\n{cmd}\n```")

    # -- checks
    checks = []
    sims = {h.get("simulator") for hs in heads.values() for h in hs.values()}
    checks.append((len(sims) == 1, f"one customer in every run: "
                   f"{', '.join('`' + str(x) + '`' for x in sorted(map(str, sims)))}"))
    # The tools a run had (tools.CHECKS; absent before v27): v3 settles a
    # cancelled order's payment (WHAT_FAILED #37), so runs on either side of
    # it are not one world.
    tools_v = {h.get("tool_checks", 1) for hs in heads.values() for h in hs.values()}
    checks.append((len(tools_v) == 1, "one tool version in every run: "
                   + ", ".join(f"v{x}" for x in sorted(tools_v))))
    temps = {role: {h.get("agent_temperature") for h in hs.values()}
             for role, hs in heads.items()}
    ok_t = all(t == {0.0} for t in temps.values())
    checks.append((ok_t, "agent temperature 0 in every scored run" if ok_t else
                   f"agent temperatures differ or are unrecorded: "
                   f"{ {r: sorted(map(str, t)) for r, t in temps.items()} }"))
    ks = {role: _passk(e)[2] for role, e in eps.items()}
    checks.append((len(set(ks.values())) == 1, f"k per task: {ks}"))
    if folds:
        moved = sorted({h["task_id"] for hs in heads.values() for h in hs.values()
                        if h.get("task_digest") and folds["task_digest"].get(h["task_id"])
                        and h["task_digest"] != folds["task_digest"][h["task_id"]]})
        checks.append((not moved, "tasks unchanged since the split" if not moved else
                       f"{len(moved)} task(s) changed since the split: {moved[:5]}"))
    if "rft" in eps:
        rows = _load_json(traces / runs["rft"] / "summary.json") or [{}]
        fm = ((rows[0] or {}).get("setup") or {}).get("fold_models")
        if not folds or not fm:
            checks.append((False, "cannot confirm the held-out routing: "
                           + ("no data/splits/folds.json" if not folds
                              else "the RFT run records no --fold-models")))
        else:
            trained = dict(kv.split("=", 1) for kv in fm.split(","))
            wrong = [h["task_id"] for h in heads["rft"].values()
                     if h.get("requested_model") !=
                     trained.get(other_fold(folds["task_fold"].get(h["task_id"], "A")))]
            n = len(heads["rft"])
            checks.append((not wrong, f"{n - len(wrong)}/{n} RFT episodes answered by "
                           f"the model that did not train on that task's fold"
                           + (f" — LEAKED: {sorted(set(wrong))[:5]}" if wrong else "")))
    out.append("\nChecks:\n")
    out += [f"- {'✓ ' if ok else '✗ **'}{msg}{'' if ok else '**'}" for ok, msg in checks]

    # -- per trap
    if "rft" in eps:
        def by_trap(e):
            d: dict[str, list[bool]] = {}
            for x in e:
                d.setdefault(x["trap"], []).append(x["passed"])
            return {t: sum(v) / len(v) for t, v in d.items()}
        bt, rt = by_trap(eps["base"]), by_trap(eps["rft"])
        out.append("\n| trap | base | RFT | change |")
        out.append("|---|---|---|---|")
        for t in sorted(bt, key=lambda t: rt.get(t, 0) - bt[t]):
            if t in rt:
                out.append(f"| `{t}` | {bt[t]:.2f} | {rt[t]:.2f} | {rt[t] - bt[t]:+.2f} |")
        out.append("\n*Per-trap moves are leads to read in the traces, not results: "
                   "each trap is ~13 tasks, and only the paired test above says "
                   "whether anything moved at all.*")
        photo = _photo_claims([({"rft": "RFT"}.get(role, role), cells[role], eps[role])
                               for role in ("base", "rft", "reference") if role in eps])
        if photo:
            out.append(f"\n*The photo trap's customer is told she has no photos. She said she "
                       f"had sent them in {photo}. Where she said it, the trap's verdict "
                       f"follows her line, not the agent's policy (WHAT_FAILED #43).*")
        vouch = _voucher_choices([({"rft": "RFT"}.get(role, role), cells[role], eps[role])
                                  for role in ("base", "rft", "reference") if role in eps])
        if vouch:
            out.append(f"\n*The voucher trap's customer, offered the voucher, declined it and "
                       f"asked for her money or a human in {vouch}. P6.3 and P10 ask for the "
                       f"escalation that follows, and the trap's check fails it: there the "
                       f"verdict follows her choice (WHAT_FAILED #46).*")

    stats = _load_json(stats_path)
    if stats and "per_trap_pass_rate" in stats:
        dead = stats.get("traps_with_zero_signal") or {}
        # Which checker picked the training episodes. Before v19 it was the
        # verdict each episode recorded, and v18's checks demanded one lookup
        # tool on two traps -- which the fine-tune then learned (#30).
        how = {"current": "today's checks at build time",
               "recorded": "the verdicts the episodes recorded"}.get(
            stats.get("checker"), "the verdicts the episodes recorded, under the "
            "checker of the day (this stats.json predates `--checker`: before v19, "
            "WHAT_FAILED #30)")
        per_trap = (stats.get("selection") or {}).get("per_trap")
        out.append(f"\nCollection (`{stats.get('source')}`): {stats['episodes']} "
                   f"episodes at temperature 1.0, pass rate {stats['pass_rate']:.3f}; "
                   f"examples per fold {stats['examples']}. Passing episodes were "
                   f"picked by {how}"
                   + (f", {per_trap} a trap and fold (WHAT_FAILED #31)." if per_trap else
                      ", at most three a task."))
        for fold, traps in dead.items():
            if traps:
                out.append(f"- fold {fold} never solved {len(traps)} trap(s), so "
                           f"rft-{fold} had nothing to learn for: "
                           f"{', '.join('`' + t + '`' for t in traps)}")
    return "\n".join(out)


def _grpo_runs(traces: Path) -> list[int]:
    """The GRPO iterations evaluated so far: N for each traces/P2-grpo-iN-e2."""
    out = []
    for d in traces.glob("P2-grpo-i*-e2"):
        n = d.name[len("P2-grpo-i"):-len("-e2")]
        if n.isdigit() and any((d / "full").glob("*.jsonl")):
            out.append(int(n))
    return sorted(out)


def _verdict(g: dict | None, better: str, worse: str) -> str:
    """A paired comparison in the words the reading fixed before the run."""
    if g is None:
        return "fewer than 5 shared tasks — not testable"
    head = (better if g["diff"] > 0 else worse) if g["resolved"] else "INCONCLUSIVE"
    return (f"**{head}** — {g['diff']:+.3f}; {g['better']} tasks better, {g['worse']} worse, "
            f"p={g['p']:.3f}")


def _routed(eps_heads: dict[str, dict], cell: Path, folds: dict | None) -> tuple[bool, str]:
    """Whether every episode was answered by the adapter that did not train on
    its task's fold (the router's rule, which the newer tasks' split reuses)."""
    from pasarbench.rl.split import other_fold
    rows = _load_json(cell.parent / "summary.json") or [{}]
    fm = ((rows[0] or {}).get("setup") or {}).get("fold_models")
    if not folds or not fm:
        return False, "cannot confirm the routing: " + (
            "no split file" if not folds else "the run records no --fold-models")
    trained = dict(kv.split("=", 1) for kv in fm.split(","))
    wrong = [h["task_id"] for h in eps_heads.values()
             if h.get("requested_model") !=
             trained.get(other_fold(folds["task_fold"].get(h["task_id"], "A")))]
    n = len(eps_heads)
    return (not wrong, f"{n - len(wrong)}/{n} episodes answered by the adapter that did "
            f"not train on the task's fold" + (f" — LEAKED: {sorted(set(wrong))[:5]}"
                                              if wrong else ""))


def section_grpo(traces: Path = Path("traces"), stats_dir: Path = Path("data/grpo2/stats"),
                 e2_split: Path = Path("data/splits/folds-e2.json"),
                 split: Path = Path("data/splits/folds.json")) -> str:
    """GRPO from round 2's fine-tunes, read as RUNBOOK phase 4 fixed before the
    run: on the 62 newer tasks, GRPO against RFT (the result) and RFT against
    the base model; on the held-out fold, GRPO against RFT (the regression
    check); and each iteration's build stats (the learning curve)."""
    from pasarbench.diagnose import load_episodes, paired_episodes

    def eps(name: str) -> list[dict]:
        cell = traces / name / "full"
        return load_episodes(cell) if cell.is_dir() else []

    base = eps("P2-base-e2")
    head = "\n\n### GRPO, from round 2's fine-tunes\n\n"
    gate = ""
    if base:
        p1, pk, _ = _passk(base)
        gate = (f"Before it ran, the base model on the 62 newer tasks: pass^1 {p1:.3f}, "
                f"pass^k {pk:.3f} — " + ("at or above 0.95, so GRPO is held until the harder "
                                         "edition gives it room."
                                         if p1 >= 0.95 else "below 0.95, so GRPO runs.") + "\n\n")
    iters = _grpo_runs(traces)
    if not iters:
        return head + gate + _missing(
            "GRPO has not been evaluated yet",
            "PASAR_ROUND=2 bash scripts/gpu_pipeline.sh grpo   # or pn.run(\"grpo\", round=2)")
    n = iters[-1]
    rows = {"base": ("base", "P2-base-e2"), "rft": ("RFT", "P2-rft-e2"),
            "grpo": (f"GRPO, {n} iteration{'s' if n > 1 else ''}", f"P2-grpo-i{n}-e2")}
    e2 = {k: eps(run) for k, (_, run) in rows.items()}
    out = [head + gate + "**The 62 newer tasks**, which no model trained on; each adapter "
           "answers its own half of their split:\n",
           "| model | run | tasks | pass^1 | pass^k | tokens per resolved |",
           "|---|---|---|---|---|---|"]
    for key, (label, run) in rows.items():
        e = e2[key]
        if not e:
            out.append(f"| {label} | `{run}` | {MISSING} | | | |")
            continue
        p1, pk, k = _passk(e)
        resolved = sum(x["passed"] for x in e)
        tpr = sum(x["tokens"] for x in e) / resolved if resolved else float("nan")
        out.append(f"| {label} | `{run}` | {len({x['task_id'] for x in e})} | {p1:.3f} | "
                   f"{pk:.3f} (k={k}) | {tpr:,.0f} |")
    g_result = (paired_episodes(e2["rft"], e2["grpo"]) if e2["rft"] and e2["grpo"]
                else None)
    g_carry = (paired_episodes(e2["base"], e2["rft"]) if e2["base"] and e2["rft"]
               else None)
    out.append("")
    out.append("- **The result**, GRPO against RFT, paired by task: "
               + _verdict(g_result, "GRPO helped", "GRPO hurt") + ".")
    out.append("- **RFT on tasks it never saw**, RFT against the base model: "
               + _verdict(g_carry, "RFT carried over", "RFT hurt") + ".")
    cell = {k: traces / run / "full" for k, (_, run) in rows.items()}
    policy = _as_policy_reads(
        [(what, "", cell[a], e2[a], cell[b], e2[b])
         for what, a, b in (("GRPO against RFT", "rft", "grpo"),
                            ("RFT against the base model", "base", "rft"))
         if e2[a] and e2[b]])
    if policy:
        from pasarbench.diagnose import voucher_choice
        line = ("- *After the fact, beside the result and never instead of it* "
                "(WHAT_FAILED #46): with each voucher-trap escalation after the customer "
                "declined the voucher read as P6.3 and P10 read it, " + "; ".join(policy))
        if base:
            rb, _ = voucher_choice(cell["base"], base)
            q1, _, _ = _passk(rb)
            line += (f"; the base model's pass^1 {q1:.3f}"
                     + (" — at or above the 0.95 that holds GRPO" if q1 >= 0.95 else ""))
        out.append(line + ".")
    vouch = _voucher_choices([(label, cell[k], e2[k]) for k, (label, _) in rows.items()
                              if e2[k]])
    if vouch:
        out.append(f"\n*On the voucher trap the customer, offered the voucher, declined it and "
                   f"asked for her money or a human in {vouch}; P6.3 and P10 ask for the "
                   f"escalation that follows, and the trap's check fails it (WHAT_FAILED "
                   f"#46).*")

    present = [k for k in rows if e2[k]]
    if len(present) > 1:
        def by_trap(e):
            d: dict[str, list[bool]] = {}
            for x in e:
                d.setdefault(x["trap"], []).append(x["passed"])
            return {t: sum(v) / len(v) for t, v in d.items()}
        rates = {k: by_trap(e2[k]) for k in present}
        out.append("\n| trap | " + " | ".join(rows[k][0] for k in present) + " |")
        out.append("|---|" + "---|" * len(present))
        for t in sorted(set().union(*(set(r) for r in rates.values()))):
            out.append(f"| `{t}` | " + " | ".join(
                f"{rates[k][t]:.2f}" if t in rates[k] else "—" for k in present) + " |")

    held = {"rft": eps("P2-rft"), "grpo": eps(f"P2-grpo-i{n}")}
    out.append("\n**The held-out fold** (the regression check: at 0.96 INCONCLUSIVE is the "
               "expected outcome; a trap that drops is read failure by failure, "
               "`scripts/read_failures.py traces/P2-rft/full traces/P2-grpo-i{n}/full`):\n"
               .replace("{n}", str(n)))
    if held["rft"] and held["grpo"]:
        a1, ak, _ = _passk(held["rft"])
        b1, bk, _ = _passk(held["grpo"])
        out.append(f"RFT {a1:.3f} (pass^k {ak:.3f}), GRPO {b1:.3f} (pass^k {bk:.3f}): "
                   + _verdict(paired_episodes(held["rft"], held["grpo"]),
                              "GRPO better", "GRPO worse") + ".")
        ra = {}
        for x in held["rft"]:
            ra.setdefault(x["trap"], []).append(x["passed"])
        rb = {}
        for x in held["grpo"]:
            rb.setdefault(x["trap"], []).append(x["passed"])
        moved = sorted((sum(rb[t]) / len(rb[t]) - sum(ra[t]) / len(ra[t]), t)
                       for t in ra if t in rb
                       and abs(sum(rb[t]) / len(rb[t]) - sum(ra[t]) / len(ra[t])) > 1e-9)
        if moved:
            out.append("Traps that moved: " + ", ".join(
                f"`{t}` {sum(ra[t]) / len(ra[t]):.2f} → {sum(rb[t]) / len(rb[t]):.2f}"
                for _, t in moved) + ".")
        held_cells = (traces / "P2-rft" / "full", traces / f"P2-grpo-i{n}" / "full")
        policy = _as_policy_reads([("GRPO against RFT", "", held_cells[0], held["rft"],
                                    held_cells[1], held["grpo"])])
        if policy:
            out.append("*After the fact* (WHAT_FAILED #46), the voucher trap's escalations "
                       "after she declined the voucher read as P6.3 and P10 read them: "
                       + policy[0] + ".")
        photo = _photo_claims([("RFT", held_cells[0], held["rft"]),
                               ("GRPO", held_cells[1], held["grpo"])])
        if photo:
            out.append(f"\n*On the photo trap the customer said she had sent photos she was "
                       f"told she does not have in {photo}; where she did, the verdict "
                       f"follows her line, not the agent's policy (WHAT_FAILED #43).*")
        vouch = _voucher_choices([("RFT", held_cells[0], held["rft"]),
                                  ("GRPO", held_cells[1], held["grpo"])])
        if vouch:
            out.append(f"\n*On the voucher trap the customer, offered the voucher, declined it "
                       f"and asked for her money or a human in {vouch}; P6.3 and P10 ask for "
                       f"the escalation that follows, and the trap's check fails it "
                       f"(WHAT_FAILED #46).*")
    else:
        out.append(MISSING + " — `P2-rft` and `P2-grpo-i" + str(n) + "` are both needed.")

    # Every point built: iteration i's rollouts are the adapter it started
    # from, so with one more than the iterations evaluated, the last GRPO
    # adapters measured on their own tasks and not trained on (grpo-measure).
    points = sorted({int(p.name[1:].split("-")[0]) for p in stats_dir.glob("i*-*.stats.json")
                     if p.name[1:].split("-")[0].isdigit()}) if stats_dir.is_dir() else []
    curve, counted, undated = [], False, False
    for i in points:
        for fold in ("A", "B"):
            st = _load_json(stats_dir / f"i{i}-{fold}.stats.json")
            if st:
                who = str(st.get("model") or "?").split(":")[-1]
                vd = st.get("voucher_declined")
                counted |= bool(vd)
                undated |= vd is None
                curve.append(f"| {i}{' (measured)' if i > n else ''} | {fold} | `{who}` | "
                             f"{st.get('episodes', 0)} on {st.get('groups', 0)} "
                             f"tasks | {st.get('pass_rate', 0):.0%} | "
                             f"{'—' if vd is None else vd} | "
                             f"{st.get('degenerate_rate', 0):.0%} | "
                             f"{0 if i > n else st.get('examples', 0)} |")
    tasks_of = {f: _load_json(stats_dir / f"tasks-{f}.json") or {} for f in ("A", "B")}
    if curve or any(tasks_of.values()):
        out.append("\n**The learning curve**: the policy at temperature 1 on its own "
                   "training tasks, each iteration's rollouts (iteration 1: the RFT "
                   "adapter's). Rising here and flat on the newer tasks is learning the "
                   "training tasks without carrying it over.\n")
        for f, d in tasks_of.items():
            if d:
                left = d.get("skipped_tasks") or []
                out.append(f"- fold {f}: {len(d.get('tasks', []))} tasks"
                           + (f"; {len(left)} of {', '.join(d.get('skipped_traps', []))} left "
                              f"out (WHAT_FAILED #43)" if left else ""))
        if curve:
            out.append("\n| iteration | fold | rolled out by | rollouts | passed (checks) | "
                       "she declined the voucher | groups with nothing to learn | "
                       "turns trained |")
            out.append("|---|---|---|---|---|---|---|---|")
            out += curve
            notes = []
            if counted:
                notes.append("\"She declined the voucher\": rollouts the checks failed where "
                             "P6.3 and P10 ask for the escalation (WHAT_FAILED #46), counted "
                             "whichever way the reward read them")
            if undated:
                notes.append("— built before the build counted the voucher she declined (v52)")
            if notes:
                out.append("\n" + "; ".join(notes) + ".")
        if any(i > n for i in points):
            m = max(points)
            out.append(f"\nPoint {m} is GRPO's last adapters on the same tasks, measured and "
                       f"not trained on (`grpo-measure`); `logs/r2-grpo-measure-i{m}.txt` pairs "
                       f"it with point {m - 1} task by task, as its reading was fixed before it "
                       f"ran (RUNBOOK phase 4, GRPO, the learning curve's second point). The "
                       f"pairing counts every rollout as the checks score it; the rates here "
                       f"leave out the few the harness stopped.")

    checks = []
    heads = {name: _headers(traces / name / "full")
             for name in ("P2-base-e2", "P2-rft-e2", f"P2-grpo-i{n}-e2", "P2-rft",
                          f"P2-grpo-i{n}") if (traces / name / "full").is_dir()}
    sims = {h.get("simulator") for hs in heads.values() for h in hs.values()}
    checks.append((len(sims) == 1, "one customer in every run: "
                   + ", ".join(f"`{x}`" for x in sorted(map(str, sims)))))
    temps = {h.get("agent_temperature") for hs in heads.values() for h in hs.values()}
    checks.append((temps == {0.0}, "agent temperature 0 in every evaluation" if temps == {0.0}
                   else f"agent temperatures differ or are unrecorded: {sorted(map(str, temps))}"))
    folds_e2, folds = _load_json(e2_split), _load_json(split)
    for name, f in ((f"P2-grpo-i{n}-e2", folds_e2), ("P2-rft-e2", folds_e2),
                    (f"P2-grpo-i{n}", folds)):
        if name in heads:
            ok, msg = _routed(heads[name], traces / name / "full", f)
            checks.append((ok, f"`{name}`: {msg}"))
    out.append("\nChecks:\n")
    out += [f"- {'✓ ' if ok else '✗ **'}{msg}{'' if ok else '**'}" for ok, msg in checks]
    return "\n".join(out)


def section_training_rounds(traces: Path = Path("traces"),
                            spec1: str = "base=P-base,rft=P-rft,reference=P-ref",
                            spec2: str = "base=P2-base,rft=P2-rft,reference=P2-ref") -> str:
    """Round 2 first once its base model has run, round 1 after it as context:
    round 1 had another simulated customer and older tools, so the two are
    never paired (RUNBOOK phase 4, round 2)."""
    runs2 = dict(kv.split("=", 1) for kv in spec2.split(",") if "=" in kv)
    base2 = traces / runs2.get("base", "P2-base") / "full"
    one = section_training(traces, spec1, Path("data/rft/stats.json"))
    if not base2.is_dir() or not any(base2.glob("*.jsonl")):
        return one
    two = section_training(traces, spec2, Path("data/rft2/stats.json")) + section_grpo(traces)
    return ("### Round 2: training data balanced by trap, today's customer and tools\n\n"
            + two + "\n\n### Round 1, as context\n\n*Another simulated customer and older "
            "tools than round 2's: its numbers are never paired against round 2's.*\n\n" + one)


def section_rescoring(traces: Path = Path("traces")) -> str:
    """Every run, under the verdicts it recorded and under today's checks.

    A checker change is a change to what every number means. This is where it
    is accounted for: per cell, what moved and which way, and per trap, where.
    A correction that relaxes a check can only move failures to passes, so a
    move the other way is flagged: unless a check got stricter, the replay
    rebuilt a different state than the episode ended in. A move in a trap
    whose checks did not change means the same; the per-trap table is there to
    be read against what changed -- the report cannot know that."""
    from collections import Counter, defaultdict

    from pasarbench.rescore import cells, rescore_dir, summary
    if not traces.is_dir():
        return _missing("no traces/ to re-score", "python scripts/rescore.py")
    rows, traps, kept = [], defaultdict(Counter), Counter()
    for cell in cells(traces):
        rs = rescore_dir(cell)
        sm = summary(rs)
        if not sm["episodes"]:
            continue
        rows.append((str(cell.relative_to(traces)), sm))
        for r in rs:
            if r.moved:
                traps[r.trap][r.moved] += 1
            if r.recorded is not None and r.current is None:
                kept[r.status] += 1
    if not rows:
        return _missing("no episodes under traces/", "python scripts/rescore.py")
    out = ["Each episode's recorded tool calls are replayed against the world it ran "
           "in, and today's checks score the rebuilt state (`python scripts/rescore.py`; "
           "no model is called). An episode whose replay does not reproduce its "
           "recording keeps its recorded verdict.\n",
           "| run | episodes | pass^1 recorded → now | pass^k recorded → now | "
           "fail → pass | pass → fail | kept as recorded |",
           "|---|---|---|---|---|---|---|"]
    for name, sm in rows:
        out.append(f"| `{name}` | {sm['episodes']} | {sm['recorded']:.3f} → "
                   f"{sm['current']:.3f} | {sm['recorded_k']:.3f} → {sm['current_k']:.3f} | "
                   f"{sm['fail_to_pass']} | {sm['pass_to_fail']} | {sm['not_rescorable']} |")
    if traps:
        out.append("\n| trap | fail → pass | pass → fail |")
        out.append("|---|---|---|")
        for trap, c in sorted(traps.items(), key=lambda kv: -sum(kv[1].values())):
            out.append(f"| `{trap}` | {c['fail→pass']} | {c['pass→fail']} |")
    else:
        out.append("\nNothing moved: today's checks give every episode the verdict it "
                   "recorded.")
    worse = sum(c["pass→fail"] for c in traps.values())
    if worse:
        out.append(f"\n> **{worse} episode(s) moved from pass to fail.** If no check "
                   f"got stricter, the replay rebuilt a different state than the run "
                   f"ended in: read them with `python scripts/rescore.py --changed`.")
    if kept:
        why = {"world_changed": "a payload differs from the recording (a world older "
                                "than the replay can rebuild)",
               "diverged": "a call succeeds on replay where it failed in the run, or "
                           "the reverse",
               "task_changed": "the task's opening or facts changed since the run",
               "unknown_task": "the task is no longer in the suite"}
        out.append("\nKept as recorded: " + "; ".join(
            f"{n} because {why.get(k, k)}" for k, n in kept.most_common()) + ".")
    return "\n".join(out)


# --------------------------------------------------------------------------

TEMPLATE = """# PasarBench — results

Generated by `scripts/make_report.py`. Every number below comes from a file on
disk. Anything marked {missing} has not been run, and the command to produce it
is shown inline. **Do not fill these in by hand** — re-run and regenerate.

{scoring}

## The suite

{suite}
{noise}
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

## 7. Re-scoring: recorded verdicts and today's checks

{rescoring}

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
    ap.add_argument("--noise-pair", default="",
                    help="two cells YOU know ran the same configuration, e.g. "
                         "H-context/full,I-tools2/full+all-20 -- reported as the "
                         "run-to-run noise floor every other comparison sits on")
    ap.add_argument("--context-run", default="",
                    help="run under traces/ for section 1 (default: largest)")
    ap.add_argument("--tools-run", default="",
                    help="run under traces/ for section 2 (default: largest)")
    ap.add_argument("--multilingual-run", default="",
                    help="run, or run/cell, for section 3 (default: largest). "
                         "Several, comma-separated, adds a replication table")
    ap.add_argument("--post-training", default="base=P-base,rft=P-rft,reference=P-ref",
                    help="runs for section 6: base, fold-swapped RFT, API reference")
    ap.add_argument("--post-training-2", default="base=P2-base,rft=P2-rft,reference=P2-ref",
                    help="round 2's runs for section 6, shown first once its base run "
                         "exists, with round 1 after it as context")
    ap.add_argument("--checker", choices=["current", "recorded"], default="current",
                    help="current: every verdict re-scored by today's checks from the "
                         "recorded tool calls (default); recorded: as each run scored "
                         "it. Section 7 compares the two either way")
    args = ap.parse_args()

    from pasarbench.diagnose import use_checker
    use_checker(args.checker)
    scoring = ("**Scoring:** today's checks, re-scored from each episode's recorded "
               "tool calls — the runs were scored by the checker of their day, and "
               "v19 corrected two of its checks (WHAT_FAILED #30). Section 7 shows "
               "what that moved; `--checker recorded` rebuilds this report on the "
               "verdicts as recorded."
               if args.checker == "current" else
               "**Scoring:** the verdicts each run recorded, by the checker of its "
               "day. Section 7 shows what today's checks would change; the default "
               "`--checker current` uses them throughout.")

    runs = sorted(p for p in Path(args.traces).glob("*") if p.is_dir()) \
        if Path(args.traces).exists() else []
    judge = _load_jsonl(Path("data/labels/judge_decomposed.jsonl"))

    sections = {
        "context": ("1. Context ablation",
                    section_context(runs, args.context_run)),
        "tools": ("2. Tool scaling", section_tools(runs, args.tools_run)),
        "multilingual": ("3. Multilingual diagnosis",
                         section_multilingual(runs, judge, args.multilingual_run)),
        "judge": ("4. Judge calibration",
                  section_judge() + section_planted()
                  + section_judge_vs_verifier(Path(args.traces))),
        "serving": ("5. Serving and cost", section_serving()),
        "training": ("6. Post-training",
                     section_training_rounds(Path(args.traces), args.post_training,
                                             args.post_training_2)),
        "rescoring": ("7. Re-scoring", section_rescoring(Path(args.traces))),
    }
    body = TEMPLATE.format(missing=MISSING, suite=section_suite(), scoring=scoring,
                           noise=section_noise(Path(args.traces), args.noise_pair),
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
