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


def section_context(runs: list[Path]) -> str:
    from pasarbench.analyze import markdown_report
    for run in runs:
        rows = _load_json(run / "summary.json")
        if rows and len(rows) > 1 and any(r.get("exposure", "default") == "default"
                                          for r in rows):
            try:
                return markdown_report(rows, baseline=rows[0]["strategy"])
            except Exception:                            # noqa: BLE001
                continue
    return _missing(
        "context ablation not run against a real model",
        "python -m pasarbench.sweep --backend openai --model <m> --suite all \\\n"
        "    --strategies full,window8,window4,trim3,notes4 --sample 2 -k 3")


def section_tools(runs: list[Path]) -> str:
    from pasarbench.diagnose import tool_scaling_report
    for run in runs:
        rows = _load_json(run / "summary.json")
        if rows and any(r.get("exposure", "default") != "default" for r in rows):
            return tool_scaling_report(rows)
    return _missing(
        "tool-scaling arms not run",
        "python -m pasarbench.sweep --backend openai --model <m> --suite all \\\n"
        "    --strategies full --exposure oracle,all-20,all-100,random-100,search-300")


def section_multilingual(runs: list[Path], judge: list[dict]) -> str:
    from pasarbench.diagnose import load_episodes, report
    for run in runs:
        for cell in sorted(run.iterdir()):
            if not cell.is_dir():
                continue
            eps = load_episodes(cell)
            if eps and len({e["language"] for e in eps}) > 1:
                return report(eps, judge or None)
    return _missing(
        "no multi-language run found in traces/",
        "python -m pasarbench.sweep --backend openai --model <m> --suite all -k 3")


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
        out.append("| criterion | p_o | kappa | CI95 | prevalence | reading |")
        out.append("|---|---|---|---|---|---|")
        for key, a in per_criterion(r1, dec, keys).items():
            ci = f"[{a.ci95[0]:.2f}, {a.ci95[1]:.2f}]" if a.ci95 else "-"
            out.append(f"| `{key}` | {a.p_o:.3f} | {a.kappa:+.3f} | {ci} | "
                       f"{a.prevalence:.2f} | {a.note.split(':')[0][:48]} |")
    else:
        out.append(_missing("decomposed judge not run against the labels",
                            "# score the labelled sample with DecomposedJudge, "
                            "write data/labels/judge_decomposed.jsonl"))

    if naive and dec:
        n_k = per_criterion(r1, naive, ["overall_acceptable"])
        if n_k:
            out.append(f"\nNaive 1-5 judge, collapsed: "
                       f"kappa = {list(n_k.values())[0].kappa:+.3f}")
    else:
        out.append("\nNaive baseline: " + MISSING +
                   " — without it the decomposed judge has nothing to beat")

    if r2:
        c = ceiling_report(r1, r2, dec or r1, keys)
        out.append(f"\n**Ceiling:** mean self-kappa {c['mean_self_kappa']:+.3f}, "
                   f"mean judge kappa {c['mean_judge_kappa']:+.3f}, "
                   f"**{(c['mean_fraction_of_ceiling'] or 0):.0%} of achievable "
                   f"agreement**")
    else:
        out.append("\n**Ceiling:** " + MISSING +
                   " — run `annotate --round 2 --retest 30` a week later, blind")
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
    args = ap.parse_args()

    runs = sorted(p for p in Path(args.traces).glob("*") if p.is_dir()) \
        if Path(args.traces).exists() else []
    judge = _load_jsonl(Path("data/labels/judge_decomposed.jsonl"))

    body = TEMPLATE.format(
        missing=MISSING,
        suite=section_suite(),
        context=section_context(runs),
        tools=section_tools(runs),
        multilingual=section_multilingual(runs, judge),
        judge=section_judge(),
        serving=section_serving(),
        training=section_training(),
    )
    Path(args.out).write_text(body)

    n_missing = body.count(MISSING)
    print(f"wrote {args.out}")
    print(f"{n_missing} result(s) still unmeasured" if n_missing
          else "every section is backed by measured data")
    if n_missing:
        print("Each one prints the command that produces it. Run those, "
              "regenerate; never hand-fill.")


if __name__ == "__main__":
    main()
