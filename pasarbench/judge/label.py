"""
Human labelling.

This is the grind week. There is no way around it: you sit down and label
about 200 transcripts by hand, and it takes several hours. That work is the
foundation for every judge number you will ever quote, and it is the reason
almost nobody has a calibrated judge.

    python -m pasarbench.judge.label sample  --traces traces/run/full --n 200
    python -m pasarbench.judge.label annotate --round 1
    # ... a week later, on a 30-item subset ...
    python -m pasarbench.judge.label annotate --round 2 --retest 30
    python -m pasarbench.judge.label status

THREE RULES THAT DECIDE WHETHER THE LABELS ARE WORTH ANYTHING
-------------------------------------------------------------
1. LABEL BEFORE YOU LOOK AT ANY JUDGE OUTPUT. Once you have seen a model's
   answer you cannot un-see it, your labels drift toward it, and the agreement
   you measure is partly your own anchoring. Everything in this file is built
   to keep judge output out of sight until labelling is done.

2. THE RETEST ROUND MUST BE BLIND. Round 2 re-shows a subset in a different
   order and never displays your round-1 answers. If you can see what you said
   last time, test-retest kappa measures your memory, not your consistency,
   and the ceiling it produces is fake.

3. STRATIFY THE SAMPLE. Uniform sampling from a suite where most episodes pass
   gives you 180 clean transcripts and 20 interesting ones, and near-zero
   signal on the criteria that matter. Sample across trap, language, and
   pass/fail so the rare failures are represented.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .rubric import CRITERIA, blank_labels, derive_verdict, is_complete

ROOT = Path("data/labels")


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def load_transcripts(trace_dir: str | Path) -> list[dict[str, Any]]:
    """Reconstruct transcripts + metadata from trace JSONL files."""
    out = []
    for f in sorted(Path(trace_dir).glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        head = next((r for r in recs if r.get("type") == "header"), {})
        foot = next((r for r in reversed(recs) if r.get("type") == "footer"), {})
        steps = [r for r in recs if r.get("type") == "step"]
        out.append({
            "transcript_id": f.stem,
            "trap": head.get("trap", "?"),
            "language": head.get("language", "?"),
            "market": head.get("market", "?"),
            "passed": bool(foot.get("passed")),
            "stop_reason": foot.get("stop_reason"),
            "n_steps": len(steps),
            "prose": "\n".join(s.get("model_content", "") for s in steps
                               if s.get("model_content")),
            "path": str(f),
        })
    return out


def stratified(items: list[dict[str, Any]], n: int, seed: int = 0
               ) -> list[dict[str, Any]]:
    """Balance across (trap, language, passed).

    Failures are massively over-sampled relative to their base rate and that is
    correct: a criterion like `no_data_leak` is violated only in the failures,
    so a representative sample would give you almost no positives to agree or
    disagree about.
    """
    rng = random.Random(seed)
    buckets: dict[tuple, list] = defaultdict(list)
    for it in items:
        buckets[(it["trap"], it["language"], it["passed"])].append(it)

    keys = sorted(buckets, key=str)
    picked, i = [], 0
    while len(picked) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            picked.append(buckets[k].pop(rng.randrange(len(buckets[k]))))
        i += 1
    return picked[:n]


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

def _path(name: str) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT / name


def save_sample(items: list[dict[str, Any]]) -> Path:
    p = _path("sample.jsonl")
    with p.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return p


def load_sample() -> list[dict[str, Any]]:
    p = _path("sample.jsonl")
    if not p.exists():
        raise SystemExit("no sample yet -- run `label sample --traces <dir>` first")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_labels(round_no: int) -> dict[str, dict[str, Any]]:
    p = _path(f"human_round{round_no}.jsonl")
    if not p.exists():
        return {}
    return {r["transcript_id"]: r
            for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def append_label(round_no: int, record: dict[str, Any]) -> None:
    with _path(f"human_round{round_no}.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Annotating
# --------------------------------------------------------------------------

HELP = """
  y / <enter>  satisfied (also use for 'not applicable')
  n            violated
  ?            show the guidance again
  s            skip this transcript
  q            save and quit
"""


def annotate(args) -> None:
    sample = load_sample()
    done = load_labels(args.round)

    todo = [s for s in sample if s["transcript_id"] not in done]
    if args.retest:
        # Round 2: a subset, RESHUFFLED, with round-1 answers never shown.
        r1 = load_labels(1)
        pool = [s for s in sample if s["transcript_id"] in r1
                and s["transcript_id"] not in done]
        rng = random.Random(args.seed + 99)
        rng.shuffle(pool)
        todo = pool[:args.retest]

    if not todo:
        print("nothing left to label in this round")
        return

    print(f"round {args.round}: {len(todo)} to label. {HELP}")
    for i, item in enumerate(todo, 1):
        print("\n" + "=" * 72)
        print(f"[{i}/{len(todo)}]  {item['transcript_id']}   trap={item['trap']}  "
              f"lang={item['language']}")
        # Deliberately NOT shown: whether the task passed, and any judge output.
        # Both would anchor the labels.
        print("=" * 72)
        print(Path(item["path"]).exists() and _render(item["path"]) or item["prose"])
        print("-" * 72)

        labels, quit_now = blank_labels(), False
        for c in CRITERIA:
            while True:
                tag = " [CRITICAL]" if c.critical else ""
                ans = input(f"  {c.key}{tag}: {c.question}\n  > ").strip().lower()
                if ans in ("", "y"):
                    labels[c.key] = True
                    break
                if ans == "n":
                    labels[c.key] = False
                    break
                if ans == "?":
                    print(f"    {c.guidance}")
                    continue
                if ans == "s":
                    labels = None
                    break
                if ans == "q":
                    quit_now = True
                    break
                print("    y / n / ? / s / q")
            if labels is None or quit_now:
                break

        if quit_now:
            print("saved. resume with the same command.")
            return
        if labels is None:
            continue

        note = input("  note (optional) > ").strip()
        append_label(args.round, {
            "transcript_id": item["transcript_id"],
            "trap": item["trap"], "language": item["language"],
            "labels": labels, "verdict": derive_verdict(labels),
            "note": note, "labeller": args.labeller, "round": args.round,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"  -> {derive_verdict(labels)}")

    print("\nround complete.")


def _render(path: str) -> str:
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    out = []
    left = False
    for r in recs:
        if r.get("type") == "step":
            if r.get("closing") and not left:
                out.append("--- the customer has left: closing check, which the customer "
                           "never sees ---")
                left = True
            if r.get("model_content"):
                tag = ("AGENT (held back by the claim guardrail, never sent)"
                       if r.get("guardrail") else
                       "AGENT (case note, never sent)" if r.get("closing") else "AGENT")
                out.append(f"{tag}: {r['model_content']}")
            for tc in r.get("tool_calls", []):
                out.append(f"   -> {tc['name']}({json.dumps(tc['arguments'])[:200]})")
            for tr in r.get("tool_results", []):
                flag = "ok" if tr.get("ok") else f"ERR {tr.get('error')}"
                out.append(f"   <- {tr['name']}: {flag}")
        elif r.get("type") == "event" and r.get("kind") == "user_turn":
            if r.get("text"):
                out.append(f"CUSTOMER: {r['text']}")
    return "\n".join(out)


# --------------------------------------------------------------------------

def status(args) -> None:
    sample = load_sample()
    print(f"sample: {len(sample)}")
    print(f"  traps      {len({s['trap'] for s in sample})}")
    print(f"  languages  {dict(Counter(s['language'] for s in sample))}")
    print(f"  passed     {sum(s['passed'] for s in sample)}/{len(sample)}")
    for r in (1, 2):
        lab = load_labels(r)
        if not lab:
            continue
        complete = sum(1 for v in lab.values() if is_complete(v["labels"]))
        print(f"round {r}: {len(lab)} labelled ({complete} complete)")
        print(f"  verdicts {dict(Counter(v['verdict'] for v in lab.values()))}")
        for c in CRITERIA:
            viol = sum(1 for v in lab.values() if v["labels"].get(c.key) is False)
            rate = viol / len(lab)
            warn = ("   <- near-zero variance; kappa will be unstable "
                    "(see the kappa paradox)" if rate < 0.05 else "")
            print(f"    {c.key:28s} violated {viol:3d} ({rate:.1%}){warn}")


def set_rater(args) -> None:
    """Record who labelled a round, after the fact.

    One rater labelling twice (test-retest) and two raters labelling once
    (inter-annotator) produce identical files and opposite biases in the
    ceiling, so make_report has to know which it is. `--labeller` defaults to
    "me", which means a second person who ran the plain command is recorded as
    you. This fixes that without touching any label.
    """
    p = _path(f"human_round{args.round}.jsonl")
    if not p.exists():
        raise SystemExit(f"no {p.name}")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    before = Counter(r.get("labeller", "?") for r in rows)
    backup = p.with_name(p.stem + ".before_set_rater.jsonl")
    if not backup.exists():                 # never overwrite the first backup
        backup.write_text(p.read_text())
    for r in rows:
        r["labeller"] = args.labeller
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"round {args.round}: {len(rows)} rows  {dict(before)} -> "
          f"{{'{args.labeller}': {len(rows)}}}")
    print(f"backup: {backup}")


def sample_cmd(args) -> None:
    items = load_transcripts(args.traces)
    if not items:
        raise SystemExit(f"no traces in {args.traces}")
    picked = stratified(items, args.n, args.seed)
    p = save_sample(picked)
    print(f"sampled {len(picked)} of {len(items)} -> {p}")
    print(f"  traps {len({x['trap'] for x in picked})}  "
          f"languages {dict(Counter(x['language'] for x in picked))}  "
          f"failures {sum(1 for x in picked if not x['passed'])}")
    print("\nLabel these BEFORE running any judge. Seeing judge output first "
          "anchors your labels and the agreement you measure will be partly "
          "your own anchoring.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--traces", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=sample_cmd)

    a = sub.add_parser("annotate")
    a.add_argument("--round", type=int, default=1)
    a.add_argument("--retest", type=int, default=0,
                   help="round 2: re-label N items blind, for the ceiling")
    a.add_argument("--labeller", default="me")
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=annotate)

    t = sub.add_parser("status")
    t.set_defaults(fn=status)

    r = sub.add_parser("set-rater",
                       help="record who labelled a round (e.g. a second person)")
    r.add_argument("--round", type=int, required=True)
    r.add_argument("--labeller", required=True)
    r.set_defaults(fn=set_rater)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
